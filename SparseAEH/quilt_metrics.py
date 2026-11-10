"""Recovery of the generating spatial maps and gene groups in the quilt simulation."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def score_quilt(method, truth, fitted, *, sign_free=False, true_labels=None, labels=None):
    truth, fitted = (np.asarray(x, dtype=float) for x in (truth, fitted))
    if (truth.ndim != 2 or fitted.ndim != 2 or truth.shape[0] != fitted.shape[0]
            or min(truth.shape + fitted.shape) < 1 or not np.isfinite(truth).all()
            or not np.isfinite(fitted).all()):
        raise ValueError("Expected finite, spot-aligned spot-by-pattern matrices.")
    truth = truth - truth.mean(axis=0)
    fitted = fitted - fitted.mean(axis=0)
    norms = np.linalg.norm(truth, axis=0)
    if (norms == 0).any():
        raise ValueError("A generating pattern is constant on the selected spots.")
    denom = norms[:, None] * np.linalg.norm(fitted, axis=0)
    corr = np.divide(truth.T @ fitted, denom, out=np.full(denom.shape, np.nan), where=denom > 0)
    corr = np.clip(corr, -1, 1)
    # An absent or constant fitted map contributes zero, not a perfect or omitted match.
    utility = np.nan_to_num(abs(corr) if sign_free else corr)
    rows, cols = linear_sum_assignment(-utility)
    details = pd.DataFrame({"true_pattern": np.arange(1, len(truth.T)+1),
                            "fitted_pattern": pd.array([None]*len(truth.T), dtype="Int64"),
                            "pearson_r": np.nan, "recovery_score": 0.0})
    details.loc[rows, "fitted_pattern"] = cols+1
    details.loc[rows, "pearson_r"] = corr[rows, cols]
    details.loc[rows, "recovery_score"] = utility[rows, cols]
    summary = dict(method=method, training_spots=len(truth), sign_free=sign_free,
                   mean_matched_r=details.recovery_score.mean(), genes_scored=0, gene_ari=np.nan)
    if labels is not None:
        if (not labels.index.is_unique or not true_labels.index.is_unique
                or not labels.index.isin(true_labels.index).all() or labels.isna().any()
                or len(labels) < 2):
            raise ValueError("Gene assignments must identify distinct known genes.")
        summary.update(genes_scored=len(labels),
                       gene_ari=adjusted_rand_score(true_labels.loc[labels.index], labels))
    return summary, details


def record_quilt(output, method, truth, fitted, **kwargs):
    output = Path(output)
    summary, details = score_quilt(method, truth, fitted, **kwargs)
    details.to_csv(output / (method.lower() + "_pattern_matches.csv"), index=False)
    path = output / "accuracy_summary.csv"
    row = pd.DataFrame([summary])
    previous = pd.read_csv(path) if path.exists() else row.iloc[:0]
    pd.concat([previous[previous.method != method], row], ignore_index=True).to_csv(path, index=False)


def clear_quilt_score(output, method):
    """Invalidate a previous score before rerunning a method in the same session."""
    path = Path(output) / 'accuracy_summary.csv'
    if path.exists():
        previous = pd.read_csv(path)
        previous.loc[previous.method != method].to_csv(path, index=False)


def summarize_quilt_repeats(output, seeds):
    """Summarize seed-level scores, not individual pattern correlations as replicates."""
    output = Path(output)
    methods = ['SparseAEH', 'SpatialDE', 'NSF', 'MEFISTO', 'PCA', 'SpatialDE SVG']
    rows = []
    for seed in seeds:
        folder = output / f'seed_{seed}'
        path = folder / 'accuracy_summary.csv'
        scores = pd.read_csv(path).set_index('method') if path.exists() else pd.DataFrame()
        records = []
        for path in sorted((folder / 'logs').glob('runs_*.jsonl')):
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        latest = {r['method']: r for r in sorted(records, key=lambda r: r['started_utc'])}
        for method in methods:
            record = latest.get(method)
            scored = method in scores.index
            status = 'not_run' if record is None else record['status']
            if status == 'ok' and not scored and method != 'SpatialDE SVG':
                status = 'metrics_missing'
            row = dict(method=method, seed=seed, status=status,
                       mean_matched_r=np.nan, gene_ari=np.nan, wall_seconds=np.nan,
                       peak_rss_mib=np.nan)
            if record is not None:
                row['error'] = record.get('error')
                row['attempt_wall_seconds'] = record['wall_seconds']
            if status == 'ok':
                row['wall_seconds'] = record['wall_seconds']
                row['peak_rss_mib'] = record['peak_process_tree_rss_bytes'] / 2**20
                if scored:
                    for metric in ['mean_matched_r', 'gene_ari']:
                        row[metric] = scores.loc[method, metric]
            rows.append(row)
    runs = pd.DataFrame(rows)
    summaries = []
    for method in methods:
        group = runs.loc[runs.method == method]
        summary = dict(method=method, requested_runs=len(seeds),
                       completed_runs=int((group.status == 'ok').sum()),
                       failed_runs=int((group.status == 'error').sum()),
                       complete=bool((group.status == 'ok').all()))
        for metric in ['mean_matched_r', 'gene_ari', 'wall_seconds', 'peak_rss_mib']:
            values = group[metric].dropna()
            summary.update({metric + '_mean': values.mean(),
                            metric + '_sd': values.std(ddof=1),
                            metric + '_n': len(values)})
        summaries.append(summary)
    summary = pd.DataFrame(summaries)
    runs.to_csv(output / 'repeated_runs.csv', index=False)
    summary.to_csv(output / 'repeated_summary.csv', index=False)
    return runs, summary
    print(row.to_string(index=False))
