"""Recovery of the generating spatial maps and gene groups in the quilt simulation."""
from pathlib import Path

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
    print(row.to_string(index=False))
