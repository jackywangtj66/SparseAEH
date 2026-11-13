"""Plot six CRC RCTD reference maps from saved weights and coordinates."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import pandas as pd


CELL_TYPES = ['Tumor I', 'Tumor II', 'Tumor III', 'Tumor V',
              'Macrophage', 'Fibroblast']


def load_profiles(result_dir, reference=None):
    result_dir = Path(result_dir)
    reference = Path(reference) if reference else result_dir / 'rctd/aligned_reference.csv'
    locations = pd.read_csv(result_dir / 'locations.csv', dtype={'location_id': str})
    locations = locations.set_index('location_id')[['array_row', 'array_col']]
    weights = pd.read_csv(reference, index_col=0)
    weights.index = weights.index.astype(str)
    weights = weights[CELL_TYPES].apply(pd.to_numeric, errors='raise')
    if locations.empty or not locations.index.is_unique or not weights.index.is_unique:
        raise ValueError('Locations must be nonempty and both files must have unique location IDs.')
    coords = locations.to_numpy(dtype=float)
    if not np.isfinite(coords).all() or not np.equal(coords, np.round(coords)).all():
        raise ValueError('locations.csv must contain finite integer array coordinates.')
    if locations.duplicated().any():
        raise ValueError('Multiple location IDs share an array coordinate.')
    matched = locations.index[locations.index.isin(weights.index)]
    values = weights.loc[matched]
    if values.empty or not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError('Matched RCTD weights must be nonempty, finite and nonnegative.')
    return locations, values, reference


def make_figure(locations, weights, upper_percentile=99, vmax=None):
    if not 0 < upper_percentile <= 100:
        raise ValueError('upper_percentile must be in (0, 100].')
    values = weights.to_numpy()
    limit = float(vmax if vmax is not None else np.percentile(values, upper_percentile))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('The color maximum must be positive; supply --vmax for zero-heavy data.')

    # Match plot_crc_patterns.py: array_row is horizontal, array_col vertical.
    # origin='lower' preserves left/right and uses the corrected CRC vertical flip.
    coords = locations.astype(int)
    x = coords.array_row.to_numpy() - coords.array_row.min()
    y = coords.array_col.to_numpy() - coords.array_col.min()
    shape = (int(y.max()) + 1, int(x.max()) + 1)
    if shape[0] * shape[1] > 10_000_000:
        raise ValueError('Grid is unexpectedly large; use array coordinates, not pixel coordinates.')
    tissue = np.full(shape, np.nan)
    tissue[y, x] = 1
    indices = locations.index.get_indexer(weights.index)
    if (indices < 0).any():
        raise ValueError('Some reference IDs have no coordinate.')
    norm = Normalize(vmin=0, vmax=limit, clip=False)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad((0, 0, 0, 0))
    missing = ListedColormap(['#d9d9d9'])
    missing.set_bad((0, 0, 0, 0))
    with plt.rc_context({'font.size': 10, 'font.family': 'DejaVu Sans',
                         'pdf.fonttype': 42, 'text.color': '#22292b',
                         'axes.labelcolor': '#22292b'}):
        fig, axes = plt.subplots(2, 3, figsize=(10, 7.7))
        fig.subplots_adjust(left=.035, right=.965, top=.90, bottom=.18,
                            wspace=.08, hspace=.15)
        fig.suptitle('CRC RCTD cell-type weights', fontsize=13, y=.975)
        for ax, name in zip(axes.flat, CELL_TYPES):
            grid = np.full(shape, np.nan)
            grid[y[indices], x[indices]] = weights[name].to_numpy()
            ax.imshow(tissue, cmap=missing, vmin=0, vmax=1, origin='lower',
                      interpolation='nearest', aspect='equal')
            im = ax.imshow(grid, cmap=cmap, norm=norm, origin='lower',
                           interpolation='nearest', aspect='equal')
            ax.set_title(name, fontsize=11, pad=6)
            ax.set_axis_off()
        cax = fig.add_axes([.26, .115, .48, .018])
        extended = bool((values > limit).any())
        bar = fig.colorbar(im, cax=cax, orientation='horizontal',
                           extend='max' if extended else 'neither')
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2, labelsize=9)
        scale = f'pooled P{upper_percentile:g}' if vmax is None else f'maximum {limit:g}'
        bar.set_label(f'RCTD weight (shared scale, 0 to {scale})', fontsize=10, labelpad=5)
        fig.text(.5, .025, f'{len(weights):,} matched bins; light gray: no RCTD estimate '
                 f'({len(locations) - len(weights):,} bins). Weights are not renormalized.',
                 ha='center', fontsize=9)
    return fig, limit


def plot_reference(result_dir, reference=None, output_dir=None, upper_percentile=99,
                   vmax=None, dpi=400):
    result_dir = Path(result_dir)
    locations, weights, reference = load_profiles(result_dir, reference)
    fig, limit = make_figure(locations, weights, upper_percentile, vmax)
    output_dir = Path(output_dir) if output_dir else result_dir / 'rctd'
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / 'crc_rctd_celltypes'
    try:
        for ext in ('png', 'pdf'):
            fig.savefig(stem.with_suffix('.' + ext), dpi=dpi, facecolor='white')
    finally:
        plt.close(fig)
    settings = {
        'reference_file': str(reference.resolve()),
        'locations_file': str((result_dir / 'locations.csv').resolve()),
        'cell_types': CELL_TYPES, 'alignment': 'location IDs; no row-order assumption',
        'expression_locations': len(locations), 'matched_locations': len(weights),
        'unmatched_expression_locations': len(locations) - len(weights),
        'horizontal': 'array_row as stored in locations.csv',
        'vertical': 'array_col as stored in locations.csv; increasing upward',
        'transform': 'none; no renormalization of the selected cell types',
        'cmap': 'viridis', 'shared_color_limits': [0.0, limit],
        'upper_percentile': upper_percentile if vmax is None else None,
        'percentile_scope': 'all six weights at matched locations pooled',
        'fraction_above_color_maximum': {c: float((weights[c] > limit).mean()) for c in CELL_TYPES},
        'missing_reference_color': '#d9d9d9', 'dpi': dpi,
    }
    stem.with_suffix('.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    print(f'Matched {len(weights):,}/{len(locations):,} bins; shared limits: 0 to {limit:.6g}')
    print(f'Saved {stem}.png, .pdf and .json')
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path, help='CRC result folder containing locations.csv')
    parser.add_argument('--reference', type=Path,
                        help='RCTD CSV with location IDs in column 1; default: result_dir/rctd/aligned_reference.csv')
    parser.add_argument('--output-dir', type=Path, help='Default: result_dir/rctd')
    parser.add_argument('--upper-percentile', type=float, default=99,
                        help='Shared upper percentile across the six maps; 100 disables percentile clipping')
    parser.add_argument('--vmax', type=float, help='Explicit shared upper color limit; overrides percentile')
    parser.add_argument('--dpi', type=int, default=400)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    plot_reference(**vars(args))


if __name__ == '__main__':
    main()
