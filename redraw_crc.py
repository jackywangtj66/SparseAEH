"""Replace a saved CRC pattern plot without loading expression data or refitting."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SparseAEH.plot import plot_clusters


FILES = {
    'sparseaeh': ('sparse_gem', 'means'),
    'full-covariance': ('full_covariance_gem', 'means'),
    'spatialde': ('spatialde_aeh', 'patterns'),
    'rsf': ('rsf_gaussian', 'factors'),
    'mefisto': ('mefisto', 'factors'),
}


def redraw(result_dir, method='sparseaeh'):
    result_dir = Path(result_dir).expanduser().resolve()
    final = result_dir / 'final' if (result_dir / 'final').is_dir() else result_dir
    root = final.parent if final.name == 'final' else final
    stem, suffix = FILES[method]
    means = pd.read_csv(final / f'{stem}_{suffix}.csv', index_col=0)
    locations = pd.read_csv(root / 'locations.csv', dtype={'location_id': str}).set_index('location_id')
    means.index = means.index.map(str)
    if not means.index.is_unique or not locations.index.is_unique:
        raise ValueError('Location IDs must be unique in the maps and locations.csv.')
    spatial = locations.loc[means.index, ['array_row', 'array_col']].to_numpy()
    count = None
    if method in ('sparseaeh', 'full-covariance'):
        with np.load(final / f'{stem}_fit.npz') as fit:
            count = np.bincount(fit['labels'].astype(int), minlength=means.shape[1])
    elif method == 'spatialde':
        genes = pd.read_csv(final / f'{stem}_genes.csv')
        count = np.bincount(genes['pattern'].to_numpy(dtype=int), minlength=means.shape[1])
    target = final / f'{stem}.png'
    temporary = target.with_suffix('.redraw.png')
    fig, _ = plot_clusters(mean=means.to_numpy(), spatial=spatial, count=count, s=2,
                           invert_y=False, show=False)
    try:
        fig.savefig(temporary)
        temporary.replace(target)
    finally:
        plt.close(fig)
    print('Replaced:', target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path, help='CRC run folder, or its final/ folder.')
    parser.add_argument('--method', choices=FILES, default='sparseaeh')
    args = parser.parse_args()
    redraw(args.result_dir, args.method)
