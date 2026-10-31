import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
from SparseAEH.base import MixedGaussian
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable 

# def plot_clusters(gaussian:MixedGaussian,label='counts',figsize=(18,16),s=15):
#     plt.figure(figsize=figsize)
#     k = gaussian.K
#     h = np.ceil(np.sqrt(k)).astype(int)
#     w = np.ceil(k/h).astype(int)
#     for i in range(gaussian.K):
#         plt.subplot(h, w, i + 1)
#         plt.scatter(gaussian.kernel.spatial[:,0],gaussian.kernel.spatial[:,1],marker = 's', c=gaussian.mean[:,i], cmap="viridis", s=s)
#         plt.axis('equal')
#         plt.gca().invert_yaxis()
#         if label == 'counts':
#             plt.title('{}'.format(np.sum(gaussian.labels==i)))
#         else:
#             plt.title('{}'.format(gaussian.pi[i]))
#         plt.gcf().set_dpi(300)

def plot_clusters(gaussian=None, mean=None, spatial=None, count=None, label='genes', figsize=None, s=15, base_subplot_size_inches=(4, 4), save=None, *, percentiles=(1, 99), show=True, rasterized=False):
    """Plot means with a shared viridis scale, pooled across all patterns.

    Percentile limits saturate colors only; input values are not modified.
    Colorbar extensions mark values outside the limits in each panel.
    Set percentiles=None for the full range. Coincident percentile limits
    fall back to the full range. Returns (figure, flat occupied axes).
    """
    if gaussian is not None:
        mean = gaussian.mean
        spatial = gaussian.kernel.spatial

    mean = np.asarray(mean, dtype=float)
    spatial = np.asarray(spatial, dtype=float)
    if mean.ndim != 2 or min(mean.shape) == 0:
        raise ValueError('mean must be a nonempty locations-by-patterns matrix.')
    if spatial.shape != (mean.shape[0], 2):
        raise ValueError('spatial must have one row per location and two columns.')
    if not np.isfinite(mean).all() or not np.isfinite(spatial).all():
        raise ValueError('mean and spatial must contain only finite values.')
    k = mean.shape[1]
    if count is not None and len(count) != k:
        raise ValueError('count must have one entry per pattern.')
    sx, sy = spatial[:, 0], spatial[:, 1]

    if percentiles is None:
        vmin, vmax = mean.min(), mean.max()
    else:
        bounds = np.asarray(percentiles, dtype=float)
        if bounds.shape != (2,) or not (0 <= bounds[0] < bounds[1] <= 100):
            raise ValueError('percentiles must satisfy 0 <= lower < upper <= 100.')
        vmin, vmax = np.percentile(mean, bounds)
        if vmin == vmax:
            vmin, vmax = mean.min(), mean.max()
    if vmin == vmax:
        # A constant profile still needs a nonzero color interval.
        padding = max(abs(vmin) * 0.01, 0.01)
        vmin, vmax = vmin - padding, vmax + padding
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Calculate grid dimensions for subplots
    h = math.ceil(math.sqrt(k))
    w = math.ceil(k / h)

    # Automatic figsize calculation if not explicitly provided
    if figsize is None:
        fig_width = w * base_subplot_size_inches[0] + (1.0 if w > 1 else 0)
        fig_height = h * base_subplot_size_inches[1] + (1.0 if h > 1 else 0)
        figsize = (fig_width, fig_height)

    # Create the figure and subplots
    # We set dpi here directly for the figure
    fig, axes = plt.subplots(h, w, figsize=figsize, dpi=300, squeeze=False)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    fig_width = fig.get_figwidth()

    base_font_size = 12
    reference_fig_width = 8
    scaling_factor_sublots = 1.0 / np.sqrt(k) if k > 0 else 1.0
    calculated_labelsize = base_font_size * (fig_width / reference_fig_width) * scaling_factor_sublots

    for i in range(k):
        ax = axes[i] # Get the current subplot axis

        sc = ax.scatter(sx, sy, marker='o', c=mean[:, i], cmap="viridis", s=s,
                        norm=norm, rasterized=rasterized)

        ax.set_aspect('equal', adjustable='box') # Ensures square aspect ratio for the plot area
        ax.invert_yaxis() 
        
        if gaussian is not None:
            ax.set_title(f'Cluster {i+1}: {np.sum(gaussian.labels==i)} {label}')
        elif count is not None:
            ax.set_title(f'Cluster {i+1}: {count[i]} {label}')
        else:
            ax.set_title(f'Cluster {i+1}')

        ax.set_xticks([])
        ax.set_yticks([])

        divider = make_axes_locatable(ax)
        
        # Append a new Axes to the right for the colorbar
        # Adjust 'size' to change the width of the colorbar
        # Adjust 'pad' to change the spacing between the plot and the colorbar
        # 'size' can be a percentage string (e.g., "5%", "10%")
        # or an absolute string (e.g., "0.2in", "0.5cm")
        cax = divider.append_axes("right", size="5%", pad=0.1) 
        
        # Create the colorbar and attach it to the new cax
        below = np.any(mean[:, i] < vmin)
        above = np.any(mean[:, i] > vmax)
        extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
        cbar = fig.colorbar(sc, cax=cax, orientation='vertical', extend=extend)
        
        # You can still modify tick parameters on the colorbar's axes
        cbar.ax.tick_params(labelsize=calculated_labelsize, size=0.01, pad=0.04)

    for j in range(k, h * w):
        fig.delaxes(axes[j]) # Remove the empty subplot

    fig.tight_layout()

    if save is not None:
        fig.savefig(save)

    if show:
        plt.show()
    return fig, axes[:k]
