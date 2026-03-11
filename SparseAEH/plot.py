import numpy as np
import matplotlib.pyplot as plt
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

def plot_clusters(gaussian=None, mean=None, spatial=None, count=None, label='genes', figsize=None, s=15, base_subplot_size_inches=(4, 4), save=None):
    
    if gaussian is not None:
        k = gaussian.K
        mean = gaussian.mean
        sx = gaussian.kernel.spatial[:, 0]
        sy = gaussian.kernel.spatial[:, 1]
    else:
        mean = mean
        k = mean.shape[-1]
        sx = spatial[:, 0]
        sy = spatial[:, 1]

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
    fig, axes = plt.subplots(h, w, figsize=figsize, dpi=300)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    vmin = np.min(mean)
    vmax = np.max(mean)

    base_font_size = 12
    reference_fig_width = 8
    scaling_factor_sublots = 1.0 / np.sqrt(k) if k > 0 else 1.0
    calculated_labelsize = base_font_size * (fig_width / reference_fig_width) * scaling_factor_sublots

    for i in range(k):
        ax = axes[i] # Get the current subplot axis

        sc = ax.scatter(sx, sy, marker='o', c=mean[:, i], cmap="viridis", s=s,vmin=vmin, vmax=vmax)

        ax.set_aspect('equal', adjustable='box') # Ensures square aspect ratio for the plot area
        ax.invert_yaxis() 
        
        if gaussian is not None:
            ax.set_title(f'Cluster {i+1}: {np.sum(gaussian.labels==i)} {label}')
        elif count is not None:
            ax.set_title(f'Cluster {i+1}: {count[i]} {label}')
        else:
            pass

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
        cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
        
        # You can still modify tick parameters on the colorbar's axes
        cbar.ax.tick_params(labelsize=calculated_labelsize, size=0.01, pad=0.04)

    for j in range(k, h * w):
        fig.delaxes(axes[j]) # Remove the empty subplot

    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    plt.show()