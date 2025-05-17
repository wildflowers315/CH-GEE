"""Shared evaluation utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def validate_data(pred_data: np.ndarray, ref_data: np.ndarray):
    """Validate data before analysis and return validation info."""
    validation_info = {
        'pred_range': (np.min(pred_data), np.max(pred_data)),
        'ref_range': (np.min(ref_data), np.max(ref_data)),
        'pred_stats': {'mean': np.mean(pred_data), 'std': np.std(pred_data)},
        'ref_stats': {'mean': np.mean(ref_data), 'std': np.std(ref_data)}
    }
    
    # Check for zero variance
    pred_std = validation_info['pred_stats']['std']
    if pred_std == 0:
        raise ValueError("Prediction data has zero variance (all values are the same). " +
                        f"All values are {pred_data[0]:.2f}")
    
    ref_std = validation_info['ref_stats']['std']
    if ref_std == 0:
        raise ValueError("Reference data has zero variance (all values are the same). " +
                        f"All values are {ref_data[0]:.2f}")
    
    # Check for reasonable value ranges
    if validation_info['pred_range'][1] < 0.01:
        raise ValueError(f"Prediction values seem too low. Max value is {validation_info['pred_range'][1]:.6f}")
    
    if validation_info['ref_range'][1] < 0.01:
        raise ValueError(f"Reference values seem too low. Max value is {validation_info['ref_range'][1]:.6f}")
    
    return validation_info


def create_plots(pred: np.ndarray, ref: np.ndarray, metrics: dict, output_dir: str):
    """Create evaluation plots and return plot paths."""
    plot_paths = {}
    
    # Scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(ref, pred, alpha=0.5, s=1)
    plt.plot([0, max(ref.max(), pred.max())], [0, max(ref.max(), pred.max())], 'r--', label='1:1 line')
    
    # Add trend line
    z = np.polyfit(ref, pred, 1)
    p = np.poly1d(z)
    plt.plot(ref, p(ref), 'b--', label=f'Trend line (y = {z[0]:.3f}x + {z[1]:.3f})')
    
    plt.xlabel('Reference Height (m)')
    plt.ylabel('Predicted Height (m)')
    plt.title('Predicted vs Reference Height\n' + \
             f'R² = {metrics["R2"]:.3f}, RMSE = {metrics["RMSE"]:.3f}m')
    plt.legend()
    plt.grid(True)
    plot_paths['scatter'] = os.path.join(output_dir, 'scatter_plot.png')
    plt.savefig(plot_paths['scatter'], dpi=300, bbox_inches='tight')
    plt.close()
    
    # Error histogram
    errors = pred - ref
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75, density=True)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    
    # Add normal distribution curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, errors.mean(), errors.std())
    plt.plot(x, p, 'k--', label='Normal Distribution')
    
    plt.xlabel('Prediction Error (m)')
    plt.ylabel('Density')
    plt.title(f'Error Distribution\n' + \
             f'Mean = {errors.mean():.3f}m, Std = {errors.std():.3f}m')
    plt.legend()
    plt.grid(True)
    plot_paths['error_hist'] = os.path.join(output_dir, 'error_hist.png')
    plt.savefig(plot_paths['error_hist'], dpi=300, bbox_inches='tight')
    plt.close()
    
    # Height distributions
    plt.figure(figsize=(10, 6))
    plt.hist(ref, bins=50, alpha=0.5, label='Reference', density=True)
    plt.hist(pred, bins=50, alpha=0.5, label='Predicted', density=True)
    plt.xlabel('Height (m)')
    plt.ylabel('Density')
    plt.title('Height Distributions')
    plt.legend()
    plt.grid(True)
    plot_paths['height_dist'] = os.path.join(output_dir, 'height_distributions.png')
    plt.savefig(plot_paths['height_dist'], dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_paths


def create_comparison_grid(ref_data, pred_data, diff_data, rgb_data, output_path, forest_mask=None):
    """Create 2x2 grid visualization and save to file."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    vmax = 40
    # Create masked versions for visualization
    if forest_mask is not None:
        ref_masked = np.ma.masked_where(~forest_mask, ref_data)
        pred_masked = np.ma.masked_where(~forest_mask, pred_data)
        diff_masked = np.ma.masked_where(~forest_mask, diff_data)
    else:
        ref_masked = ref_data
        pred_masked = pred_data
        diff_masked = diff_data
    
    # Plot reference data
    im0 = axes[0,0].imshow(ref_masked, cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
    axes[0,0].set_title('Reference Heights')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    # Plot prediction data
    im1 = axes[0,1].imshow(pred_masked, cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
    axes[0,1].set_title('Predicted Heights')
    plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    # Plot difference map
    im2 = axes[1,0].imshow(diff_masked, cmap='RdYlBu', vmin=-10, vmax=10, aspect='equal')
    axes[1,0].set_title('Height Difference (Pred - Ref)')
    plt.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    
    # Plot RGB or empty plot
    if rgb_data is not None:
        axes[1,1].imshow(rgb_data, aspect='equal')
        axes[1,1].set_title('RGB Composite')
        # if forest_mask is not None:
    #         # Apply forest mask to RGB data
    #         rgb_masked = rgb_data.copy()
    #         for i in range(3):
    #             rgb_masked[:,:,i] = np.where(forest_mask, rgb_data[:,:,i], 0)
            # axes[1,1].imshow(rgb_masked, aspect='equal')
    #     else:
    #         axes[1,1].imshow(rgb_data, aspect='equal')
    #     axes[1,1].set_title('RGB Composite')
    else:
        axes[1,1].imshow(np.zeros_like(pred_data), cmap='gray', aspect='equal')
        axes[1,1].set_title('RGB Not Available')
    
    # Remove axes ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()