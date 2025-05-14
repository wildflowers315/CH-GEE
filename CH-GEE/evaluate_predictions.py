"""Module for evaluating canopy height predictions."""

import os
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import norm
import rasterio
from datetime import datetime

from save_evaluation_pdf import save_evaluation_to_pdf
from raster_utils import load_and_align_rasters


def check_predictions(pred_path: str):
    """Check if predictions are valid before proceeding."""
    with rasterio.open(pred_path) as src:
        data = src.read(1)
        if np.all(data == src.nodata):
            print(f"\nError: The prediction file {os.path.basename(pred_path)} contains only nodata values.")
            print("Please ensure the prediction generation completed successfully.")
            return False
        return True


def validate_data(pred_data: np.ndarray, ref_data: np.ndarray):
    """Validate data before analysis."""
    # Check for zero variance
    pred_std = np.std(pred_data)
    if pred_std == 0:
        raise ValueError("Prediction data has zero variance (all values are the same). " +
                        f"All values are {pred_data[0]:.2f}")
    
    ref_std = np.std(ref_data)
    if ref_std == 0:
        raise ValueError("Reference data has zero variance (all values are the same). " +
                        f"All values are {ref_data[0]:.2f}")
    
    # Check for reasonable value ranges
    if np.max(pred_data) < 0.01:
        raise ValueError(f"Prediction values seem too low. Max value is {np.max(pred_data):.6f}")
    
    if np.max(ref_data) < 0.01:
        raise ValueError(f"Reference values seem too low. Max value is {np.max(ref_data):.6f}")
    
    print("Data validation passed:")
    print(f"Prediction range: {np.min(pred_data):.2f} to {np.max(pred_data):.2f}")
    print(f"Reference range: {np.min(ref_data):.2f} to {np.max(ref_data):.2f}")


def calculate_metrics(pred: np.ndarray, ref: np.ndarray):
    """Calculate evaluation metrics."""
    mse = mean_squared_error(ref, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ref, pred)
    r2 = r2_score(ref, pred)
    
    # Calculate additional statistics
    errors = pred - ref
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(np.abs(errors))
    
    # Calculate percentage of predictions within different error ranges
    within_1m = np.mean(np.abs(errors) <= 1.0) * 100
    within_2m = np.mean(np.abs(errors) <= 2.0) * 100
    within_5m = np.mean(np.abs(errors) <= 5.0) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Mean Error': mean_error,
        'Std Error': std_error,
        'Max Absolute Error': max_error,
        'Within 1m (%)': within_1m,
        'Within 2m (%)': within_2m,
        'Within 5m (%)': within_5m
    }


def create_plots(pred: np.ndarray, ref: np.ndarray, metrics: dict, output_dir: str):
    """Create evaluation plots."""
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
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(output_dir, 'error_hist.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(output_dir, 'height_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate canopy height predictions against reference data')
    parser.add_argument('--pred', type=str, help='Path to prediction raster', default='chm_outputs/predictions.tif')
    parser.add_argument('--ref', type=str, help='Path to reference raster', default='chm_outputs/dchm_09id4.tif')
    parser.add_argument('--output', type=str, help='Output directory', default='chm_outputs/evaluation')
    parser.add_argument('--pdf', action='store_true', help='Generate PDF report with 2x2 comparison grid')
    parser.add_argument('--training', type=str, help='Path to training data CSV for additional metadata', default='chm_outputs/training_data.csv')
    parser.add_argument('--merged', type=str, help='Path to merged data raster for RGB visualization', default=None)
    args = parser.parse_args()
    
    # Set paths
    pred_path = args.pred
    ref_path = args.ref
    output_dir = args.output
    generate_pdf = args.pdf
    training_data_path = args.training if os.path.exists(args.training) else None
    merged_data_path = args.merged if args.merged and os.path.exists(args.merged) else None
    
    # First check if predictions are valid
    if not check_predictions(pred_path):
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # date with YYYYMMDD
    date = datetime.now().strftime("%Y%m%d")
    # update output directory with date
    output_dir = os.path.join(output_dir, date)
        
    try:
        print("Loading and preprocessing rasters...")
        pred_data, ref_data, transform = load_and_align_rasters(pred_path, ref_path, output_dir)
        
        # Create masks for no data values and outliers
        print("\nCreating valid data masks...")
        pred_mask = (pred_data >= 0) & (pred_data <= 50)  # Reasonable height range for trees
        ref_mask = (ref_data >= 0) & (ref_data <= 50)     # Same range for reference
        
        # Combine all masks
        mask = pred_mask & ref_mask
        
        valid_pixels = np.sum(mask)
        total_pixels = mask.size
        print(f"Valid pixels: {valid_pixels:,} of {total_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
        area_ha = np.sum(mask) * (transform[0] * transform[4]) / 10000  # Convert to hectares
        print(f"Area of valid pixels: {area_ha:.2f} ha")
        if valid_pixels == 0:
            raise ValueError("No valid pixels in intersection area")
        
        # Apply masks to get valid data only
        pred_data = pred_data[mask]
        ref_data = ref_data[mask]
        
        # Print statistics
        print("\nStatistics for valid pixels (filtered to 0-50m range):")
        print("Prediction - Min: {:.2f}, Max: {:.2f}, Mean: {:.2f}, Std: {:.2f}".format(
            np.min(pred_data), np.max(pred_data), np.mean(pred_data), np.std(pred_data)))
        print("Reference - Min: {:.2f}, Max: {:.2f}, Mean: {:.2f}, Std: {:.2f}".format(
            np.min(ref_data), np.max(ref_data), np.mean(ref_data), np.std(ref_data)))
        
        # Validate data
        print("\nValidating data...")
        validate_data(pred_data, ref_data)
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_metrics(pred_data, ref_data)
        
        print("Generating visualizations...")
        if generate_pdf:
            # Create PDF report with 2x2 visualization grid
            print("\nGenerating PDF report...")
            pdf_path = save_evaluation_to_pdf(
                pred_path,
                ref_path,
                pred_data,
                ref_data,
                metrics,
                output_dir,
                mask=mask,
                training_data_path=training_data_path,
                merged_data_path=merged_data_path,
                mask=mask, area_ha=area_ha
            )
            print(f"PDF report saved to: {pdf_path}")
        else:
            # Generate individual plots
            print("Generating plots...")
            create_plots(pred_data, ref_data, metrics, output_dir)
        
        # Print results
        print("\nEvaluation Results (for heights between 0-50m):")
        print("-" * 50)
        for metric, value in metrics.items():
            if metric.endswith('(%)'):
                print(f"{metric:<20}: {value:>7.1f}%")
            else:
                print(f"{metric:<20}: {value:>7.3f}")
        print("-" * 50)
        
        print("\nOutputs saved to:", output_dir)
        
    except ValueError as e:
        print(f"\nValidation Error: {str(e)}")
        print("\nPlease check that both rasters contain valid height values.")
        raise
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()