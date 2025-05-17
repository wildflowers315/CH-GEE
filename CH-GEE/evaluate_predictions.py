"""Module for evaluating canopy height predictions."""

import os
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import rasterio
from datetime import datetime
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from save_evaluation_pdf import save_evaluation_to_pdf
from raster_utils import load_and_align_rasters
from evaluation_utils import validate_data, create_plots


def check_predictions(pred_path: str):
    """Check if predictions are valid before proceeding."""
    with rasterio.open(pred_path) as src:
        data = src.read(1)
        if np.all(data == src.nodata):
            print(f"\nError: The prediction file {os.path.basename(pred_path)} contains only nodata values.")
            print("Please ensure the prediction generation completed successfully.")
            return False
        return True


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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate canopy height predictions against reference data')
    parser.add_argument('--pred', type=str, help='Path to prediction raster', default='chm_outputs/predictions.tif')
    parser.add_argument('--ref', type=str, help='Path to reference raster', default='chm_outputs/dchm_09id4.tif')
    parser.add_argument('--forest-mask', type=str, help='Path to forest mask raster', default=None)
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Add date to output directory
    date = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(output_dir, date)
    os.makedirs(output_dir, exist_ok=True)
    
    # First check if predictions are valid
    if not check_predictions(pred_path):
        return 1

    try:
        
        print("Loading and preprocessing rasters...")
        pred_data, ref_data, transform, forest_mask = load_and_align_rasters(
            pred_path, ref_path, args.forest_mask, output_dir)
        
        # Create masks for no data values and outliers
        print("\nCreating valid data masks...")
        pred_mask = (pred_data >= 0) & (pred_data <= 50) & ~np.isnan(pred_data)  # Reasonable height range for trees
        ref_mask = (ref_data >= 0) & (ref_data <= 50) & ~np.isnan(ref_data)      # Same range for reference
        
        # Combine all masks
        mask = pred_mask & ref_mask
        if forest_mask is not None:
            mask = mask & forest_mask
            print(f"Applied forest mask - {np.sum(forest_mask):,} forest pixels")
        
        valid_pixels = np.sum(mask)
        total_pixels = mask.size
        print(f"Valid pixels: {valid_pixels:,} of {total_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
        # Calculate area using geographic coordinates
        with rasterio.open(pred_path) as src:
            # Get the CRS of the prediction
            if src.crs.is_geographic:
                # For geographic coordinates, calculate approximate area using UTM
                center_lat = (src.bounds.bottom + src.bounds.top) / 2
                center_lon = (src.bounds.left + src.bounds.right) / 2
                utm_zone = int((center_lon + 180) / 6) + 1
                utm_epsg = 32600 + utm_zone + (0 if center_lat >= 0 else 100)
                utm_crs = CRS.from_epsg(utm_epsg)
                
                # Transform bounds to UTM
                bounds_utm = transform_bounds(src.crs, utm_crs, *src.bounds)
                width_m = bounds_utm[2] - bounds_utm[0]
                height_m = bounds_utm[3] - bounds_utm[1]
                
                # Calculate pixel size in meters
                pixel_width_m = width_m / src.width
                pixel_height_m = height_m / src.height
                pixel_area_m2 = pixel_width_m * pixel_height_m
            else:
                # For projected coordinates, use transform directly
                pixel_area_m2 = abs(transform[0] * transform[4])
        
        area_ha = (np.sum(mask) * pixel_area_m2) / 10000  # Convert to hectares
        print(f"Area of valid pixels: {area_ha:.2f} ha")
        
        if valid_pixels == 0:
            raise ValueError("No valid pixels in intersection area")
        
        # Create masked copies for statistics
        pred_masked = pred_data[mask]
        ref_masked = ref_data[mask]
        
        # Print statistics
        print("\nStatistics for valid pixels (filtered to 0-50m range):")
        print("Prediction - Min: {:.2f}, Max: {:.2f}, Mean: {:.2f}, Std: {:.2f}".format(
            np.min(pred_masked), np.max(pred_masked), np.mean(pred_masked), np.std(pred_masked)))
        print("Reference - Min: {:.2f}, Max: {:.2f}, Mean: {:.2f}, Std: {:.2f}".format(
            np.min(ref_masked), np.max(ref_masked), np.mean(ref_masked), np.std(ref_masked)))
        
        # Validate data and get statistics
        print("\nValidating data...")
        validation_info = validate_data(pred_masked, ref_masked)
        print("Data validation passed:")
        print(f"Prediction range: {validation_info['pred_range'][0]:.2f} to {validation_info['pred_range'][1]:.2f}")
        print(f"Reference range: {validation_info['ref_range'][0]:.2f} to {validation_info['ref_range'][1]:.2f}")
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_metrics(pred_masked, ref_masked)
        
        print("Generating visualizations...")
        # Always generate plots for masked data
        plot_paths = create_plots(pred_masked, ref_masked, metrics, output_dir)
        
        if generate_pdf:
            # Create PDF report with all visualizations
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
                area_ha=area_ha,
                validation_info=validation_info,
                plot_paths=plot_paths
            )
            print(f"PDF report saved to: {pdf_path}")
        
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
        
        return 0
        
    except ValueError as e:
        print(f"\nValidation Error: {str(e)}")
        print("\nPlease check that both rasters contain valid height values.")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())