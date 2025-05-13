import rasterio
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.mask import mask
from shapely.geometry import box, mapping
from scipy.stats import norm
import os

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

def get_intersection_bounds(pred_path: str, ref_path: str):
    """Get the intersection bounds of two rasters."""
    with rasterio.open(pred_path) as pred_src:
        pred_crs = pred_src.crs
        pred_bounds = pred_src.bounds
        print(f"\nPrediction bounds ({pred_crs}):")
        print(f"Left: {pred_bounds.left:.6f}, Bottom: {pred_bounds.bottom:.6f}")
        print(f"Right: {pred_bounds.right:.6f}, Top: {pred_bounds.top:.6f}")
        
        with rasterio.open(ref_path) as ref_src:
            ref_bounds_orig = ref_src.bounds
            print(f"\nReference bounds ({ref_src.crs}):")
            print(f"Left: {ref_bounds_orig.left:.6f}, Bottom: {ref_bounds_orig.bottom:.6f}")
            print(f"Right: {ref_bounds_orig.right:.6f}, Top: {ref_bounds_orig.top:.6f}")
            
            # Transform reference bounds to prediction CRS
            if ref_src.crs != pred_crs:
                print(f"\nTransforming reference bounds to {pred_crs}")
                ref_bounds = transform_bounds(ref_src.crs, pred_crs, *ref_src.bounds)
            else:
                ref_bounds = ref_src.bounds
            
            # Create boxes for intersection
            pred_box = box(*pred_bounds)
            ref_box = box(*ref_bounds)
            
            # Get intersection
            intersection = pred_box.intersection(ref_box)
            
            if intersection.is_empty:
                raise ValueError("No intersection between prediction and reference rasters")
            
            bounds = intersection.bounds
            print(f"\nIntersection bounds: {bounds}")
            return bounds

def clip_and_resample(src_path: str, bounds: tuple, target_transform=None, target_crs=None, 
                     target_shape=None, output_path: str = None):
    """Clip raster to bounds and optionally resample to target resolution."""
    with rasterio.open(src_path) as src:
        print(f"\nProcessing: {src_path}")
        print(f"Original shape: {src.shape}")
        print(f"Original resolution: {src.res}")
        
        # Transform bounds if CRS differs
        if target_crs and src.crs != target_crs:
            bounds = transform_bounds(target_crs, src.crs, *bounds)
        
        # Create window from bounds
        window = from_bounds(*bounds, src.transform)
        
        # Read data in window
        data = src.read(1, window=window)
        print(f"Clipped shape: {data.shape}")
        
        # Get transform for clipped data
        clip_transform = rasterio.transform.from_bounds(*bounds, data.shape[1], data.shape[0])
        
        # If target parameters are provided, resample the data
        if all(x is not None for x in [target_transform, target_crs, target_shape]):
            # Create destination array
            dest = np.zeros(target_shape, dtype=np.float32)
            
            # Reproject and resample
            reproject(
                source=data,
                destination=dest,
                src_transform=clip_transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.average
            )
            
            data = dest
            out_transform = target_transform
            out_crs = target_crs
            print(f"Resampled shape: {data.shape}")
        else:
            out_transform = clip_transform
            out_crs = src.crs
        
        # Print value range
        valid_mask = data != src.nodata
        if np.any(valid_mask):
            print(f"Value range: {np.min(data[valid_mask]):.2f} to {np.max(data[valid_mask]):.2f}")
        
        # Save if output path provided
        if output_path:
            profile = src.profile.copy()
            profile.update({
                'height': data.shape[0],
                'width': data.shape[1],
                'transform': out_transform,
                'crs': out_crs
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            print(f"Saved to: {output_path}")
        
        return data, out_transform

def load_and_preprocess_rasters(pred_path: str, ref_path: str, output_dir: str):
    """Load and preprocess both rasters with better outlier handling."""
    print("\nFinding intersection bounds...")
    bounds = get_intersection_bounds(pred_path, ref_path)
    
    # Get prediction properties to use as target
    with rasterio.open(pred_path) as pred_src:
        target_transform = pred_src.transform
        target_crs = pred_src.crs
        target_shape = pred_src.shape
        pred_nodata = pred_src.nodata if pred_src.nodata is not None else -9999
    
    # Create clipped and resampled versions
    print("\nProcessing prediction raster...")
    pred_clip_path = os.path.join(output_dir, 'pred_clipped.tif')
    pred_data, _ = clip_and_resample(
        pred_path, bounds,
        target_transform=target_transform,
        target_crs=target_crs,
        target_shape=target_shape,
        output_path=pred_clip_path
    )
    
    print("\nProcessing reference raster...")
    ref_clip_path = os.path.join(output_dir, 'ref_clipped.tif')
    ref_data, _ = clip_and_resample(
        ref_path, bounds,
        target_transform=target_transform,
        target_crs=target_crs,
        target_shape=target_shape,
        output_path=ref_clip_path
    )
    
    # Create masks for no data values and outliers
    print("\nCreating valid data masks...")
    pred_mask = pred_data != pred_nodata
    ref_mask = ref_data != -32767  # Reference nodata value
    
    # Add reasonable range filters
    pred_range_mask = (pred_data >= 0) & (pred_data <= 50)  # Reasonable height range for trees
    ref_range_mask = (ref_data >= 0) & (ref_data <= 50)     # Same range for reference
    
    # Combine all masks
    mask = pred_mask & ref_mask & pred_range_mask & ref_range_mask
    
    valid_pixels = np.sum(mask)
    total_pixels = mask.size
    print(f"Valid pixels: {valid_pixels:,} of {total_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
    
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
    
    return pred_data, ref_data, target_transform, mask

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

def create_difference_map(pred_path: str, ref_path: str, output_dir: str):
    """Create and save difference map."""
    # Load and preprocess data
    pred_data, ref_data, transform, mask = load_and_preprocess_rasters(pred_path, ref_path, output_dir)
    
    # Reshape the data back to 2D
    with rasterio.open(pred_path) as pred_src:
        diff_data = pred_src.read(1).copy()
        ref_reshaped = np.full_like(diff_data, -9999)
        mask_indices = np.where(mask)
        ref_reshaped[mask_indices] = ref_data
        
        # Calculate difference
        diff_data = np.where((diff_data != pred_src.nodata) & (ref_reshaped != -9999) &
                           (diff_data >= 0) & (diff_data <= 50) &
                           (ref_reshaped >= 0) & (ref_reshaped <= 50),
                           diff_data - ref_reshaped,
                           -9999)
        
        # Save difference map
        diff_profile = pred_src.profile.copy()
        diff_profile.update(nodata=-9999)
        diff_path = os.path.join(output_dir, 'height_difference.tif')
        
        with rasterio.open(diff_path, 'w', **diff_profile) as dst:
            dst.write(diff_data, 1)
            
        return diff_path

def main():
    # Set paths
    pred_path = 'chm_outputs/chm_export_1747130473.tif'
    ref_path = 'chm_outputs/dchm_09id4.tif'
    output_dir = 'chm_outputs/evaluation'
    
    # First check if predictions are valid
    if not check_predictions(pred_path):
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing rasters...")
        pred_data, ref_data, transform, mask = load_and_preprocess_rasters(pred_path, ref_path, output_dir)
        
        # Validate data
        print("\nValidating data...")
        validate_data(pred_data, ref_data)
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_metrics(pred_data, ref_data)
        
        # Generate plots
        print("Generating plots...")
        create_plots(pred_data, ref_data, metrics, output_dir)
        
        # Create difference map
        print("Creating difference map...")
        diff_path = create_difference_map(pred_path, ref_path, output_dir)
        
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