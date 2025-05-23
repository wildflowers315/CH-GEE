"""Shared raster processing utilities."""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds
import numpy as np
import os

def clip_and_resample_raster(src_path: str, bounds: tuple, target_transform=None, 
                           target_crs=None, target_shape=None, output_path: str = None):
    """Clip raster to bounds and optionally resample to target resolution."""
    with rasterio.open(src_path) as src:
        # Transform bounds if CRS differs
        if target_crs and src.crs != target_crs:
            clip_bounds = transform_bounds(target_crs, src.crs, *bounds)
        else:
            clip_bounds = bounds
        
        # Calculate output dimensions based on bounds and resolution
        west, south, east, north = clip_bounds
        output_width = max(1, int(round((east - west) / abs(src.res[0]))))
        output_height = max(1, int(round((north - south) / abs(src.res[1]))))
        
        # Create window for clipping
        col_start = int((west - src.bounds.left) / src.res[0])
        row_start = int((src.bounds.top - north) / src.res[1])
        col_stop = int((east - src.bounds.left) / src.res[0])
        row_stop = int((src.bounds.top - south) / src.res[1])
        
        window = Window(
            col_off=col_start,
            row_off=row_start,
            width=max(1, col_stop - col_start),
            height=max(1, row_stop - row_start)
        )
        
        # Read data in window
        data = src.read(1, window=window)
        
        # Get transform for clipped data
        clip_transform = rasterio.transform.from_bounds(
            west, south, east, north,
            output_width, output_height
        )
        
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
        else:
            out_transform = clip_transform
            out_crs = src.crs
        
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
        
        return data, out_transform
def get_common_bounds(pred_path: str, ref_path: str):
    """Get intersection bounds of two rasters in the prediction CRS."""
    with rasterio.open(pred_path) as pred_src:
        pred_crs = pred_src.crs
        pred_bounds = pred_src.bounds
        print(f"\nPrediction bounds ({pred_crs}):")
        print(f"Left: {pred_bounds.left:.6f}, Bottom: {pred_bounds.bottom:.6f}")
        print(f"Right: {pred_bounds.right:.6f}, Top: {pred_bounds.top:.6f}")
        
        with rasterio.open(ref_path) as ref_src:
            print(f"\nReference bounds ({ref_src.crs}):")
            print(f"Left: {ref_src.bounds.left:.6f}, Bottom: {ref_src.bounds.bottom:.6f}")
            print(f"Right: {ref_src.bounds.right:.6f}, Top: {ref_src.bounds.top:.6f}")
            
            if ref_src.crs != pred_crs:
                print(f"\nTransforming reference bounds to {pred_crs}")
                ref_bounds = transform_bounds(ref_src.crs, pred_crs, *ref_src.bounds)
            else:
                ref_bounds = ref_src.bounds
            
            # Find intersection
            west = max(pred_bounds.left, ref_bounds[0])
            south = max(pred_bounds.bottom, ref_bounds[1])
            east = min(pred_bounds.right, ref_bounds[2])
            north = min(pred_bounds.top, ref_bounds[3])
            
            bounds = (west, south, east, north)
            print(f"\nIntersection bounds: {bounds}")
            return bounds


def load_and_align_rasters(pred_path: str, ref_path: str, forest_mask_path: str = None, output_dir: str = None):
    """Load and align rasters to same CRS and resolution, optionally applying forest mask."""
    # Get intersection bounds in prediction CRS
    bounds = get_common_bounds(pred_path, ref_path)
    
    # Get prediction properties to use as target
    with rasterio.open(pred_path) as pred_src:
        target_transform = pred_src.transform
        target_crs = pred_src.crs
        target_shape = pred_src.shape
    
    pred_filename = os.path.basename(pred_path)
    ref_filename = os.path.basename(ref_path)
    
 
    if output_dir:
        # Create paths for clipped files
        pred_clip_path = os.path.join(output_dir, f"{os.path.splitext(pred_filename)[0]}_clipped.tif")
        ref_clip_path = os.path.join(output_dir, f"{os.path.splitext(ref_filename)[0]}_clipped.tif")
    else:
        pred_clip_path = ref_clip_path = None
        
    print("\nProcessing prediction raster...")
    pred_data, _ = clip_and_resample_raster(
        pred_path, bounds,
        target_transform=target_transform,
        target_crs=target_crs,
        target_shape=target_shape,
        output_path=pred_clip_path
    )
    
    print("\nProcessing reference raster...")
    ref_data, _ = clip_and_resample_raster(
        ref_path, bounds,
        target_transform=target_transform,
        target_crs=target_crs,
        target_shape=target_shape,
        output_path=ref_clip_path
    )
    
    # Load and apply forest mask if provided
    forest_mask = None
    if forest_mask_path and os.path.exists(forest_mask_path):
        print("\nProcessing forest mask...")
        mask_data, _ = clip_and_resample_raster(
            forest_mask_path, bounds,
            target_transform=target_transform,
            target_crs=target_crs,
            target_shape=target_shape
        )
        # Create binary mask
        forest_mask = (mask_data > 0)
        
        # Apply forest mask to both prediction and reference data
        pred_data[~forest_mask] = np.nan
        ref_data[~forest_mask] = np.nan
        
        print(f"Forest mask applied - {np.sum(forest_mask):,} forest pixels")

    return pred_data, ref_data, target_transform, forest_mask