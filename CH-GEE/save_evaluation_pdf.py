"""Module for generating PDF evaluation reports."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from raster_utils import load_and_align_rasters
from evaluate_predictions import create_plots, validate_data

# --- ヘルパー関数: スケーリング、コントラスト・ガンマ調整 ---
def scale_adjust_band(band_data, min_val, max_val, contrast=1.0, gamma=1.0):
    """Min/Maxスケール、コントラスト、ガンマ調整を行い、uint8に変換"""
    # 0. NaNを一時的に特定の値（例：-9999）に置き換え（計算後元に戻す）
    nan_mask = np.isnan(band_data)
    temp_nodata = -9999
    work_data = band_data.copy()
    if np.any(work_data[~nan_mask] == temp_nodata):
        valid_min = np.min(work_data[~nan_mask]) if not nan_mask.all() else -1
        temp_nodata = valid_min - 1
        print(f"警告: 一時的なNoData値 {temp_nodata} を使用します。")

    work_data[nan_mask] = temp_nodata
    work_data = work_data.astype(np.float32)

    # 1. Min/Max スケーリング (0-1 float)
    if max_val == min_val:
        scaled_data = np.zeros_like(work_data, dtype=np.float32)
    else:
        # temp_nodata は計算に影響しないようにする (例: スケール後に処理)
        mask_valid = (work_data != temp_nodata)
        scaled_data = np.zeros_like(work_data, dtype=np.float32)
        # Ensure division by zero is avoided if max_val == min_val (already handled above, but safe)
        if max_val - min_val != 0:
              scaled_data[mask_valid] = (work_data[mask_valid] - min_val) / (max_val - min_val)
        else:
              scaled_data[mask_valid] = 0 # Or handle as appropriate if max=min

    # 0-1の範囲にクリッピング (temp_nodata以外)
    scaled_data[mask_valid] = np.clip(scaled_data[mask_valid], 0, 1)

    # 2. コントラスト調整 (中心を0.5として調整)
    if contrast != 1.0:
        scaled_data[mask_valid] = 0.5 + contrast * (scaled_data[mask_valid] - 0.5)
        # 再度クリッピング
        scaled_data[mask_valid] = np.clip(scaled_data[mask_valid], 0, 1)

    # 3. ガンマ補正
    if gamma != 1.0 and gamma > 0: # gammaは正の値である必要あり
        # ガンマ補正は正の値にのみ適用
        gamma_mask = mask_valid & (scaled_data > 0)
        with np.errstate(invalid='ignore'):
            scaled_data[gamma_mask] = scaled_data[gamma_mask]**(1.0 / gamma)
        # ガンマ補正は0-1範囲を維持するはずだが念のため
        scaled_data[gamma_mask] = np.clip(scaled_data[gamma_mask], 0, 1)
    elif gamma <= 0:
        print(f"警告: ガンマ値 ({gamma}) は正の値である必要があります。ガンマ補正はスキップされました。")

    # 4. uint8変換
    # temp_nodataだった箇所（元々NaNだった箇所）は0にする
    scaled_data[~mask_valid] = 0 # temp_nodata (元NaN) を 0 に
    scaled_uint8 = (scaled_data * 255).astype(np.uint8)

    return scaled_uint8

def create_2x2_visualization(ref_path, pred_path, merged_path, output_path, mask=None):
    """Create 2x2 grid with reference, prediction, difference and RGB data."""
    # Load and align rasters
    pred_data, ref_data, transform = load_and_align_rasters(pred_path, ref_path)
    
    # Calculate difference
    diff_data = pred_data - ref_data
    
    # Create figure with 2x2 layout with fixed aspect ratio
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Plot reference data
    im0 = axes[0,0].imshow(ref_data, cmap='viridis', vmin=0, vmax=50, aspect='equal')
    axes[0,0].set_title('Reference Heights')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    # Plot prediction data
    im1 = axes[0,1].imshow(pred_data, cmap='viridis', vmin=0, vmax=50, aspect='equal')
    axes[0,1].set_title('Predicted Heights')
    plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    # Plot difference map
    im2 = axes[1,0].imshow(diff_data, cmap='RdYlBu', vmin=-10, vmax=10, aspect='equal')
    axes[1,0].set_title('Height Difference (Pred - Ref)')
    plt.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    # Plot RGB if available
    if merged_path and os.path.exists(merged_path):
        try:
            # Check if the merged file has enough bands for RGB
            with rasterio.open(merged_path) as src:
                band_count = src.count
                
                if band_count >= 3:
                    # For Sentinel-2, use bands 2,1,0 (BGR → RGB)
                    rgb_bands = [2, 1, 0]  # Traditional RGB order
                    
                    # Create RGB composite - use same dimensions as pred_data
                    rgb = np.zeros((pred_data.shape[0], pred_data.shape[1], 3), dtype=np.float32)
                    
                    for i, band in enumerate(rgb_bands):
                        with rasterio.open(merged_path) as src:
                            band_data = src.read(band + 1)  # +1 because rasterio uses 1-based indexing
                            # Always resample to match prediction shape
                            from rasterio.warp import reproject, Resampling
                            band_resampled = np.zeros(pred_data.shape, dtype=band_data.dtype)
                            reproject(
                                band_data,
                                band_resampled,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=src.crs,
                                resampling=Resampling.bilinear
                            )
                            rgb[:, :, i] = band_resampled
                    
                    # Normalize for Sentinel-2 display
                    rgb_min = 0 
                    rgb_max = 2000
                    contrast = 1.0
                    gamma = 1.0
                    
                    # Create a normalized RGB array
                    rgb_norm = np.zeros_like(rgb, dtype=np.uint8)
                    
                    # Process each channel
                    for i in range(3):
                        rgb_norm[:,:,i] = scale_adjust_band(
                            rgb[:,:,i], rgb_min, rgb_max, contrast, gamma)
                    
                    # Display with aspect='equal' to match other plots
                    axes[1,1].imshow(rgb_norm, aspect='equal')
                    axes[1,1].set_title('RGB Composite')
                else:
                    axes[1,1].set_title('RGB Not Available (Insufficient Bands)')
                    # Add empty plot with correct aspect ratio
                    axes[1,1].imshow(np.zeros_like(pred_data), cmap='gray', aspect='equal')
        except Exception as e:
            print(f"Error creating RGB composite: {e}")
            axes[1,1].set_title('RGB Error')
            # Add empty plot with correct aspect ratio
            axes[1,1].imshow(np.zeros_like(pred_data), cmap='gray', aspect='equal')
    else:
        axes[1,1].set_title('RGB Not Available')
        # Add empty plot with correct aspect ratio
        axes[1,1].imshow(np.zeros_like(pred_data), cmap='gray', aspect='equal')
    
    # Remove axes ticks to save space
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout to prevent cut-off
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def get_training_info(csv_path):
    """Extract information from training data."""
    if not os.path.exists(csv_path):
        return {
            'sample_size': 0,
            'band_names': [],
            'height_range': (0, 0)
        }
        
    df = pd.read_csv(csv_path)
    bands = [col for col in df.columns if col not in ['rh', 'longitude', 'latitude']]
    
    return {
        'sample_size': len(df),
        'band_names': sorted(bands),
        'height_range': (df['rh'].min(), df['rh'].max())
    }


def calculate_area(bounds: tuple, crs: CRS):
    """Calculate area in hectares from bounds."""
    # For geographic coordinates, convert to appropriate UTM zone
    if crs.is_geographic:
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone + (0 if center_lat >= 0 else 100)
        utm_crs = CRS.from_epsg(utm_epsg)
        bounds = transform_bounds(crs, utm_crs, *bounds)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area_m2 = width * height
    else:
        # For projected coordinates, directly calculate area
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area_m2 = width * height
        
    # Convert to hectares
    return area_m2 / 10000


def save_evaluation_to_pdf(pred_path, ref_path, pred_data, ref_data, metrics, 
                         output_dir, training_data_path=None, merged_data_path=None,
                         mask=None, area_ha=None):
    """Create PDF report with evaluation results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 2x2 visualization grid
    grid_path = os.path.join(output_dir, 'comparison_grid.png')
    create_2x2_visualization(ref_path, pred_path, merged_data_path, grid_path, mask)
    
    # create plots
    create_plots(pred_data, ref_data, metrics, output_dir)
    
    # validate data info
    validate_data(pred_data, ref_data)
    
    # Get area if not provided
    if area_ha is None:
        with rasterio.open(pred_path) as src:
            area_ha = calculate_area(src.bounds, src.crs)
    
    # Get training info if available
    if training_data_path and os.path.exists(training_data_path):
        train_info = get_training_info(training_data_path)
    else:
        train_info = {'sample_size': 0, 'band_names': [], 'height_range': (0, 0)}
    
    # Format filename with date and area
    date = datetime.now().strftime("%Y%m%d")
    n_bands = len(train_info['band_names']) if train_info['band_names'] else 'X'
    pdf_name = f"{date}_b{n_bands}_{int(area_ha)}ha.pdf"
    pdf_path = os.path.join(output_dir, pdf_name)
    
    # Create PDF with multiple pages if needed
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # First page - Info and metrics
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Canopy Height Model Evaluation Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height-70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Add training data info
    y = height-100
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Training Data:")
    c.setFont("Helvetica", 10)
    y -= 15
    c.drawString(70, y, f"Sample Size: {train_info['sample_size']:,}")
    y -= 15
    c.drawString(70, y, f"Input Bands: {', '.join(train_info['band_names'])}")
    y -= 15
    c.drawString(70, y, f"Height Range: {train_info['height_range'][0]:.1f}m to {train_info['height_range'][1]:.1f}m")
    
    # Add area info
    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Area Information:")
    c.setFont("Helvetica", 10)
    y -= 15
    c.drawString(70, y, f"Total Area: {area_ha:,.1f} hectares")
    
    # Add metrics
    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Evaluation Metrics:")
    c.setFont("Helvetica", 10)
    y -= 15
    for metric, value in metrics.items():
        if metric.endswith('(%)'):
            c.drawString(70, y, f"{metric}: {value:,.1f}%")
        else:
            c.drawString(70, y, f"{metric}: {value:,.3f}")
        y -= 15
    
    # Add note about visualization
    y -= 25
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, y, "See next page for visualization grid")
    
    # Save current page and start a new one for the visualization
    c.showPage()
    
    # Second page - Full-page visualization grid
    # Add a title to the second page
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-40, "Canopy Height Model Comparison Grid")
    
    # Position the grid to use most of the page
    if os.path.exists(grid_path):
        # Allow more space for the grid by using most of the page
        grid_height = height - 80  # Leave space for the title
        grid_width = width - 100   # Leave margins
        c.drawImage(grid_path, 50, height-grid_height-40, width=grid_width, height=grid_height, preserveAspectRatio=True)
    
    # Save and close the PDF
    c.save()
    return pdf_path