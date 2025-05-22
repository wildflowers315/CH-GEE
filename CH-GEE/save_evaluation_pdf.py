"""Module for generating PDF evaluation reports."""

import os
import json
import pandas as pd
import numpy as np
import rasterio
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from raster_utils import load_and_align_rasters
from utils import get_latest_file


def scale_adjust_band(band_data, min_val, max_val, contrast=1.0, gamma=1.0):
    """Adjust band data with min/max scaling, contrast, and gamma."""
    # Handle NaN values
    nan_mask = np.isnan(band_data)
    temp_nodata = -9999
    work_data = band_data.copy()
    
    if np.any(work_data[~nan_mask] == temp_nodata):
        valid_min = np.min(work_data[~nan_mask]) if not nan_mask.all() else -1
        temp_nodata = valid_min - 1

    work_data[nan_mask] = temp_nodata
    work_data = work_data.astype(np.float32)

    # Min/Max scaling
    mask_valid = (work_data != temp_nodata)
    scaled_data = np.zeros_like(work_data, dtype=np.float32)
    if max_val - min_val != 0:
        scaled_data[mask_valid] = (work_data[mask_valid] - min_val) / (max_val - min_val)
    scaled_data[mask_valid] = np.clip(scaled_data[mask_valid], 0, 1)

    # Contrast adjustment
    if contrast != 1.0:
        scaled_data[mask_valid] = 0.5 + contrast * (scaled_data[mask_valid] - 0.5)
        scaled_data[mask_valid] = np.clip(scaled_data[mask_valid], 0, 1)

    # Gamma correction
    if gamma != 1.0 and gamma > 0:
        gamma_mask = mask_valid & (scaled_data > 0)
        with np.errstate(invalid='ignore'):
            scaled_data[gamma_mask] = scaled_data[gamma_mask]**(1.0 / gamma)
        scaled_data[gamma_mask] = np.clip(scaled_data[gamma_mask], 0, 1)

    # Convert to uint8
    scaled_data[~mask_valid] = 0
    scaled_uint8 = (scaled_data * 255).astype(np.uint8)
    return scaled_uint8


def load_rgb_composite(merged_path, target_shape, transform, temp_dir=None):
    """Load and process RGB composite from merged data."""
    merged_file_name = os.path.basename(merged_path)
    if temp_dir is None:
        temp_dir = os.path.dirname(merged_path)
    merged_clipped_path = os.path.join(temp_dir, f"{merged_file_name.split('.')[0]}_clipped.tif")
    os.makedirs(os.path.dirname(merged_clipped_path), exist_ok=True)
        
    try:
        with rasterio.open(merged_path) as src:
            if src.count >= 4:  # Check if we have enough bands
                # Use S2 bands 4,3,2 (R,G,B) for natural color
                rgb_bands = [3, 2, 1]  # B4 (R, 665nm), B3 (G, 560nm), B2 (B, 490nm)
                rgb = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.float32)
                
                from rasterio.warp import reproject, Resampling
                for i, band in enumerate(rgb_bands):
                    band_data = src.read(band)  # Band numbers are 1-based
                    band_resampled = np.zeros(target_shape, dtype=np.float32)
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
                
                # Apply band-specific scaling for Sentinel-2 reflectance values
                rgb_norm = np.zeros_like(rgb, dtype=np.uint8)
                # Sentinel-2 L2A typical reflectance ranges
                scale_params = [
                    {'min': 0, 'max': 3000, 'contrast': 1.2, 'gamma': 0.8},  # Red (B4)
                    {'min': 0, 'max': 3000, 'contrast': 1.2, 'gamma': 0.8},  # Green (B3)
                    {'min': 0, 'max': 3000, 'contrast': 1.2, 'gamma': 0.8}    # Blue (B2)
                ]
                for i in range(3):
                    rgb_norm[:,:,i] = scale_adjust_band(
                        rgb[:,:,i],
                        scale_params[i]['min'],
                        scale_params[i]['max'],
                        contrast=scale_params[i]['contrast'],
                        gamma=scale_params[i]['gamma']
                    )
                
                # save the RGB composite
                profile = src.profile.copy()
                profile.update({
                    'height': target_shape[0],
                    'width': target_shape[1],
                    'transform': transform,
                    'count': 3,
                    'dtype': 'uint8'
                })
                try:
                    with rasterio.open(merged_clipped_path, 'w', **profile) as dst:
                        # Write bands in correct order (R,G,B)
                        dst.write(rgb_norm.transpose(2, 1, 0))
                except Exception as e:
                    print(f"Warning: Could not save RGB composite: {e}")
                    print(f"Attempted to save to: {merged_clipped_path}")
                    # Continue even if saving fails - we can still use the RGB data in memory
                return rgb_norm
    except Exception as e:
        print(f"Error creating RGB composite: {e}")
    return None


def create_2x2_visualization(ref_data, pred_data, diff_data, merged_path, transform, output_path, mask=None, forest_mask=None, temp_dir=None):
    """Create 2x2 grid with reference, prediction, difference and RGB data."""
    
    # Load RGB composite if available
    rgb_norm = None
    if merged_path and os.path.exists(merged_path):
        rgb_norm = load_rgb_composite(merged_path, pred_data.shape, transform, temp_dir)
    else:
        print("Merged data not found or invalid. Skipping RGB composite creation.")
    # Apply mask if provided
    # Combine validity mask with forest mask if provided
    final_mask = mask if mask is not None else np.ones_like(pred_data, dtype=bool)
    if forest_mask is not None:
        final_mask = final_mask & forest_mask
        
    # if final_mask is not None and rgb_norm is not None:
    #     # Create a 3D mask by expanding dimensions
    #     mask_3d = np.repeat(final_mask[:, :, np.newaxis], 3, axis=2)
    #     # Apply mask - set masked areas to 0
    #     rgb_norm = np.where(mask_3d, rgb_norm, 0)
        
    from evaluation_utils import create_comparison_grid
    create_comparison_grid(ref_data, pred_data, diff_data, rgb_norm, output_path, 
                           forest_mask=final_mask)
    return output_path


def format_band_names(bands, line_length=80):
    """Format band names into multiple lines."""
    lines = []
    current_line = []
    current_length = 0
    
    for band in bands:
        if current_length + len(band) + 2 > line_length:  # +2 for comma and space
            lines.append(', '.join(current_line))
            current_line = [band]
            current_length = len(band)
        else:
            current_line.append(band)
            current_length += len(band) + 2  # +2 for comma and space
            
    if current_line:
        lines.append(', '.join(current_line))
    
    return '\n'.join(lines)

def get_training_info(csv_path):
    """Extract information from training data."""
    if not os.path.exists(csv_path):
        return {'sample_size': 0, 'band_names': [], 'height_range': (0, 0)}
        
    df = pd.read_csv(csv_path)
    bands = [col for col in df.columns if col not in ['rh', 'longitude', 'latitude']]
    
    return {
        'sample_size': len(df),
        'band_names': sorted(bands),
        'height_range': (df['rh'].min(), df['rh'].max())
    }


def calculate_area(bounds: tuple, crs: CRS):
    """Calculate area in hectares from bounds."""
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
    return area_m2 / 10000


def create_feature_importance_chart(data, width, height):
    """Create a bar chart showing feature importance."""
    drawing = Drawing(width, height)
    
    # Sort data by importance value
    sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
    values = list(sorted_data.values())[:10]  # Top 10 features
    names = list(sorted_data.keys())[:10]
    
    # Create and customize the chart
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 50
    chart.height = height - 100
    chart.width = width - 100
    chart.data = [values]
    chart.categoryAxis.categoryNames = names
    chart.categoryAxis.labels.boxAnchor = 'ne'
    chart.categoryAxis.labels.angle = 45
    chart.categoryAxis.labels.dx = -10
    chart.categoryAxis.labels.dy = -20
    chart.bars[0].fillColor = colors.blue
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(values) * 1.1
    chart.valueAxis.valueStep = max(values) / 5
    
    drawing.add(chart)
    return drawing

def save_evaluation_to_pdf(pred_path, ref_path, pred_data, ref_data, metrics,
                          output_dir, training_data_path=None, merged_data_path=None,
                          mask=None, forest_mask=None, area_ha=None, validation_info=None, plot_paths=None):
    """Create PDF report with evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate difference for visualization
    diff_data = pred_data - ref_data
    
    # Create comparison grid visualization
    grid_path = os.path.join(output_dir, 'comparison_grid.png')
    with rasterio.open(pred_path) as src:
        transform = src.transform
    # Create a temp directory for RGB composites within output_dir
    rgb_temp_dir = os.path.join(output_dir, 'rgb_temp')
    os.makedirs(rgb_temp_dir, exist_ok=True)
    
    create_2x2_visualization(
        ref_data, pred_data, diff_data,
        merged_data_path, transform, grid_path,
        mask=mask, forest_mask=forest_mask,
        temp_dir=rgb_temp_dir
    )
    
    # Get area if not provided
    if area_ha is None:
        with rasterio.open(pred_path) as src:
            area_ha = calculate_area(src.bounds, src.crs)
    
    # Get training info
    train_info = get_training_info(training_data_path) if training_data_path else {
        'sample_size': 0, 'band_names': [], 'height_range': (0, 0)
    }
    
    # Create PDF
    date = datetime.now().strftime("%Y%m%d")
    n_bands = len(train_info['band_names']) if train_info['band_names'] else 'X'
    pdf_name = f"{date}_b{n_bands}_{int(area_ha)}ha.pdf"
    pdf_path = os.path.join(output_dir, pdf_name)
    
    # Initialize PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # First page - Summary information
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Canopy Height Model Evaluation Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height-70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Add validation info
    y = height-100
    if validation_info:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Data Statistics:")
        c.setFont("Helvetica", 10)
        y -= 15
        c.drawString(70, y, "Prediction Data:")
        y -= 15
        c.drawString(90, y, f"Range: {validation_info['pred_range'][0]:.2f}m to {validation_info['pred_range'][1]:.2f}m")
        y -= 15
        c.drawString(90, y, f"Mean: {validation_info['pred_stats']['mean']:.2f}m, Std: {validation_info['pred_stats']['std']:.2f}m")
        y -= 15
        c.drawString(70, y, "Reference Data:")
        y -= 15
        c.drawString(90, y, f"Range: {validation_info['ref_range'][0]:.2f}m to {validation_info['ref_range'][1]:.2f}m")
        y -= 15
        c.drawString(90, y, f"Mean: {validation_info['ref_stats']['mean']:.2f}m, Std: {validation_info['ref_stats']['std']:.2f}m")
        y -= 25
    
    # Add training info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Training Data:")
    c.setFont("Helvetica", 10)
    y -= 15
    c.drawString(70, y, f"Sample Size: {train_info['sample_size']:,}")
    y -= 15
    
    # Format band names into multiple lines
    c.drawString(70, y, "Input Bands:")
    y -= 15
    formatted_bands = format_band_names(train_info['band_names'], line_length=80)
    for line in formatted_bands.split('\n'):
        c.drawString(90, y, line)
        y -= 15
    
    c.drawString(70, y, f"Height Range: {train_info['height_range'][0]:.1f}m to {train_info['height_range'][1]:.1f}m")
    
    # Add training metrics if available
    chm_outputs_dir = os.path.dirname(os.path.dirname(output_dir))  # Get chm_outputs directory
    print(f"Looking for model_evaluation.json in: {chm_outputs_dir}")
    model_eval_path = get_latest_file(chm_outputs_dir, 'model_evaluation', required=False)
    if model_eval_path:
        print(f"Found model evaluation file at: {model_eval_path}")
    else:
        print("No model evaluation file found")
    if model_eval_path and os.path.exists(model_eval_path):
        try:
            with open(model_eval_path) as f:
                model_data = json.load(f)
            
            if 'train_metrics' in model_data:
                y -= 25
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, "Training Metrics:")
                c.setFont("Helvetica", 10)
                y -= 15
                for metric, value in model_data['train_metrics'].items():
                    metric_name = metric.replace('_', ' ').title()
                    if isinstance(value, float):
                        if metric.endswith('(%)'):
                            c.drawString(70, y, f"{metric_name}: {value:.1f}%")
                        else:
                            c.drawString(70, y, f"{metric_name}: {value:.3f}")
                    else:
                        c.drawString(70, y, f"{metric_name}: {value}")
                    y -= 15
        except Exception as e:
            print(f"Warning: Could not load training metrics: {e}")
    
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
    
    c.showPage()
    
    # Second page - Comparison grid
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-40, "Canopy Height Model Comparison Grid")
    
    if os.path.exists(grid_path):
        grid_height = height - 80
        grid_width = width - 100
        c.drawImage(grid_path, 50, height-grid_height-40, width=grid_width, height=grid_height, preserveAspectRatio=True)
    
    c.showPage()
    
    # Third page - Analysis plots
    if plot_paths:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height-40, "Detailed Analysis")
        
        y = height - 60
        plot_height = (height - 100) / 2
        
        if os.path.exists(plot_paths.get('scatter', '')):
            c.drawImage(plot_paths['scatter'], 50, y-plot_height, width=width/2-60, height=plot_height, preserveAspectRatio=True)
        
        if os.path.exists(plot_paths.get('error_hist', '')):
            c.drawImage(plot_paths['error_hist'], width/2, y-plot_height, width=width/2-60, height=plot_height, preserveAspectRatio=True)
        
        if os.path.exists(plot_paths.get('height_dist', '')):
            y -= plot_height + 20
            c.drawImage(plot_paths['height_dist'], width/4, y-plot_height, width=width/2-60, height=plot_height, preserveAspectRatio=True)
        
        c.showPage()
    
    def draw_feature_importance_table(c, data, x, y, width):
        """Draw a table of feature importance values."""
        # Sort features by importance
        sorted_features = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
        total_importance = sum(sorted_features.values())
        
        # Calculate column widths
        feature_width = width * 0.6
        value_width = width * 0.2
        percent_width = width * 0.2
        
        # Draw table border
        c.rect(x, y-14, width, 30)  # Header box
        
        # Table header
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x + 5, y, "Feature")
        c.drawRightString(x + feature_width + value_width - 5, y, "Importance")
        c.drawRightString(x + width - 5, y, "Percent")
        y -= 15
        
        # Draw horizontal line under header
        c.setLineWidth(0.5)
        c.line(x, y+2, x + width, y+2)
        y -= 15
        
        # Draw vertical lines
        c.line(x + feature_width, y+32, x + feature_width, y-14*len(sorted_features))  # After Feature
        c.line(x + feature_width + value_width, y+32, x + feature_width + value_width, y-14*len(sorted_features))  # After Importance
        
        # Table content
        row = 0
        c.setFont("Helvetica", 9)
        for feature, importance in sorted_features.items():
            # Alternate row colors
            if row % 2 == 0:
                c.setFillColorRGB(0.95, 0.95, 0.95)
                c.rect(x, y-3, width, 14, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)
            
            # Draw row content with padding
            c.drawString(x + 5, y, feature)
            c.drawRightString(x + feature_width + value_width - 5, y, f"{importance:.4f}")
            percentage = (importance / total_importance) * 100
            c.drawRightString(x + width - 5, y, f"{percentage:.1f}%")
            
            # Draw horizontal line after each row
            if row < len(sorted_features) - 1:
                c.setLineWidth(0.1)
                c.line(x, y-7, x + width, y-7)
            
            y -= 14
            row += 1
            
            if y < 50:  # Start new column if near bottom
                y = height - 100
                x += width + 20
                row = 0  # Reset row counter for new column
                
        return y

    # Fourth page - Feature Importance
    chm_outputs_dir = os.path.dirname(os.path.dirname(output_dir))  # Get chm_outputs directory
    model_eval_path = get_latest_file(chm_outputs_dir, 'model_evaluation', required=False)
    if model_eval_path:
        try:
            with open(model_eval_path) as f:
                model_data = json.load(f)
            
            if 'feature_importance' in model_data:
                # Add title
                c.setFont("Helvetica-Bold", 14)
                # c.drawString(50, height-40, "Feature Importance Analysis")
                
                # # Add chart
                # chart = create_feature_importance_chart(model_data['feature_importance'], width-100, height/2)
                # chart.drawOn(c, 50, height/2)
                
                # Add table below chart
                table_y = height/2 - 20
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, table_y, "Feature Importance Values")
                table_y -= 20
                draw_feature_importance_table(c, model_data['feature_importance'], 50, table_y, width/2-70)
                
                c.showPage()
        except Exception as e:
            print(f"Warning: Could not load feature importance data: {e}")
    # if model_eval_path and os.path.exists(model_eval_path):
    #     try:
    #         with open(model_eval_path) as f:
    #             model_data = json.load(f)
            
    #         if 'feature_importance' in model_data:
    #             c.setFont("Helvetica-Bold", 14)
    #             c.drawString(50, height-40, "Feature Importance Analysis")
                
    #             # Create and add feature importance chart
    #             chart = create_feature_importance_chart(model_data['feature_importance'], width-100, height-100)
    #             chart.drawOn(c, 50, 50)
                
    #             c.showPage()
    #     except Exception as e:
    #         print(f"Warning: Could not load feature importance data: {e}")
    
    c.save()
    return pdf_path