import os
import pandas as pd
import rasterio
from rasterio.warp import transform, reproject
import numpy as np
from utils import get_latest_file
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
import json

def analyze_heights(df: pd.DataFrame, height_columns: list, ref_column: str = 'reference_height'):
    """
    Analyze relationships between reference heights and other height columns.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(height_columns, list):
        raise TypeError("height_columns must be a list")
    if ref_column not in df.columns:
        raise ValueError(f"Reference column '{ref_column}' not found in DataFrame")
        
    # Filter out nodata and -32767 values from reference height
    valid_mask = (df[ref_column] != -32767) & df[ref_column].notna()
    valid_data = df[valid_mask].copy()
    
    print(f"\nHeight Analysis (using {len(valid_data)} valid points):")
    print("-" * 50)
    
    # Create a figure with subplots
    n_plots = len(height_columns)
    if n_plots > 0:
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(12, 5*n_rows))
        axes = []
    else:
        print("No height columns to analyze")
        return None, None
    
    stats_dict = {}
    error_matrix = {}
    
    # Process each height column
    for idx, col in enumerate(height_columns):
        if col in valid_data.columns:
            print(f"\nAnalyzing {col}...")
            # Get valid pairs (no NaN in either column)
            valid_pairs = valid_data[[ref_column, col]].dropna()
            n_valid = len(valid_pairs)
            print(f"Found {n_valid} valid pairs for {col}")
            
            if n_valid < 2:
                print(f"Skipping {col} - insufficient valid pairs")
                continue
                
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            axes.append(ax)
            
            try:
                # Convert to numpy arrays and ensure float type
                ref_vals = valid_pairs[ref_column].values.astype(float)
                col_vals = valid_pairs[col].values.astype(float)
                
                # Calculate statistics
                rmse = float(np.sqrt(mean_squared_error(ref_vals, col_vals)))
                mae = float(mean_absolute_error(ref_vals, col_vals))
                bias = float(np.mean(col_vals - ref_vals))
                correlation = float(np.corrcoef(ref_vals, col_vals)[0, 1])
                
                # Calculate error statistics
                errors = col_vals - ref_vals
                error_std = float(np.std(errors))
                error_percentiles = [float(x) for x in np.percentile(errors, [5, 25, 50, 75, 95])]
                
                # Calculate regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(ref_vals, col_vals)
                r_squared = r_value ** 2
                
                # Store statistics
                stats_dict[col] = {
                    'N': n_valid,
                    'Correlation': correlation,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Bias': bias,
                    'R-squared': r_squared,
                    'Slope': float(slope),
                    'Intercept': float(intercept),
                    'Error_std': error_std,
                    'Error_p5': error_percentiles[0],
                    'Error_p25': error_percentiles[1],
                    'Error_p50': error_percentiles[2],
                    'Error_p75': error_percentiles[3],
                    'Error_p95': error_percentiles[4]
                }
                error_matrix[col] = {'RMSE': rmse}
                
                # Print results
                print(f"Correlation: {correlation:.3f}")
                print(f"RMSE: {rmse:.3f} m")
                print(f"MAE: {mae:.3f} m")
                print(f"Bias: {bias:.3f} m")
                print(f"R-squared: {r_squared:.3f}")
                print(f"Linear fit: y = {slope:.3f}x + {intercept:.3f}")
                
                # Create scatter plot
                ax.scatter(ref_vals, col_vals, alpha=0.5, s=10)
                
                # Add identity line
                min_val = min(np.nanmin(ref_vals), np.nanmin(col_vals))
                max_val = max(np.nanmax(ref_vals), np.nanmax(col_vals))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
                
                # Add regression line
                x_range = np.array([min_val, max_val])
                ax.plot(x_range, slope * x_range + intercept, 'g-', 
                       label=f'Regression (R²={r_squared:.3f})')
                
                ax.set_xlabel(f'Reference Height (m)')
                ax.set_ylabel(f'{col} (m)')
                ax.set_title(f'Reference vs {col}\nRMSE={rmse:.2f}m, Bias={bias:.2f}m')
                ax.grid(True)
                ax.legend()
                
            except Exception as e:
                print(f"Error analyzing {col}: {str(e)}")
                print(f"Data types - ref: {type(ref_vals)}, col: {type(col_vals)}")
                print(f"Data ranges - ref: [{np.nanmin(ref_vals)}, {np.nanmax(ref_vals)}], "
                      f"col: [{np.nanmin(col_vals)}, {np.nanmax(col_vals)}]")
                continue
    
    stats_dict['error_matrix'] = error_matrix
    
    if len(axes) > 0:
        plt.tight_layout()
        return stats_dict, fig
    else:
        return None, None

def combine_heights_with_training(output_dir: str, reference_path: str):
    """Combine reference heights with training data coordinates."""
    # Input validation
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")
    if not os.path.exists(reference_path):
        raise ValueError(f"Reference file does not exist: {reference_path}")
    
    # Get latest training data file
    training_file = get_latest_file(output_dir, 'training_data')
    print(f"Using training file: {training_file}")
    
    # Read training data
    df = pd.read_csv(training_file)
    
    # Open reference raster
    with rasterio.open(reference_path) as src:
        print(f"Reference CRS: {src.crs}")
        
        # Create lists of coordinates
        lons = df['longitude'].values
        lats = df['latitude'].values
        
        # Transform coordinates from EPSG:4326 to reference CRS
        xs, ys = transform('EPSG:4326', src.crs, lons, lats)
        
        # Sample the raster at transformed coordinates
        coords = list(zip(xs, ys))
        samples = list(src.sample(coords))
        heights = [sample[0] for sample in samples]
    
    # Create a Series with the heights
    height_series = pd.Series(heights)
    
    # Replace -32767 with pd.NA
    height_series = height_series.replace(-32767, pd.NA)
    
    # Add heights to dataframe
    df['reference_height'] = height_series
    
    # Analyze relationships with other height columns
    height_columns = ['rh', 'ht2021', 'ht2022', 'ht2023']
    
    # Check column existence
    available_columns = [col for col in height_columns if col in df.columns]
    if not available_columns:
        print("\nWarning: None of the specified height columns found in the data")
        print(f"Available columns: {df.columns.tolist()}")
    else:
        print(f"\nAnalyzing height columns: {available_columns}")
        stats, fig = analyze_heights(df, available_columns)
        
        if stats is not None:
            # Save the statistics to a JSON file
            stats_file = os.path.join(output_dir, 'trainingData_height_analysis.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"\nStatistics saved to: {stats_file}")
            
            # Save the analysis plot
            plot_path = os.path.join(output_dir, 'trainingData_height_comparison.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Analysis plot saved to: {plot_path}")
            
            # Print error matrix
            print("\nError Matrix (RMSE between reference and height columns):")
            error_df = pd.DataFrame.from_dict({col: data['RMSE'] 
                                   for col, data in stats['error_matrix'].items()}, 
                                   orient='index', columns=['RMSE'])
            print(error_df.round(3))
    
    # Save combined data with explicit NA handling
    output_file = os.path.join(output_dir, 'trainingData_with_heights.csv')
    df.to_csv(output_file, index=False, na_rep='NA')
    
    # Print summary statistics
    valid_heights = df['reference_height'].dropna()
    print(f"\nSummary:")
    print(f"Total points: {len(df)}")
    print(f"Valid heights: {len(valid_heights)}")
    print(f"No data points: {len(df) - len(valid_heights)}")
    if len(valid_heights) > 0:
        print(f"Height range: {valid_heights.min():.2f} to {valid_heights.max():.2f}")
    print(f"\nCombined data saved to: {output_file}")

if __name__ == "__main__":
    # Use chm_outputs directory
    output_dir = "chm_outputs"
    reference = f"{output_dir}/dchm_09id4.tif"
    combine_heights_with_training(output_dir, reference)