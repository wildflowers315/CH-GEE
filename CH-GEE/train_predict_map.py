import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import rasterio
from rasterio.mask import geometry_mask
from shapely.geometry import Point, box
from shapely.ops import transform
import geopandas as gpd
import os
from pathlib import Path
from typing import Tuple, Optional
import warnings
import argparse
warnings.filterwarnings('ignore')

from evaluate_predictions import calculate_metrics

def load_training_data(csv_path: str, mask_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from CSV file and optionally mask with forest mask.
    
    Args:
        csv_path: Path to training data CSV
        mask_path: Optional path to forest mask TIF
        
    Returns:
        X: Feature matrix
        y: Target variable (rh)
    """
    # Read training data
    df = pd.read_csv(csv_path)
    
    # Create GeoDataFrame from points
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
        crs="EPSG:4326"
    )
    
    if mask_path:
        with rasterio.open(mask_path) as mask_src:
            # Check CRS
            mask_crs = mask_src.crs
            if mask_crs != gdf.crs:
                gdf = gdf.to_crs(mask_crs)
            
            # Get bounds of mask
            mask_bounds = box(*mask_src.bounds)
            
            # First filter points by mask bounds
            gdf = gdf[gdf.geometry.within(mask_bounds)]
            
            if len(gdf) == 0:
                raise ValueError("No training points fall within the mask bounds")
            
            # Convert points to pixel coordinates
            pts_pixels = []
            valid_indices = []
            for idx, point in enumerate(gdf.geometry):
                row, col = rasterio.transform.rowcol(mask_src.transform, 
                                                   point.x, 
                                                   point.y)
                if (0 <= row < mask_src.height and 
                    0 <= col < mask_src.width):
                    pts_pixels.append((row, col))
                    valid_indices.append(idx)
            
            if not pts_pixels:
                raise ValueError("No training points could be mapped to valid pixels")
            
            # Read forest mask values at pixel locations
            mask_values = [mask_src.read(1)[r, c] for r, c in pts_pixels]
            
            # Filter points by mask values
            mask_indices = [i for i, v in enumerate(mask_values) if v == 1]
            if not mask_indices:
                raise ValueError("No training points fall within the forest mask")
            
            final_indices = [valid_indices[i] for i in mask_indices]
            gdf = gdf.iloc[final_indices]
    
    # Convert back to original CRS if needed
    if mask_path and mask_crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Separate features and target
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    y = df['rh'].values
    X = df.drop(['rh', 'longitude', 'latitude'], axis=1, errors='ignore').values
    
    return X, y

def load_prediction_data(stack_path: str, mask_path: Optional[str] = None) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """
    Load prediction data from stack TIF and optionally apply forest mask.
    
    Args:
        stack_path: Path to stack TIF file
        mask_path: Optional path to forest mask TIF
        
    Returns:
        X: Feature matrix for prediction
        src: Rasterio dataset for writing results
    """
    # Read stack file
    with rasterio.open(stack_path) as src:
        stack = src.read()
        stack_crs = src.crs
        
        # Reshape stack to 2D array (bands x pixels)
        n_bands, height, width = stack.shape
        X = stack.reshape(n_bands, -1).T
        
        # Apply mask if provided
        if mask_path:
            with rasterio.open(mask_path) as mask_src:
                # Check CRS
                if mask_src.crs != stack_crs:
                    raise ValueError(f"CRS mismatch: stack {stack_crs} != mask {mask_src.crs}")
                
                # Check dimensions
                if mask_src.shape != (height, width):
                    raise ValueError(f"Shape mismatch: stack {(height, width)} != mask {mask_src.shape}")
                
                mask = mask_src.read(1)
                mask = mask.reshape(-1)
                X = X[mask == 1]
        
        src_copy = rasterio.open(stack_path)
        return X, src_copy

def train_model(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> RandomForestRegressor:
    """
    Train Random Forest model with optional validation split.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data to use for validation
        
    Returns:
        Trained model
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    rf = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Print validation score
    val_score = rf.score(X_val, y_val)
    y_pred = rf.predict(X_val)
    train_metrix = calculate_metrics(y_pred, y_val)
    # print(f"Validation R² score: {val_score:.3f}")
    for matrix, value in train_metrix.items():
        print (f"{matrix}: {value:.3f}")
    
    return rf, train_metrix

def save_predictions(predictions: np.ndarray, src: rasterio.DatasetReader, output_path: str,
                    mask_path: Optional[str] = None) -> None:
    """
    Save predictions to a GeoTIFF file.
    
    Args:
        predictions: Model predictions
        src: Source rasterio dataset for metadata
        output_path: Path to save predictions
        mask_path: Optional path to forest mask TIF
    """
    # Create output profile
    profile = src.profile.copy()
    profile.update(count=1, dtype='float32')
    
    # Initialize prediction array
    height, width = src.height, src.width
    pred_array = np.zeros((height, width), dtype='float32')
    
    if mask_path:
        # Apply predictions only to masked areas
        with rasterio.open(mask_path) as mask_src:
            # Check CRS
            if mask_src.crs != src.crs:
                raise ValueError(f"CRS mismatch: source {src.crs} != mask {mask_src.crs}")
            
            mask = mask_src.read(1)
            mask_idx = np.where(mask.reshape(-1) == 1)[0]
            pred_array.reshape(-1)[mask_idx] = predictions
    else:
        # Apply predictions to all pixels
        pred_array = predictions.reshape(height, width)
    
    try:
        # Save predictions
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(pred_array, 1)
    finally:
        src.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train model and generate canopy height predictions')
    
    # Input paths
    parser.add_argument('--training-data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--stack', type=str, required=True,
                       help='Path to stack TIF file')
    parser.add_argument('--mask', type=str, required=True,
                       help='Path to forest mask TIF')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='chm_outputs',
                       help='Output directory for predictions')
    parser.add_argument('--output-filename', type=str, default='canopy_height_predictions.tif',
                       help='Output filename for predictions')
    
    # Model parameters
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for validation')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    X, y = load_training_data(args.training_data, args.mask)
    print(f"Loaded training data with {X.shape[1]} features and {len(y)} samples")
    
    # Train model
    print("Training model...")
    model, train_metrix = train_model(X, y, args.test_size)
    
    # Load prediction data
    print("Loading prediction data...")
    X_pred, src = load_prediction_data(args.stack, args.mask)
    print(f"Loaded prediction data with shape: {X_pred.shape}")
    
    # Make predictions
    print("Generating predictions...")
    predictions = model.predict(X_pred)
    print(f"Generated {len(predictions)} predictions")
    output_path = Path(args.stack).stem.replace('stack_', 'predictCH')
    
    # Save predictions
    # output_path = os.path.join(args.output_dir, output_filename)
    print(f"Saving predictions to: {output_path}")
    save_predictions(predictions, src, output_path, args.mask)
    print("Done!")

if __name__ == "__main__":
    main()