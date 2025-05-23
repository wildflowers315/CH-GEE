import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from dl_models import MLPRegressionModel, create_normalized_dataloader
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
from tqdm import tqdm
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
            gdf_masked = gdf[gdf.geometry.within(mask_bounds)]
            
            if len(gdf_masked) == 0:
                raise ValueError("No training points fall within the mask bounds")
            else:
                gdf = gdf_masked
            
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

def save_metrics_and_importance(metrics: dict, importance_data: dict, output_dir: str) -> None:
    """
    Save training metrics and feature importance to JSON file, ensuring all values are JSON serializable.
    """
    # Convert any non-serializable values to Python native types
    serializable_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # Handle numpy/torch numbers
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = float(value)
    
    serializable_importance = {}
    for key, value in importance_data.items():
        if hasattr(value, 'item'):  # Handle numpy/torch numbers
            serializable_importance[key] = value.item()
        else:
            serializable_importance[key] = float(value)
    """
    Save training metrics and feature importance to JSON file.
    
    Args:
        metrics: Dictionary of training metrics
        importance_data: Dictionary of feature importance data
        output_dir: Directory to save JSON file
    """
    import json
    from pathlib import Path
    
    # Combine metrics and importance data
    output_data = {
        "train_metrics": serializable_metrics,
        "feature_importance": serializable_importance
    }
    
    # Create output path
    output_path = Path(output_dir) / "model_evaluation.json"
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved model evaluation data to: {output_path}")

def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf', batch_size: int = 64,
                test_size: float = 0.2, feature_names: Optional[list] = None,
                n_bands: Optional[int] = None) -> Tuple[object, dict, dict]:
    """
    Train model with optional validation split.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_type: Type of model ('rf' or 'mlp')
        batch_size: Batch size for MLP training
        test_size: Proportion of data to use for validation
        feature_names: Optional list of feature names
        
    Returns:
        Trained model, training metrics, and feature importance/weights
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    if model_type == 'rf':
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_val)
        train_metrics = calculate_metrics(y_pred, y_val)
        
        # Get feature importance
        importance = model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_data = {
            name: float(imp) for name, imp in zip(feature_names, importance)
        }
        
    else:  # MLP model
        # Create normalized dataloaders
        train_loader, val_loader, scaler_mean, scaler_std = create_normalized_dataloader(
            X_train, X_val, y_train, y_val, batch_size=batch_size, n_bands=n_bands
        )
        
        # Initialize model
        model = MLPRegressionModel(input_size=X.shape[1])
        if torch.cuda.is_available():
            model = model.cuda()
            
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        num_epochs = 100
        best_val_loss = float('inf')
        
        # Training loop with tqdm progress bar
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            model.train()
            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    outputs = model(batch_X)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_metrics = calculate_metrics(val_predictions, val_targets)
            val_loss = val_metrics['RMSE']
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                train_metrics = val_metrics
        
        # Get feature importance (using weights of first layer as proxy)
        with torch.no_grad():
            weights = model.layers[0].weight.abs().mean(dim=0).cpu().numpy()
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(weights))]
            importance_data = {}
            for name, weight in zip(feature_names, weights):
                # Convert numpy.float32 to Python float
                weight_value = weight.item() if hasattr(weight, 'item') else float(weight)
                importance_data[name] = weight_value
        
        # Store normalization parameters with model
        model.scaler_mean = scaler_mean
        model.scaler_std = scaler_std
    
    # Sort importance by value
    importance_data = dict(sorted(importance_data.items(), key=lambda x: x[1], reverse=True))
    
    # Print metrics and top features
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    print("\nTop 5 Important Features:")
    for name, imp in list(importance_data.items())[:5]:
        print(f"{name}: {imp:.3f}")
    
    return model, train_metrics, importance_data

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
    # parser.add_argument('--output-filename', type=str, default='canopy_height_predictions.tif',
    #                    help='Output filename for predictions')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'mlp'],
                       help='Model type: random forest (rf) or MLP neural network (mlp)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for MLP training')
    parser.add_argument('--n-bands', type=int, default=None,
                       help='Number of spectral bands for band-wise normalization')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for validation')
    parser.add_argument('--apply-forest-mask', action='store_true',
                       help='Apply forest mask to predictions')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    df = pd.read_csv(args.training_data)
    feature_names = [col for col in df.columns if col not in ['rh', 'longitude', 'latitude']]
    X, y = load_training_data(args.training_data, args.mask)
    print(f"Loaded training data with {X.shape[1]} features and {len(y)} samples")
    
    # Train model
    print("Training model...")
    model, train_metrics, importance_data = train_model(
        X, y,
        model_type=args.model,
        batch_size=args.batch_size,
        test_size=args.test_size,
        feature_names=feature_names,
        n_bands=args.n_bands
    )
    
    # Save metrics and importance
    save_metrics_and_importance(train_metrics, importance_data, args.output_dir)
    
    # Load prediction data
    print("Loading prediction data...")
    X_pred, src = load_prediction_data(args.stack, args.mask)
    print(f"Loaded prediction data with shape: {X_pred.shape}")
    
    # Make predictions
    print("Generating predictions...")
    if args.model == 'rf':
        predictions = model.predict(X_pred)
    else:  # MLP model
        model.eval()
        with torch.no_grad():
            # Normalize prediction data
            X_pred_tensor = torch.FloatTensor(X_pred)
            X_pred_normalized = (X_pred_tensor - model.scaler_mean) / model.scaler_std
            
            # Make predictions in batches
            predictions = []
            for i in range(0, len(X_pred), args.batch_size):
                batch = X_pred_normalized[i:i + args.batch_size]
                if torch.cuda.is_available():
                    batch = batch.cuda()
                pred = model(batch)
                predictions.extend(pred.cpu().numpy())
            predictions = np.array(predictions)
    print(f"Generated {len(predictions)} predictions")
    output_path = Path(args.output_dir) / f"{Path(args.stack).stem.replace('stack_', 'predictCH')}.tif"
    
    # Save predictions
    # output_path = os.path.join(args.output_dir, output_filename)
    print(f"Saving predictions to: {output_path}")
    save_predictions(predictions, src, output_path, args.mask)
    print("Done!")

if __name__ == "__main__":
    main()