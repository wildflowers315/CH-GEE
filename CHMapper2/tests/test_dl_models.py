import pytest
import torch
import numpy as np
from dl_models import create_normalized_dataloader, MLPRegressionModel

def test_create_normalized_dataloader_independent():
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)  # 100 samples, 10 features
    X_val = np.random.randn(20, 10)     # 20 samples, 10 features
    y_train = np.random.randn(100)
    y_val = np.random.randn(20)
    
    # Get normalized dataloaders
    train_loader, val_loader, scaler_mean, scaler_std = create_normalized_dataloader(
        X_train, X_val, y_train, y_val, batch_size=32
    )
    
    # Check shapes
    assert scaler_mean.shape == (10,)
    assert scaler_std.shape == (10,)
    
    # Check normalization parameters with relaxed tolerance for float32/float64 differences
    np.testing.assert_allclose(scaler_mean.numpy(), X_train.mean(axis=0), rtol=1e-2, atol=1e-4)
    np.testing.assert_allclose(scaler_std.numpy(), X_train.std(axis=0), rtol=1e-2, atol=1e-4)
    
    # Check normalized data statistics
    batch_X, batch_y = next(iter(train_loader))
    batch_mean = batch_X.mean(dim=0)
    batch_std = batch_X.std(dim=0)
    
    # Check if normalized data has approximately zero mean and unit variance
    np.testing.assert_allclose(batch_mean.numpy(), np.zeros_like(batch_mean), atol=0.5)
    np.testing.assert_allclose(batch_std.numpy(), np.ones_like(batch_std), atol=0.5)

def test_create_normalized_dataloader_band():
    # Create sample data with band structure
    np.random.seed(42)
    n_samples_train = 100
    n_samples_val = 20
    n_groups = 3    # e.g., VV, VH, and HH polarizations
    n_bands = 4     # e.g., temporal snapshots
    
    # Create data with known band structure
    X_train = np.random.randn(n_samples_train, n_groups * n_bands)
    X_val = np.random.randn(n_samples_val, n_groups * n_bands)
    y_train = np.random.randn(n_samples_train)
    y_val = np.random.randn(n_samples_val)
    
    # Get normalized dataloaders
    train_loader, val_loader, scaler_mean, scaler_std = create_normalized_dataloader(
        X_train, X_val, y_train, y_val, batch_size=32, n_bands=n_bands
    )
    
    # Check shapes
    assert scaler_mean.shape == (n_groups * n_bands,)
    assert scaler_std.shape == (n_groups * n_bands,)
    
    # Reshape data to check band-wise normalization
    X_train_reshaped = X_train.reshape(n_samples_train, n_groups, n_bands)
    expected_mean = X_train_reshaped.mean(axis=0).reshape(-1)
    expected_std = X_train_reshaped.std(axis=0).reshape(-1)
    
    # Check normalization parameters with relaxed tolerance for float32/float64 differences
    np.testing.assert_allclose(scaler_mean.numpy(), expected_mean, rtol=1e-2, atol=1e-4)
    np.testing.assert_allclose(scaler_std.numpy(), expected_std, rtol=1e-2, atol=1e-4)
    
    # Check normalized data
    batch_X, batch_y = next(iter(train_loader))
    batch_reshaped = batch_X.reshape(-1, n_groups, n_bands)
    
    # Check each band group separately
    for g in range(n_groups):
        batch_mean = batch_reshaped[:, g, :].mean(dim=0)
        batch_std = batch_reshaped[:, g, :].std(dim=0)
        
        # Check if normalized data has approximately zero mean and unit variance
        np.testing.assert_allclose(batch_mean.numpy(), np.zeros_like(batch_mean), atol=0.5)
        np.testing.assert_allclose(batch_std.numpy(), np.ones_like(batch_std), atol=0.5)

def test_mlp_regression_model():
    # Test model initialization and forward pass
    input_size = 12
    batch_size = 4
    model = MLPRegressionModel(input_size=input_size)
    
    # Create sample input
    x = torch.randn(batch_size, input_size)
    
    # Test forward pass
    output = model(x)
    assert output.shape == (batch_size,)
    
    # Test model architecture
    assert len(model.layers) > 0
    assert isinstance(model.layers[0], torch.nn.Linear)
    assert model.layers[0].in_features == input_size

if __name__ == '__main__':
    pytest.main([__file__])