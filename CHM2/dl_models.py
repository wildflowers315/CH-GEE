import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

def create_normalized_dataloader(X_train, X_val, y_train, y_val, batch_size=64, n_bands=None):
    """
    Create normalized dataloaders for training and validation data with band-by-band normalization.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets
        batch_size: Batch size for dataloaders
        n_bands: Number of spectral bands per feature. If None, treats each feature independently.
        
    Returns:
        train_loader: Normalized training dataloader
        val_loader: Normalized validation dataloader
        scaler_mean: Feature means for denormalization (shape: n_features)
        scaler_std: Feature standard deviations for denormalization (shape: n_features)
    """
    # Convert to torch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_train_tensor = torch.FloatTensor(y_train)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # If n_bands is provided, reshape tensors to group by bands
    if n_bands is not None:
        n_train_samples = X_train_tensor.shape[0]
        n_val_samples = X_val_tensor.shape[0]
        n_features = X_train_tensor.shape[1]
        n_groups = n_features // n_bands
        
        # Reshape to (samples, n_groups, n_bands)
        X_train_tensor = X_train_tensor.view(n_train_samples, n_groups, n_bands)
        X_val_tensor = X_val_tensor.view(n_val_samples, n_groups, n_bands)
        
        # Calculate normalization parameters for each band within each group
        scaler_mean = X_train_tensor.mean(dim=0)  # Shape: (n_groups, n_bands)
        scaler_std = X_train_tensor.std(dim=0)    # Shape: (n_groups, n_bands)
        scaler_std[scaler_std == 0] = 1  # Prevent division by zero
        
        # Normalize band by band
        X_train_normalized = (X_train_tensor - scaler_mean) / scaler_std
        X_val_normalized = (X_val_tensor - scaler_mean) / scaler_std
        
        # Reshape back to original format
        X_train_normalized = X_train_normalized.reshape(n_train_samples, -1)
        X_val_normalized = X_val_normalized.reshape(n_val_samples, -1)
        scaler_mean = scaler_mean.reshape(-1)
        scaler_std = scaler_std.reshape(-1)
    else:
        # Calculate normalization parameters for each feature independently
        scaler_mean = X_train_tensor.mean(dim=0)
        scaler_std = X_train_tensor.std(dim=0)
        scaler_std[scaler_std == 0] = 1  # Prevent division by zero
        
        # Normalize data
        X_train_normalized = (X_train_tensor - scaler_mean) / scaler_std
        X_val_normalized = (X_val_tensor - scaler_mean) / scaler_std
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_normalized, y_train_tensor)
    val_dataset = TensorDataset(X_val_normalized, y_val_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, scaler_mean, scaler_std

class MLPRegressionModel(torch.nn.Module):
    def __init__(self, input_size: int, num_layers: int = 3, nodes: int = 1024,
                dropout: float = 0.2, is_nodes_half: bool = False):
        super().__init__()
        self.num_features = input_size
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activation = nn.ReLU()

        self.layers.append(nn.Linear(self.num_features, nodes))
        self.batch_norms.append(nn.BatchNorm1d(nodes))
        
        self.dropout = nn.Dropout(dropout)
        
        if is_nodes_half:
            for i in range(num_layers):
                in_features = int(nodes / (i+1))
                out_features = int(nodes / (i+2))
                self.layers.append(nn.Linear(in_features, out_features))
                self.batch_norms.append(nn.BatchNorm1d(out_features))
            self.head = nn.Linear(int(nodes / (num_layers+1)), 1)
        else:
            for _ in range(num_layers):
                self.layers.append(nn.Linear(nodes, nodes))
                self.batch_norms.append(nn.BatchNorm1d(nodes))
            self.head = nn.Linear(nodes, 1)
    
    def forward(self, x):
        for layer, bn in zip(self.layers, self.batch_norms):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.head(x)
        return x.squeeze(1)
