"""Unit tests for evaluate_predictions module."""

import unittest
import numpy as np
import os
import tempfile
import shutil
import rasterio
from rasterio.transform import from_origin

from evaluate_predictions import (
    check_predictions,
    calculate_metrics
)
from evaluation_utils import validate_data, create_plots
from raster_utils import load_and_align_rasters

class TestEvaluatePredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temporary test data."""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create test rasters
        cls.ref_path = os.path.join(cls.temp_dir, 'ref.tif')
        cls.pred_path = os.path.join(cls.temp_dir, 'pred.tif')
        cls.forest_mask_path = os.path.join(cls.temp_dir, 'forest_mask.tif')
        
        # Sample data
        # Generate data within valid range (0-50m)
        cls.ref_data = np.clip(np.random.normal(25, 5, (100, 100)), 0, 50)
        cls.pred_data = np.clip(cls.ref_data + np.random.normal(0, 2, (100, 100)), 0, 50)
        cls.forest_mask = np.random.choice([0, 1], size=(100, 100), p=[0.3, 0.7])  # 70% forest coverage
        
        # Create sample rasters
        transform = from_origin(10.0, 50.0, 0.001, 0.001)
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999,
            'width': 100,
            'height': 100,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform
        }
        
        # Write reference raster
        with rasterio.open(cls.ref_path, 'w', **profile) as dst:
            dst.write(cls.ref_data.astype('float32'), 1)
            
        # Write prediction raster
        with rasterio.open(cls.pred_path, 'w', **profile) as dst:
            dst.write(cls.pred_data.astype('float32'), 1)
            
        # Write forest mask raster
        with rasterio.open(cls.forest_mask_path, 'w', **profile) as dst:
            dst.write(cls.forest_mask.astype('float32'), 1)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.temp_dir)

    def test_check_predictions(self):
        """Test prediction file validation."""
        # Test valid predictions
        self.assertTrue(check_predictions(self.pred_path))
        
        # Test invalid predictions (all nodata)
        invalid_path = os.path.join(self.temp_dir, 'invalid.tif')
        with rasterio.open(self.pred_path) as src:
            profile = src.profile.copy()
            with rasterio.open(invalid_path, 'w', **profile) as dst:
                dst.write(np.full((100, 100), -9999, dtype=np.float32), 1)
        
        self.assertFalse(check_predictions(invalid_path))

    def test_validate_data(self):
        """Test data validation checks."""
        # Test valid data
        validate_data(self.pred_data.flatten(), self.ref_data.flatten())
        
        # Test zero variance data
        with self.assertRaises(ValueError):
            validate_data(np.zeros(100), self.ref_data.flatten())
            
        # Test too low values
        with self.assertRaises(ValueError):
            validate_data(np.full(100, 0.001), self.ref_data.flatten())

    def test_load_and_align_rasters(self):
        """Test loading and aligning raster data."""
        # Create output directory
        out_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(out_dir, exist_ok=True)
        
        # Test without forest mask
        # Test without forest mask
        pred_data, ref_data, transform, forest_mask = load_and_align_rasters(
            self.pred_path,
            self.ref_path,
            None,  # No forest mask
            out_dir
        )
        
        # Basic checks
        self.assertEqual(pred_data.shape, ref_data.shape)
        self.assertEqual(pred_data.shape, (100, 100))
        # Check if any non-NaN values are outside valid range
        valid_pred = pred_data[~np.isnan(pred_data)]
        valid_ref = ref_data[~np.isnan(ref_data)]
        self.assertTrue(np.all(valid_pred >= 0))  # Check valid range
        self.assertTrue(np.all(valid_pred <= 50))
        self.assertTrue(np.all(valid_ref >= 0))
        self.assertTrue(np.all(valid_ref <= 50))
        # Test with forest mask (remove duplicate line)
        pred_data, ref_data, transform, mask = load_and_align_rasters(
            self.pred_path,
            self.ref_path,
            self.forest_mask_path,
            out_dir
        )
        
        # Additional checks for forest mask
        self.assertEqual(pred_data.shape, ref_data.shape)
        
        # Check if any non-NaN values are outside valid range
        valid_pred = pred_data[~np.isnan(pred_data)]
        valid_ref = ref_data[~np.isnan(ref_data)]
        self.assertTrue(np.all(valid_pred >= 0))  # Check valid range
        self.assertTrue(np.all(valid_pred <= 50))
        self.assertTrue(np.all(valid_ref >= 0))
        self.assertTrue(np.all(valid_ref <= 50))
        
        # Forest mask specific checks
        self.assertIsNotNone(mask)  # Forest mask should be present
        self.assertEqual(mask.shape, pred_data.shape)  # Mask should match data shape
        
        # Check that non-forest areas are masked
        self.assertTrue(np.all(np.isnan(pred_data[~mask])))
        self.assertTrue(np.all(np.isnan(ref_data[~mask])))
        
        # Check that forest areas have valid data
        self.assertTrue(np.any(~np.isnan(pred_data[mask])))
        self.assertTrue(np.any(~np.isnan(ref_data[mask])))

    def test_calculate_metrics(self):
        """Test calculation of evaluation metrics."""
        # Create sample data with known properties
        ref = np.array([10, 15, 20, 25, 30])
        pred = np.array([11, 14, 21, 24, 31])
        
        metrics = calculate_metrics(pred, ref)
        
        # Check all required metrics are present
        required_metrics = [
            'MSE', 'RMSE', 'MAE', 'R2', 
            'Mean Error', 'Std Error', 'Max Absolute Error',
            'Within 1m (%)', 'Within 2m (%)', 'Within 5m (%)'
        ]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            
        # Check some specific metric properties
        self.assertGreaterEqual(metrics['R2'], 0)  # R² should be non-negative
        self.assertGreaterEqual(metrics['RMSE'], metrics['MAE'])  # RMSE ≥ MAE

    def test_create_plots(self):
        """Test creation of evaluation plots."""
        output_dir = os.path.join(self.temp_dir, 'plots')
        os.makedirs(output_dir, exist_ok=True)
        
        # Test without forest mask first
        mask = (self.pred_data >= 0) & (self.pred_data <= 50) & ~np.isnan(self.pred_data)
        pred_masked = self.pred_data[mask]
        ref_masked = self.ref_data[mask]
        metrics = calculate_metrics(pred_masked.flatten(), ref_masked.flatten())
        
        # Create plots without forest mask
        create_plots(pred_masked.flatten(), ref_masked.flatten(), metrics, output_dir)
        
        # Test plots with forest mask
        forest_mask = self.forest_mask == 1
        pred_masked_forest = self.pred_data[forest_mask]
        ref_masked_forest = self.ref_data[forest_mask]
        forest_metrics = calculate_metrics(pred_masked_forest.flatten(), ref_masked_forest.flatten())
        create_plots(pred_masked_forest.flatten(), ref_masked_forest.flatten(), forest_metrics, output_dir)
        
        # Verify plots were created
        expected_plots = ['scatter_plot.png', 'error_hist.png', 'height_distributions.png']
        for plot in expected_plots:
            plot_path = os.path.join(output_dir, plot)
            self.assertTrue(os.path.exists(plot_path))
            self.assertGreater(os.path.getsize(plot_path), 0)

if __name__ == '__main__':
    unittest.main()