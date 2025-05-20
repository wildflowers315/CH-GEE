import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import os
from pathlib import Path
import tempfile
import shutil
from shapely.geometry import Point

from train_predict_map import (
    load_training_data,
    load_prediction_data,
    train_model,
    save_predictions
)

class TestTrainPredictMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temporary test data"""
        cls.test_dir = tempfile.mkdtemp()
        
        # Define raster properties
        cls.height, cls.width = 10, 10
        cls.bounds = (0, 0, 1, 1)  # xmin, ymin, xmax, ymax
        cls.transform = rasterio.transform.from_bounds(
            *cls.bounds, cls.width, cls.height
        )
        
        # Calculate pixel coordinates for points we want to be inside forest mask
        row, col = 2, 2  # Position in the upper left quadrant
        x, y = rasterio.transform.xy(cls.transform, row, col)
        row2, col2 = 3, 3  # Another position in the upper left quadrant
        x2, y2 = rasterio.transform.xy(cls.transform, row2, col2)
        
        # Create test points ensuring they fall within masked area
        points = np.array([
            [x, y],     # Inside forest mask (upper left)
            [x2, y2],   # Inside forest mask (upper left)
            [0.8, 0.8], # Outside forest mask
            [0.9, 0.9]  # Outside forest mask
        ])
        
        cls.csv_data = pd.DataFrame({
            'rh': np.random.rand(len(points)),
            'band1': np.random.rand(len(points)),
            'band2': np.random.rand(len(points)),
            'longitude': points[:, 0],
            'latitude': points[:, 1]
        })
        cls.csv_path = os.path.join(cls.test_dir, 'test_training.csv')
        cls.csv_data.to_csv(cls.csv_path, index=False)
        
        # Create raster profile
        profile = {
            'driver': 'GTiff',
            'height': cls.height,
            'width': cls.width,
            'count': cls.n_bands if hasattr(cls, 'n_bands') else 2,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': cls.transform
        }
        
        # Stack TIF
        cls.stack_path = os.path.join(cls.test_dir, 'test_stack.tif')
        with rasterio.open(cls.stack_path, 'w', **profile) as dst:
            for i in range(profile['count']):
                dst.write(np.random.rand(cls.height, cls.width).astype('float32'), i + 1)
        
        # Forest mask TIF (1s in upper left quadrant, 0s elsewhere)
        mask_profile = profile.copy()
        mask_profile['count'] = 1
        cls.mask_path = os.path.join(cls.test_dir, 'test_mask.tif')
        with rasterio.open(cls.mask_path, 'w', **mask_profile) as dst:
            mask = np.zeros((cls.height, cls.width), dtype='float32')
            mask[:cls.height//2, :cls.width//2] = 1  # Set upper left quadrant to 1
            dst.write(mask, 1)
        
        # Create mask with different CRS for testing
        cls.mask_path_diff_crs = os.path.join(cls.test_dir, 'test_mask_diff_crs.tif')
        mask_profile['crs'] = 'EPSG:3857'
        with rasterio.open(cls.mask_path_diff_crs, 'w', **mask_profile) as dst:
            dst.write(mask, 1)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test data"""
        shutil.rmtree(cls.test_dir)
        
    def test_debug_point_locations(self):
        """Debug test to verify point locations"""
        with rasterio.open(self.mask_path) as src:
            # Print mask array
            mask_data = src.read(1)
            print("\nMask array:")
            print(mask_data)
            
            # Print point coordinates and their pixel locations
            for idx, row in self.csv_data.iterrows():
                x, y = row['longitude'], row['latitude']
                r, c = rasterio.transform.rowcol(src.transform, x, y)
                mask_val = mask_data[r, c] if 0 <= r < src.height and 0 <= c < src.width else -1
                print(f"\nPoint {idx}: ({x}, {y}) -> pixel ({r}, {c}), mask value: {mask_val}")

    def test_load_training_data(self):
        """Test loading and masking training data"""
        # First run debug test
        self.test_debug_point_locations()
        
        # Test without mask
        X, y = load_training_data(self.csv_path)
        self.assertEqual(X.shape[0], len(self.csv_data))
        self.assertEqual(X.shape[1], 2)  # band1, band2
        
        # Test with mask
        X, y = load_training_data(self.csv_path, self.mask_path)
        self.assertEqual(X.shape[0], 2)  # Should have 2 points inside forest mask
        self.assertEqual(X.shape[1], 2)  # band1, band2
        
        # Test with empty intersection
        empty_df = pd.DataFrame({
            'rh': [1.0],
            'band1': [1.0],
            'band2': [1.0],
            'longitude': [2.0],  # Outside mask bounds
            'latitude': [2.0]
        })
        empty_path = os.path.join(self.test_dir, 'empty.csv')
        empty_df.to_csv(empty_path, index=False)
        
        with self.assertRaises(ValueError) as context:
            load_training_data(empty_path, self.mask_path)
        self.assertTrue('No training points fall within the mask bounds' in str(context.exception))

    def test_load_prediction_data(self):
        """Test loading prediction data with CRS checks"""
        # Test without mask
        X, src = load_prediction_data(self.stack_path)
        self.assertEqual(X.shape[0], self.height * self.width)
        self.assertEqual(X.shape[1], 2)  # Number of bands
        src.close()
        
        # Test with matching CRS mask
        X, src = load_prediction_data(self.stack_path, self.mask_path)
        expected_pixels = (self.height * self.width) // 4  # Quarter of pixels are masked
        self.assertEqual(X.shape[0], expected_pixels)
        src.close()
        
        # Test with different CRS mask
        with self.assertRaises(ValueError) as context:
            X, src = load_prediction_data(self.stack_path, self.mask_path_diff_crs)
            src.close()
        self.assertTrue('CRS mismatch' in str(context.exception))

    def test_train_model(self):
        """Test model training"""
        X, y = load_training_data(self.csv_path)
        model, metrix = train_model(X, y, test_size=0.2)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(metrix)
        self.assertTrue(hasattr(model, 'predict'))
        
        test_pred = model.predict(X[:2])
        self.assertEqual(len(test_pred), 2)

    def test_save_predictions(self):
        """Test saving predictions with CRS checks"""
        output_path = os.path.join(self.test_dir, 'test_output.tif')
        
        # Get masked prediction data and keep src open
        with rasterio.open(self.stack_path) as src:
            X_pred, _ = load_prediction_data(self.stack_path, self.mask_path)
            predictions = np.random.rand(len(X_pred))
            
            # Test with matching CRS
            save_predictions(predictions, src, output_path, self.mask_path)
        
        self.assertTrue(os.path.exists(output_path))
        with rasterio.open(output_path) as dst:
            self.assertEqual(dst.shape, (self.height, self.width))
            self.assertEqual(dst.count, 1)
        
        # Test with different CRS
        with self.assertRaises(ValueError) as context, \
             rasterio.open(self.stack_path) as src:
            save_predictions(predictions, src, output_path, self.mask_path_diff_crs)
        self.assertTrue('CRS mismatch' in str(context.exception))

if __name__ == '__main__':
    unittest.main()