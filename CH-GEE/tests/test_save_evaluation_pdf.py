"""Unit tests for save_evaluation_pdf module."""

import unittest
import numpy as np
import os
import pandas as pd
import tempfile
import rasterio
from rasterio.transform import from_origin
import shutil
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from save_evaluation_pdf import (
    create_2x2_visualization,
    get_training_info,
    calculate_area,
    save_evaluation_to_pdf
)

class TestSaveEvaluationPDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temporary test data."""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create test rasters
        cls.ref_path = os.path.join(cls.temp_dir, 'ref.tif')
        cls.pred_path = os.path.join(cls.temp_dir, 'pred.tif')
        cls.merged_path = os.path.join(cls.temp_dir, 'merged.tif')
        
        # Sample data
        ref_data = np.random.normal(15, 5, (100, 100))
        pred_data = ref_data + np.random.normal(0, 2, (100, 100))
        
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
            dst.write(ref_data.astype('float32'), 1)
            
        # Write prediction raster
        with rasterio.open(cls.pred_path, 'w', **profile) as dst:
            dst.write(pred_data.astype('float32'), 1)
            
        # Create merged raster with 3 bands for RGB
        profile['count'] = 3
        rgb_data = np.random.randint(0, 255, (3, 100, 100)).astype('float32')
        with rasterio.open(cls.merged_path, 'w', **profile) as dst:
            dst.write(rgb_data)
        
        # Create sample training data CSV
        cls.train_path = os.path.join(cls.temp_dir, 'training_data.csv')
        train_df = pd.DataFrame({
            'rh': np.random.normal(15, 5, 1000),
            'B1': np.random.random(1000),
            'B2': np.random.random(1000),
            'B3': np.random.random(1000),
            'longitude': np.random.uniform(10, 11, 1000),
            'latitude': np.random.uniform(50, 51, 1000)
        })
        train_df.to_csv(cls.train_path, index=False)
        
        # Sample metrics
        cls.metrics = {
            'RMSE': 2.5,
            'MAE': 1.8,
            'R2': 0.85,
            'Within 1m (%)': 65.0,
            'Within 2m (%)': 85.0
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.temp_dir)

    def test_create_2x2_visualization(self):
        """Test creation of 2x2 comparison grid."""
        output_path = os.path.join(self.temp_dir, 'grid.png')
        result = create_2x2_visualization(
            self.ref_path,
            self.pred_path,
            self.merged_path,
            output_path
        )
        self.assertTrue(os.path.exists(result))
        self.assertEqual(result, output_path)

    def test_get_training_info(self):
        """Test extraction of training data information."""
        info = get_training_info(self.train_path)
        
        self.assertEqual(info['sample_size'], 1000)
        self.assertListEqual(sorted(info['band_names']), ['B1', 'B2', 'B3'])
        self.assertEqual(len(info['height_range']), 2)
        
        # Test with non-existent file
        info = get_training_info('nonexistent.csv')
        self.assertEqual(info['sample_size'], 0)
        self.assertEqual(len(info['band_names']), 0)

    def test_calculate_area(self):
        """Test area calculation from bounds."""
        # Test with geographic coordinates
        bounds = (10.0, 50.0, 11.0, 51.0)  # ~8500 km² at 50°N
        area = calculate_area(bounds, rasterio.crs.CRS.from_epsg(4326))
        # One degree is approximately 111km at equator, area should be roughly 111km * 111km * cos(50°)
        self.assertGreater(area, 700000)  # Should be around 750,000 ha at 50°N
        
        # Test with projected coordinates
        bounds = (0, 0, 1000, 1000)  # 1 km²
        area = calculate_area(bounds, rasterio.crs.CRS.from_epsg(32632))  # UTM 32N
        self.assertAlmostEqual(area, 100)  # Should be 100 ha

    def test_save_evaluation_to_pdf(self):
        """Test full PDF report generation."""
        # Create test data
        pred_data = np.random.normal(15, 5, 1000)
        ref_data = pred_data + np.random.normal(0, 2, 1000)
        
        # Generate PDF
        pdf_path = save_evaluation_to_pdf(
            self.pred_path,
            self.ref_path,
            pred_data,
            ref_data,
            self.metrics,
            self.temp_dir,
            training_data_path=self.train_path,
            merged_data_path=self.merged_path
        )
        
        # Verify PDF was created
        self.assertTrue(os.path.exists(pdf_path))
        
        # Verify PDF filename format
        filename = os.path.basename(pdf_path)
        date = datetime.now().strftime("%Y%m%d")
        self.assertTrue(filename.startswith(date))
        self.assertTrue(filename.endswith('ha.pdf'))
        
        # Verify PDF contains required components by checking file size
        file_size = os.path.getsize(pdf_path)
        self.assertGreater(file_size, 10000)  # Should be substantial in size

if __name__ == '__main__':
    unittest.main()