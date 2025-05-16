import unittest
import os
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import ee
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path to import chm_main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chm_main import parse_args, load_aoi, export_training_data, export_tif_via_ee, initialize_ee

class TestCHMMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        
        # Initialize Earth Engine
        try:
            initialize_ee()
        except Exception as e:
            print(f"Warning: Could not initialize Earth Engine: {e}")
    
    @classmethod
    def tearDownClass(cls):
        # Remove temporary directory
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        # Create sample AOI GeoJSONs for different formats
        # 1. Simple Polygon
        self.polygon_path = os.path.join(self.test_dir, 'test_polygon.geojson')
        self.polygon_data = {
            "type": "Polygon",
            "coordinates": [[
                [13.0, 52.0],
                [13.1, 52.0],
                [13.1, 52.1],
                [13.0, 52.1],
                [13.0, 52.0]
            ]]
        }
        with open(self.polygon_path, 'w') as f:
            json.dump(self.polygon_data, f)
        
        # 2. FeatureCollection with MultiPolygon
        self.fc_multipolygon_path = os.path.join(self.test_dir, 'test_fc_multipolygon.geojson')
        self.fc_multipolygon_data = {
            "type": "FeatureCollection",
            "name": "test_aoi",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": [{
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[
                            [139.59, 36.30],
                            [139.74, 36.30],
                            [139.74, 36.41],
                            [139.59, 36.41],
                            [139.59, 36.30]
                        ]]
                    ]
                }
            }]
        }
        with open(self.fc_multipolygon_path, 'w') as f:
            json.dump(self.fc_multipolygon_data, f)
    
    def test_load_aoi_polygon(self):
        """Test loading simple Polygon GeoJSON."""
        aoi = load_aoi(self.polygon_path)
        self.assertIsInstance(aoi, ee.Geometry)
    
    def test_load_aoi_fc_multipolygon(self):
        """Test loading FeatureCollection with MultiPolygon."""
        aoi = load_aoi(self.fc_multipolygon_path)
        self.assertIsInstance(aoi, ee.Geometry)
    
    def test_load_aoi_nonexistent(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_aoi('nonexistent.geojson')
    
    def test_load_aoi_invalid_type(self):
        """Test loading invalid GeoJSON type."""
        invalid_path = os.path.join(self.test_dir, 'invalid.geojson')
        invalid_data = {
            "type": "Point",
            "coordinates": [13.0, 52.0]
        }
        with open(invalid_path, 'w') as f:
            json.dump(invalid_data, f)
        
        with self.assertRaises(ValueError):
            load_aoi(invalid_path)
    
    def test_export_training_data(self):
        """Test exporting training data to CSV."""
        # Create mock FeatureCollection
        mock_features = {
            'features': [
                {
                    'properties': {
                        'rh': 25.5,
                        'B1': 0.1,
                        'B2': 0.2
                    },
                    'geometry': {
                        'coordinates': [13.05, 52.05]
                    }
                },
                {
                    'properties': {
                        'rh': 30.2,
                        'B1': 0.15,
                        'B2': 0.25
                    },
                    'geometry': {
                        'coordinates': [13.06, 52.06]
                    }
                }
            ]
        }
        mock_fc = MagicMock()
        mock_fc.getInfo.return_value = mock_features
        
        # Create output directory
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Mock features data
        mock_features = {
            'features': [
                {
                    'properties': {'rh': 25.5, 'B1': 0.1},
                    'geometry': {'coordinates': [13.05, 52.05]}
                },
                {
                    'properties': {'rh': 30.2, 'B1': 0.2},
                    'geometry': {'coordinates': [13.06, 52.06]}
                }
            ]
        }
        mock_fc.getInfo.return_value = mock_features
        
        # Test export
        output_path = export_training_data(mock_fc, self.test_dir)
        
        # Verify output
        self.assertTrue(output_path.endswith('.csv'))
        self.assertTrue(os.path.exists(output_path))
        
        # Read and verify the CSV content
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 2)  # Verify we have 2 rows
        self.assertTrue('longitude' in df.columns)
        self.assertTrue('latitude' in df.columns)
        self.assertTrue('rh' in df.columns)
        self.assertTrue('B1' in df.columns)
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 2)
        self.assertTrue('longitude' in df.columns)
        self.assertTrue('latitude' in df.columns)
        self.assertTrue('rh' in df.columns)
    
    def test_export_tif_via_ee(self):
        """Test exporting predictions to GeoTIFF using Earth Engine."""
        # Create mock image with required methods
        mock_image = MagicMock()
        mock_image.bandNames().getInfo.return_value = ['classification']
        mock_image.select.return_value = mock_image
        
        # Use the polygon AOI for testing
        aoi = ee.Geometry(self.polygon_data)
        
        # Test export
        output_path = os.path.join(self.test_dir, 'test_predictions.tif')
        
        # Mock the Earth Engine export task
        mock_task = MagicMock()
        with patch('ee.batch.Export.image.toDrive') as mock_export:
            mock_export.return_value = mock_task
            export_tif_via_ee(mock_image, aoi, output_path, scale=30)
            
            # Verify export parameters
            mock_export.assert_called_once()
            args = mock_export.call_args[1]
            self.assertEqual(args['scale'], 30)
            self.assertEqual(args['maxPixels'], 1e10)
            self.assertEqual(args['fileFormat'], 'GeoTIFF')
            mock_task.start.assert_called_once()
    
    def test_parse_args(self):
        """Test argument parsing."""
        with patch('sys.argv', ['chm_main.py',
                            '--aoi', 'test.geojson',
                            '--year', '2020',
                            '--output-dir', 'outputs',
                            '--export-training',
                            '--export-predictions',
                            '--mask-type', 'NDVI']):
            args = parse_args()
            self.assertEqual(args.aoi, 'test.geojson')
            self.assertEqual(args.year, 2020)
            self.assertEqual(args.output_dir, 'outputs')
            self.assertTrue(args.export_training)
            self.assertTrue(args.export_predictions)
            self.assertEqual(args.mask_type, 'NDVI')
            self.assertEqual(args.year, 2020)
            self.assertEqual(args.output_dir, 'outputs')
            self.assertTrue(args.export_training)
            self.assertTrue(args.export_predictions)

if __name__ == '__main__':
    unittest.main()