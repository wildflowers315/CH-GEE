import unittest
from unittest.mock import patch, MagicMock
import ee
import pandas as pd
import os
from canopyht_source import get_canopyht_data
from chm_main import export_training_data

class TestCanopyHeightSource(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize Earth Engine for testing."""
        try:
            ee.Initialize(project='my-project-423921')
        except Exception as e:
            print(f"Earth Engine initialization failed: {e}")
            raise

    def setUp(self):
        """Set up test data before each test."""
        # Create test points in an area where we expect different canopy heights
        # Using points in a forested area for meaningful values
        self.test_points = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([11.0, 48.0]), {}),  # Munich area
            ee.Feature(ee.Geometry.Point([11.1, 48.1]), {}),  # Different location
            ee.Feature(ee.Geometry.Point([11.2, 48.2]), {})   # Another location
        ])
        
        # Create boundary for the area containing the points
        self.test_aoi = self.test_points.geometry().bounds()

    def test_canopy_height_values(self):
        """Test that canopy height values can be extracted and are reasonable."""
        # Get canopy height data
        canopy_ht = get_canopyht_data(aoi=self.test_aoi)
        
        # Sample points
        sampled_points = canopy_ht.sampleRegions(
            collection=self.test_points,
            scale=30,
            geometries=True
        )
        
        # Create temporary directory for test output
        test_output_dir = "test_output"
        os.makedirs(test_output_dir, exist_ok=True)
        
        try:
            # Export and read the data
            output_path = export_training_data(sampled_points, test_output_dir)
            
            # Read the exported data
            df = pd.read_csv(output_path)
            
            # Basic validations
            self.assertGreater(len(df), 0, "No data points were exported")
            self.assertTrue('ht2023' in df.columns, "Canopy height column not found")
            
            # Check that we have different height values
            unique_heights = df['ht2023'].nunique()
            self.assertGreater(unique_heights, 1, 
                             "All points have the same height value")
            
            # Check height values are within reasonable range (0-100 meters)
            self.assertTrue(all(df['ht2023'].between(0, 100)), 
                          "Height values outside reasonable range")
            
            # Check if heights are not all zeros
            self.assertFalse(all(df['ht2023'] == 0), 
                           "All height values are zero")
            
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(test_output_dir):
                os.rmdir(test_output_dir)

if __name__ == '__main__':
    unittest.main()