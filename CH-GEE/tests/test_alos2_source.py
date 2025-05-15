"""
Unit tests for the ALOS-2 PALSAR source module.
"""

import unittest
import os
import ee
import numpy as np
from alos2_source import get_alos2_data, get_alos2_mosaic

class TestALOS2Source(unittest.TestCase):
    """Test cases for ALOS-2 PALSAR source functions."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize Earth Engine for all tests."""
        # Initialize Earth Engine if not already initialized
        try:
            ee.Initialize(project='my-project-423921')
        except Exception as e:
            # If already initialized or using service account credential
            pass
        
        # Create a test AOI - using a small area in Japan where ALOS data is definitely available
        cls.test_aoi = ee.Geometry.Rectangle([139.5, 35.6, 139.6, 35.7])  # Small area near Tokyo
        
        # Define test years
        cls.test_year = 2017  # Known year with good ALOS data coverage
        cls.test_years = [2015, 2016, 2017]
    
    def test_get_alos2_data_basic(self):
        """Test basic ALOS-2 data retrieval with minimal parameters."""
        try:
            result = get_alos2_data(
                aoi=self.test_aoi,
                year=self.test_year,
                include_texture=False,
                speckle_filter=False
            )
            
            # Verify it's an ee.Image
            self.assertIsInstance(result, ee.Image)
            
            # Verify it has the expected bands
            band_names = result.bandNames().getInfo()
            self.assertIn('HH', band_names)
            self.assertIn('HV', band_names)
            # self.assertIn('ratio', band_names)
            # self.assertIn('ndri', band_names)
            # self.assertIn('HH_dB', band_names)
            # self.assertIn('HV_dB', band_names)
            
            # Verify it doesn't have texture bands
            texture_band_prefixes = ['HH_contrast', 'HV_dissimilarity']
            for prefix in texture_band_prefixes:
                self.assertFalse(any(band.startswith(prefix) for band in band_names))
            
        except Exception as e:
            self.fail(f"get_alos2_data raised exception: {e}")
    
    # def test_get_alos2_data_with_texture(self):
    #     """Test ALOS-2 data retrieval with texture metrics."""
    #     try:
    #         result = get_alos2_data(
    #             aoi=self.test_aoi,
    #             year=self.test_year,
    #             include_texture=True,
    #             speckle_filter=False
    #         )
            
    #         # Verify it has texture bands
    #         band_names = result.bandNames().getInfo()
            
    #         # Check for texture bands
    #         texture_bands = [
    #             'HH_contrast_3', 'HH_dissimilarity_3', 'HH_homogeneity_3',
    #             'HV_contrast_3', 'HV_dissimilarity_3', 'HV_homogeneity_3',
    #             'HH_contrast_5', 'HH_dissimilarity_5', 'HH_homogeneity_5',
    #             'HV_contrast_5', 'HV_dissimilarity_5', 'HV_homogeneity_5'
    #         ]
            
    #         # Check at least some texture bands are present
    #         texture_found = False
    #         for band in texture_bands:
    #             if band in band_names:
    #                 texture_found = True
    #                 break
            
    #         self.assertTrue(texture_found, "No texture bands found in the output image")
            
    #     except Exception as e:
    #         self.fail(f"get_alos2_data with texture raised exception: {e}")
    
    # def test_get_alos2_data_with_speckle_filter(self):
    #     """Test ALOS-2 data retrieval with speckle filtering."""
    #     try:
    #         result = get_alos2_data(
    #             aoi=self.test_aoi,
    #             year=self.test_year,
    #             include_texture=False,
    #             speckle_filter=True
    #         )
            
    #         # Verify it has filtered bands
    #         band_names = result.bandNames().getInfo()
    #         self.assertIn('HH_filtered', band_names)
    #         self.assertIn('HV_filtered', band_names)
            
    #     except Exception as e:
    #         self.fail(f"get_alos2_data with speckle filter raised exception: {e}")
    
    def test_get_alos2_data_date_range(self):
        """Test ALOS-2 data retrieval with custom date range."""
        try:
            result = get_alos2_data(
                aoi=self.test_aoi,
                year=self.test_year,
                start_date="06-01",
                end_date="12-31",
                include_texture=False,
                speckle_filter=False
            )
            
            # Verify it's still an ee.Image (can't verify date range easily in unit test)
            self.assertIsInstance(result, ee.Image)
            
        except Exception as e:
            self.fail(f"get_alos2_data with custom date range raised exception: {e}")
    
    # def test_get_alos2_mosaic(self):
    #     """Test multi-year ALOS-2 mosaic creation."""
    #     try:
    #         result = get_alos2_mosaic(
    #             aoi=self.test_aoi,
    #             years=self.test_years,
    #             include_texture=False,
    #             speckle_filter=False
    #         )
            
    #         # Verify it's an ee.Image
    #         self.assertIsInstance(result, ee.Image)
            
    #         # Verify it has the expected bands
    #         band_names = result.bandNames().getInfo()
    #         self.assertIn('HH', band_names)
    #         self.assertIn('HV', band_names)
            
    #     except Exception as e:
    #         self.fail(f"get_alos2_mosaic raised exception: {e}")
    
    # def test_get_alos2_mosaic_auto_years(self):
    #     """Test multi-year ALOS-2 mosaic with automatic year determination."""
    #     try:
    #         result = get_alos2_mosaic(
    #             aoi=self.test_aoi,
    #             include_texture=False,
    #             speckle_filter=False
    #         )
            
    #         # Verify it's an ee.Image
    #         self.assertIsInstance(result, ee.Image)
            
    #     except Exception as e:
    #         self.fail(f"get_alos2_mosaic with auto years raised exception: {e}")
    
    def test_export_to_asset(self):
        """Test if the result can be exported to an asset (without actually exporting)."""
        try:
            result = get_alos2_data(
                aoi=self.test_aoi,
                year=self.test_year,
                include_texture=False,
                speckle_filter=False
            )
            
            # Set up an export task (but don't actually start it)
            task = ee.batch.Export.image.toAsset(
                image=result,
                description='test_alos_export',
                assetId='projects/your-project/assets/test_alos_export',
                region=self.test_aoi,
                scale=30,
                maxPixels=1e9
            )
            
            # Verify the task was created successfully
            self.assertIsInstance(task, ee.batch.Task)
            
        except Exception as e:
            self.fail(f"Export task creation raised exception: {e}")
    
    def test_image_properties(self):
        """Test that the output image has expected properties."""
        try:
            result = get_alos2_data(
                aoi=self.test_aoi,
                year=self.test_year,
                include_texture=False,
                speckle_filter=False
            )
            
            # Get image scale
            scale = result.projection().nominalScale().getInfo()
            
            # Typical ALOS PALSAR data has 25m resolution
            # Allow some flexibility (might be resampled in some cases)
            self.assertGreaterEqual(scale, 20)  # Should be at least 20m
            self.assertLessEqual(scale, 30)     # Should be at most 30m
            
        except Exception as e:
            self.fail(f"Image properties test raised exception: {e}")


if __name__ == '__main__':
    unittest.main()