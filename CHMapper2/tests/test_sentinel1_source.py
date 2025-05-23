import unittest
import ee
import geemap
import os
from sentinel1_source import (
    # mask_border_noise,
    # gamma_map_filter,
    # terrain_flattening,
    get_sentinel1_data
)

class TestSentinel1Source(unittest.TestCase):
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
        # Load the forest polygon from local GeoJSON file
        forest_polygon_path = os.path.join('..', 'downloads', 'new_aoi.geojson')
        
        # Check if file exists
        if not os.path.exists(forest_polygon_path):
            raise FileNotFoundError(f"Forest polygon file not found at: {forest_polygon_path}")
        
        # Load GeoJSON using geemap
        forest_polygon = geemap.geojson_to_ee(forest_polygon_path)
        # Get the geometry from the FeatureCollection and then buffer it
        geometry = forest_polygon.geometry()
        expanded_forest_polygon = geometry.buffer(100000)
        self.test_aoi = expanded_forest_polygon

    # def test_mask_border_noise(self):
    #     """Test border noise masking function."""
    #     # Get a sample Sentinel-1 image
    #     s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
    #         .filterDate('2022-01-01', '2022-01-31') \
    #         .filterBounds(self.test_aoi) \
    #         .first()
        
    #     # Apply border noise mask
    #     masked = mask_border_noise(s1)
        
    #     # Check if the result is a ComputedObject
    #     self.assertIsInstance(masked, ee.ComputedObject)
        
    #     # Check if VV and VH bands exist
    #     band_names = masked.bandNames().getInfo()
    #     self.assertIn('VV', band_names)
    #     self.assertIn('VH', band_names)

    # def test_gamma_map_filter(self):
    #     """Test Gamma Map filter function."""
    #     # Get a sample Sentinel-1 image
    #     s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
    #         .filterDate('2022-01-01', '2022-01-31') \
    #         .filterBounds(self.test_aoi) \
    #         .first()
        
    #     # Apply Gamma Map filter
    #     filtered = gamma_map_filter(s1, 15, 10)
        
    #     # Check if the result is a ComputedObject
    #     self.assertIsInstance(filtered, ee.ComputedObject)
        
    #     # Check if VV and VH bands exist
    #     band_names = filtered.bandNames().getInfo()
    #     self.assertIn('VV', band_names)
    #     self.assertIn('VH', band_names)

    # def test_terrain_flattening(self):
    #     """Test terrain flattening function."""
    #     # Get a sample Sentinel-1 image
    #     s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
    #         .filterDate('2022-01-01', '2022-01-31') \
    #         .filterBounds(self.test_aoi) \
    #         .first()
        
    #     # Apply terrain flattening
    #     flattened = terrain_flattening(s1, 'USGS/SRTMGL1_003')
        
    #     # Check if the result is a ComputedObject
    #     self.assertIsInstance(flattened, ee.ComputedObject)
        
    #     # Check if VV and VH bands exist
    #     band_names = flattened.bandNames().getInfo()
    #     self.assertIn('VV', band_names)
    #     self.assertIn('VH', band_names)

    def test_get_sentinel1_data(self):
        """Test main Sentinel-1 data retrieval function."""
        # Get Sentinel-1 data
        s1_data = get_sentinel1_data(
            self.test_aoi,
            2022,
            '01-01',
            '12-31'
        )
        
        # Check if the result is a ComputedObject
        self.assertIsInstance(s1_data, ee.ComputedObject)
        
        # Check if VV and VH bands exist
        band_names = s1_data.bandNames().getInfo()
        self.assertIn('VV', band_names)
        self.assertIn('VH', band_names)
        
        # Test visualization
        Map = geemap.Map()
        Map.centerObject(self.test_aoi, 15)
        
        # Add Sentinel-1 data with visualization
        vis_params = {
            'bands': ['VV', 'VH', 'VV'],
            'min': [-25, -25, -25],
            'max': [0, 0, 0]
        }
        
        # Add the layers
        Map.addLayer(s1_data, vis_params, 'Sentinel-1 Data')
        Map.addLayer(self.test_aoi, {'color': 'white'}, 'AOI')
        
        # Save visualization
        output_dir = os.path.join('..', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        html_file = os.path.join(output_dir, "sentinel1_data_visualization.html")
        Map.to_html(filename=html_file, title="Sentinel-1 Data Visualization", width="100%", height="880px")
        
        # Verify output file
        self.assertTrue(os.path.exists(html_file))

    # def test_no_data_available(self):
    #     """Test when no Sentinel-1 data is available for the area."""
    #     # Use an area where we know there won't be data (middle of ocean)
    #     ocean_aoi = ee.Geometry.Rectangle([-180, -90, -179.9, -89.9])
        
    #     s1_data = get_sentinel1_data(
    #         ocean_aoi,
    #         2022,
    #         '01-01',
    #         '12-31'
    #     )
        
    #     # Check if the result is a ComputedObject
    #     self.assertIsInstance(s1_data, ee.ComputedObject)
        
    #     # Check if VV and VH bands exist
    #     band_names = s1_data.bandNames().getInfo()
    #     self.assertIn('VV', band_names)
    #     self.assertIn('VH', band_names)

if __name__ == '__main__':
    unittest.main() 