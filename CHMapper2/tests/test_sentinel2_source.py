import unittest
import ee
import geemap
import os
from sentinel2_source import get_sentinel2_data

class TestSentinel2Source(unittest.TestCase):
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
        self.test_aoi = forest_polygon.geometry()
        
        # Create a simple mask raster (all 1s)
        # self.mask_raster = ee.Image.constant(1).clip(self.test_aoi)

    def test_basic_functionality(self):
        """Test basic Sentinel-2 data retrieval."""
        s2_data = get_sentinel2_data(
            aoi=self.test_aoi,
            year=2023,
            start_date='01-01',
            end_date='01-07',  # One month of data
            clouds_th=20,  # 20% cloud threshold
            # geometry=self.test_aoi
        )
        
        # Check if the result is an Image
        self.assertIsInstance(s2_data, ee.Image)
        
        # Check if all required bands are present
        band_names = s2_data.bandNames().getInfo()
        required_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
                         'NDVI']
                        #  , 'EVI', 'NDWI', 'NDMI']
        for band in required_bands:
            self.assertIn(band, band_names)

    def test_cloud_filtering(self):
        """Test cloud filtering functionality."""
        # Test with very strict cloud threshold
        s2_data = get_sentinel2_data(
            aoi=self.test_aoi,
            year=2023,
            start_date='01-01',
            end_date='01-31',
            clouds_th=5,  # Very strict cloud threshold
            # geometry=self.test_aoi
        )
        
        # Check if the result is an Image
        self.assertIsInstance(s2_data, ee.Image)
        
        # Get the number of valid pixels
        valid_pixels = s2_data.select('B2').mask().reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.test_aoi,
            scale=10
        ).getInfo()
        
        # Verify that we have some valid pixels
        self.assertGreater(valid_pixels.get('B2', 0), 0)

    def test_vegetation_indices(self):
        """Test vegetation indices calculation."""
        s2_data = get_sentinel2_data(
            aoi=self.test_aoi,
            year=2023,
            start_date='01-01',
            end_date='01-31',
            clouds_th=20,
            # geometry=self.test_aoi
        )
        
        # Check if vegetation indices are within expected ranges
        indices = ['NDVI'] #, 'EVI', 'NDWI', 'NDMI']
        for index in indices:
            stats = s2_data.select(index).reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=self.test_aoi,
                scale=10
            ).getInfo()
            
            # NDVI, EVI, NDWI, and NDMI should be between -1 and 1
            self.assertGreaterEqual(stats.get(f'{index}_max', 0), -1)
            self.assertLessEqual(stats.get(f'{index}_max', 0), 1)
            self.assertGreaterEqual(stats.get(f'{index}_min', 0), -1)
            self.assertLessEqual(stats.get(f'{index}_min', 0), 1)

    def test_visualize_results(self):
        """Test visualization of Sentinel-2 data."""
        # Get Sentinel-2 data
        s2_data = get_sentinel2_data(
            aoi=self.test_aoi,
            year=2022,
            start_date='08-01',
            end_date='10-31',
            clouds_th=70,
            # geometry=self.test_aoi
        )
        
        output_dir = os.path.join('..', 'outputs')
        os.makedirs(output_dir, exist_ok=True)

        vis_list = ['html']#, 'png']
        for vis in vis_list:
            # Create a map
            Map = geemap.Map()
            Map.centerObject(self.test_aoi, 12)
            
            # Add RGB composite
            rgb_vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
            Map.addLayer(s2_data, rgb_vis, 'RGB')
            
            # Add NDVI
            ndvi_vis = {'bands': ['NDVI'], 'min': -1, 'max': 1, 'palette': ['white', 'green']}
            Map.addLayer(s2_data, ndvi_vis, 'NDVI')
            
            # Save the map
            if vis == 'html':
                html_file = os.path.join(output_dir, "sentinel2_visualization.html")
                Map.to_html(filename=html_file, title="sentinel2_visualization", width="100%", height="880px")
            # elif vis == 'png':  
            #     png_file = os.path.join(output_dir, "sentinel2_visualization.png")
            #     Map.to_image(filename=png_file, monitor=1)
        # Verify that the output file was created
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'sentinel2_visualization.html')))

if __name__ == '__main__':
    unittest.main() 