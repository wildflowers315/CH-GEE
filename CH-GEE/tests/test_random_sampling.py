import unittest
import ee
import geemap
import os
from random_sampling import create_training_data
from l2a_gedi_source import get_gedi_data
from sentinel1_source import get_sentinel1_data
from sentinel2_source import get_sentinel2_data

class TestRandomSampling(unittest.TestCase):
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
        self.test_aoi = forest_polygon.geometry().buffer(5000)
        
        # Get test data
        self.gedi_data = get_gedi_data(
            self.test_aoi,
            '2022-01-01',
            '2022-12-31',
            'rh98'
        )
        
        self.gedi_geometry = self.gedi_data.geometry()
        
        self.s1_data = get_sentinel1_data(
            self.gedi_geometry,
            year=2022,
            start_date='01-01',
            end_date='12-31'
        )
        
        self.s2_data = get_sentinel2_data(
            self.gedi_geometry,
            year=2022,
            start_date='01-01',
            end_date='12-31',
            clouds_th=70,
            geometry=self.gedi_geometry
        )

    def test_gedi_data_size(self):
        """Test the size of the GEDI data."""
        size = self.gedi_data.size().getInfo()
        print(f"GEDI data size: {size}")
        self.assertGreater(size, 0)

    def test_s1_data_bands(self):
        """Test the Sentinel-1 data bands."""
        bands = self.s1_data.bandNames().getInfo()
        print(f"S1 bands: {bands}")
        self.assertGreater(len(bands), 0)

    def test_s2_data_bands(self):
        """Test the Sentinel-2 data bands."""
        bands = self.s2_data.bandNames().getInfo()
        print(f"S2 bands: {bands}")
        self.assertGreater(len(bands), 0)

    def test_create_training_data(self):
        """Test creation of training data."""
        # Print GEDI data info
        gedi_info = self.gedi_data.first().getInfo()
        # print(f"GEDI data info: {gedi_info}")
        
        # Print Sentinel data info
        s1_info = self.s1_data.getInfo()
        s2_info = self.s2_data.getInfo()
        # print(f"S1 data info: {s1_info}")
        # print(f"S2 data info: {s2_info}")
        
        # Create training data
        training_data = create_training_data(
            self.gedi_data,
            self.s1_data,
            self.s2_data
        )
        
        # Check if the result is a FeatureCollection
        self.assertIsInstance(training_data, ee.FeatureCollection)
        
        # Get the size of the training data
        size = training_data.size().getInfo()
        print(f"Training data size: {size}")
        self.assertGreater(size, 0)
        
        # Get a sample of features to check properties
        sample = training_data.limit(1).getInfo()
        first_feature = sample['features'][0]
        print(f"First feature properties: {first_feature['properties']}")
        
        # Check if it has the required properties
        self.assertIn('rh', first_feature['properties'])
        # self.assertIn('height', first_feature['properties'])
        
        # Check if it has Sentinel bands
        sentinel_bands = self.s1_data.bandNames().cat(self.s2_data.bandNames()).getInfo()
        for band in sentinel_bands:
            self.assertIn(band, first_feature['properties'])

    def test_training_data_visualization(self):
        """Test visualization of training data."""
        # Create training data
        training_data = create_training_data(
            self.gedi_data,
            self.s1_data,
            self.s2_data
        )
        
        # Create a map
        Map = geemap.Map()
        Map.centerObject(self.test_aoi, 15)
        
        # Add layers
        rgb_vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
        Map.addLayer(self.s2_data, rgb_vis, 'Sentinel-2')
        
        # Add training points with proper visualization
        point_vis = {
            'color': 'red',
            'pointSize': 3,
            'pointShape': 'circle',
            'width': 2
        }
        
        # Convert to FeatureCollection if needed
        if isinstance(training_data, ee.ImageCollection):
            training_data = ee.FeatureCollection(training_data)
        
        Map.addLayer(training_data, point_vis, 'Training Points')
        Map.addLayer(self.test_aoi, {'color': 'white'}, 'AOI')
        
        # Save visualization
        output_dir = os.path.join('..', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        html_file = os.path.join(output_dir, "training_data_visualization.html")
        Map.to_html(filename=html_file, title="Training Data Visualization", width="100%", height="880px")
        
        # Verify output file
        self.assertTrue(os.path.exists(html_file))

if __name__ == '__main__':
    unittest.main() 