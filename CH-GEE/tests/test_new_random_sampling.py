import unittest
import ee
import numpy as np
from new_random_sampling import generate_sampling_sites, create_training_data

class TestNewRandomSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize Earth Engine for testing."""
        try:
            ee.Initialize(project='my-project-423921')
        except:
            ee.Authenticate()
            ee.Initialize(project='my-project-423921')
        
        # Create test geometries with appropriate sizes for sampling
        cls.small_aoi = ee.Geometry.Rectangle([0, 0, 0.01, 0.01])    # ~100 hectares
        cls.medium_aoi = ee.Geometry.Rectangle([0, 0, 0.1, 0.1])     # ~1000 hectares
        cls.large_aoi = ee.Geometry.Rectangle([0, 0, 1, 1])          # ~10000 hectares
        
        # Create test mask
        cls.test_mask = ee.Image.constant(1)
        
        # Create test data - keep as collections
        cls.test_gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').limit(10)
        
        # Get Sentinel data
        s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD').limit(1)
        cls.test_s1 = ee.Image(s1_collection.first())
        
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
        cls.test_s2 = ee.Image(s2_collection.first())

    def test_generate_sampling_sites_small_area(self):
        """Test sampling site generation for small areas."""
        result = generate_sampling_sites(self.small_aoi, 100, 1, self.test_mask)
        
        # Check if result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check if buffer exists in result
        self.assertIn('buffer', result)
        
        # Check if buffer is a FeatureCollection
        self.assertIsInstance(result['buffer'], ee.FeatureCollection)
        
        # Check if buffer has features
        buffer_size = result['buffer'].size().getInfo()
        self.assertGreater(buffer_size, 0)

    def test_generate_sampling_sites_large_area(self):
        """Test sampling site generation for large areas."""
        result = generate_sampling_sites(self.large_aoi, 50000, 1, self.test_mask)
        
        # Check if result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check if buffer exists in result
        self.assertIn('buffer', result)
        
        # Check if buffer is a FeatureCollection
        self.assertIsInstance(result['buffer'], ee.FeatureCollection)
        
        # Check if buffer has features
        buffer_size = result['buffer'].size().getInfo()
        self.assertGreater(buffer_size, 0)

    def test_create_training_data_small_area(self):
        """Test training data creation for small areas."""
        result = create_training_data(
            self.test_gedi,
            self.test_s1,
            self.test_s2,
            self.small_aoi,
            self.test_mask
        )
        
        # Check if result is a FeatureCollection
        self.assertIsInstance(result, ee.FeatureCollection)
        
        # Check if result has features
        result_size = result.size().getInfo()
        self.assertGreater(result_size, 0)
        
        # Check if random column exists
        properties = result.first().getInfo()['properties']
        self.assertIn('random', properties)

    def test_create_training_data_large_area(self):
        """Test training data creation for large areas."""
        result = create_training_data(
            self.test_gedi,
            self.test_s1,
            self.test_s2,
            self.large_aoi,
            self.test_mask
        )
        
        # Check if result is a FeatureCollection
        self.assertIsInstance(result, ee.FeatureCollection)
        
        # Check if result has features
        result_size = result.size().getInfo()
        self.assertGreater(result_size, 0)
        
        # Check if random column exists
        properties = result.first().getInfo()['properties']
        self.assertIn('random', properties)

    def test_create_training_data_no_mask(self):
        """Test training data creation without mask."""
        result = create_training_data(
            self.test_gedi,
            self.test_s1,
            self.test_s2,
            self.medium_aoi
        )
        
        # Check if result is a FeatureCollection
        self.assertIsInstance(result, ee.FeatureCollection)
        
        # Check if result has features
        result_size = result.size().getInfo()
        self.assertGreater(result_size, 0)

    def test_create_training_data_invalid_inputs(self):
        """Test training data creation with invalid inputs."""
        with self.assertRaises(ValueError):
            create_training_data(
                None,  # Invalid GEDI data
                self.test_s1,
                self.test_s2,
                self.medium_aoi
            )
        
        with self.assertRaises(ValueError):
            create_training_data(
                self.test_gedi,
                None,  # Invalid S1 data
                self.test_s2,
                self.medium_aoi
            )
        
        with self.assertRaises(ValueError):
            create_training_data(
                self.test_gedi,
                self.test_s1,
                None,  # Invalid S2 data
                self.medium_aoi
            )

    def test_generate_sampling_sites_invalid_inputs(self):
        """Test sampling site generation with invalid inputs."""
        with self.assertRaises(ValueError):
            generate_sampling_sites(None, 100, 1, self.test_mask)  # Invalid region
        
        with self.assertRaises(ValueError):
            generate_sampling_sites(self.small_aoi, -100, 1, self.test_mask)  # Invalid cell size
        
        with self.assertRaises(ValueError):
            generate_sampling_sites(self.small_aoi, 100, 1, None)  # Invalid mask

if __name__ == '__main__':
    unittest.main() 