import unittest
import ee
import geemap
import os
from ch_gee_main import canopy_height_mapper, get_variable_importance
from new_random_sampling import create_training_data
from l2a_gedi_source import get_gedi_data
from sentinel1_source import get_sentinel1_data
from sentinel2_source import get_sentinel2_data
from for_forest_masking import apply_forest_mask

class TestCanopyHeightMapper(unittest.TestCase):
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
        
        # Test parameters
        self.year = 2022
        self.start_date = '01-01'
        self.end_date = '12-31'
        self.start_date_gedi = '2022-01-01'
        self.end_date_gedi = '2022-12-31'
        self.clouds_th = 70
        self.quantile = 'rh98'
        self.training_buffer = 5000

    def test_training_data_creation(self):
        """Test the creation and validation of training data."""
        # Create training AOI with buffer
        training_aoi = self.test_aoi.buffer(self.training_buffer)
        
        # Get GEDI data
        gedi_data = get_gedi_data(
            training_aoi,
            self.start_date_gedi,
            self.end_date_gedi,
            self.quantile
        )
        
        # Verify GEDI data exists
        gedi_size = gedi_data.size().getInfo()
        print(f"Number of GEDI points: {gedi_size}")
        self.assertGreater(gedi_size, 0, "No GEDI data found in the area")
        
        # Filter GEDI data to ensure it has required properties
        gedi_data = gedi_data.filter(ee.Filter.notNull(['rh']))
        gedi_size = gedi_data.size().getInfo()
        print(f"Number of GEDI points with valid height data: {gedi_size}")
        self.assertGreater(gedi_size, 0, "No GEDI points with valid height data found")
        
        # Print GEDI data properties for debugging
        first_gedi = gedi_data.first().getInfo()
        print("First GEDI point properties:", first_gedi.get('properties', {}))
        
        # Get Sentinel data
        s1_data = get_sentinel1_data(
            training_aoi,
            self.year,
            self.start_date,
            self.end_date
        )
        s2_data = get_sentinel2_data(
            training_aoi,
            self.year,
            self.start_date,
            self.end_date,
            self.clouds_th,
            training_aoi
        )
        
        # Print Sentinel data bands for debugging
        print("Sentinel-1 bands:", s1_data.bandNames().getInfo())
        print("Sentinel-2 bands:", s2_data.bandNames().getInfo())
        
        # Try to create forest mask, but don't fail if it's not available
        try:
            forest_mask = apply_forest_mask(
                training_aoi,
                'DW',  # Dynamic World mask
                training_aoi,
                self.year,
                self.start_date,
                self.end_date
            )
            print("Forest mask created successfully")
        except Exception as e:
            print(f"Warning: Could not create forest mask: {e}")
            forest_mask = None
        
        # Create training data
        print("Creating training data...")
        training_data = create_training_data(
            gedi_data=gedi_data,
            s1_data=s1_data,
            s2_data=s2_data,
            aoi=self.test_aoi,
            mask=forest_mask
        )
        
        # Verify training data properties
        self.assertIsInstance(training_data, ee.FeatureCollection)
        
        # Check if training data has features
        training_size = training_data.size().getInfo()
        print(f"Number of training features: {training_size}")
        
        if training_size == 0:
            # Print more diagnostic information
            print("\nDiagnostic Information:")
            print("1. GEDI data size:", gedi_size)
            print("2. Training AOI area:", training_aoi.area().getInfo(), "square meters")
            print("3. Forest mask status:", "Created" if forest_mask else "Not created")
            print("4. Sentinel-1 data available:", s1_data.bandNames().getInfo())
            print("5. Sentinel-2 data available:", s2_data.bandNames().getInfo())
            
            # Try to get a sample of GEDI points
            sample_points = gedi_data.limit(5).getInfo()
            print("\nSample GEDI points:")
            for point in sample_points.get('features', []):
                print(point.get('properties', {}))
        
        self.assertGreater(training_size, 0, "No training features created")
        
        # Get first feature to check properties
        first_feature = training_data.first().getInfo()
        
        # Verify required properties exist
        self.assertIn('properties', first_feature)
        properties = first_feature['properties']
        
        # Check for required bands
        required_bands = ['rh']  # Changed from 'rh98' to 'rh'
        for band in required_bands:
            self.assertIn(band, properties, f"Missing required band: {band}")
        
        # Check for valid values
        for band in required_bands:
            value = properties[band]
            self.assertIsNotNone(value, f"Null value found for band: {band}")
            self.assertIsInstance(value, (int, float), f"Invalid value type for band: {band}")
        
        # Split into training and validation
        split = 0.7
        training = training_data.filter(ee.Filter.lt('random', split))
        validation = training_data.filter(ee.Filter.gte('random', split))
        
        # Verify split sizes
        training_size = training.size().getInfo()
        validation_size = validation.size().getInfo()
        total_size = training_size + validation_size
        
        # Check if split is approximately correct
        training_ratio = training_size / total_size
        self.assertAlmostEqual(training_ratio, split, delta=0.1,
                             msg="Training/validation split ratio is incorrect")

    def test_rf_model(self):
        """Test Random Forest model."""
        # Run canopy height mapper with RF model
        result = canopy_height_mapper(
            aoi=self.test_aoi,
            year=self.year,
            start_date=self.start_date,
            end_date=self.end_date,
            start_date_gedi=self.start_date_gedi,
            end_date_gedi=self.end_date_gedi,
            clouds_th=self.clouds_th,
            quantile=self.quantile,
            model='RF',
            mask='none',  # Don't use forest mask for testing
            gedi_type='singleGEDI',
            num_trees_rf=10,
            var_split_rf=2,
            min_leaf_pop_rf=1,
            bag_frac_rf=0.5,
            max_nodes_rf=100,
            num_trees_gbm=10,
            shr_gbm=0.1,
            sampling_rate_gbm=0.5,
            max_nodes_gbm=100,
            loss_gbm='LeastSquares',
            max_nodes_cart=100,
            min_leaf_pop_cart=1,
            training_buffer=self.training_buffer
        )
        
        # Check if result is an Image
        self.assertIsInstance(result, ee.Image)
        
        # Check if result has the expected band
        bands = result.bandNames().getInfo()
        self.assertIn('classification', bands)
        
        # Check if result has valid values
        stats = result.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=self.test_aoi,
            scale=30,
            maxPixels=1e8
        ).getInfo()
        self.assertIsNotNone(stats)
        
        # Save stats to csv
        import pandas as pd
        pd.DataFrame(stats).to_csv('../outputs/stats.csv', index=False)
        
        # Export the result to Google Drive
        task = ee.batch.Export.image.toDrive(
            image=result,
            description='canopy_height_map',
            folder='CH-GEE_Outputs',
            fileNamePrefix='canopy_height_demo',
            scale=100,
            region=self.test_aoi    
        )
        task.start()
        print("Export task started. Check your Google Drive for the results.")


    # def test_gbm_model(self):
    #     """Test Gradient Boosting Machine model."""
    #     # Run canopy height mapper with GBM model
    #     result = canopy_height_mapper(
    #         aoi=self.test_aoi,
    #         year=self.year,
    #         start_date=self.start_date,
    #         end_date=self.end_date,
    #         start_date_gedi=self.start_date_gedi,
    #         end_date_gedi=self.end_date_gedi,
    #         clouds_th=self.clouds_th,
    #         quantile=self.quantile,
    #         model='GBM',
    #         mask='none',
    #         gedi_type='singleGEDI',
    #         num_trees_rf=10,
    #         var_split_rf=2,
    #         min_leaf_pop_rf=1,
    #         bag_frac_rf=0.5,
    #         max_nodes_rf=100,
    #         num_trees_gbm=10,
    #         shr_gbm=0.1,
    #         sampling_rate_gbm=0.5,
    #         max_nodes_gbm=100,
    #         loss_gbm='LeastSquares',
    #         max_nodes_cart=100,
    #         min_leaf_pop_cart=1,
    #         training_buffer=self.training_buffer
    #     )
        
    #     # Check if result is an Image
    #     self.assertIsInstance(result, ee.Image)
        
    #     # Check if result has the expected band
    #     bands = result.bandNames().getInfo()
    #     self.assertIn('classification', bands)

    # def test_cart_model(self):
    #     """Test CART model."""
    #     # Run canopy height mapper with CART model
    #     result = canopy_height_mapper(
    #         aoi=self.test_aoi,
    #         year=self.year,
    #         start_date=self.start_date,
    #         end_date=self.end_date,
    #         start_date_gedi=self.start_date_gedi,
    #         end_date_gedi=self.end_date_gedi,
    #         clouds_th=self.clouds_th,
    #         quantile=self.quantile,
    #         model='CART',
    #         mask='none',
    #         gedi_type='singleGEDI',
    #         num_trees_rf=10,
    #         var_split_rf=2,
    #         min_leaf_pop_rf=1,
    #         bag_frac_rf=0.5,
    #         max_nodes_rf=100,
    #         num_trees_gbm=10,
    #         shr_gbm=0.1,
    #         sampling_rate_gbm=0.5,
    #         max_nodes_gbm=100,
    #         loss_gbm='LeastSquares',
    #         max_nodes_cart=100,
    #         min_leaf_pop_cart=1,
    #         training_buffer=self.training_buffer
    #     )
        
    #     # Check if result is an Image
    #     self.assertIsInstance(result, ee.Image)
        
    #     # Check if result has the expected band
    #     bands = result.bandNames().getInfo()
    #     self.assertIn('classification', bands)

    # def test_forest_mask(self):
    #     """Test forest masking functionality."""
    #     # Run canopy height mapper with forest mask
    #     result = canopy_height_mapper(
    #         aoi=self.test_aoi,
    #         year=self.year,
    #         start_date=self.start_date,
    #         end_date=self.end_date,
    #         start_date_gedi=self.start_date_gedi,
    #         end_date_gedi=self.end_date_gedi,
    #         clouds_th=self.clouds_th,
    #         quantile=self.quantile,
    #         model='RF',
    #         mask='FNF',  # Use Forest/Non-Forest mask
    #         gedi_type='singleGEDI',
    #         num_trees_rf=10,
    #         var_split_rf=2,
    #         min_leaf_pop_rf=1,
    #         bag_frac_rf=0.5,
    #         max_nodes_rf=100,
    #         num_trees_gbm=10,
    #         shr_gbm=0.1,
    #         sampling_rate_gbm=0.5,
    #         max_nodes_gbm=100,
    #         loss_gbm='LeastSquares',
    #         max_nodes_cart=100,
    #         min_leaf_pop_cart=1,
    #         training_buffer=self.training_buffer
    #     )
        
    #     # Check if result is an Image
    #     self.assertIsInstance(result, ee.Image)
        
    #     # Check if result has the expected band
    #     bands = result.bandNames().getInfo()
    #     self.assertIn('classification', bands)

    # def test_variable_importance(self):
    #     """Test variable importance calculation."""
    #     # Run canopy height mapper with RF model
    #     result = canopy_height_mapper(
    #         aoi=self.test_aoi,
    #         year=self.year,
    #         start_date=self.start_date,
    #         end_date=self.end_date,
    #         start_date_gedi=self.start_date_gedi,
    #         end_date_gedi=self.end_date_gedi,
    #         clouds_th=self.clouds_th,
    #         quantile=self.quantile,
    #         model='RF',
    #         mask='none',
    #         gedi_type='singleGEDI',
    #         num_trees_rf=10,
    #         var_split_rf=2,
    #         min_leaf_pop_rf=1,
    #         bag_frac_rf=0.5,
    #         max_nodes_rf=100,
    #         num_trees_gbm=10,
    #         shr_gbm=0.1,
    #         sampling_rate_gbm=0.5,
    #         max_nodes_gbm=100,
    #         loss_gbm='LeastSquares',
    #         max_nodes_cart=100,
    #         min_leaf_pop_cart=1,
    #         training_buffer=self.training_buffer
    #     )
        
    #     # Get variable importance
    #     importance = get_variable_importance(result)
        
    #     # Check if importance is a dictionary
    #     self.assertIsInstance(importance, dict)
        
    #     # Check if importance has values
    #     self.assertGreater(len(importance), 0)

if __name__ == '__main__':
    unittest.main() 