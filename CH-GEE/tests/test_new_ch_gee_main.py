import unittest
import ee
import geemap
import os
from new_ch_gee_main import canopy_height_mapper
from new_random_sampling import create_training_data, generate_sampling_sites
from l2a_gedi_source import get_gedi_data
from sentinel1_source import get_sentinel1_data
from sentinel2_source import get_sentinel2_data
from for_forest_masking import apply_forest_mask

class TestNewCanopyHeightMapper(unittest.TestCase):
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
        self.training_buffer = 10

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
            mask='DW',  # Don't use forest mask for testing
            gedi_type='singleGEDI',
            num_trees_rf=500,
            var_split_rf=None, 
            min_leaf_pop_rf=1,
            bag_frac_rf=0.5,
            max_nodes_rf=5,
            num_trees_gbm=10,
            shr_gbm=0.1,
            sampling_rate_gbm=0.5,
            max_nodes_gbm=100,
            loss_gbm='LeastSquares',
            max_nodes_cart=100,
            min_leaf_pop_cart=1,
            training_buffer=self.training_buffer,
            # gedi_geojson_path=os.path.join('..', 'downloads', 'gedi_data_rh98_2022-01-01_2022-12-31_scale25.geojson')
        )
        
        # Check if result is a list containing an Image
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ee.Image)
        
        # Check if result has valid values
        stats = result[0].reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=self.test_aoi,
            scale=25,
            maxPixels=1e8,
            bestEffort=True,
        ).getInfo()
        self.assertIsNotNone(stats)
        print("Stats type:", type(stats))
        print("Stats value:", stats)
        # Check if stats is valid before creating DataFrame
        if stats and isinstance(stats, dict) and len(stats) > 0:
            # Save stats to csv
            import pandas as pd
            stats_path = os.path.join('..', 'outputs', 'stats_new.csv')
            pd.DataFrame([stats]).to_csv(stats_path, index=False)  # Note the [stats] to make it a list
            print(f"Stats saved to {stats_path}")
        else:
            print("Warning: No valid statistics returned from Earth Engine")
        
        # Export the result to Google Drive
        task = ee.batch.Export.image.toDrive(
            image=result[0],
            description='canopy_height_map_new',
            folder='CH-GEE_Outputs',
            fileNamePrefix='canopy_height_demo_new',
            scale=100,
            region=self.test_aoi    
        )
        task.start()
        print("Export task started. Check your Google Drive for the results.")


    # def test_training_data_creation(self):
    #     """Test the creation and validation of training data."""
    #     # Create training AOI with buffer
    #     training_aoi = self.test_aoi.buffer(self.training_buffer)
        
    #     # Get GEDI data
    #     gedi_data = get_gedi_data(
    #         training_aoi,
    #         self.start_date_gedi,
    #         self.end_date_gedi,
    #         self.quantile
    #     )
    #     # gedi_data = gedi_data.select("rh").reduce(ee.Reducer.firstNonNull()).rename("rh")
        
    #     # Verify GEDI data exists
    #     assert gedi_data is not None, "GEDI data is None"
    #     # print("GEDI data:", gedi_data.getInfo())
    #     # gedi_size = gedi_data.size().getInfo()
    #     # print(f"Number of GEDI points: {gedi_size}")
    #     # self.assertGreater(gedi_size, 0, "No GEDI data found in the area")
        
    #     # # Filter GEDI data to ensure it has required properties
    #     # gedi_data = gedi_data.filter(ee.Filter.notNull(['rh']))
    #     # gedi_size = gedi_data.size().getInfo()
    #     # print(f"Number of GEDI points with valid height data: {gedi_size}")
    #     # self.assertGreater(gedi_size, 0, "No GEDI points with valid height data found")
        
    #     # Get Sentinel data
    #     s1_data = get_sentinel1_data(
    #         training_aoi,
    #         self.year,
    #         self.start_date,
    #         self.end_date
    #     )
    #     s2_data = get_sentinel2_data(
    #         training_aoi,
    #         self.year,
    #         self.start_date,
    #         self.end_date,
    #         self.clouds_th,
    #         training_aoi
    #     )
        
    #     # Print Sentinel data bands for debugging
    #     print("Sentinel-1 bands:", s1_data.bandNames().getInfo())
    #     print("Sentinel-2 bands:", s2_data.bandNames().getInfo())
        
    #     # Merge all data sources
    #     merged = s2_data.select(ee.List.sequence(0, 11, 1)).addBands(gedi_data).addBands(s1_data)
        
    #     # Try to create forest mask, but don't fail if it's not available
    #     try:
    #         forest_mask = apply_forest_mask(
    #             training_aoi,
    #             'DW',  # Dynamic World mask
    #             training_aoi,
    #             self.year,
    #             self.start_date,
    #             self.end_date
    #         )
    #         print("Forest mask created successfully")
    #     except Exception as e:
    #         print(f"Warning: Could not create forest mask: {e}")
    #         forest_mask = None
        
    #     # Create training data
    #     print("Creating training data...")
    #     generated_points = generate_sampling_sites(
    #         self.test_aoi,
    #         cell_size=4000,
    #         seed=1,
    #         mask_raster=forest_mask
    #     )
    #     # aoi_buffer = generated_points.buffer
    #     # aoi_prova = aoi_buffer.geometry()
    #     # ample(region, scale, projection, factor, numPixels, seed, dropNulls, tileScale, geometries)
        
    #     reference = merged.sample(
    #         # region=merged,
    #         scale=10,
    #         projection=merged.projection(),
    #         numPixels=1e13,
    #         seed=0,            
    #         dropNulls=True,
    #         geometries=True
    #     )
        
    #     # Verify training data
    #     reference_size = reference.size().getInfo()
    #     print(f"Number of reference features created: {reference_size}")
    #     if reference_size == 0:
    #         raise ValueError("No valid reference features were created")
        
    #     # Verify training data properties
    #     self.assertIsInstance(reference, ee.FeatureCollection)
    #     self.assertGreater(reference_size, 0, "No training features created")

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
        
    #     # Check if result is a list containing an Image
    #     self.assertIsInstance(result, list)
    #     self.assertEqual(len(result), 1)
    #     self.assertIsInstance(result[0], ee.Image)
        
    #     # Check if result has valid values
    #     stats = result[0].reduceRegion(
    #         reducer=ee.Reducer.minMax(),
    #         geometry=self.test_aoi,
    #         scale=30,
    #         maxPixels=1e8
    #     ).getInfo()
    #     self.assertIsNotNone(stats)

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
        
    #     # Check if result is a list containing an Image
    #     self.assertIsInstance(result, list)
    #     self.assertEqual(len(result), 1)
    #     self.assertIsInstance(result[0], ee.Image)
        
    #     # Check if result has valid values
    #     stats = result[0].reduceRegion(
    #         reducer=ee.Reducer.minMax(),
    #         geometry=self.test_aoi,
    #         scale=30,
    #         maxPixels=1e8
    #     ).getInfo()
    #     self.assertIsNotNone(stats)

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
        
    #     # Check if result is a list containing an Image
    #     self.assertIsInstance(result, list)
    #     self.assertEqual(len(result), 1)
    #     self.assertIsInstance(result[0], ee.Image)
        
    #     # Check if result has valid values
    #     stats = result[0].reduceRegion(
    #         reducer=ee.Reducer.minMax(),
    #         geometry=self.test_aoi,
    #         scale=30,
    #         maxPixels=1e8
    #     ).getInfo()
    #     self.assertIsNotNone(stats)

if __name__ == '__main__':
    unittest.main() 