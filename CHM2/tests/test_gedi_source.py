import unittest
import ee
import geemap
import os
import json
from l2a_gedi_source import get_gedi_data

class TestGEDISource(unittest.TestCase):
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
        expanded_forest_polygon = geometry.buffer(5000)
        self.test_aoi = expanded_forest_polygon

    def test_basic_gedi_retrieval(self):
        """Test basic GEDI data retrieval."""
        gedi_data = get_gedi_data(
            self.test_aoi,
            '2022-01-01',
            '2022-12-31',
            'rh98'
        )
        
        # Check if the result is an Image
        self.assertIsInstance(gedi_data, ee.Image)
        
        # Check if the data has the expected band
        band_names = gedi_data.bandNames().getInfo()
        self.assertIn('rh', band_names)
        
        # gedi_geom = gedi_data.geometry()
        # Get a sample of the data
        sample = gedi_data.sample(
            region=self.test_aoi,
            scale=30,
            numPixels=100,
            seed=42
        ).getInfo()
        
        if len(sample['features']) > 0:
            # Check if we have valid height values
            heights = [f['properties']['rh'] for f in sample['features']]
            self.assertTrue(any(h > 0 for h in heights), "No valid height values found")
        else:
            print("Warning: No GEDI data found for the test area. This might be due to:")
            print("1. No GEDI coverage in the area")
            print("2. Quality filters being too strict")
            print("3. Time period not having data")
            self.skipTest("No GEDI data available for testing")

    def test_different_quantiles(self):
        """Test GEDI data retrieval with different quantiles."""
        quantiles = ['rh100', 'rh98', 'rh95', 'rh90']#['rh100', 'rh98', 'rh95', 'rh90', 'rh75', 'rh50', 'rh25', 'rh10']
        
        for quantile in quantiles:
            gedi_data = get_gedi_data(
                self.test_aoi,
                '2022-01-01',
                '2022-12-31',
                quantile
            )
            
            # Check if the result is an Image
            self.assertIsInstance(gedi_data, ee.Image)
            
            # Check if the data has the expected band
            band_names = gedi_data.bandNames().getInfo()
            self.assertIn('rh', band_names)
            
            # gedi_geom = gedi_data.geometry()
            # Get a sample of the data
            sample = gedi_data.sample(
                region=self.test_aoi,
                scale=30,
                numPixels=100,
                seed=42
            ).getInfo()
            
            if len(sample['features']) > 0:
                # Check if we have valid height values
                heights = [f['properties']['rh'] for f in sample['features']]
                self.assertTrue(any(h > 0 for h in heights), f"No valid height values found for {quantile}")
            else:
                print(f"Warning: No GEDI data found for {quantile}")
                self.skipTest(f"No GEDI data available for {quantile}")

    def test_quality_filters(self):
        """Test if quality filters are properly applied."""
        gedi_data = get_gedi_data(
            self.test_aoi,
            '2022-01-01',
            '2022-12-31',
            'rh98'
        )
        
        # Check if the image has the expected band
        band_names = gedi_data.bandNames().getInfo()
        self.assertIn('rh', band_names)
        
        # Get a sample of the data with more points and larger area
        sample = gedi_data.sample(
            region=self.test_aoi,
            scale=30,
            numPixels=1000,
            seed=42
        ).getInfo()
        
        print(f"Number of sample points: {len(sample['features'])}")
        
        if len(sample['features']) > 0:
            # Check if we have valid height values
            heights = [f['properties']['rh'] for f in sample['features']]
            self.assertTrue(any(h > 0 for h in heights), "No valid height values found")
            
            # Check if heights are within reasonable range (0-100 meters)
            self.assertTrue(all(0 <= h <= 100 for h in heights), "Height values outside reasonable range")
        else:
            print("Warning: No sample points found. This might indicate:")
            print("1. No GEDI data in the area")
            print("2. Quality filters too strict")
            print("3. Sampling parameters need adjustment")
            self.skipTest("No valid data points found in sample")

    def test_export_gedi_to_geojson(self):
        """Test exporting GEDI data to GeoJSON format."""
        
        start_date_gedi = '2022-01-01'
        end_date_gedi = '2022-12-31'
        quantile = 'rh98'
        scale = 25 # GEDI resolution is 25m  # Using a smaller scale to get more detailed points
        
        gedi_data = get_gedi_data(
            self.test_aoi,
            start_date_gedi,
            end_date_gedi,
            quantile
        )
        
        # Get all GEDI points within the AOI
        gedi_points = gedi_data.sample(
            region=self.test_aoi,
            scale=scale,
            geometries=True,
            dropNulls=True,  # Only get points with valid data
            seed=42  # Fixed seed for reproducibility
        )
        
        # Export the data as CSV
        csv_task = ee.batch.Export.table.toDrive(
            collection=gedi_points,
            description='GEDI_Data_Export_CSV',
            folder='CH-GEE_Outputs',
            fileNamePrefix=f'gedi_data_{quantile}_{start_date_gedi}_{end_date_gedi}_scale{scale}',
            fileFormat='CSV'
        )
        csv_task.start()
        
        # Export the data as GeoJSON
        geojson_task = ee.batch.Export.table.toDrive(
            collection=gedi_points,
            description='GEDI_Data_Export_GeoJSON',
            folder='CH-GEE_Outputs',
            fileNamePrefix=f'gedi_data_{quantile}_{start_date_gedi}_{end_date_gedi}_scale{scale}',
            fileFormat='GeoJSON'
        )
        geojson_task.start()
        
        # Print task information
        print(f"CSV Export Task ID: {csv_task.id}")
        print(f"GeoJSON Export Task ID: {geojson_task.id}")
        print("Export tasks started. Check Earth Engine Tasks panel to download the files.")
        
        # if task.status()['state'] == 'COMPLETED':
        #     print("Export completed successfully")
            
        #     # Verify the exported data
        #     self.assertTrue(os.path.exists(geojson_file), "GeoJSON file was not created")
            
        #     # Read and validate the GeoJSON
        #     with open(geojson_file, 'r') as f:
        #         geojson_data = json.load(f)
                
        #     # Check if it's valid GeoJSON
        #     self.assertIn('features', geojson_data)
        #     self.assertGreater(len(geojson_data['features']), 0)
            
        #     # Check if features have required properties
        #     for feature in geojson_data['features']:
        #         self.assertIn('properties', feature)
        #         self.assertIn('rh', feature['properties'])
        #         self.assertIn('geometry', feature)
                
        #         # Check if height value is a float
        #         self.assertIsInstance(feature['properties']['rh'], float)
                
        #         # Check if geometry is valid
        #         self.assertIn('type', feature['geometry'])
        #         self.assertIn('coordinates', feature['geometry'])
        # else:
        #     print(f"Export failed: {task.status()['error_message']}")
        #     self.fail("Export task failed")

    def test_visualization(self):
        """Test visualization of GEDI data."""
        gedi_data = get_gedi_data(
            self.test_aoi,
            '2022-01-01',
            '2022-12-31',
            'rh98'
        )
        
        # Create a map
        Map = geemap.Map()
        Map.centerObject(self.test_aoi, 15)
        
        # Add GEDI data with height visualization
        height_vis = {
            'bands': ['rh'],
            'min': 0,
            'max': 50,
            'palette': ['#FFFFFF', '#90EE90', '#006400'],
            'breaks': [0, 5, 50]
        }
        
        # Add the layers
        Map.addLayer(gedi_data, height_vis, 'GEDI Data')
        Map.addLayer(self.test_aoi, {'color': 'white'}, 'AOI')
        
        # Save visualization
        output_dir = os.path.join('..', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        html_file = os.path.join(output_dir, "gedi_data_visualization.html")
        Map.to_html(filename=html_file, title="GEDI Data Visualization", width="100%", height="880px")
        
        # Verify output file
        self.assertTrue(os.path.exists(html_file))

    # def test_no_data_available(self):
    #     """Test when no GEDI data is available for the area."""
    #     # Use an area where we know there won't be data (middle of ocean)
    #     ocean_aoi = ee.Geometry.Rectangle([0, 0, 0.000001, 0.000001])
        
    #     gedi_data = get_gedi_data(
    #         ocean_aoi,
    #         '2022-01-01',
    #         '2022-12-31',
    #         'rh98'
    #     )
        
    #     # Check if the result is an ImageCollection
    #     self.assertIsInstance(gedi_data, ee.ImageCollection)
        
    #     # Get the size of the data
    #     size = gedi_data.size().getInfo()
    #     print(f"Number of GEDI images found in ocean area: {size}")
        
    #     if size > 0:
    #         # Even if we have images, check if they contain any valid data
    #         first_image = gedi_data.first()
    #         sample = first_image.sample(
    #             region=ocean_aoi,
    #             scale=30,
    #             numPixels=100
    #         ).getInfo()
            
    #         # Should have no valid data points
    #         self.assertEqual(len(sample['features']), 0)
    #     else:
    #         self.assertEqual(size, 0)  # No images at all

if __name__ == '__main__':
    unittest.main() 