import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

def initialize_earth_engine():
    """Initialize Earth Engine with proper authentication."""
    try:
        ee.Initialize()
    except:
        print("Please authenticate with Earth Engine first by running:")
        print("earthengine authenticate")
        raise

def check_gedi_data(aoi, start_date, end_date):
    """Check if GEDI data is available for the given area and time period."""
    gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
    filtered = gedi.filterBounds(aoi).filterDate(start_date, end_date)
    count = filtered.size().getInfo()
    print(f"Found {count} GEDI data points in the specified area and time period")
    return count > 0

def load_aoi_from_geojson(geojson_path):
    """Load AOI from GeoJSON file and ensure it's a valid geometry."""
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Ensure we have a valid polygon
        if geojson_data['type'] != 'Polygon':
            raise ValueError("GeoJSON must be a Polygon type")
        
        # Create Earth Engine geometry
        aoi = ee.Geometry.Polygon(geojson_data['coordinates'])
        
        # Get the bounds of the AOI
        bounds = aoi.bounds().getInfo()
        print(f"AOI bounds: {bounds}")
        
        return aoi
    except Exception as e:
        print(f"Error loading AOI: {str(e)}")
        raise

def create_demo_map():
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Load AOI from GeoJSON file
    aoi_path = 'downloads/aoi.geojson'
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI file not found at {aoi_path}")
    
    aoi = load_aoi_from_geojson(aoi_path)
    
    # Create a map centered on the AOI
    Map = geemap.Map(center=[36.55, 139.81], zoom=10)
    
    # Add AOI to map
    Map.addLayer(aoi, {'color': 'red'}, 'Area of Interest')
    
    # Set parameters for canopy height mapping
    year = 2023
    start_date = "01-01"
    end_date = "12-31"
    start_date_gedi = "2023-01-01"
    end_date_gedi = "2023-12-31"
    
    # Check if GEDI data is available
    print("Checking GEDI data availability...")
    if not check_gedi_data(aoi, start_date_gedi, end_date_gedi):
        print("No GEDI data found for 2023. Trying 2022...")
        start_date_gedi = "2022-01-01"
        end_date_gedi = "2022-12-31"
        if not check_gedi_data(aoi, start_date_gedi, end_date_gedi):
            print("No GEDI data found for 2022. Trying 2021...")
            start_date_gedi = "2021-01-01"
            end_date_gedi = "2021-12-31"
            if not check_gedi_data(aoi, start_date_gedi, end_date_gedi):
                raise Exception("No GEDI data available for this area in 2021-2023. Please try a different location.")
    
    clouds_th = 20
    quantile = "rh100"
    model = "RF"  # Using Random Forest model
    mask = "none"
    gedi_type = "singleGEDI"
    
    # Model parameters
    num_trees_rf = 100
    var_split_rf = 3
    min_leaf_pop_rf = 5
    bag_frac_rf = 0.5
    max_nodes_rf = 1000
    
    # These parameters are not used for RF but required by the function
    num_trees_gbm = 100
    shr_gbm = 0.1
    sampling_rate_gbm = 0.5
    max_nodes_gbm = 1000
    loss_gbm = "squared"
    max_nodes_cart = 1000
    min_leaf_pop_cart = 5
    
    # Import the canopy height mapper function
    from ch_gee_main import canopy_height_mapper
    
    # Run the canopy height mapper
    print("Running canopy height mapper...")
    try:
        tree_heights = canopy_height_mapper(
            aoi=aoi,
            year=year,
            start_date=start_date,
            end_date=end_date,
            start_date_gedi=start_date_gedi,
            end_date_gedi=end_date_gedi,
            clouds_th=clouds_th,
            quantile=quantile,
            model=model,
            mask=mask,
            gedi_type=gedi_type,
            num_trees_rf=num_trees_rf,
            var_split_rf=var_split_rf,
            min_leaf_pop_rf=min_leaf_pop_rf,
            bag_frac_rf=bag_frac_rf,
            max_nodes_rf=max_nodes_rf,
            num_trees_gbm=num_trees_gbm,
            shr_gbm=shr_gbm,
            sampling_rate_gbm=sampling_rate_gbm,
            max_nodes_gbm=max_nodes_gbm,
            loss_gbm=loss_gbm,
            max_nodes_cart=max_nodes_cart,
            min_leaf_pop_cart=min_leaf_pop_cart
        )
        
        # Add the canopy height layer to the map
        vis_params = {
            'min': 0,
            'max': 50,
            'palette': ['green', 'yellow', 'red']
        }
        Map.addLayer(tree_heights, vis_params, 'Canopy Height (m)')
        
        # Export the result to Google Drive
        task = ee.batch.Export.image.toDrive(
            image=tree_heights,
            description='canopy_height_map',
            folder='CH-GEE_Outputs',
            fileNamePrefix='canopy_height_demo',
            scale=30,
            region=aoi
        )
        task.start()
        print("Export task started. Check your Google Drive for the results.")
        
        # Display the map
        Map.show()
        
        return Map, tree_heights
        
    except Exception as e:
        print(f"Error during canopy height mapping: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have a stable internet connection")
        print("2. Verify that you have access to Google Earth Engine")
        print("3. Try a different time period")
        print("4. Check if the AOI file is valid")
        raise

if __name__ == "__main__":
    try:
        # Run the demo
        Map, tree_heights = create_demo_map()
        
        # Print instructions
        print("\nInstructions:")
        print("1. The map will open in your default web browser")
        print("2. The canopy height layer will be displayed with a color scale from green (low) to red (high)")
        print("3. The results will be exported to your Google Drive in the 'CH-GEE_Outputs' folder")
        print("4. You can use the map controls to zoom, pan, and toggle layers")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please try again with different parameters or check the AOI file.") 