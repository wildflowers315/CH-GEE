import ee
import geemap
import numpy as np
import json
import os
from pathlib import Path
import pandas as pd
import datetime
import argparse
from typing import Union, List, Dict, Any
import time
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

# Import custom functions
from l2a_gedi_source import get_gedi_data
from sentinel1_source import get_sentinel1_data
from sentinel2_source import get_sentinel2_data
from for_forest_masking import apply_forest_mask, get_forest_mask
from alos2_source import get_alos2_data
from new_random_sampling import create_training_data, generate_sampling_sites

def load_aoi(aoi_path: str) -> ee.Geometry:
    """
    Load AOI from GeoJSON file. Handles both simple Polygon/MultiPolygon and FeatureCollection formats.
    
    Args:
        aoi_path: Path to GeoJSON file
    
    Returns:
        ee.Geometry: Earth Engine geometry object
    """
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")
    
    with open(aoi_path, 'r') as f:
        geojson_data = json.load(f)
    
    def create_geometry(geom_type: str, coords: List) -> ee.Geometry:
        """Helper function to create ee.Geometry objects."""
        if geom_type == 'Polygon':
            return ee.Geometry.Polygon(coords)
        elif geom_type == 'MultiPolygon':
            # MultiPolygon coordinates are nested one level deeper than Polygon
            return ee.Geometry.MultiPolygon(coords[0])
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")
    
    # Handle FeatureCollection
    if geojson_data['type'] == 'FeatureCollection':
        if not geojson_data['features']:
            raise ValueError("Empty FeatureCollection")
        
        # Get the first feature's geometry
        geometry = geojson_data['features'][0]['geometry']
        return create_geometry(geometry['type'], geometry['coordinates'])
    
    # Handle direct Polygon/MultiPolygon
    elif geojson_data['type'] in ['Polygon', 'MultiPolygon']:
        return create_geometry(geojson_data['type'], geojson_data['coordinates'])
    else:
        raise ValueError(f"Unsupported GeoJSON type: {geojson_data['type']}")

def parse_args():
    parser = argparse.ArgumentParser(description='Canopy Height Mapping using Earth Engine')
    # Basic parameters
    parser.add_argument('--aoi', type=str, required=True, help='Path to AOI GeoJSON file')
    parser.add_argument('--year', type=int, required=True, help='Year for analysis')
    parser.add_argument('--start-date', type=str, default='01-01', help='Start date (MM-DD)')
    parser.add_argument('--end-date', type=str, default='12-31', help='End date (MM-DD)')
    parser.add_argument('--clouds-th', type=float, default=65, help='Cloud threshold')
    parser.add_argument('--scale', type=int, default=30, help='Output resolution in meters')
    
    # GEDI parameters
    parser.add_argument('--gedi-start-date', type=str, help='GEDI start date (YYYY-MM-DD)')
    parser.add_argument('--gedi-end-date', type=str, help='GEDI end date (YYYY-MM-DD)')
    parser.add_argument('--quantile', type=str, default='098', help='GEDI height quantile')
    parser.add_argument('--gedi-type', type=str, default='singleGEDI', help='GEDI data type')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='RF', choices=['RF', 'GBM', 'CART'],
                       help='Machine learning model type')
    parser.add_argument('--num-trees-rf', type=int, default=100,
                       help='Number of trees for Random Forest')
    parser.add_argument('--min-leaf-pop-rf', type=int, default=1,
                       help='Minimum leaf population for Random Forest')
    parser.add_argument('--bag-frac-rf', type=float, default=0.5,
                       help='Bagging fraction for Random Forest')
    parser.add_argument('--max-nodes-rf', type=int, default=None,
                       help='Maximum nodes for Random Forest')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for CSV and TIF files')
    parser.add_argument('--export-training', action='store_true',
                       help='Export training data as CSV')
    parser.add_argument('--export-predictions', action='store_true',
                       help='Export predictions as TIF')
    
    args = parser.parse_args()
    return args

def initialize_ee():
    """Initialize Earth Engine with project ID."""
    EE_PROJECT_ID = "my-project-423921"
    ee.Initialize(project=EE_PROJECT_ID)

def export_training_data(reference_data: ee.FeatureCollection, output_path: str):
    """Export training data as CSV."""
    # Get feature properties as a list of dictionaries
    features = reference_data.getInfo()['features']
    
    # Extract properties and coordinates
    data = []
    for feature in features:
        properties = feature['properties']
        geometry = feature['geometry']['coordinates']
        properties['longitude'] = geometry[0]
        properties['latitude'] = geometry[1]
        data.append(properties)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Training data exported to: {output_path}")

def export_tif2(image: ee.Image, aoi: ee.Geometry, output_path: str, scale: int):
    """Export predicted canopy height map as GeoTIFF."""
    # Get image data and projection
    data = image.getInfo()
    projection = image.projection().getInfo()
    
    # Get bounds
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    min_lon = min(p[0] for p in bounds)
    max_lon = max(p[0] for p in bounds)
    min_lat = min(p[1] for p in bounds)
    max_lat = max(p[1] for p in bounds)
    
    # Calculate dimensions
    width = int((max_lon - min_lon) / (scale / 111000))  # approximately degrees to meters
    height = int((max_lat - min_lat) / (scale / 111000))
    
    # Get image data as array
    data_array = image.sampleRectangle(
        region=aoi,
        properties=image.bandNames(),
        defaultValue=-9999
    ).getInfo()
    
    # Find the actual band name in the data array
    band_name = list(data_array.keys())[0]  # Get the first band name
    
    # Create the GeoTIFF
    transform = from_origin(min_lon, max_lat, scale/111000, scale/111000)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(4326),
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(data_array[band_name], 1)
    except Exception as e:
        print('Use classification band instead')
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(4326),
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(data_array['classification'], 1)
    print(f"Predictions exported to: {output_path}")

def export_tif(image: ee.Image, aoi: ee.Geometry, output_path: str, scale: int):
    """Export predicted canopy height map as GeoTIFF."""
    # Rename the classification band for clarity (optional)
    if 'classification' in image.bandNames().getInfo():
        image = image.select(['classification'], ['canopy_height'])
    
    # Get bounds
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    min_lon = min(p[0] for p in bounds)
    max_lon = max(p[0] for p in bounds)
    min_lat = min(p[1] for p in bounds)
    max_lat = max(p[1] for p in bounds)
    
    # Calculate dimensions
    width = int((max_lon - min_lon) / (scale / 111000))  # approximately degrees to meters
    height = int((max_lat - min_lat) / (scale / 111000))
    
    # Get image data as array using getRegion instead of sampleRectangle
    region = aoi.bounds()
    data = image.getRegion(region, scale).getInfo()
    
    # Convert the data to a numpy array
    # The first row contains column headers
    headers = data[0]
    values = data[1:]
    
    # Find the band index (either 'classification' or the first band)
    band_index = None
    if 'classification' in headers:
        band_index = headers.index('classification')
    elif 'canopy_height' in headers:
        band_index = headers.index('canopy_height')
    else:
        # Get the first band that's not id, longitude, latitude, or time
        for i, header in enumerate(headers):
            if header not in ['id', 'longitude', 'latitude', 'time']:
                band_index = i
                break
    
    if band_index is None:
        raise ValueError("Could not find band data in the image")
    
    # Create a 2D grid to hold the data
    grid = np.full((height, width), -9999, dtype=np.float32)
    
    # Extract coordinates and values
    x_coords = [row[1] for row in values]  # longitude
    y_coords = [row[2] for row in values]  # latitude
    pixel_values = [row[band_index] for row in values]
    
    # Convert geographic coordinates to pixel coordinates
    x_pixel = np.floor((np.array(x_coords) - min_lon) / ((max_lon - min_lon) / width)).astype(int)
    y_pixel = np.floor((max_lat - np.array(y_coords)) / ((max_lat - min_lat) / height)).astype(int)
    
    # Assign values to the grid
    for i in range(len(x_pixel)):
        if 0 <= x_pixel[i] < width and 0 <= y_pixel[i] < height:
            grid[y_pixel[i], x_pixel[i]] = pixel_values[i]
    
    # Create the GeoTIFF
    transform = from_origin(min_lon, max_lat, (max_lon - min_lon) / width, (max_lat - min_lat) / height)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(grid, 1)
    
    print(f"Predictions exported to: {output_path}")


def export_tif_via_ee(image: ee.Image, aoi: ee.Geometry, output_path: str, scale: int):
    """Export predicted canopy height map as GeoTIFF using Earth Engine export."""
    # Rename the classification band for clarity
    if 'classification' in image.bandNames().getInfo():
        image = image.select(['classification'], ['canopy_height'])
    
    # Generate a unique task ID
    task_id = f"chm_export_{int(time.time())}"
    
    # Set export parameters
    export_params = {
        'image': image,
        'description': task_id,
        'fileNamePrefix': task_id,
        'scale': scale,
        'region': aoi,
        'fileFormat': 'GeoTIFF',
        'maxPixels': 1e10
    }
    
    # Start the export task
    task = ee.batch.Export.image.toDrive(**export_params)
    task.start()
    
    print(f"Export started with task ID: {task_id}")
    print("The file will be available in your Google Drive once the export completes.")
    print(f"You can manually download it and save to: {output_path}")
    
    # # Optionally monitor the task
    # while task.status()['state'] in ['READY', 'RUNNING']:
    #     print(f"Task status: {task.status()['state']}")
    #     time.sleep(5)
    
    # print(f"Task completed with status: {task.status()['state']}")
    
def main():
    """Main function to run the canopy height mapping process."""
    # Parse arguments
    args = parse_args()
    
    # Initialize Earth Engine
    initialize_ee()
    
    # Load AOI
    aoi = load_aoi(args.aoi)
    
    # Set dates
    start_date = f"{args.year}-{args.start_date}"
    end_date = f"{args.year}-{args.end_date}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get GEDI data
    print("Loading GEDI data...")
    gedi = get_gedi_data(aoi, args.gedi_start_date, args.gedi_end_date, args.quantile)
    
    # Sample GEDI points
    gedi_points = gedi.sample(
        region=aoi,
        scale=args.scale,
        geometries=True,
        dropNulls=True,
        seed=42
    )
    
    # Get satellite data
    print("Collecting satellite data...")
    s1 = get_sentinel1_data(aoi, args.year, args.start_date, args.end_date)
    s2 = get_sentinel2_data(aoi, args.year, args.start_date, args.end_date, args.clouds_th)
    
    # Get terrain data
    try:
        dem = ee.Image("USGS/GMTED2010_FULL").select(['min'], ['elevation']) #"USGS/GMTED2010" was deprecated.
        slope = ee.Terrain.slope(dem)
        aspect = ee.Terrain.aspect(dem)
        dem_data = dem.addBands(slope).addBands(aspect).select(['elevation', 'slope', 'aspect'])
    except Exception as e:
        # print(f"Downloading SRTM data instead{}")
        print("Donwloading SRTM data instead")
        dem = ee.Image("USGS/SRTMGL1_003").select('elevation')
        slope = ee.Terrain.slope(dem)
        aspect = ee.Terrain.aspect(dem)
        dem_data = dem.addBands(slope).addBands(aspect).select(['elevation', 'slope', 'aspect'])

    # Import ALOS2 sar data
    alos2 = get_alos2_data(aoi, args.year, args.start_date, args.end_date,include_texture=False,
                speckle_filter=False)
    
    # ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2") # Uncomment if using ALOS data
    
    # Reproject datasets to the same projection
    s2_projection = s2.projection()
    # Convert to Float32
    dem_data = dem_data.reproject(s2_projection).float()  # Convert to Float32
    s1 = s1.reproject(s2_projection).float()              # Convert to Float32
    s2 = s2.float()                                       # Convert to Float32
    alos2 = alos2.reproject(s2_projection).float()        # Convert to Float32
    # Merge datasets
    merged = s2.addBands(s1).addBands(dem_data).addBands(alos2)
    
    # Sample points
    reference_data = merged.sampleRegions(
        collection=gedi_points,
        scale=args.scale,
        projection=s2_projection,
        tileScale=1,
        geometries=True,
    )
    
    # Export training data if requested
    if args.export_training:
        training_path = os.path.join(args.output_dir, 'training_data.csv')
        export_training_data(reference_data, training_path)
        stack_tif_path = os.path.join(args.output_dir, 'stacked_tif.tif')
        try:
            export_tif(merged, aoi, stack_tif_path, args.scale)
        except Exception as e:
            print('Exporting via Earth Engine instead')
            export_tif_via_ee(merged, aoi, stack_tif_path, args.scale)
        
    # Train model
    print("Training model...")
    predictor_names = merged.bandNames()
    if args.model == "RF":
        var_split_rf = int(np.sqrt(predictor_names.size().getInfo()).round())
        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=args.num_trees_rf,
            variablesPerSplit=var_split_rf,
            minLeafPopulation=args.min_leaf_pop_rf,
            bagFraction=args.bag_frac_rf,
            maxNodes=args.max_nodes_rf
        ).setOutputMode("Regression") \
         .train(reference_data, "rh", predictor_names)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = merged.classify(classifier)
    
    # Export predictions if requested
    if args.export_predictions:
        prediction_path = os.path.join(args.output_dir, 'predictions.tif')
        try:
            export_tif(predictions, aoi, prediction_path, args.scale)
        except Exception as e:
            print('Exporting via Earth Engine instead')
            export_tif_via_ee(predictions, aoi, prediction_path, args.scale)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()