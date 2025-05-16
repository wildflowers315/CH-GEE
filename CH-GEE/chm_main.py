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
from for_forest_masking import apply_forest_mask, create_forest_mask
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
    parser.add_argument('--mask-type', type=str, default='NDVI',
                       choices=['DW', 'FNF', 'NDVI', 'ALL', 'none'],
                       help='Type of forest mask to apply')
    
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

def export_training_data(reference_data: ee.FeatureCollection, output_dir: str):
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
    band_length = len(df.columns) - 3  # Exclude 'rh', 'longitude' and 'latitude'
    df_size = len(df)
    output_path = os.path.join(output_dir, f"training_data_b{band_length}_{df_size}.csv")
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Training data exported to: {output_path}")
    return output_path

def export_tif_via_ee(image: ee.Image, aoi: ee.Geometry, prefix: str, scale: int):
    """Export predicted canopy height map as GeoTIFF using Earth Engine export."""
    # Rename the classification band for clarity
    # if 'classification' in image.bandNames().getInfo():
    #     image = image.select(['classification'], ['canopy_height'])
    band_count = image.bandNames().size().getInfo()
    # Get the first band name
    first_band = image.bandNames().get(0)
    region = image.geometry().bounds()
    info = image.select(first_band).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=10,
        maxPixels=1e13
    )
    pixel_count = info.get(first_band)
    
    # Generate a unique task ID
    task_id = f"{prefix}_b{band_count}_s{scale}_p{pixel_count}"
    
    # Set export parameters
    export_params = {
        'image': image,
        'description': task_id,
        'fileNamePrefix': task_id,
        'folder': 'GEE_exports',
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
    # print(f"You can manually download it and save to: {output_path}")
    
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
    
    # Create and apply forest mask
    print(f"Creating and applying forest mask (type: {args.mask_type})...")
    forest_mask = create_forest_mask(args.mask_type, aoi,
                                   ee.Date(f"{args.year}-{args.start_date}"),
                                   ee.Date(f"{args.year}-{args.end_date}"))
    
    # Calculate forest area in hectares
    mask_area = forest_mask.multiply(ee.Image.pixelArea().divide(10000))
    forest_area_ha = mask_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=args.scale,
        maxPixels=1e10
    ).get('constant')
    
    # Apply forest mask to reference data and merged data
    reference_data = apply_forest_mask(reference_data, args.mask_type, aoi,
                                     args.year, args.start_date, args.end_date)
    merged = apply_forest_mask(merged, args.mask_type, aoi,
                             args.year, args.start_date, args.end_date)
    
    # Export forest mask using export_tif_via_ee
    forest_mask_filename = f'forest_mask_{forest_area_ha.getInfo():.1f}ha'
    forest_mask_path = os.path.join(args.output_dir, f'{forest_mask_filename}.tif')
    print(f"Exporting forest mask as {forest_mask_filename}...")
    export_tif_via_ee(forest_mask, aoi, 'forest-mask', args.scale)
    
    predictor_names = merged.bandNames()
    
    # Export training data if requested
    if args.export_training:
        # training_path = os.path.join(args.output_dir, 'training_data.csv')
        training_path = export_training_data(reference_data, args.output_dir,)
        stack_tif_path = os.path.join(args.output_dir, 'stacked_tif.tif')
        
        print('Exporting training data and tif through Earth Engine')
        export_tif_via_ee(merged, aoi, 'stack', args.scale)
        
    # Train model
    print("Training model...")
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
        print('Exporting via Earth Engine instead')
        export_tif_via_ee(predictions, aoi, 'chm-prediction', args.scale)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()