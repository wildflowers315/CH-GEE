import ee
from typing import Union, List

def get_sentinel1_data(
    aoi: ee.Geometry,
    year: int,
    start_date: str,
    end_date: str
) -> ee.Image:
    """
    Get Sentinel-1 data for the specified area and time period.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        year: Year for analysis
        start_date: Start date for Sentinel-1 data
        end_date: End date for Sentinel-1 data
    
    Returns:
        ee.Image: Processed Sentinel-1 data
    """
    # Import Sentinel-1 dataset
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    
    # Filter by date and region
    s1_filtered = s1.filterDate(f"{year}-{start_date}", f"{year}-{end_date}") \
                    .filterBounds(aoi)
    
    # Filter by instrument mode and polarization
    s1_filtered = s1_filtered.filter(ee.Filter.eq('instrumentMode', 'IW')) \
                             .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                             .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    
    # Select VV and VH bands
    s1_filtered = s1_filtered.select(['VV', 'VH'])
    
    # Convert to linear scale
    s1_filtered = s1_filtered.map(lambda img: ee.Image(10).pow(img.divide(10)))
    
    # Calculate VV/VH ratio
    s1_filtered = s1_filtered.map(lambda img: img.addBands(
        img.select('VV').divide(img.select('VH')).rename('VVVH')
    ))
    
    # Calculate temporal statistics
    s1_mean = s1_filtered.mean()
    s1_std = s1_filtered.reduce(ee.Reducer.stdDev())
    
    # Combine mean and standard deviation
    s1_processed = s1_mean.addBands(s1_std)
    
    return s1_processed 