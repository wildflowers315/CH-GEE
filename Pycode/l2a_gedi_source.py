import ee
from typing import Union, List

def get_gedi_data(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    quantile: str
) -> ee.FeatureCollection:
    """
    Get GEDI L2A data for the specified area and time period.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        start_date: Start date for GEDI data
        end_date: End date for GEDI data
        quantile: Quantile for GEDI data (e.g., 'rh100')
    
    Returns:
        ee.FeatureCollection: GEDI data points
    """
    # Import GEDI L2A dataset
    gedi = ee.FeatureCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
    
    # Filter by date and region
    gedi_filtered = gedi.filterDate(start_date, end_date) \
                        .filterBounds(aoi)
    
    # Select quality metrics and height metrics
    gedi_filtered = gedi_filtered.select([
        'quality_flag',
        'degrade_flag',
        'sensitivity',
        'solar_elevation',
        'rh100',
        'rh98',
        'rh95',
        'rh90',
        'rh75',
        'rh50',
        'rh25',
        'rh10'
    ])
    
    # Filter by quality
    gedi_filtered = gedi_filtered.filter(ee.Filter.eq('quality_flag', 1)) \
                                 .filter(ee.Filter.eq('degrade_flag', 0)) \
                                 .filter(ee.Filter.gt('sensitivity', 0.95))
    
    # Filter by solar elevation
    gedi_filtered = gedi_filtered.filter(ee.Filter.gt('solar_elevation', 0))
    
    # Select the specified quantile
    gedi_filtered = gedi_filtered.select([quantile])
    
    return gedi_filtered 