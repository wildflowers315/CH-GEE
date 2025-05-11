import ee
from typing import Union, List

import ee
import math

# def mask_border_noise(img):
#     """Border Noise Correction"""
#     # Check if mask band exists
#     bands = img.bandNames().getInfo()
#     if 'mask' in bands:
#         mask = img.select(['mask']).eq(0)  # Mask border noise pixels
#         return img.updateMask(mask).copyProperties(img, ['system:time_start'])
#     return img.copyProperties(img, ['system:time_start'])

def gamma_map_filter(image, kernel_size, enl):
    """Speckle Filter - Gamma Map Multi-temporal"""
    bands = ['VV', 'VH']
    
    def filter_band(band):
        mean = image.select([band]).reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=ee.Kernel.square(kernel_size/2, 'pixels')
        )
        variance = image.select([band]).reduceNeighborhood(
            reducer=ee.Reducer.variance(),
            kernel=ee.Kernel.square(kernel_size/2, 'pixels')
        )
        # Convert to numbers for mathematical operations
        mean_num = mean.toFloat()
        variance_num = variance.toFloat()
        enl_num = ee.Number(enl)
        
        # Calculate coefficient of variation
        cv = variance_num.divide(mean_num.pow(2)).sqrt()
        
        # Calculate weight
        weight = cv.pow(-2).divide(cv.pow(-2).add(enl_num))
        
        # Apply filter
        return mean_num.multiply(weight).add(
            image.select([band]).toFloat().multiply(ee.Number(1).subtract(weight))
        )
    
    filtered = ee.Image(ee.List(bands).map(filter_band))
    return image.addBands(filtered.rename(bands), None, True)

def terrain_flattening(img, dem):
    """Terrain Flattening"""
    dem = ee.Image(dem)
    theta_i = img.select(['angle']).toFloat().multiply(ee.Number(math.pi/180))
    slope = ee.Terrain.slope(dem).toFloat().multiply(ee.Number(math.pi/180))
    aspect = ee.Terrain.aspect(dem).toFloat().multiply(ee.Number(math.pi/180))
    
    # Calculate projection angle
    phi_i = ee.Algorithms.If(
        ee.String(img.get('orbitProperties_pass')).equals('ASCENDING'),
        ee.Number(0),
        ee.Number(math.pi)
    )
    
    # Calculate d_phi using image operations
    d_phi = aspect.subtract(ee.Image.constant(phi_i))
    
    # Calculate cos_theta_s
    cos_theta_s = theta_i.cos().multiply(slope.cos()) \
        .add(theta_i.sin().multiply(slope.sin()).multiply(d_phi.cos()))
    theta_s = cos_theta_s.acos()
    
    # Apply volume model correction
    correction = theta_s.sin().divide(theta_i.sin())
    
    # Select and apply correction
    return img.select(['VV', 'VH']).toFloat() \
        .multiply(correction) \
        .set('system:time_start', img.get('system:time_start'))

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
                            .filter(ee.Filter.eq('resolution_meters', 10)) \
                            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    
    # Select VV and VH bands
    s1_filtered = s1_filtered.select(['VV', 'VH', 'angle'])
    
    # Processing chain
    s1_processed = s1_filtered \
        # .map(lambda img: gamma_map_filter(img, 15, 10)) \
        # .map(lambda img: terrain_flattening(img, 'USGS/SRTMGL1_003'))
        # .map(mask_border_noise) \
    
    # Calculate temporal statistics
    s1_median = s1_processed.select(['VV', 'VH']).median()
    s1_median = s1_median.clip(aoi)
    
    return s1_median 