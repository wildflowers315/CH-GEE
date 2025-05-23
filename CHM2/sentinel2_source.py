import ee
from typing import Union, List

def get_sentinel2_data(
    aoi: ee.Geometry,
    year: int,
    start_date: str,
    end_date: str,
    clouds_th: int,
) -> ee.Image:
    """
    Get Sentinel-2 data for the specified area and time period.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        year: Year for analysis
        start_date: Start date for Sentinel-2 data
        end_date: End date for Sentinel-2 data
        clouds_th: Cloud threshold (0-100)
    
    Returns:
        ee.Image: Processed Sentinel-2 data
    """
    # Format dates properly for Earth Engine
    start_date_ee = ee.Date(f'{year}-{start_date}')
    end_date_ee = ee.Date(f'{year}-{end_date}')
    
    # Import Sentinel-2 dataset
    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') #original code used 'COPERNICUS/S2_SR_HARMONIZED')
    
    # Filter by date and region
    s2_filtered = s2.filterDate(start_date_ee, end_date_ee) \
                    .filterBounds(aoi) #\
                    # .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', clouds_th))
    
    # Get cloud probability data
    S2_CLOUD_PROBABILITY = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
        .filterDate(start_date_ee, end_date_ee) \
        .filterBounds(aoi)
    
    s2_filtered = ee.ImageCollection(s2_filtered) \
                    .map(lambda img: img.addBands(S2_CLOUD_PROBABILITY.filter(ee.Filter.equals('system:index', img.get('system:index'))).first()))
    # For S2, get cloud probability called 'cloud_mask' from the S2_CLOUD_PROBABILITY collection without joining
    # s2_filtered = ee.ImageCollection(s2_filtered).merge(S2_CLOUD_PROBABILITY.select('cloud_mask'))
        
    # # Join with cloud probability data
    # join_filter = ee.Filter.And(
    #     ee.Filter.equals('system:index', 'system:index'),
    #     ee.Filter.equals('system:time_start', 'system:time_start')
    # )
    
    # s2_filtered = ee.Join.saveFirst('cloud_mask').apply(
    #     primary=s2_filtered,
    #     secondary=S2_CLOUD_PROBABILITY,
    #     condition=join_filter
    # )
    
    # s2_filtered = ee.ImageCollection(s2_filtered)
    
    def maskClouds(img):
        clouds = ee.Image(img).select('probability')
        # ee.Image(img.get('cloud_mask')).select('probability')
        isNotCloud = clouds.lt(clouds_th)
        return img.mask(isNotCloud)
    
    def maskEdges(s2_img):
        return s2_img.updateMask(
            s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask())
        )#.updateMask(mask_raster.eq(1))

    s2_filtered = s2_filtered.map(maskEdges) 
    s2_filtered = s2_filtered.map(maskClouds) 
    
    # Select relevant bands
    s2_filtered = s2_filtered.select([
        'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'
    ])
    
    # Calculate vegetation indices
    def add_indices(img):
        # Normalized Difference Vegetation Index (NDVI)
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # # Enhanced Vegetation Index (EVI)
        # evi = img.expression(
        #     '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        #     {
        #         'NIR': img.select('B8'),
        #         'RED': img.select('B4'),
        #         'BLUE': img.select('B2')
        #     }
        # ).rename('EVI')
        
        # # Normalized Difference Water Index (NDWI)
        # ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # # Normalized Difference Moisture Index (NDMI)
        # ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
        
        # return img.addBands([ndvi, evi, ndwi, ndmi])
        return img.addBands([ndvi])

    
    s2_filtered = s2_filtered.map(add_indices)
    
    # Calculate temporal statistics
    s2_processed = s2_filtered.median()
    s2_processed = s2_processed.clip(aoi)
    # s2_processed = s2_processed.clip(geometry)
    
    return s2_processed 