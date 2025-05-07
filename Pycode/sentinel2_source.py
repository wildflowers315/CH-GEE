import ee
from typing import Union, List

def get_sentinel2_data(
    aoi: ee.Geometry,
    year: int,
    start_date: str,
    end_date: str,
    clouds_th: float
) -> ee.Image:
    """
    Get Sentinel-2 data for the specified area and time period.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        year: Year for analysis
        start_date: Start date for Sentinel-2 data
        end_date: End date for Sentinel-2 data
        clouds_th: Cloud threshold (0-1)
    
    Returns:
        ee.Image: Processed Sentinel-2 data
    """
    # Import Sentinel-2 dataset
    s2 = ee.ImageCollection('COPERNICUS/S2_SR')
    
    # Filter by date and region
    s2_filtered = s2.filterDate(f"{year}-{start_date}", f"{year}-{end_date}") \
                    .filterBounds(aoi)
    
    # Filter by cloud coverage
    s2_filtered = s2_filtered.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', clouds_th * 100))
    
    # Select relevant bands
    s2_filtered = s2_filtered.select([
        'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'
    ])
    
    # Calculate vegetation indices
    def add_indices(img):
        # Normalized Difference Vegetation Index (NDVI)
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # Enhanced Vegetation Index (EVI)
        evi = img.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': img.select('B8'),
                'RED': img.select('B4'),
                'BLUE': img.select('B2')
            }
        ).rename('EVI')
        
        # Normalized Difference Water Index (NDWI)
        ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # Normalized Difference Moisture Index (NDMI)
        ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
        
        return img.addBands([ndvi, evi, ndwi, ndmi])
    
    s2_filtered = s2_filtered.map(add_indices)
    
    # Calculate temporal statistics
    s2_mean = s2_filtered.mean()
    s2_std = s2_filtered.reduce(ee.Reducer.stdDev())
    
    # Combine mean and standard deviation
    s2_processed = s2_mean.addBands(s2_std)
    
    return s2_processed 