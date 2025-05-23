import ee
from typing import Union, List

def get_gedi_data(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    quantile: str
) -> ee.ImageCollection:
    """
    Get GEDI L2A data for the specified area and time period.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        start_date: Start date for GEDI data
        end_date: End date for GEDI data
        quantile: Quantile for GEDI data (e.g., 'rh100')
    
    Returns:
        ee.ImageCollection: GEDI data points
    """
    # Import GEDI L2A dataset
    gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
    
    # Filter by date and region
    gedi_filtered = gedi.filterDate(start_date, end_date) \
                        .filterBounds(aoi)

    # Select quality metrics and height metrics
    # gedi_filtered = gedi_filtered.select([
    #     'quality_flag',
    #     'degrade_flag',
    #     'sensitivity',
    #     'solar_elevation',
    #     'rh100',
    #     'rh98',
    #     'rh95',
    #     'rh90',
    #     'rh75',
    #     'rh50',
    #     'rh25',
    #     'rh10'
    # ])
    
    def qualityMask(img):
        # First check if we have valid data
        has_data = img.select(quantile).gt(0)
        # Then apply quality filters
        quality_ok = img.select("quality_flag").eq(1)
        degrade_ok = img.select("degrade_flag").eq(0)
        sensitivity_ok = img.select('sensitivity').gt(0.95)
        fullPowerBeam_ok = img.select('beam').gt(4)
        # Combine all conditions
        return img.updateMask(has_data) \
                 .updateMask(quality_ok) \
                 .updateMask(degrade_ok) \
                .updateMask(sensitivity_ok) \
                .updateMask(fullPowerBeam_ok)
    
    # Select and rename the quantile
    # def rename_property(image):
    #     return image.select([quantile]).rename('rh')
    
    # gedi_filtered = gedi_filtered.map(rename_property)
    
    # Then apply quality mask
    gedi_filtered = gedi_filtered.map(qualityMask)
    gedi_filtered = gedi_filtered.select(quantile).mosaic().rename("rh")
    
    # Get all valid points by using reduce(ee.Reducer.toCollection())
    # Specify the property names we want to keep
    # gedi_points = gedi_filtered.select([quantile, 'quality_flag', 'degrade_flag']).reduce(
    #     ee.Reducer.toCollection([quantile])
    #     # ee.Reducer.toCollection(['quality_flag', 'degrade_flag', quantile])
    # )
    
    # Rename the quantile band to 'rh'
    # gedi_points = gedi_points.rename(quantile, "rh")
    
    return gedi_filtered.select('rh') 