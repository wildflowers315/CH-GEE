import ee
from typing import Union, List, Tuple

def apply_forest_mask(
    data: ee.FeatureCollection,
    mask_type: str,
    aoi: ee.Geometry,
    year: int,
    start_date: str,
    end_date: str,
) -> ee.FeatureCollection:
    """
    Apply forest mask to the data.
    
    Args:
        data: Input data as Earth Engine FeatureCollection
        mask_type: Type of mask to apply ('DW' for Dynamic World, 'FNF' for Forest/Non-Forest)
        aoi: Area of interest as Earth Engine Geometry
        year: Year for analysis
        start_date: Start date for Sentinel-2 data
        end_date: End date for Sentinel-2 data
    
    Returns:
        ee.FeatureCollection: Masked data
        
    Raises:
        ValueError: If mask_type is not one of 'DW', 'FNF', or 'none'
        ee.ee_exception.EEException: If no data is available for the specified area and date range
    """
    if mask_type not in ['DW', 'FNF', 'none']:
        raise ValueError(f"Invalid mask_type: {mask_type}. Must be one of 'DW', 'FNF', or 'none'")
    
    # Format dates properly for Earth Engine
    start_date_ee = ee.Date(f'{year}-{start_date}')
    end_date_ee = ee.Date(f'{year}-{end_date}')
    
    if mask_type == 'DW':
        # Import Dynamic World dataset
        aoi_center = aoi.centroid(maxError=1)
        
        colFilter = ee.Filter.And(
            ee.Filter.bounds(aoi_center),
            ee.Filter.date(start_date_ee, end_date_ee))
        
        dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
            .filter(colFilter)
            # .filterBounds(aoi) \
            # .filterDate(start_date_ee, end_date_ee)
        
        # Check if we have any data
        count = dw.size().getInfo()
        if count == 0:
            raise ee.ee_exception.EEException("No Dynamic World data available for the specified area and date range")
        
        # Initiate all forest mask
        forest_mask = ee.Image(1).clip(aoi)
        # Get median image and select forest class (class 1)
        dw_median = dw.median().clip(aoi)
        # non 1 value (==0, or >=2 ) is non forest class
        non_forest_mask = dw_median.select('label').eq(0).Or(dw_median.select('label').gte(2))
        FNF = forest_mask.where(non_forest_mask, 0)
        # nodata_mask = dw_median.select('label').mask().Not()  # Get NoData pixels
        # nodata_mask = nodata_mask.unmask(1) 
        # nodata_mask = nodata_mask.where(nodata_mask, 1)
        # FNF = forest_mask.Or(nodata_mask)  # Combine forest and NoData pixels
        
    elif mask_type == 'FNF':
        # Import ALOS/PALSAR dataset
        fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF4") \
            .filterBounds(aoi) \
            .filterDate(start_date_ee, end_date_ee)
        
        # Check if we have any data
        count = fnf.size().getInfo()
        if count == 0:
            print("No Dence ALOS/PALSAR FNF data available for the specified area and date range")
            fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF") \
                .filterBounds(aoi) \
                .filterDate(start_date_ee, end_date_ee)
            count = fnf.size().getInfo()
            if count == 0:
                print("No ALOS/PALSAR data available for the specified area and date range")
                FNF = ee.Image(1).clip(aoi)
            else:
                print("ALOS/PALSAR data available for the specified area and date range")
                FNF = fnf_median.select('fnf').eq(1)
        else:
            print("ALOS/PALSAR data available for the specified area and date range")
            # Get median image and process forest mask
            fnf_median = fnf.median().clip(aoi)
            FNF = fnf_median.select('fnf').eq(1).Or(fnf_median.select('fnf').eq(2))
        
    else:  # mask_type == 'none'
        # No mask applied
        FNF = ee.Image(1).clip(aoi)
    
    # Filter features that intersect with the forest mask
    binary_forest_mask = FNF.gt(0.0)
    
    def update_forest_mask(feature_or_image):
        """Update forest mask for a feature or image."""
        if isinstance(feature_or_image, ee.Feature):
            # For GEDI feature, convert feature to data
            # quantile_list = ['rh100', 'rh98', 'rh95', 'rh90', 'rh75', 'rh50', 'rh25', 'rh10']
            # get quantile from feature
            
            # convert height to image, mask, then back to feature
            height = feature_or_image.get('rh')  # Use rh98 for GEDI data
            if isinstance(height, ee.Image):
                height = height.select('rh')
            height_image = ee.Image.constant(height).rename('rh')
            masked_image = height_image.updateMask(binary_forest_mask)
            # Get the masked height value
            masked_height = ee.Algorithms.If(
                masked_image.mask().reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=feature_or_image.geometry(),
                    scale=10
                ).get('rh'),
                masked_image.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=feature_or_image.geometry(),
                    scale=10
                ).get('rh'),
                0
            )
            return ee.Feature(feature_or_image.geometry(), {'rh': masked_height})
        elif isinstance(feature_or_image, ee.Geometry):
            # For GEDI geometry, create a constant image and apply mask
            height_image = ee.Image.constant(0).rename('rh')  # Default height
            masked_image = height_image.updateMask(binary_forest_mask)
            return masked_image
        else:
            # For S2 images, just apply the mask
            return feature_or_image.updateMask(binary_forest_mask)
    
    # Apply the mask based on data type
    if isinstance(data, ee.FeatureCollection):
        masked_data = data.map(update_forest_mask)
    elif isinstance(data, ee.ImageCollection):
        masked_data = data.map(update_forest_mask)
    elif isinstance(data, (ee.Image, ee.Geometry)):
        masked_data = update_forest_mask(data)
    else:
        print("not GEDI geometry nor S1, S2 image")
        raise ValueError(f"Invalid data type: {type(data)}")
    
    return masked_data 