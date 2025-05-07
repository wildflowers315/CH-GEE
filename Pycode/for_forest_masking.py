import ee
from typing import Union, List

def apply_forest_mask(
    data: ee.FeatureCollection,
    mask_type: str
) -> ee.FeatureCollection:
    """
    Apply forest mask to the data.
    
    Args:
        data: Input data as Earth Engine FeatureCollection
        mask_type: Type of mask to apply ('DW' for Dynamic World, 'FNF' for Forest/Non-Forest)
    
    Returns:
        ee.FeatureCollection: Masked data
    """
    if mask_type == 'DW':
        # Import Dynamic World dataset
        dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        
        # Get the most recent image
        dw_image = dw.sort('system:time_start', False).first()
        
        # Select forest class (class 1)
        forest_mask = dw_image.select('label').eq(1)
        
        # Apply mask to data
        masked_data = data.filterBounds(forest_mask)
        
    elif mask_type == 'FNF':
        # Import Hansen Global Forest Change dataset
        hansen = ee.Image('UMD/hansen/global_forest_change_2021_v1_9')
        
        # Get forest mask (tree cover > 30%)
        forest_mask = hansen.select('treecover2000').gt(30)
        
        # Apply mask to data
        masked_data = data.filterBounds(forest_mask)
        
    else:
        # No mask applied
        masked_data = data
    
    return masked_data 