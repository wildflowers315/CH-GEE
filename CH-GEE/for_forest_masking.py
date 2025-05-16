import ee
# import geopandas as gpd
import pandas as pd
from typing import Union, List, Tuple

def create_forest_mask(
    mask_type: str,
    aoi: ee.Geometry,
    start_date_ee: ee.Date,
    end_date_ee: ee.Date,
    ndvi_threshold: float = 0.2
) -> ee.Image:
    """
    Create a forest mask based on specified mask type.
    
    Args:
        mask_type: Type of mask to create ('DW', 'FNF', 'NDVI', 'ALL', 'none')
        aoi: Area of interest as Earth Engine Geometry
        start_date_ee: Start date as ee.Date
        end_date_ee: End date as ee.Date
        ndvi_threshold: NDVI threshold for forest classification (default: 0.2)
    
    Returns:
        ee.Image: Binary forest mask (1 for forest, 0 for non-forest)
    """
    # Initialize masks with default (all ones)
    dw_mask = ee.Image(1).clip(aoi)
    fnf_mask = ee.Image(1).clip(aoi)
    ndvi_mask = ee.Image(1).clip(aoi)
    
    # Create a buffered version of the AOI to ensure we get all relevant tiles
    buffered_aoi = aoi.buffer(10000)  # Buffer by 5km
    
    def clip_image(image):
        """Clip image to the AOI."""
        return image.clip(aoi)
    # Create Dynamic World mask if requested
    if mask_type in ['DW', 'ALL']:
        # Import Dynamic World dataset using buffered AOI
        dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
            .filterBounds(buffered_aoi) \
            .filterDate(start_date_ee, end_date_ee)
        # dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        #     .filterDate(start_date_ee, end_date_ee) \
        #     .map(clip_image)
        # Check if we have any data
        count = dw.size().getInfo()
        if count == 0:
            print("No Dynamic World data available for the specified area and date range")
        else:
            # Get median image and select forest class (class 1)
            dw_median = dw.median().clip(aoi)
            # non 1 value (==0, or >=2 ) is non forest class
            non_forest_mask = dw_median.select('label').eq(0).Or(dw_median.select('label').gte(2))
            dw_mask = ee.Image(1).clip(aoi).where(non_forest_mask, 0)
    
    # Create Forest/Non-Forest mask if requested
    if mask_type in ['FNF', 'ALL']:
        # This FNF data sets is only available until 2018
        # FNF4 is only available from 2018 to 2021
        # mannually assign FNF start and end date to 2018-01-01 and 2021-12-31
        fnf_start_date = ee.Date('2020-01-01')
        fnf_end_date = ee.Date('2020-12-31')
        fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF4") \
            .filterBounds(buffered_aoi) \
            .filterDate(fnf_start_date, fnf_end_date)
        
        # # Import ALOS/PALSAR dataset using buffered AOI
        # fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF4") \
        #     .filterBounds(buffered_aoi) \
        #     .filterDate(start_date_ee, end_date_ee)
        # fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF") \
        #     .filterDate(start_date_ee, end_date_ee) \
        #     .map(clip_image)
        
        # Check if we have any data
        count = fnf.size().getInfo()
        if count == 0:
            print("No Dense ALOS/PALSAR FNF4 data available for the specified area and date range")
            fnf_start_date = ee.Date('2017-01-01')
            fnf_end_date = ee.Date('2017-12-31')
            fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF") \
                .filterBounds(buffered_aoi) \
                .filterDate(start_date_ee, end_date_ee)
            count = fnf.size().getInfo()
            if count == 0:
                print("No ALOS/PALSAR FNF data available for the specified area and date range")
            else:
                print("ALOS/PALSAR FNF data available for the specified area and date range")
                fnf_median = fnf.median().clip(aoi)
                fnf_mask = fnf_median.select('fnf').eq(1)
        else:
            print("Dense ALOS/PALSAR FNF4 data available for the specified area and date range")
            # Get median image and process forest mask
            fnf_median = fnf.median().clip(aoi)
            fnf_mask = fnf_median.select('fnf').eq(1).Or(fnf_median.select('fnf').eq(2))
    
    # Create NDVI-based mask if requested
    if mask_type in ['NDVI', 'ALL']:
        # Import Sentinel-2 dataset using buffered AOI
        s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterBounds(buffered_aoi) \
            .filterDate(start_date_ee, end_date_ee)
        # Get cloud probability data
        S2_CLOUD_PROBABILITY = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterDate(start_date_ee, end_date_ee) \
            .filterBounds(aoi)
            
        s2 = ee.ImageCollection(s2) \
                    .map(lambda img: img.addBands(S2_CLOUD_PROBABILITY.filter(ee.Filter.equals('system:index', img.get('system:index'))).first()))
        def maskClouds(img):
            clouds = ee.Image(img).select('probability')
            # ee.Image(img.get('cloud_mask')).select('probability')
            isNotCloud = clouds.lt(70)
            return img.mask(isNotCloud)
        
        def maskEdges(s2_img):
            return s2_img.updateMask(
                s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask())
            )#.updateMask(mask_raster.eq(1))

        s2 = s2.map(maskEdges) 
        s2 = s2.map(maskClouds) 
        
        # Check if we have any data
        count = s2.size().getInfo()
        if count == 0:
            print("No Sentinel-2 data available for the specified area and date range for NDVI calculation")
        else:
            # Calculate NDVI for each image
            def add_ndvi(img):
                ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
                return img.addBands(ndvi)
            
            s2_with_ndvi = s2.map(add_ndvi)
            
            # Get median NDVI
            ndvi_median = s2_with_ndvi.select('NDVI').median().clip(aoi)
            
            # Create forest mask based on NDVI threshold
            ndvi_mask = ndvi_median.gte(ndvi_threshold)
    
    # Determine final mask based on mask_type
    if mask_type == 'DW':
        forest_mask = dw_mask
    elif mask_type == 'FNF':
        forest_mask = fnf_mask
    elif mask_type == 'NDVI':
        forest_mask = ndvi_mask
    elif mask_type == 'ALL':
        # Combine all masks (if ANY mask indicates forest, treat as forest)
        forest_mask = dw_mask.And(fnf_mask).And(ndvi_mask)  # Changed to AND for strictness
    else:  # mask_type == 'none'
        # No mask applied
        forest_mask = ee.Image(1).clip(aoi)
    
    return forest_mask

def apply_forest_mask(
        # data: Union[ee.FeatureCollection, ee.Image, ee.ImageCollection, ee.Geometry, pd.DataFrame, gpd.GeoDataFrame],
        data: Union[ee.FeatureCollection, ee.Image, ee.ImageCollection, ee.Geometry],
        mask_type: str,
        aoi: ee.Geometry,
        year: int,
        start_date: str,
        end_date: str,
        ndvi_threshold: float = 0.2,
        scale: int = 30,
    ) -> Union[ee.FeatureCollection, ee.Image, ee.ImageCollection]:
        """
        Apply forest mask to the data.
        
        Args:
            data: Input data (FeatureCollection, Image, ImageCollection, Geometry, DataFrame, or GeoDataFrame)
            mask_type: Type of mask to apply ('DW', 'FNF', 'NDVI', 'ALL', 'none')
            aoi: Area of interest as Earth Engine Geometry
            year: Year for analysis
            start_date: Start date for Sentinel-2 data (for Earth Engine data)
            end_date: End date for Sentinel-2 data (for Earth Engine data)
            ndvi_threshold: NDVI threshold for forest classification (default: 0.2)
        
        Returns:
            Masked data of the same type as input. For DataFrames/GeoDataFrames, returns a copy with
            non-forest areas having rh=0. For Earth Engine data, returns masked data with masked out non-forest areas.
            
        Raises:
            ValueError: If mask_type is not one of 'DW', 'FNF', 'NDVI', 'ALL', or 'none'
            ee.ee_exception.EEException: If no data is available for the specified area and date range
        """
        if mask_type not in ['DW', 'FNF', 'NDVI', 'ALL', 'none']:
            raise ValueError(f"Invalid mask_type: {mask_type}. Must be one of 'DW', 'FNF', 'NDVI', 'ALL', or 'none'")
        
        # Format dates properly for Earth Engine
        start_date_ee = ee.Date(f'{year}-{start_date}')
        end_date_ee = ee.Date(f'{year}-{end_date}')
        
        # Create forest mask
        forest_mask = create_forest_mask(mask_type, aoi, start_date_ee, end_date_ee, ndvi_threshold)
        
        # Filter features that intersect with the forest mask
        binary_forest_mask = forest_mask.gt(0.0)
        
        def update_forest_mask(feature_or_image):
            """Update forest mask for a feature or image using server-side operations."""
            element = ee.Element(feature_or_image)
            element_type = ee.Algorithms.ObjectType(element)
            
            def handle_feature():
                # Get the feature and preserve all properties
                feature = ee.Feature(element)
                props = feature.toDictionary()
                
                # Check if point is in forest
                is_forest = binary_forest_mask.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=feature.geometry(),
                    scale=scale
                ).get(binary_forest_mask.bandNames().get(0))
                
                # Update height based on forest mask
                height = ee.Algorithms.If(
                    ee.Algorithms.IsEqual(is_forest, 1),
                    feature.get('rh'),
                    0
                )
                
                # Keep the original properties but update rh
                return ee.Feature(feature.geometry(), props.set('rh', height))
            
            def handle_image():
                return ee.Image(element).updateMask(binary_forest_mask)
            
            return ee.Algorithms.If(
                ee.Algorithms.IsEqual(element_type, 'Feature'),
                handle_feature(),
                handle_image()
            )
        
        # Apply the mask based on data type
        if isinstance(data, ee.FeatureCollection):
            masked_data = data.map(update_forest_mask)
        elif isinstance(data, ee.ImageCollection):
            masked_data = data.map(update_forest_mask)
        elif isinstance(data, (ee.Image, ee.Geometry)):
            masked_data = update_forest_mask(data)
        else:
            # # Handle pandas DataFrame or GeoDataFrame
            # if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            #     if 'rh' not in data.columns:
            #         raise ValueError("DataFrame must contain 'rh' column")
                
            #     # Create a binary mask for forest (all True for 'none' mask type)
            #     if mask_type == 'none':
            #         forest_mask = pd.Series(True, index=data.index)
            #     else:
            #         # Apply masking criteria based on mask_type
            #         forest_mask = pd.Series(True, index=data.index)
                    
            #         if 'ndvi' in data.columns and mask_type in ['NDVI', 'ALL']:
            #             forest_mask &= data['ndvi'] >= ndvi_threshold
                    
            #         if 'dw_class' in data.columns and mask_type in ['DW', 'ALL']:
            #             forest_mask &= data['dw_class'] == 1
                    
            #         if 'fnf' in data.columns and mask_type in ['FNF', 'ALL']:
            #             forest_mask &= data['fnf'].isin([1, 2])  # Assuming 1,2 are forest classes
                
            #     # Apply mask and set non-forest heights to 0
            #     masked_data = data.copy()
            #     masked_data.loc[~forest_mask, 'rh'] = 0
            #     return masked_data
            # else:
            print(f"Unsupported data type: {type(data)}")
            raise ValueError(f"Invalid data type: {type(data)}. Must be one of ee.FeatureCollection, ee.Image, ee.ImageCollection, ee.Geometry, pd.DataFrame, or gpd.GeoDataFrame")
        
        return masked_data
