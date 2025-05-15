"""
Module for retrieving and processing ALOS-2 PALSAR SAR data from Google Earth Engine.
"""

import ee
from typing import Union, List

def get_alos2_data(
    aoi: ee.Geometry,
    year: int,
    start_date: str = "01-01",
    end_date: str = "12-31",
    include_texture: bool = True,
    speckle_filter: bool = True,
) -> ee.Image:
    """
    Get ALOS-2 PALSAR data for the specified area and time period.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        year: Year for analysis
        start_date: Start date for ALOS-2 data (format: MM-DD, default: 01-01)
        end_date: End date for ALOS-2 data (format: MM-DD, default: 12-31)
        include_texture: Whether to include texture metrics (GLCM)
        speckle_filter: Whether to apply speckle filtering
    
    Returns:
        ee.Image: Processed ALOS-2 PALSAR data
    """
    # Format dates properly for Earth Engine
    start_date_ee = ee.Date(f'{year}-{start_date}')
    end_date_ee = ee.Date(f'{year}-{end_date}')
    
    # Import ALOS-2 PALSAR dataset
    # Using the ALOS/PALSAR/YEARLY collection (annual mosaics)
    # alos = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR')"JAXA/ALOS/PALSAR/YEARLY/SAR"
    alos = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR")
    alos.select(['HH', 'HV'])  # Select HH and HV bands
    
    # Filter by date and region
    alos_filtered = alos.filterDate(start_date_ee, end_date_ee) \
                       .filterBounds(aoi)
    
    # Apply pre-processing
    def preprocess_sar(img):
        # Extract date information
        date = ee.Date(img.get('system:time_start'))
        
        # Basic preprocessing - convert to natural units (from dB)
        # HH and HV bands are stored in dB, convert to natural values for processing
        hh = ee.Image(10.0).pow(img.select('HH').divide(10.0))
        hv = ee.Image(10.0).pow(img.select('HV').divide(10.0))
                
        # Ratio (HH/HV)
        ratio = hh.divide(hv).rename('ratio')
        
        # Normalized difference (HH-HV)/(HH+HV)
        ndri = hh.subtract(hv).divide(hh.add(hv)).rename('ndri')
        
        # Combine all bands
        processed = img
        
        processed = processed.addBands([hh, hv, ratio, ndri])
        
        return processed
    
    # Apply preprocessing to all images
    # alos_processed = alos_filtered.map(preprocess_sar)
    
    # Apply speckle filtering if requested
    if speckle_filter:
        def apply_speckle_filter(img):
            # Use Refined Lee filter
            kernel_size = 3  # Use 3x3 kernel
            hh_filtered = img.select('HH').focal_mean(kernel_size, 'square')
            hv_filtered = img.select('HV').focal_mean(kernel_size, 'square')
            
            # Replace original bands with filtered ones
            return img.addBands([
                hh_filtered.rename('HH_filtered'),
                hv_filtered.rename('HV_filtered')
            ], None, True)
        
        alos_processed = alos_processed.map(apply_speckle_filter)
    
    # Add texture metrics if requested
    if include_texture:
        def add_texture(img):
            # Define windows sizes for texture calculation
            windows = [3, 5]
            
            texture_bands = []
            
            for window in windows:
                # Calculate GLCM texture metrics for HH and HV bands
                glcm_hh = img.select('HH').glcmTexture(window)
                glcm_hv = img.select('HV').glcmTexture(window)
                
                # Rename bands to include window size
                glcm_hh = glcm_hh.rename([
                    f'HH_contrast_{window}', f'HH_dissimilarity_{window}', 
                    f'HH_homogeneity_{window}', f'HH_ASM_{window}', 
                    f'HH_energy_{window}', f'HH_max_{window}', 
                    f'HH_entropy_{window}', f'HH_correlation_{window}'
                ])
                
                glcm_hv = glcm_hv.rename([
                    f'HV_contrast_{window}', f'HV_dissimilarity_{window}', 
                    f'HV_homogeneity_{window}', f'HV_ASM_{window}', 
                    f'HV_energy_{window}', f'HV_max_{window}', 
                    f'HV_entropy_{window}', f'HV_correlation_{window}'
                ])
                
                texture_bands.extend([glcm_hh, glcm_hv])
            
            # Add texture bands to the image
            return img.addBands(texture_bands)
        
        alos_processed = alos_processed.map(add_texture)
    
    alos_processed = alos_filtered
    # Create median composite
    alos_median = alos_processed.median()
    
    # Clip to area of interest
    alos_median = alos_median.clip(aoi)
    
    # Convert HH and HV bands back to dB for final output
    # This makes it consistent with typical SAR data representation
    def convert_to_db(img):
        hh_db = ee.Image(10).multiply(img.select('HH').log10()).rename('HH_dB')
        hv_db = ee.Image(10).multiply(img.select('HV').log10()).rename('HV_dB')
        return img.addBands([hh_db, hv_db])
    
    # alos_median = convert_to_db(alos_median)
    
    return alos_median

def get_alos2_mosaic(
    aoi: ee.Geometry,
    years: List[int] = None,
    include_texture: bool = True,
    speckle_filter: bool = True,
) -> ee.Image:
    """
    Get multi-year ALOS-2 PALSAR mosaic for the specified area.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        years: List of years to include (default: most recent 3 years)
        include_texture: Whether to include texture metrics
        speckle_filter: Whether to apply speckle filtering
    
    Returns:
        ee.Image: Multi-year ALOS-2 PALSAR mosaic
    """
    # If years not specified, use most recent 3 years
    if years is None:
        # Get current date using JavaScript Date
        # ee.Date() without arguments gives the current date/time
        current_year = ee.Date().get('year').subtract(1)  # Last complete year
        years = [
            current_year.subtract(2).getInfo(), 
            current_year.subtract(1).getInfo(), 
            current_year.getInfo()
        ]
    
    # Get data for each year
    year_images = []
    for year in years:
        year_img = get_alos2_data(
            aoi=aoi,
            year=year,
            include_texture=include_texture,
            speckle_filter=speckle_filter
        )
        year_images.append(year_img)
    
    # Create a collection and reduce to median
    alos_collection = ee.ImageCollection.fromImages(year_images)
    alos_mosaic = alos_collection.median()
    
    return alos_mosaic.clip(aoi)

# Example usage:
# ee.Initialize()
# aoi = ee.Geometry.Rectangle([139.5, 35.6, 140.0, 36.0])
# alos_data = get_alos2_data(aoi, 2020)
# multi_year_mosaic = get_alos2_mosaic(aoi, [2018, 2019, 2020])