import ee
from typing import Union, List

def create_training_data(
    gedi_data: ee.FeatureCollection,
    s1_data: ee.Image,
    s2_data: ee.Image
) -> ee.FeatureCollection:
    """
    Create training data by sampling GEDI points and extracting features from Sentinel data.
    
    Args:
        gedi_data: GEDI data points as Earth Engine FeatureCollection
        s1_data: Sentinel-1 data as Earth Engine Image
        s2_data: Sentinel-2 data as Earth Engine Image
    
    Returns:
        ee.FeatureCollection: Training data with features and labels
    """
    # Combine Sentinel data
    sentinel_data = s1_data.addBands(s2_data)
    
    # Sample points from GEDI data
    def sample_point(feature):
        # Get the point geometry
        point = feature.geometry()
        
        # Sample Sentinel data at the point
        sentinel_sample = sentinel_data.sample(
            region=point,
            scale=30,
            numPixels=1,
            dropNulls=True
        )
        
        # Get the first sample
        sentinel_sample = sentinel_sample.first()
        
        # Add GEDI height as label
        height = feature.get('rh100')
        
        # Combine features
        return sentinel_sample.set('height', height)
    
    # Apply sampling to all GEDI points
    training_data = gedi_data.map(sample_point)
    
    # Filter out null values
    training_data = training_data.filter(ee.Filter.notNull(['height']))
    
    return training_data 