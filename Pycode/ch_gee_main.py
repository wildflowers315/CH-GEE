import ee
import geemap
import numpy as np
from typing import Union, List, Dict, Any

def canopy_height_mapper(
    aoi: ee.Geometry,
    year: int,
    start_date: str,
    end_date: str,
    start_date_gedi: str,
    end_date_gedi: str,
    clouds_th: float,
    quantile: str,
    model: str,
    mask: str,
    gedi_type: str,
    num_trees_rf: int,
    var_split_rf: int,
    min_leaf_pop_rf: int,
    bag_frac_rf: float,
    max_nodes_rf: int,
    num_trees_gbm: int,
    shr_gbm: float,
    sampling_rate_gbm: float,
    max_nodes_gbm: int,
    loss_gbm: str,
    max_nodes_cart: int,
    min_leaf_pop_cart: int
) -> ee.Image:
    """
    Main function for canopy height mapping using Earth Engine.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        year: Year for analysis
        start_date: Start date for satellite data
        end_date: End date for satellite data
        start_date_gedi: Start date for GEDI data
        end_date_gedi: End date for GEDI data
        clouds_th: Cloud threshold
        quantile: Quantile for GEDI data
        model: Model type ('RF', 'GBM', or 'CART')
        mask: Mask type ('DW', 'FNF', or 'none')
        gedi_type: GEDI data type ('singleGEDI' or 'meanGEDI')
        num_trees_rf: Number of trees for Random Forest
        var_split_rf: Variable split for Random Forest
        min_leaf_pop_rf: Minimum leaf population for Random Forest
        bag_frac_rf: Bagging fraction for Random Forest
        max_nodes_rf: Maximum nodes for Random Forest
        num_trees_gbm: Number of trees for GBM
        shr_gbm: Shrinkage for GBM
        sampling_rate_gbm: Sampling rate for GBM
        max_nodes_gbm: Maximum nodes for GBM
        loss_gbm: Loss function for GBM
        max_nodes_cart: Maximum nodes for CART
        min_leaf_pop_cart: Minimum leaf population for CART
    
    Returns:
        ee.Image: Classified canopy height map
    """
    # Calculate area
    polygon_area = aoi.area({'maxError': 1})
    polygon_area = polygon_area.divide(10000).round()
    
    # Import required modules
    from .l2a_gedi_source import get_gedi_data
    from .sentinel1_source import get_sentinel1_data
    from .sentinel2_source import get_sentinel2_data
    from .for_forest_masking import apply_forest_mask
    from .random_sampling import create_training_data
    
    # Get input data
    gedi_data = get_gedi_data(aoi, start_date_gedi, end_date_gedi, quantile)
    s1_data = get_sentinel1_data(aoi, year, start_date, end_date)
    s2_data = get_sentinel2_data(aoi, year, start_date, end_date, clouds_th)
    
    # Apply forest mask if needed
    if mask != 'none':
        gedi_data = apply_forest_mask(gedi_data, mask)
    
    # Create training data
    training_data = create_training_data(gedi_data, s1_data, s2_data)
    
    # Train model based on selected type
    if model == 'RF':
        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=num_trees_rf,
            variablesPerSplit=var_split_rf,
            minLeafPopulation=min_leaf_pop_rf,
            bagFraction=bag_frac_rf,
            maxNodes=max_nodes_rf
        )
    elif model == 'GBM':
        classifier = ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=num_trees_gbm,
            shrinkage=shr_gbm,
            samplingRate=sampling_rate_gbm,
            maxNodes=max_nodes_gbm,
            loss=loss_gbm
        )
    else:  # CART
        classifier = ee.Classifier.smileCart(
            maxNodes=max_nodes_cart,
            minLeafPopulation=min_leaf_pop_cart
        )
    
    # Train the classifier
    trained_classifier = classifier.train(
        features=training_data,
        classProperty='height',
        inputProperties=s1_data.bandNames().cat(s2_data.bandNames())
    )
    
    # Classify the image
    classified = s1_data.addBands(s2_data).classify(trained_classifier)
    
    # Scale the output
    classified = classified.multiply(50).round()
    
    return classified

def get_variable_importance(classifier: ee.Classifier) -> Dict[str, float]:
    """
    Get variable importance from the trained classifier.
    
    Args:
        classifier: Trained Earth Engine classifier
    
    Returns:
        Dict[str, float]: Dictionary of variable names and their importance scores
    """
    return classifier.explain().get('importance').getInfo() 