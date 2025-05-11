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
    min_leaf_pop_cart: int,
    training_buffer: int
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
        training_buffer: Training buffer for sampling

    Returns:
        ee.Image: Classified canopy height map
    """
    # Calculate area
    area_m2 = aoi.area(maxError=1)
    area_ha = area_m2.divide(10000).round().getInfo()
    print(f"Area of the AOI: {area_ha} hectares")
    
    # Import required modules
    from l2a_gedi_source import get_gedi_data
    from sentinel1_source import get_sentinel1_data
    from sentinel2_source import get_sentinel2_data
    from for_forest_masking import apply_forest_mask
    from new_random_sampling import create_training_data
    
    training_aoi = aoi.buffer(training_buffer)
    
    # Get GEDI data
    print("Collecting GEDI data...")
    gedi_data = get_gedi_data(training_aoi, start_date_gedi, end_date_gedi, quantile)
    
    # Check if GEDI data is empty
    gedi_size = gedi_data.size().getInfo()
    print(f"Number of GEDI points: {gedi_size}")
    if gedi_size == 0:
        raise ValueError("No GEDI data found in the area of interest")
    
    # Filter GEDI data to ensure it has required properties
    gedi_data = gedi_data.filter(ee.Filter.notNull(['rh']))
    gedi_size = gedi_data.size().getInfo()
    print(f"Number of GEDI points with valid height data: {gedi_size}")
    if gedi_size == 0:
        raise ValueError("No GEDI points with valid height data found")
    
    # Create forest mask if needed
    forest_mask = None
    if mask != 'none':
        try:
            forest_mask = apply_forest_mask(training_aoi, mask, training_aoi, year, start_date, end_date)
            print("Forest mask created successfully")
        except Exception as e:
            print(f"Warning: Could not create forest mask: {e}")
    
    # Get Sentinel data
    print("Collecting Sentinel data...")
    s1_data = get_sentinel1_data(training_aoi, year, start_date, end_date)
    s2_data = get_sentinel2_data(training_aoi, year, start_date, end_date, clouds_th, training_aoi)
    
    # Print available bands for debugging
    print("Sentinel-1 bands:", s1_data.bandNames().getInfo())
    print("Sentinel-2 bands:", s2_data.bandNames().getInfo())
    
    # Create training data using new sampling approach
    print("Creating training dataset...")
    training_data = create_training_data(
        gedi_data=gedi_data,
        s1_data=s1_data,
        s2_data=s2_data,
        aoi=aoi,
        mask=forest_mask
    )
    
    # Verify training data
    training_size = training_data.size().getInfo()
    print(f"Number of training features created: {training_size}")
    if training_size == 0:
        raise ValueError("No valid training features were created")
    
    # Split into training and validation
    split = 0.7
    training = training_data.filter(ee.Filter.lt('random', split))
    validation = training_data.filter(ee.Filter.gte('random', split))
    
    # Get predictor names
    predictors = s2_data.bandNames().cat(s1_data.bandNames())
    
    # Train model based on selected type
    if model == 'RF':
        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=num_trees_rf,
            variablesPerSplit=var_split_rf,
            minLeafPopulation=min_leaf_pop_rf,
            bagFraction=bag_frac_rf,
            maxNodes=max_nodes_rf
        ).setOutputMode('Regression')
    elif model == 'GBM':
        classifier = ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=num_trees_gbm,
            shrinkage=shr_gbm,
            samplingRate=sampling_rate_gbm,
            maxNodes=max_nodes_gbm,
            loss=loss_gbm
        ).setOutputMode('Regression')
    else:  # CART
        classifier = ee.Classifier.smileCart(
            maxNodes=max_nodes_cart,
            minLeafPopulation=min_leaf_pop_cart
        ).setOutputMode('Regression')
    
    # Train the classifier
    print("Training the model...")
    trained_classifier = classifier.train(
        features=training,
        classProperty='rh',
        inputProperties=predictors
    )
    
    # Classify the mapping area
    print("Classifying the mapping area...")
    classified = s1_data.addBands(s2_data).classify(trained_classifier)
    
    # Scale the output
    classified = classified.multiply(50).round().float()
    
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