# ***************************************************************************************************************
# ********************************************* Canopy Height Mapper  *******************************************
# ***************************************************************************************************************
import ee
import geemap
import numpy as np
import json
import os
from typing import Union, List, Dict, Any

from l2a_gedi_source import get_gedi_data
from sentinel1_source import get_sentinel1_data
from sentinel2_source import get_sentinel2_data
from for_forest_masking import apply_forest_mask
from new_random_sampling import create_training_data, generate_sampling_sites

def load_gedi_from_geojson(geojson_path: str) -> ee.FeatureCollection:
    """
    Load GEDI data from a GeoJSON file.
    
    Args:
        geojson_path: Path to the GeoJSON file containing GEDI data
        
    Returns:
        ee.FeatureCollection: GEDI data points as a FeatureCollection
    """
    # Check if file exists
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GEDI GeoJSON file not found at: {geojson_path}")
    
    # Read the GeoJSON file
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    # Remove geodesic property from each feature's geometry
    for feature in geojson_data['features']:
        if 'geometry' in feature and 'geodesic' in feature['geometry']:
            del feature['geometry']['geodesic']
    
    # Create Earth Engine FeatureCollection
    gedi_fc = ee.FeatureCollection(geojson_data)
    
    # Verify the data has required properties
    first_feature = gedi_fc.first()
    properties = first_feature.getInfo()['properties']
    if 'rh' not in properties:
        raise ValueError("GeoJSON file does not contain 'rh' property")
    
    return gedi_fc

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
    training_buffer: int,
    gedi_geojson_path: str = None  # New parameter for GeoJSON path
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
        gedi_geojson_path: Path to the GeoJSON file containing GEDI data (optional)

    Returns:
        ee.Image: Classified canopy height map
    """

    # aoi, year, start_date, end_date, start_date_gedi, end_date_gedi, clouds_th, quantile, 
    # model, mask, gedi_type, num_trees_rf, var_split_rf, min_leaf_popu_rf, bag_frac_rf, 
    # max_nodes_rf, num_trees_gbm, shr_gbm, sam_ling_rate_gbm, max_nodes_gbm, loss_gbm, 
    # max_nodes_cart, min_leaf_pop_cart
    # ):
    
    # Import required modules
    from l2a_gedi_source import get_gedi_data
    from sentinel1_source import get_sentinel1_data
    from sentinel2_source import get_sentinel2_data
    from for_forest_masking import apply_forest_mask
    from new_random_sampling import create_training_data, generate_sampling_sites
    
    # ***************************************************************************************************************
    #  Input Data
    # ***************************************************************************************************************
    
    start_date_with_year = f"{year}-{start_date}"
    end_date_with_year = f"{year}-{end_date}"
    
    # ***************************************************************************************************************
    #  Importing Area of Insterest (AOI) through drawing or uploading
    # ***************************************************************************************************************
    aoi2 = ee.Geometry(ee.FeatureCollection(aoi).geometry())
    area_m2 = aoi2.area(maxError=1)
    radius = area_m2.sqrt().divide(4).getInfo()
    print(f"Radius: {radius} m")
    polygon_area = area_m2.divide(10000).round().getInfo()
    print(f"Original Polygon area: {polygon_area} ha")

    training_buffer = max(training_buffer, -radius)
    print(f"Training buffer: {training_buffer} m")
    training_aoi = aoi2.buffer(training_buffer)
    training_aoi_ha = training_aoi.area(maxError=1).divide(10000).round().getInfo()
    print(f"Training AOI area: {training_aoi_ha} ha")
    polygon_area = training_aoi_ha
    # ***************************************************************************************************************
    #  Adjusting the Visualization Settings for the AOI
    # ***************************************************************************************************************
    Map = geemap.Map()
    if polygon_area < 2000:
        scale = 25
        Map.centerObject(aoi, 14)
    elif polygon_area >= 2000 and polygon_area < 10000:
        scale = 50
        Map.centerObject(aoi, 12)
    elif polygon_area >= 10000 and polygon_area < 20000:
        scale = 100
        Map.centerObject(aoi, 10)
    elif polygon_area >= 10000 and polygon_area < 330000:
        scale = 100
        Map.centerObject(aoi, 8)
    elif polygon_area >= 330000 and polygon_area < 2200000:
        scale = 200
        Map.centerObject(aoi, 8)
    elif polygon_area >= 2200000 and polygon_area < 10000000:
        scale = 250
        Map.centerObject(aoi, 8)
    else:
        scale = 250
        Map.centerObject(aoi, 6)
    
    # ***************************************************************************************************************
    #  Selecting one of the three Forest Masks within the AOI
    # ***************************************************************************************************************
    
    fnf = None
    if mask != 'none':
        try:
            fnf = apply_forest_mask(training_aoi, mask, training_aoi, year, start_date, end_date)
            print("Forest mask created successfully")
        except Exception as e:
            print(f"Warning: Could not create forest mask: {e}")
    
    # ***************************************************************************************************************
    #  GEDI Relative Height (RH) metrics
    # ***************************************************************************************************************    
    print("Loading GEDI data...")
    if gedi_geojson_path:
        # Load GEDI data from GeoJSON
        gedi_points = load_gedi_from_geojson(gedi_geojson_path)
        print(f"Loaded GEDI data from {gedi_geojson_path}")
    else:
        # Original GEDI data collection code
        gedi = get_gedi_data(training_aoi, start_date_gedi, end_date_gedi, quantile)
        gedi_points = gedi.sample(
            region=training_aoi,
            scale=scale,  # GEDI resolution
            geometries=True,
            dropNulls=True,
            seed=42
        )
        print("GEDI data loaded from Earth Engine")
    
    gedi_points_size = gedi_points.size().getInfo()
    print(f"Number of GEDI points: {gedi_points_size}")
    
    # Filter GEDI data to ensure it has required properties
    # gedi = gedi.filter(ee.Filter.notNull(['rh']))
    # gedi_size = gedi.size().getInfo()
    # print(f"Number of GEDI points with valid height data: {gedi_size}")
    # if gedi_size == 0:
    #     raise ValueError("No GEDI points with valid height data found")
    
    # print(f"GEDI bands: {gedi.bandNames().getInfo()}")
    # gedi_geom = gedi.first().geometry()
    # gedi_geom = gedi.geometry()
    # ***************************************************************************************************************
    #  Selecting Independent variables
    # ***************************************************************************************************************
    # Get Sentinel data
    print("Collecting Sentinel data...")
    s1 = get_sentinel1_data(training_aoi, year, start_date, end_date)
    s2 = get_sentinel2_data(training_aoi, year, start_date, end_date, clouds_th)
    
    # Print available bands for debugging
    print("Sentinel-1 bands:", s1.bandNames().getInfo())
    print("Sentinel-2 bands:", s2.bandNames().getInfo())
        
    def reproject_feature(feature):
        # Get the original geometry
        original_geom = feature.geometry()
        # Transform the geometry to the target projection
        reprojected_geom = original_geom.transform(s2_projection)
        # Return the feature with the new geometry
        return feature.setGeometry(reprojected_geom)

    # Get Sentinel-2 projection
    s2_projection = s2.projection()
    # Reproject GEDI points to Sentinel-2 projection
    gedi_points = gedi_points.map(reproject_feature)
    # Check if the projection is correct
    gedi_projection = gedi_points.first().geometry().projection()
    print("S2 projection:", s2_projection.getInfo())
    print("GEDI points projection:", gedi_projection.getInfo())
    # After reprojection, check if points remain in the analysis area
    gedi_points = gedi_points.filter(ee.Filter.bounds(training_aoi))
    gedi_points_size_after = gedi_points.size().getInfo()
    print(f"GEDI points after reprojection and area filter: {gedi_points_size_after}")
    if gedi_points_size_after == 0:
        raise ValueError("All GEDI points were outside the buffered area. Try a larger buffer value.")
    
    # Apply forest mask to GEDI points if available
    # if fnf is not None:
    #     # Sample the forest mask at GEDI point locations
    #     forest_mask_points = fnf.sampleRegions(
    #         collection=gedi_points,
    #         scale=scale,
    #         projection=s2_projection,
    #         tileScale=1
    #     )
        
    #     # Filter GEDI points to keep only those in forest areas
    #     gedi_points = gedi_points.filter(
    #         ee.Filter.equals('coordinates', forest_mask_points.filter(ee.Filter.eq('forest', 1)).get('coordinates'))
    #     )
        
    #     print(f"Number of GEDI points after forest mask: {gedi_points.size().getInfo()}")
    
    # ***************************************************************************************************************
    # Global Multi-resolution Terrain Elevation Data 2010
    # ***************************************************************************************************************
    dem = ee.Image("USGS/GMTED2010")
    mask_terrain = dem.gt(0)
    dem_masked = dem.mask(mask_terrain).rename('dem')
    slope = ee.Terrain.slope(dem_masked)
    aspect = ee.Terrain.aspect(dem_masked)
    
    dem_band_clip = dem_masked.addBands(slope).addBands(aspect)
    dem_band_clip = dem_band_clip.clip(training_aoi)
    print("Bands of dem_band_clip: ", dem_band_clip.bandNames().getInfo())
    
    # ***************************************************************************************************************
    #  Creating a dataset using dependent and independent variables
    # ***************************************************************************************************************
    # ReProjection to Sentinel2 projection before merging
    dem_band_clip = dem_band_clip.reproject(s2_projection)
    s1 = s1.reproject(s2_projection)
    # gedi = gedi.reproject(s2_projection)
    
    # merged = s2.addBands(gedi).addBands(dem_band_clip).addBands(s1)
    merged = s2.addBands(s1).addBands(dem_band_clip)
    #check if the projection is correct
    merged_projection = merged.projection()
    print("Merged projection:", merged_projection.getInfo())
    
    # make gedi image from feature collection
    # gedi_image = gedi_points.mosaic() #.getInfo()
    # # add merged data to gedi points
    # merged = gedi_image.addBands(merged)

    # select image form merged image where matched gedi points geometry
    merged_points = merged.sampleRegions(
        collection=gedi_points,
        scale=scale,
        projection=s2_projection,
        tileScale=1,
        geometries=True,
    )
    merged_points_size = merged_points.size().getInfo()
    print(f"Number of merged points: {merged_points_size}")
    
    reference = gedi_points.map(lambda feature: feature.set(
        ee.Dictionary(merged_points.filter(ee.Filter.equals('coordinates', feature.get('coordinates'))).first().toDictionary())
    ))
    
    # Function to match points based on coordinates with tolerance
    def matchPoints(feature):
        # Get the coordinates of the GEDI point
        gedi_coords = feature.geometry().coordinates()
        
        # Find matching point in merged_points
        def findMatch(merged_feature):
            merged_coords = merged_feature.geometry().coordinates()
            # Check if coordinates are within tolerance (approximately 1 meter)
            lon_match = ee.Number(gedi_coords.get(0)).subtract(merged_coords.get(0)).abs().lt(0.00001)
            lat_match = ee.Number(gedi_coords.get(1)).subtract(merged_coords.get(1)).abs().lt(0.00001)
            return lon_match.And(lat_match)
        
        # Filter merged_points to find matching point
        matching_point = merged_points.filter(ee.Filter.and_(
            ee.Filter.equals('system:index', feature.get('system:index')),
            ee.Filter.equals('geometry', feature.geometry())
        )).first()
        
        # If no match found, try coordinate-based matching
        matching_point = ee.Algorithms.If(
            matching_point,
            matching_point,
            merged_points.filter(ee.Filter.function(findMatch)).first()
        )
        
        # Get properties from matching point
        properties = ee.Dictionary(matching_point.toDictionary())
        
        # Add original GEDI properties
        return feature.set(properties)
    
    # Apply the matching function to each GEDI point
    # reference = gedi_points.map(matchPoints)
    
    # print("Bands of merged ee.Image: ", merged.bandNames().getInfo())
    # assert merged.size().getInfo() > 0, "Merged ee.Image is empty"
    # print("Merged ee.Image: ", merged.getInfo())
    
    # ***************************************************************************************************************
    #  Application of random sampling in large areas
    # ***************************************************************************************************************
    if scale == 100:
        cell_size = 4000
    else:
        if polygon_area < 5000:
            cell_size = 100
        elif polygon_area >= 5000 and polygon_area < 1000000:
            cell_size = 4000  # 4000 # Riducendo questo valore aumenti l'accuratezza
        elif polygon_area >= 1000000 and polygon_area < 2000000:
            cell_size = 6000  # 6000
        elif polygon_area >= 2000000 and polygon_area < 3000000:
            cell_size = 6000
        else:
            cell_size = 50000
    
    
    # library_rs = ee.Algorithms.require("users/calvites1990/CH-GEE:RandomSampling")
    # generated_points = library_rs.generateSamplingSites(aoi2, cell_size, 1, fnf)
    # aoi_buffer = generated_points.buffer
    # aoi_prova = aoi_buffer.geometry()
    generated_points = generate_sampling_sites(aoi2, cell_size, 1, fnf)
    aoi_buffer = generated_points['buffer']
    # aoi_buffer = generated_points.buffer
    aoi_prova = aoi_buffer.geometry()
    
    # ***************************************************************************************************************
    #  Sampling configuration for small and large area (4000 km2 is used as the threshold)
    # ***************************************************************************************************************
    # gedi_geom = gedi.geometry()
    # if polygon_area <= 4000:
    #     reference = merged.sample(
    #         region=gedi_geom,
    #         scale=scale,
    #         projection=merged.projection(),
    #         numPixels=1e13,
    #         seed=0,            
    #         dropNulls=True,
    #         geometries=True
    #     )
    # else:
    #     reference = merged.sample(
    #         region=aoi_prova,
    #         scale=scale,
    #         projection=merged.projection(),
    #         numPixels=1e13,
    #         tileScale=16,
    #         seed=0,
    #         dropNulls=True,
    #         geometries=True
    #         )
    
    # Apply the matching function to each GEDI point
    # reference = gedi_points.map(matchPoints)
    # Verify training data
    reference_size = reference.size().getInfo()
    print(f"Number of reference features created: {reference_size}")
    if reference_size == 0:
        raise ValueError("No valid reference features were created")
    
    # ***************************************************************************************************************
    #  Splitting the dataset into training and validation sets 
    # ***************************************************************************************************************
    reference = reference.randomColumn('random')
    split = 0.7
    training = reference.filter(ee.Filter.lt('random', split))
    validation = reference.filter(ee.Filter.gte('random', split))
  
    # ***************************************************************************************************************
    # Colnames all of used variables
    # ***************************************************************************************************************
    predictors_names = s2 \
        .addBands(dem_masked).addBands(slope).addBands(aspect) \
        .addBands(s1).bandNames()
    
    # ***************************************************************************************************************
    # Configurating the hyperparameters in each of the three machine learning algorithms:
    # RF   - Random Forest 
    # CART - Classification And Regression Trees classifier
    # GB   - Gradient Tree Boost 
    # ***************************************************************************************************************
    # Get band names as a client-side list first
    predictor_names_list = predictors_names.getInfo()
    var_split_rf = ee.Number(np.sqrt(len(predictor_names_list)).round())
    # var_split_rf = ee.Number(predictors_names.size()).sqrt().round()
    if model == "RF":
        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=ee.Number(num_trees_rf),
            variablesPerSplit=var_split_rf,
            minLeafPopulation=ee.Number(min_leaf_pop_rf),
            bagFraction=ee.Number(bag_frac_rf),
            maxNodes=ee.Number(max_nodes_rf)
        ).setOutputMode("Regression") \
          .train(training, "rh", predictors_names)
    
    # ***************************************************************************************************************
    if model == "CART":
        classifier = ee.Classifier.smileCart(
            maxNodes=ee.Number(max_nodes_cart),
            minLeafPopulation=ee.Number(min_leaf_pop_cart)
        ).train(training, "rh", predictors_names) \
          .setOutputMode("Regression")
    # ***************************************************************************************************************
    if model == "GBM":
        classifier = ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=ee.Number(num_trees_gbm),
            shrinkage=ee.Number(shr_gbm),
            samplingRate=ee.Number(sampling_rate_gbm),
            maxNodes=ee.Number(max_nodes_gbm)
        ).train(training, "rh", predictors_names) \
          .setOutputMode("Regression")
    
    # ***************************************************************************************************************
    # Model Validation
    # ***************************************************************************************************************
    print("Validating model...")
    # Apply the trained classifier to the validation data
    validated = validation.classify(classifier)

    # Calculate regression metrics
    def calculate_metrics(observed_prediction_fc):
        # Get the predictions and actual values
        predictions = observed_prediction_fc.aggregate_array('classification')
        actuals = observed_prediction_fc.aggregate_array('rh')
        
        # Combine into a single array for calculations
        combined = predictions.zip(actuals)
        
        # Calculate difference and squared difference
        # Fix: Access list elements by index, not with 'get'
        diff = combined.map(lambda pair: ee.Number(ee.List(pair).get(0)).subtract(ee.Number(ee.List(pair).get(1))))
        sq_diff = diff.map(lambda d: ee.Number(d).multiply(ee.Number(d)))
        abs_diff = diff.map(lambda d: ee.Number(d).abs())
        
        # Calculate metrics
        n = combined.size()
        rmse = ee.Number(sq_diff.reduce(ee.Reducer.mean())).sqrt()
        mae = ee.Number(abs_diff.reduce(ee.Reducer.mean()))
        
        # Calculate R²
        mean_actual = actuals.reduce(ee.Reducer.mean())
        total_sum_sq = actuals.map(lambda a: ee.Number(a).subtract(mean_actual).pow(2)).reduce(ee.Reducer.sum())
        residual_sum_sq = sq_diff.reduce(ee.Reducer.sum())

        # Fix: Ensure both values are explicit ee.Number objects
        total_sum_sq = ee.Number(total_sum_sq)
        residual_sum_sq = ee.Number(residual_sum_sq)
        r_squared = ee.Number(1).subtract(residual_sum_sq.divide(total_sum_sq))
        
        # For classification metrics (if needed)
        # We can create binary classes based on height thresholds
        # For example, classify as "tall" if height > 10m
        # threshold = 10
        # actual_class = actuals.map(lambda a: ee.Number(a).gt(threshold))
        # pred_class = predictions.map(lambda p: ee.Number(p).gt(threshold))
        
        # # Calculate confusion matrix components
        # combined_class = actual_class.zip(pred_class)
        # tp = combined_class.map(lambda pair: ee.Number(ee.List(pair).get(0)).And(ee.Number(ee.List(pair).get(1)))).reduce(ee.Reducer.sum())
        # tn = combined_class.map(lambda pair: ee.Number(ee.List(pair).get(0)).Not().And(ee.Number(ee.List(pair).get(1)).Not())).reduce(ee.Reducer.sum())
        # fp = combined_class.map(lambda pair: ee.Number(ee.List(pair).get(0)).Not().And(ee.Number(ee.List(pair).get(1)))).reduce(ee.Reducer.sum())
        # fn = combined_class.map(lambda pair: ee.Number(ee.List(pair).get(0)).And(ee.Number(ee.List(pair).get(1)).Not())).reduce(ee.Reducer.sum())        
        # # Calculate accuracy and F1 score
        # accuracy = tp.add(tn).divide(n)
        # precision = tp.divide(tp.add(fp))
        # recall = tp.divide(tp.add(fn))
        # f1_score = precision.multiply(recall).multiply(2).divide(precision.add(recall))
        
        # Return all metrics as a dictionary
        return {
            'RMSE': rmse.getInfo(),
            'MAE': mae.getInfo(),
            'R_squared': r_squared.getInfo(),
            'Sample_Size': n.getInfo()
        }
    
    try:
        # Calculate and print validation metrics
        validation_metrics = calculate_metrics(validated)
        print("Validation Metrics:")
        for metric, value in validation_metrics.items():
            print(f"  {metric}: {value}")
    except Exception as e:
        print(f"Error calculating validation metrics: {e}")
    
    
    # ***************************************************************************************************************
    # Prediction of canopy heights 
    # ***************************************************************************************************************
    classified = merged.classify(classifier)
  
    # ***************************************************************************************************************
    # Scatter Plot 
    # ***************************************************************************************************************
    # library4 = ee.Algorithms.require("users/calvites1990/CH-GEE:ForPlots")
    # validated = validation.classify(classifier)
    # RMSE = library4.CalculationRMSE(validated)
    # library4.SCPLOT(validated, RMSE)
    
    # from new_for_plots import ForPlots
    # validated = validation.classify(classifier)
    # RMSE = library4.CalculationRMSE(validated)
    # library4.SCPLOT(validated, RMSE)
    
    # ***************************************************************************************************************
    # Variable Importance Plot
    # ***************************************************************************************************************
    # library4.VARIMP(classifier)  # variable important function
    
    # ***************************************************************************************************************
    # Canopy height map
    # ***************************************************************************************************************
    # library4.scalecolor(0, 50, classified)
    
    return [classified]