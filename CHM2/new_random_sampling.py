import ee
from typing import Union, List, Dict, Any

def generate_sampling_sites(region: ee.Geometry, cell_size: int, seed: int, mask_raster: ee.Image) -> Dict[str, Any]:
    """
    Generate sampling sites using a grid-based approach with random points.
    
    Args:
        region: Area of interest
        cell_size: Size of sampling cells in meters
        seed: Random seed for reproducibility
        mask_raster: Forest mask raster
    
    Returns:
        Dict containing buffer geometry
    
    Raises:
        ValueError: If any input parameters are invalid
    """
    # Input validation
    if not isinstance(region, ee.Geometry):
        raise ValueError("region must be an ee.Geometry")
    if not isinstance(cell_size, (int, float)) or cell_size <= 0:
        raise ValueError("cell_size must be a positive number")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(mask_raster, ee.Image):
        raise ValueError("mask_raster must be an ee.Image")
    
    # Generate a random image of integers in the specified region and projection
    proj = ee.Projection('EPSG:4326').atScale(cell_size)
    cells = ee.Image.random(seed).multiply(1000000).int().clip(region).reproject(proj)
    
    # Create random values for finding local maximums
    random = ee.Image.random(seed).multiply(1000000).int()
    maximum = cells.addBands(random).reduceConnectedComponents(ee.Reducer.max())
    
    # Find points that are local maximums
    points = random.eq(maximum).selfMask().clip(region)
    
    # Create mask to remove pixels with even coordinates
    mask_img = ee.Image.pixelCoordinates(proj).expression(
        "!((b('x') + 0.5) % 2 != 0 || (b('y') + 0.5) % 2 != 0)"
    )
    
    # Apply masks
    strict_cells = cells.updateMask(mask_img).updateMask(
        mask_img.updateMask(mask_raster.eq(1))
    ).reproject(proj)
    
    # Find strict maximums
    strict_max = strict_cells.addBands(random).reduceConnectedComponents(ee.Reducer.max())
    strict_points = random.eq(strict_max).selfMask().clip(region)
    
    # Convert to vectors
    samples = strict_points.reduceToVectors(
        reducer=ee.Reducer.countEvery(),
        geometry=region,
        crs=proj.scale(1/16, 1/16),
        geometryType='centroid',
        maxPixels=1e9
    )
    
    # Add buffer around each point
    buffer = samples.map(lambda f: f.buffer(ee.Number(cell_size).divide(2)))
    
    return {'buffer': buffer}

def create_training_data(
    gedi_data: Union[ee.ImageCollection, ee.FeatureCollection],
    s1_data: ee.Image,
    s2_data: ee.Image,
    aoi: ee.Geometry,
    mask: ee.Image = None
) -> ee.FeatureCollection:
    """
    Create training data using grid-based sampling.
    
    Args:
        gedi_data: GEDI data points as ImageCollection or FeatureCollection
        s1_data: Sentinel-1 data
        s2_data: Sentinel-2 data
        aoi: Area of interest
        mask: Forest mask (optional)
    
    Returns:
        ee.FeatureCollection: Training data
    
    Raises:
        ValueError: If any input parameters are invalid
    """
    # Input validation
    if not isinstance(gedi_data, (ee.ImageCollection, ee.FeatureCollection)):
        raise ValueError("gedi_data must be an ee.ImageCollection or ee.FeatureCollection")
    if not isinstance(s1_data, ee.Image):
        raise ValueError("s1_data must be an ee.Image")
    if not isinstance(s2_data, ee.Image):
        raise ValueError("s2_data must be an ee.Image")
    if not isinstance(aoi, ee.Geometry):
        raise ValueError("aoi must be an ee.Geometry")
    if mask is not None and not isinstance(mask, ee.Image):
        raise ValueError("mask must be an ee.Image")
    
    # Calculate area to determine sampling strategy
    area_m2 = aoi.area(maxError=1)
    area_ha = area_m2.divide(10000).round().getInfo()
    print(f"Area of the AOI: {area_ha} hectares")
    
    # Determine cell size based on area
    if area_ha <= 4000:
        cell_size = 100
        scale = 10
    elif area_ha <= 1000000:
        cell_size = 4000
        scale = 25
    elif area_ha <= 2000000:
        cell_size = 6000
        scale = 50
    else:
        cell_size = 50000
        scale = 100
    
    # Generate sampling sites
    sampling_sites = generate_sampling_sites(aoi, cell_size, 1, mask if mask else ee.Image.constant(1))
    sampling_geometry = sampling_sites['buffer'].geometry()
    
    # Convert GEDI data to Image if it's a FeatureCollection
    if isinstance(gedi_data, ee.FeatureCollection):
        gedi_image = gedi_data.reduceToImage(
            properties=['rh'],
            reducer=ee.Reducer.first()
        )
    else:
        gedi_image = gedi_data.first()
    
    # Combine all data
    merged = s2_data.addBands(gedi_image).addBands(s1_data)
    
    # Sample based on area size
    if area_ha <= 4000:
        reference = merged.sample(
            region=aoi,
            scale=scale,
            dropNulls=True,
            numPixels=1e13,
            tileScale=4,
            seed=0,
            geometries=True
        )
    else:
        reference = merged.sample(
            region=sampling_geometry,
            scale=scale,
            dropNulls=True,
            numPixels=1e13,
            tileScale=16,
            seed=0,
            geometries=True
        )
    
    # Add random column for train/validation split
    reference = reference.randomColumn('random')
    
    return reference 