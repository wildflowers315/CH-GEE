import ee

def get_canopyht_data(
    aoi: ee.Geometry,
    # year: int,
    # start_date: str,
    # end_date: str
) -> ee.Image:
    
    # https://gee-community-catalog.org/projects/meta_trees/#dataset-citation
    canopy_ht_2023 = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight") \
        .filterBounds(aoi).mosaic().select([0],['ht2023'])
    # treenotree = canopy_ht_2023.updateMask(canopy_ht_2023.gte(0))
    
    # https://gee-community-catalog.org/projects/canopy/#earth-engine-snippet
    canopy_ht_2022 = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1") \
        .select([0],['ht2022'])
    standard_deviation_2022 = ee.Image("users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1") \
        .select([0],['std2022'])
    
    # https://www.sciencedirect.com/science/article/pii/S0034425720305381
    umdheight = ee.ImageCollection("users/potapovpeter/GEDI_V27") \
        .filterBounds(aoi).mosaic().select([0],['ht2021'])
    
    # Merge the images
    canopy_ht = canopy_ht_2023.addBands(canopy_ht_2022) \
        .addBands(umdheight) 
        # .addBands(standard_deviation_2022) \

    return canopy_ht.clip(aoi)