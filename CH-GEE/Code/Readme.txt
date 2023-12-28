//***********************************************************************************************
//************************** Canopy Height Mapper GEE web app   *********************************
*************************************************************************************************

Canopy Height Mapper is a Google Earth Engine (CH-GEE) app that predicts canopy heights by integrating 
GEDI (Global Ecosystem Dynamics Investigation) Rh metrics with Sentinel-1/2 and topographical variables.
This folder contains the JavaScripts used in the CH-GEE web app. Users can upload their desired area of interest (AOI) 
in shapefile format and download the canopy height map in GeoTIFF files using the "ForUploadDownload.js" script.
Afterwards, users can access and pre-process all canopy height footprints acquired by GEDI's NASA mission through 
the scripts in "L2A_GEDI_source.js". Following that, users can access, pre-process, and process Sentinel-1/2 using 
the JavaScripts in "Sentinel1_source.js" and "Sentinel2_source.js". It's worth noting that for larger extents of 4000 km²,
 a random sampling approach was implemented through the JavaScripts in "RandomSampling.js".
Moreover, to automatically visualize the map accuracy obtained using our locally calibrated models, users can use "ForPlots.js".
 Finally, users can execute all scripts through "CH-GEE_main.js," which can be visualized in the User Interface "CH-GEE_UI.js".
