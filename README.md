# Canopy Height Mapper
## Logo for the Canopy Height Mapper Google Earth Engine (CH-GEE) web app.
![RES_LOGO_4](https://github.com/Cesarito2021/TH-GEE/assets/81155556/65470bbe-32ab-48be-bc24-159cd73ee3da)
## Interface
<img width="960" alt="FIG_1" src="https://github.com/Cesarito2021/TH-GEE/assets/81155556/699fdee1-fc6f-4637-872d-cb279a912be9">

## Overview
The Canopy Height Mapper is a Google Earth Engine 🌎⛰️🌳🌲web application (CH-GEE) combining Global Ecosystem Dynamics Investigation (GEDI) data with Sentinel 1/2 and topographical data. 
The GEDI mission can monitor nearest Earth's forests through widespread laser shots of ~25 m of diameters (between 51.6° N and >51.6° S). 

  - Web App: https://code.earthengine.google.com/
  - Github: (https://github.com/Cesarito2021/TH-GEE.git)

## Vision
The vision of the CH-GEE web app is to be the leading platform for accessing high-resolution Canopy Height maps of Earth's forests. We aim to empower individuals, organisations, and researchers worldwide with the tools and data they need to make informed decisions, protect forests, and address critical environmental challenges.

## Tutorial 
Note: if the area of interest is larger than 10,000 grid dimension and the GeoTIFF file exceeds 32 MB, please follow this code:

```javascript
// ***************** Run CH-GEE code ***********************
var aoi = yourAOI; // Define your area of interest

// Import the CH-GEE library
var library = require("users/calvites1990/CH-GEE:CH-GEE_main");

// Configure the CanopyHeightMapper function
var CH_GEE = library.CanopyHeightMapper(
  aoi,          // Area of Interest (AOI)
  2019,         // YYYY - Sentinel collection year
  "06-01",      // MM-DD of start Sentinel collection
  "08-31",      // MM-DD of end Sentinel collection
  "2019-01-01", // YYYY-MM-DD of start GEDI collection
  "2020-01-01", // YYYY-MM-DD of end GEDI collection
  70,           // Cloud coverage percentage ("70")
  "rh95",       // GEDI metric, e.g., "rh95"
  "RF",         // Model type: "RF" (Random Forest), "GBM" (Gradient Boosting Machine), or "CART" (Classification and Regression Trees)
  "FNF",        // Forest mask: "FNF" or "DW"
  "singleGEDI", // GEDI processing type: "SingleGEDI" or "meanGEDI"
  500,          // Number of trees in Random Forest model (NumTreesRF)
  5,            // Split ratio in Random Forest model (varSplitRF)
  1,            // Minimum leaf population in Random Forest model (minLeafPopuRF)
  0.5,          // Bagging fraction in Random Forest model (bagFracRF)
  5,            // Maximum number of nodes in Random Forest model (maxNodesRF)
  "null",       // Number of trees in GBM model (numTreesGBM)
  "null",       // Shrinkage rate in GBM model (shrGBM)
  "null",       // Learning rate in GBM model (samLingRateGBM)
  "null",       // Maximum number of nodes in GBM model (maxNodesGBM)
  "null",       // Loss function in GBM model (lossGBM)
  "null",       // Maximum number of nodes in CART model (maxNodesCART)
  "null"        // Minimum leaf population in CART model (minLeafPopCART)
);

// ***************** Zoom to CH-GEE map *****************    
Map.centerObject(aoi, 14);
// ******************************************************
```

## Developers
1. Alvites Cesar
2. Bazzato Erika
3. O'Sullivan Hannah
4. Francini Saverio

## Acknowledgment
This GEE application was initially conceived during the 2nd edition Google Earth Engine Summer School organized by the Laboratory of Forest Geomatics, University of florence (September).

# CH-GEE web application configuration
## Input/Output options
Area of Interest (AOI) definition: Users can define the AOI by either 1) drawing it manually or 2) uploading a polygon in format Shapefile. For projection details, users can refer to the official GEE guide.
## Choose Forest Mask 
Users can select one of the three options for forest masks available in the CH-GEE web app: 1.Forest mask available at "GOOGLE/DYNAMICWORLD/V1" and 2. Forest mask available at "JAXA/ALOS/PALSAR/YEARLY/FNF". In contrast, the CH-GEE will assume that the full AOI is covered by forest.
## Choose GEDI Canopy Heights 
Users can select between 1) single GEDI Relative Height (Rh; m) metric ranging from 1% to 100%, or 2) The average GEDI Rh metric among 75%, 90%, 95%, and 100%.
## Choose Machine Learning (ML) Algorithm
Users can pick from 1) Random Forests (RF), 2) Gradient Tree Boosting (GB), and 3) Classification and Regression Trees (CART). Hyperparameters for RF include the number of decision trees (numberOfTrees), variables per split (variablesPerSplit), minimum training samples in each leaf node (minLeafPopulation), input fraction for bagging per tree (bagFraction), and maximum leaf nodes per tree (maxNodes). For GB, parameters encompass the number of decision trees (numberOfTrees), learning rate (shrinkage), sampling rate for stochastic tree boosting (samplingRate), maximum leaf nodes per tree (maxNodes), and loss function for regression (loss). CART parameters include maximum leaf nodes per tree (maxNodes) and minimum training samples in each leaf node (minLeafPopulation).
## Set period for Sentinel 1/2 collection 
Specify the start and end dates (Year/Month/Day) for the Sentinel 1/2 time series used for Canopy Height mapping.
## Set Cloud Coverage Percentage for Sentinel-2
Users can set the maximum acceptable percentage of cloud cover in Sentinel-2 images using a scoring function.
## Download Canopy Height Map 
Configure Google Drive folder settings and specify the image name (.GeoTIFF) for saving tree height map images.
## Run CH-GEE
Compute the CH-GEE web app with the selected parameters for the defined study area. Users can then automatically generate a Canopy Height map along with corresponding scatter plots and variable importance graphs.
## Run CH-GEE
Clear previously set parameters and study area configurations.







