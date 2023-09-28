# Tree-height-Mapper
![FigureAs](https://github.com/Cesarito2021/TH-GEE/assets/81155556/6004f5b0-63ce-4338-96c9-c128eee32519)

## Overview
The Tree Height Mapper is a Google Earth Engine 🌎⛰️🌳🌲web application (TH-GEE) combining Global Ecosystem Dynamics Investigation (GEDI) data with Sentinel 1 and Sentinel 2 atopographical data. 
The GEDI mission can monitor nearest Earth's forests through widespread laser shots of ~25 m of diameters (between 51.6° N and >51.6° S). 

  - Web App: https://code.earthengine.google.com/
  - Github: https://github.com/Cesarito2021/Global-Tree-height-Mapper

## Vision
The vision of the TH-GEE web app is to be the leading platform for accessing high-resolution tree height maps of Earth's forests. We aim to empower individuals, organisations, and researchers worldwide with the tools and data they need to make informed decisions, protect forests, and address critical environmental challenges.

## Developers
1. Alvites Cesar
2. Bazzato Erika
3. O'Sullivan Hannah
4. Francini Saverio

## Acknowledgment
This GEE application was initially conceived during the 2nd edition Google Earth Engine Summer School organized by the Laboratory of Forest Geomatics, University of florence (September).

# TH-GEE web application configuration
## Input/Output options
Study Area Definition: Users can define the study area by either 1) drawing it manually or 2) uploading a polygon or points shapefile. For projection details, users can refer to the official GEE guide.
## Choose Forest Mask (Optional)
Users can select one of the two forest masks available in the TH-GEE web app: 1.Forest mask available at "GOOGLE/DYNAMICWORLD/V1" and 2. Forest mask available at "JAXA/ALOS/PALSAR/YEARLY/FNF". In contrast, the TH-GEE will assume that the full study area is covered by forest.
## Choose GEDI Canopy Heights 
Users can select eithe 1) a GEDI metric ranging from 1% to 100%, or 2) the average value among 75%, 90%, 95%, and 100%.
## Choose Machine Learning (ML) Algorithm
Users can pick from 1) Random Forests (RF), 2) Gradient Tree Boosting (GB), and 3) Classification and Regression Trees (CART). Hyperparameters for RF include the number of decision trees (numberOfTrees), variables per split (variablesPerSplit), minimum training samples in each leaf node (minLeafPopulation), input fraction for bagging per tree (bagFraction), and maximum leaf nodes per tree (maxNodes). For GB, parameters encompass the number of decision trees (numberOfTrees), learning rate (shrinkage), sampling rate for stochastic tree boosting (samplingRate), maximum leaf nodes per tree (maxNodes), and loss function for regression (loss). CART parameters include maximum leaf nodes per tree (maxNodes) and minimum training samples in each leaf node (minLeafPopulation).
## Set Start and End Date for Sentinel-2 Time Window 
Specify the start and end dates for the Sentinel-2 image time series used for tree height mapping.
## Set Cloud Coverage Percentage for Sentinel-2
Users can set the maximum acceptable percentage of cloud cover in Sentinel-2 images using a scoring function.
## Download Tree Height Map Images
Configure Google Drive folder settings and specify the image name (.tiff) for saving tree height map images.
## Download Tree Height Measurements
Users can download tree height measurements for specific points. To export these results, users must upload the study area in the format of point.shp.
## Run TH-GEE
Compute the TH-GEE web app with the selected parameters for the defined study area. Users can then automatically generate a tree height map along with corresponding scatter plots and variable importance graphs.
## Run TH-GEE
Clear previously set parameters and study area configurations.







