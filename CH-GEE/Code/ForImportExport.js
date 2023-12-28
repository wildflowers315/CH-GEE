//************************************************************************************************************
//************************************** Importing Area of Intereset in Shapefile format *********************
//************************************************************************************************************
var InputShapefile = function(shapefile,type){
  if(type == "Point"){
    var outputPTS    = ee.Geometry(shapefile.geometry().convexHull());
    var outputPTS100 = outputPTS.buffer({'distance': 100});
    return outputPTS100}
  if(type == "Boundary"){
    var outputBOUN    = ee.Geometry(shapefile.geometry().convexHull());
    return outputBOUN}  
  }
 exports.InputShapefile = InputShapefile;
 
//************************************************************************************************************
//************************************** Downloading map  in GeoTIFF format **********************************
//************************************************************************************************************

var DownloadImg = function(dataset,description,folder,region){
          //var CHMap  = ee.Image(ee.List(dataset).get(4));
          var CHMap  = ee.Image(ee.List(dataset).get(0));
          var regionInfo  = CHMap.getInfo();
          var CrsInfo     = regionInfo.bands[0].crs;
          Export.image.toDrive({
          image: CHMap, description: description, folder: folder,
          region: region, scale: 10,maxPixels: 1e13, crs: CrsInfo})};
exports.DownloadImg = DownloadImg;             

//************************************************************************************************************
//************************************** Downloading map  in GeoTIFF format **********************************
//************************************************************************************************************

/*
var DownloadCSV = function(dataset,description,folder,region){
          var THMap       =   ee.Image(ee.List(dataset).get(4));
          //var THMap  = ee.Image(dataset);
          var regionInfo  = THMap.getInfo();
          var CrsInfo     = regionInfo.bands[0].crs;  
          var output = THMap.reduceRegions({ collection: region, reducer: ee.Reducer.first().setOutputs(["_PixelValue"]),
          scale: 10,crs: CrsInfo });
         Export.table.toDrive({collection: output,
         description: description, folder: folder, 
         fileFormat:'CSV' });
         };
exports.DownloadCSV  = DownloadCSV;
*/
//************************************************ Ends   **********************************************************************