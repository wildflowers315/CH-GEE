//************************************************************************************************************
//************************************** Downloading map  in GeoTIFF format v1 **********************************
//************************************************************************************************************

var DownloadImg = function(dataset,description,folder,region){
  //var CHMap  = ee.Image(ee.List(dataset).get(4));
  var CHMap  = ee.Image(ee.List(dataset).get(0));
  var regionInfo  = CHMap.getInfo();
  var CrsInfo     = regionInfo.bands[0].crs;
  Export.image.toDrive({
  image: CHMap, description: description, folder: folder,
  region: region, scale: 10,maxPixels: 1e13,// crs: CrsInfo
  crs: 'EPSG:4326'
  })};
exports.DownloadImg = DownloadImg;             

//************************************************************************************************************
//************************************** Link to downloading map  in GeoTIFF format **********************************
//************************************************************************************************************

var DownloadImgDirect = function(dataset, description, region) {
return new Promise(function(resolve, reject) {
try {
    // Upload CH-GEE map
    var CHMap = ee.Image(ee.List(dataset).get(0)).rename('PredictedMap');
    // Calculate the box for the Area Of Interest
    var boundingBox = ee.Geometry(ee.FeatureCollection(region).geometry()).bounds()
    // Evaluate the bounding box 
    boundingBox.evaluate(function(geoJSONRegion) {
        if (!geoJSONRegion || !geoJSONRegion.coordinates) {
            reject("Error: Invalid or empty region geometry.");
            return;
        }
        // Set up download arguments
        var downloadArgs = {
            name: description,
            crs: 'EPSG:4326',
            scale: 10,
            region: geoJSONRegion 
        };

        // Generate URL for the download
        var url = CHMap.getDownloadURL(downloadArgs);
        // Return URL
        resolve(url);
    });
} catch (e) {
    reject("Error during download: " + e.message);
}
});
};
exports.DownloadImgDirect = DownloadImgDirect;
