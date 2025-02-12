//***************************************************************************************************************
//********************************************* Canopy Height Mapper  *******************************************
//***************************************************************************************************************

var CanopyHeightMapper = function(aoi, year, start_date, end_date,startDateGEDI,endDateGEDI,cloudsTh, quantile, model, mask, gedi_type,
numTreesRF,varSplitRF,minLeafPopuRF,bagFracRF,maxNodesRF,numTreesGBM,shrGBM,samLingRateGBM,maxNodesGBM,lossGBM,maxNodesCART,minLeafPopCART){
  
  //***************************************************************************************************************
  //  Input Data
  //***************************************************************************************************************

  var startDateWithYear = year+"-"+start_date; 
  var endDateWithYear   = year+"-"+end_date;
  
  //***************************************************************************************************************
  //  Importing Area of Insterest (AOI) through drawing or uploading
  //***************************************************************************************************************

  var aoi2 = ee.Geometry(ee.FeatureCollection(aoi).geometry())//.geometries()
  var polygonArea = aoi2.area({'maxError': 1});
  polygonArea = (polygonArea.divide(10000).round()).getInfo();
  
  //***************************************************************************************************************
  // Private codes
  //***************************************************************************************************************
  
  library4.VARIMP(classifier);//* variable important function
  //***************************************************************************************************************
  // Canopy height map
  //***************************************************************************************************************
  library4.scalecolor(0,50,classified);
  //
   return([classified])
  };
exports.CanopyHeightMapper = CanopyHeightMapper;

//***************************************************** End *****************************************************