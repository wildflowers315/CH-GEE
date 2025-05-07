import ee
import geemap
import os
from typing import Union, List, Dict
from datetime import datetime

def export_to_asset(
    image: ee.Image,
    description: str,
    asset_id: str,
    scale: int = 30,
    max_pixels: int = 1e13
) -> Dict[str, str]:
    """
    Export an Earth Engine image to an Earth Engine asset.
    
    Args:
        image: Earth Engine image to export
        description: Description of the export task
        asset_id: Asset ID for the export
        scale: Scale in meters
        max_pixels: Maximum number of pixels to export
    
    Returns:
        Dict[str, str]: Task information
    """
    task = ee.batch.Export.image.toAsset(
        image=image,
        description=description,
        assetId=asset_id,
        scale=scale,
        maxPixels=max_pixels
    )
    
    task.start()
    
    return {
        'task_id': task.id,
        'description': description,
        'asset_id': asset_id
    }

def export_to_drive(
    image: ee.Image,
    description: str,
    folder: str,
    file_name: str,
    scale: int = 30,
    max_pixels: int = 1e13,
    file_format: str = 'GeoTIFF'
) -> Dict[str, str]:
    """
    Export an Earth Engine image to Google Drive.
    
    Args:
        image: Earth Engine image to export
        description: Description of the export task
        folder: Google Drive folder name
        file_name: Output file name
        scale: Scale in meters
        max_pixels: Maximum number of pixels to export
        file_format: Output file format
    
    Returns:
        Dict[str, str]: Task information
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=file_name,
        scale=scale,
        maxPixels=max_pixels,
        fileFormat=file_format
    )
    
    task.start()
    
    return {
        'task_id': task.id,
        'description': description,
        'folder': folder,
        'file_name': file_name
    }

def download_to_local(
    image: ee.Image,
    output_dir: str,
    file_name: str,
    scale: int = 30,
    region: ee.Geometry = None
) -> str:
    """
    Download an Earth Engine image to local storage.
    
    Args:
        image: Earth Engine image to download
        output_dir: Local output directory
        file_name: Output file name
        scale: Scale in meters
        region: Region to download (if None, uses image bounds)
    
    Returns:
        str: Path to downloaded file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get download URL
    if region is None:
        region = image.geometry()
    
    url = image.getDownloadURL({
        'scale': scale,
        'region': region,
        'format': 'GeoTIFF'
    })
    
    # Download file
    output_path = os.path.join(output_dir, f"{file_name}.tif")
    geemap.download_ee_image(url, output_path)
    
    return output_path 