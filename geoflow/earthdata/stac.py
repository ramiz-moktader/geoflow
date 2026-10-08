"""
Cloud-Native Data Streaming via STAC (SpatioTemporal Asset Catalog).
Provides direct streaming of analysis-ready data (Zarr / Cloud-Optimized GeoTIFFs) 
from AWS, Planetary Computer, and Earth Search without downloading files.
"""

from typing import Any, Dict, List, Optional, Union
import logging

from pystac_client import Client
import odc.stac

from geoflow.geometry.base import Geometry
from geoflow.raster.image import Image
from geoflow.raster.collection import ImageCollection

logger = logging.getLogger(__name__)

# Common STAC Catalogs
CATALOGS = {
    "planetary-computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "earth-search": "https://earth-search.aws.element84.com/v1",
}


def search_stac(
    collection: str,
    catalog: str = "planetary-computer",
    geometry: Optional[Union[Geometry, Any]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100,
    query: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Search a STAC catalog and return a pystac ItemCollection.
    """
    url = CATALOGS.get(catalog, catalog)
    client = Client.open(url)
    
    search_params = {
        "collections": [collection],
        "max_items": limit,
    }
    
    if geometry is not None:
        if hasattr(geometry, "bounds"):
            search_params["bbox"] = geometry.bounds
        elif isinstance(geometry, (list, tuple)) and len(geometry) == 4:
            search_params["bbox"] = geometry
            
    if start or end:
        s = start or "2000-01-01"
        e = end or "2030-01-01"
        search_params["datetime"] = f"{s}/{e}"
        
    if query:
        search_params["query"] = query
        
    logger.info(f"Searching STAC Catalog: {url} for {collection}")
    return client.search(**search_params).item_collection()


def load_stac_collection(
    items: Any,
    bands: Optional[List[str]] = None,
    resolution: Optional[Union[int, float]] = None,
    bbox: Optional[List[float]] = None,
    crs: str = "EPSG:4326",
    chunks: Dict[str, int] = {"x": 1024, "y": 1024, "time": 1},
) -> ImageCollection:
    """
    Lazily loads a STAC ItemCollection into a GeoFlow ImageCollection using odc-stac and dask.
    Streams data directly from the cloud bucket (S3 / Azure Blob).
    """
    import xarray as xr
    
    # Load into lazy dask-backed xarray DataArray via odc-stac
    load_params = {
        "items": items,
        "chunks": chunks,
        "groupby": "solar_day", # Group items by pass
    }
    
    if bands:
        load_params["assets"] = bands
    if resolution:
        load_params["resolution"] = resolution
    if bbox:
        load_params["bbox"] = bbox
    if crs:
        load_params["crs"] = crs
        
    # Lazy load (No data is downloaded yet)
    ds = odc.stac.load(**load_params)
    
    # odc-stac returns a Dataset (variables=bands). 
    # Convert to DataArray (band dimension) for GeoFlow compatibility
    da = ds.to_array(dim="band")
    
    # Wrap xarray slices into GeoFlow Images
    images = []
    for t_idx in range(da.sizes["time"]):
        slice_da = da.isel(time=t_idx)
        # Create GeoFlow Image wrapping the lazy DataArray
        img = Image.from_xarray(slice_da)
        # Preserve STAC metadata
        item_meta = items[t_idx].properties if t_idx < len(items) else {}
        img._metadata.update(item_meta)
        img._metadata["datetime"] = str(slice_da.time.values)
        images.append(img)
        
    collection_name = items[0].collection_id if items else "STAC_Collection"
    
    # Return a new ImageCollection
    return ImageCollection(images, dataset_name=collection_name)
