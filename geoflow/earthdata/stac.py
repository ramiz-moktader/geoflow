"""
Cloud-Native Data Streaming via STAC (SpatioTemporal Asset Catalog).
Provides direct streaming of analysis-ready data (Zarr / Cloud-Optimized GeoTIFFs) 
from AWS, Planetary Computer, and Earth Search without downloading files.
"""

from typing import Any, Dict, List, Optional, Union
import logging

from geoflow.core.crs import CRS, reproject_bounds
from geoflow.geometry.base import Geometry
from geoflow.raster.image import Image
from geoflow.raster.collection import ImageCollection

logger = logging.getLogger(__name__)

_INSTALL_HINT = 'STAC streaming requires optional dependencies. Install them with: pip install "geoflow[stac]"'


def _wgs84_bbox(geometry: Any) -> Optional[List[float]]:
    """STAC APIs expect bbox in WGS84 lon/lat; reproject a Geometry's bounds if needed."""
    if geometry is None:
        return None
    if isinstance(geometry, Geometry):
        b = geometry.bounds
        return list(b if geometry.crs == CRS("EPSG:4326") else reproject_bounds(b, geometry.crs, "EPSG:4326"))
    if hasattr(geometry, "bounds"):
        return list(geometry.bounds)
    if isinstance(geometry, (list, tuple)) and len(geometry) == 4:
        return list(geometry)
    raise TypeError(f"Unsupported geometry for STAC search: {type(geometry)}")

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
    try:
        from pystac_client import Client
    except ImportError:
        raise ImportError(_INSTALL_HINT) from None
    url = CATALOGS.get(catalog, catalog)
    client = Client.open(url)

    search_params = {
        "collections": [collection],
        "max_items": limit,
    }

    bbox = _wgs84_bbox(geometry)
    if bbox is not None:
        search_params["bbox"] = bbox

    if start or end:
        s = start or "2000-01-01"
        e = end or "2030-01-01"
        search_params["datetime"] = f"{s}/{e}"
        
    if query:
        search_params["query"] = query
        
    logger.info(f"Searching STAC Catalog: {url} for {collection}")
    items = client.search(**search_params).item_collection()
            
    return items

def load_stac_collection(
    items: Any,
    bands: Optional[List[str]] = None,
    resolution: Optional[Union[int, float]] = None,
    bbox: Optional[List[float]] = None,
    crs: str = "EPSG:4326",
    chunks: Dict[str, int] = {"x": 1024, "y": 1024, "time": 1},
    patch_url: Optional[Any] = None,
) -> ImageCollection:
    """
    Lazily loads a STAC ItemCollection into a GeoFlow ImageCollection using odc-stac and dask.
    Streams data directly from the cloud bucket (S3 / Azure Blob).
    """
    try:
        import odc.stac
        import pandas as pd
    except ImportError:
        raise ImportError(_INSTALL_HINT) from None

    # Load into lazy dask-backed xarray DataArray via odc-stac
    load_params = {
        "items": items,
        "chunks": chunks,
        "groupby": "solar_day", # Group items by pass
    }
    
    if bands:
        load_params["bands"] = bands
    if resolution:
        load_params["resolution"] = resolution
    if bbox:
        load_params["bbox"] = bbox
    if crs:
        load_params["crs"] = crs
    if patch_url:
        load_params["patch_url"] = patch_url
        
    # Lazy load (No data is downloaded yet)
    ds = odc.stac.load(**load_params)
    
    # odc-stac returns a Dataset (variables=bands). 
    # Convert to DataArray (band dimension) for GeoFlow compatibility
    da = ds.to_array(dim="band")
    
    # Items grouped into each time slice (groupby="solar_day" merges items from the same pass),
    # so metadata is matched by date rather than by position
    def naive_utc(dt):
        if dt is None:
            return None
        ts = pd.Timestamp(dt)
        return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts

    item_list = list(items)
    item_times = [naive_utc(it.datetime) for it in item_list]

    # Wrap xarray slices into GeoFlow Images
    images = []
    for t_idx in range(da.sizes["time"]):
        slice_da = da.isel(time=t_idx)
        # Create GeoFlow Image wrapping the lazy DataArray
        img = Image.from_xarray(slice_da)
        # odc-stac records the grid on its geobox; prefer it over coordinate-derived guesses
        geobox = getattr(getattr(slice_da, "odc", None), "geobox", None)
        if geobox is not None and geobox.crs is not None:
            img._crs = CRS(str(geobox.crs))
            img._transform = geobox.affine

        slice_time = pd.Timestamp(slice_da.time.values)
        matched = [it for it, t in zip(item_list, item_times)
                   if t is not None and abs((t - slice_time).total_seconds()) < 86400]
        if not matched and item_list:
            matched = [min(zip(item_list, item_times), key=lambda p: abs((p[1] - slice_time).total_seconds())
                           if p[1] is not None else float("inf"))[0]]
        if matched:
            img._metadata.update(matched[0].properties)
            img._metadata["stac_item_ids"] = [it.id for it in matched]
        img._metadata["datetime"] = str(slice_da.time.values)
        images.append(img)
        
    collection_name = items[0].collection_id if items else "STAC_Collection"
    
    # Return a new ImageCollection
    return ImageCollection(images, dataset_name=collection_name)
