"""
Data Discovery and Search Engine for NASA Earthdata and CMR.
Provides high-level search across GEDI, Sentinel, Landsat, MODIS, and other collections.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import datetime
import requests

from geoflow.geometry.base import Geometry
from geoflow.core.crs import CRS


# Common NASA concept IDs / short names
DATASET_SHORTNAMES = {
    "GEDI/L2A": "GEDI02_A",
    "GEDI/L2B": "GEDI02_B",
    "GEDI02_A": "GEDI02_A",
    "GEDI02_B": "GEDI02_B",
    "SENTINEL-2": "HLSS30",
    "LANDSAT-8": "HLSL30",
    "MODIS": "MOD09GA",
    "NASADEM": "NASADEM_HGT",
}


def search(
    dataset: str,
    region: Optional[Union[Geometry, List[float], Any]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 50,
    cloud_cover: Optional[Union[float, Tuple[float, float]]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Search NASA Earthdata granules using CMR API or earthaccess backend.
    """
    short_name = DATASET_SHORTNAMES.get(dataset, dataset)

    # Try earthaccess backend first if installed
    try:
        import earthaccess
        params = {"short_name": short_name, "count": limit}
        if region:
            bounds = region.bounds if hasattr(region, "bounds") else region
            params["bounding_box"] = bounds
        if start or end:
            temporal = (start or "1970-01-01", end or datetime.date.today().isoformat())
            params["temporal"] = temporal
        if cloud_cover is not None:
            params["cloud_cover"] = cloud_cover
            
        params.update(kwargs)

        results = earthaccess.search_data(**params)
        return results
    except (ImportError, Exception):
        pass

    # Direct NASA CMR Search API query
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    params = {
        "short_name": short_name,
        "page_size": min(limit, 2000),
    }

    if region:
        b = region.bounds if hasattr(region, "bounds") else region
        # CMR expects: lower_left_lon,lower_left_lat,upper_right_lon,upper_right_lat
        params["bounding_box"] = f"{b[0]},{b[1]},{b[2]},{b[3]}"

    if start or end:
        s_str = f"{start}T00:00:00Z" if start else "1970-01-01T00:00:00Z"
        e_str = f"{end}T23:59:59Z" if end else datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT23:59:59Z")
        params["temporal"] = f"{s_str},{e_str}"

    if cloud_cover is not None:
        params["cloud_cover"] = f"0,{cloud_cover}"

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    entries = data.get("feed", {}).get("entry", [])
    granules = []
    for entry in entries:
        links = [l["href"] for l in entry.get("links", []) if "href" in l and not l.get("inherited")]
        granules.append({
            "id": entry.get("id"),
            "title": entry.get("title"),
            "time_start": entry.get("time_start"),
            "time_end": entry.get("time_end"),
            "urls": links,
            "raw": entry,
        })
    return granules

search_data = search
