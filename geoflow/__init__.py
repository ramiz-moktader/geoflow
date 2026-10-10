"""
GeoFlow (gf)
============
A Google Earth Engine-like Earth Observation, GIS, Spatial Statistics,
Research Design, and Geospatial Machine Learning Framework.
"""

from __future__ import annotations

__version__ = "0.1.5"

# Core abstractions
from geoflow.core.filter import Filter
from geoflow.core.reducer import Reducer
from geoflow.core.crs import CRS, reproject_bounds

# Geometry
from geoflow.geometry.base import Geometry
from geoflow.geometry.point import Point, MultiPoint
from geoflow.geometry.polygon import Polygon, MultiPolygon, Rectangle

# Raster & ImageCollection
from geoflow.raster.image import Image, Raster
from geoflow.raster.collection import ImageCollection, RasterCollection

# Vector & FeatureCollection
from geoflow.vector.feature import Feature
from geoflow.vector.collection import FeatureCollection, Vector

# Visualization
from geoflow.visualization.map import Map

# Earthdata & Auth
from geoflow.auth.session import login, logout, get_session
from geoflow.earthdata.search import search, search_data
from geoflow.earthdata.downloader import download

# Datasets
from geoflow.datasets.gedi import GEDI

# Subpackages
from geoflow import spatial
from geoflow import validation
from geoflow import ml
from geoflow import sampling
from geoflow import stats
from geoflow import viz
from geoflow.viz.layout_editor import create_layout_editor as edit_layout

# Convenience aliases
import sys
ne = sys.modules[__name__]  # Alias geoflow as ne for full backward compatibility
gf = sys.modules[__name__]  # Standard alias gf

__all__ = [
    "__version__",
    "Filter",
    "Reducer",
    "CRS",
    "Geometry",
    "Point",
    "MultiPoint",
    "Polygon",
    "MultiPolygon",
    "Rectangle",
    "Image",
    "Raster",
    "ImageCollection",
    "RasterCollection",
    "Feature",
    "Vector",
    "FeatureCollection",
    "Map",
    "login",
    "logout",
    "search",
    "search_data",
    "download",
    "spatial",
    "validation",
    "ml",
    "sampling",
    "stats",
    "viz",
]
