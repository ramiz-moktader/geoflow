"""
GEE-compatible FeatureCollection implementation for GeoFlow, wrapping GeoPandas.
"""

from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Union
from pathlib import Path
import json
import math
import geopandas as gpd
import pandas as pd
import shapely.geometry
import matplotlib.pyplot as plt

from geoflow.core.base import EarthObject
from geoflow.core.crs import CRS
from geoflow.core.filter import Filter
from geoflow.core.reducer import Reducer
from geoflow.geometry.base import Geometry
from geoflow.vector.feature import Feature


class FeatureCollection(EarthObject):
    """GEE-like FeatureCollection wrapping GeoPandas GeoDataFrame."""

    def __init__(self, data: Union[gpd.GeoDataFrame, List[Feature], str, Path], crs: Union[str, int, CRS] = "EPSG:4326", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(metadata=metadata)

        if isinstance(data, (str, Path)):
            self._gdf = gpd.read_file(data)
            if self._gdf.crs is None:
                self._gdf.set_crs(str(crs), inplace=True)
        elif isinstance(data, gpd.GeoDataFrame):
            self._gdf = data.copy()
            if self._gdf.crs is None:
                self._gdf.set_crs(str(crs), inplace=True)
        elif isinstance(data, list):
            rows = []
            for item in data:
                if isinstance(item, Feature):
                    d = dict(item.properties)
                    d["geometry"] = item.geometry.shapely
                    rows.append(d)
                elif isinstance(item, dict):
                    rows.append(item)
            self._gdf = gpd.GeoDataFrame(rows, crs=str(crs))
        else:
            raise TypeError(f"Cannot initialize FeatureCollection from {type(data)}")

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        return self._gdf

    @property
    def crs(self) -> CRS:
        return CRS(self._gdf.crs.to_string() if self._gdf.crs else "EPSG:4326")

    @property
    def columns(self) -> List[str]:
        return list(self._gdf.columns)

    @property
    def bounds(self):
        """Total bounding box (minx, miny, maxx, maxy)."""
        return tuple(self._gdf.total_bounds)

    @property
    def geometry(self) -> Geometry:
        return Geometry(self._gdf.unary_union, crs=self.crs)

    def __len__(self) -> int:
        return len(self._gdf)

    def size(self) -> int:
        return len(self._gdf)

    def __iter__(self) -> Iterator[Feature]:
        for _, row in self._gdf.iterrows():
            props = row.drop("geometry").to_dict()
            yield Feature(row["geometry"], properties=props)

    def first(self) -> Optional[Feature]:
        if len(self._gdf) == 0:
            return None
        row = self._gdf.iloc[0]
        return Feature(row["geometry"], properties=row.drop("geometry").to_dict())

    @property
    def __geo_interface__(self) -> Dict[str, Any]:
        """GeoJSON-like __geo_interface__ protocol support."""
        return self._gdf.__geo_interface__

    # -------------------------------------------------------------
    # GEE Filtering
    # -------------------------------------------------------------
    def filter(self, filter_obj: Filter) -> Any:
        """Filter features using Filter predicate."""
        mask = [filter_obj(row) for _, row in self._gdf.iterrows()]
        return self.__class__(self._gdf[mask].copy(), crs=self.crs)

    def filterBounds(self, geometry: Union[Geometry, Any]) -> Any:
        """Filter features that spatially intersect geometry."""
        geom_shapely = geometry.to_shapely() if hasattr(geometry, "to_shapely") else geometry
        filtered_gdf = self._gdf[self._gdf.intersects(geom_shapely)].copy()
        return self.__class__(filtered_gdf, crs=self.crs)

    def select(self, columns: List[str]) -> Any:
        """Select specific property columns (retaining geometry)."""
        cols = [c for c in columns if c in self._gdf.columns]
        if "geometry" not in cols:
            cols.append("geometry")
        return self.__class__(self._gdf[cols].copy(), crs=self.crs)

    def sort(self, column: str, ascending: bool = True) -> Any:
        return self.__class__(self._gdf.sort_values(by=column, ascending=ascending).copy(), crs=self.crs)

    def limit(self, n: int) -> Any:
        return self.__class__(self._gdf.iloc[:n].copy(), crs=self.crs)

    # -------------------------------------------------------------
    # Spatial Operations
    # -------------------------------------------------------------
    def buffer(self, distance: float) -> Any:
        new_gdf = self._gdf.copy()
        new_gdf["geometry"] = new_gdf.geometry.buffer(distance)
        return self.__class__(new_gdf, crs=self.crs)

    def centroid(self) -> Any:
        new_gdf = self._gdf.copy()
        new_gdf["geometry"] = new_gdf.geometry.centroid
        return self.__class__(new_gdf, crs=self.crs)

    def simplify(self, tolerance: float) -> Any:
        new_gdf = self._gdf.copy()
        new_gdf["geometry"] = new_gdf.geometry.simplify(tolerance)
        return self.__class__(new_gdf, crs=self.crs)

    def dissolve(self, by: Optional[str] = None, aggfunc: str = "first") -> Any:
        new_gdf = self._gdf.dissolve(by=by, aggfunc=aggfunc).reset_index()
        return self.__class__(new_gdf, crs=self.crs)

    def spatial_join(self, other: FeatureCollection, how: str = "inner", predicate: str = "intersects") -> Any:
        joined = gpd.sjoin(self._gdf, other.gdf, how=how, predicate=predicate)
        return self.__class__(joined, crs=self.crs)

    # -------------------------------------------------------------
    # Raster Sampling
    # -------------------------------------------------------------
    def sample_raster(self, image: Any, bands: Optional[List[str]] = None, scale: Optional[float] = None) -> FeatureCollection:
        """
        Sample raster pixel values at feature geometry locations and add as columns.
        Supports both points (exact sampling) and polygons (centroid / zonal sample).
        """
        import rasterio.sample

        target_image = image if self.crs == image.crs else image.reproject(self.crs)
        target_bands = bands or target_image.bands
        selected_img = target_image.select(target_bands)

        new_gdf = self._gdf.copy()

        # Extract coordinates for sampling
        coords = []
        for geom in new_gdf.geometry:
            pt = geom.centroid if geom.geom_type != "Point" else geom
            coords.append((pt.x, pt.y))

        # Sample from array using affine inverse transform
        sampled_vals = {b: [] for b in target_bands}
        inv_transform = ~selected_img.transform

        for x, y in coords:
            col, row = inv_transform @ (x, y)
            r, c = int(math.floor(row)), int(math.floor(col))
            for b_idx, b_name in enumerate(target_bands):
                if 0 <= r < selected_img.height and 0 <= c < selected_img.width:
                    val = float(selected_img._data[b_idx, r, c])
                else:
                    val = float("nan")
                sampled_vals[b_name].append(val)

        for b_name, vals in sampled_vals.items():
            new_gdf[b_name] = vals

        return self.__class__(new_gdf, crs=self.crs)

    # -------------------------------------------------------------
    # Reducers over Attributes
    # -------------------------------------------------------------
    def reduceColumns(self, reducer: Reducer, selectors: List[str]) -> Dict[str, Any]:
        """Aggregate specified property columns."""
        out = {}
        for col in selectors:
            if col in self._gdf.columns:
                arr = self._gdf[col].to_numpy()
                out[col] = reducer.reduce(arr)
        return out

    # -------------------------------------------------------------
    # Visualization & Export
    # -------------------------------------------------------------
    def plot(
        self,
        column: Optional[str] = None,
        cmap: str = "viridis",
        alpha: float = 0.8,
        legend: bool = True,
        ax: Any = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ):
        """Plot features using GeoPandas / Matplotlib."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        self._gdf.plot(column=column, cmap=cmap, alpha=alpha, legend=legend, ax=ax, **kwargs)
        if title:
            ax.set_title(title)
        ax.set_xlabel(f"Longitude ({self.crs.to_string()})")
        ax.set_ylabel(f"Latitude ({self.crs.to_string()})")
        return ax

    def plot_map(self, **kwargs: Any) -> Any:
        """
        Render a publication-ready cartographic map complete with
        North Arrow, dynamic Scale Bar, degree-formatted Lat/Long ticks, and Legend.
        """
        from geoflow.viz.cartography import plot_carto_map
        return plot_carto_map(self, **kwargs)

    def edit_layout(self, **kwargs: Any) -> Any:
        """
        Open the interactive in-notebook ArcGIS-style Layout Editor.
        Allows real-time visual customization of North Arrow, Scale Bar, Graticules,
        Colors, Titles, and publication export directly inside Jupyter and Google Colab.
        """
        from geoflow.viz.layout_editor import create_layout_editor
        return create_layout_editor(self, **kwargs)

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return self._gdf.copy()

    def to_file(self, filepath: Union[str, Path], driver: Optional[str] = None):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._gdf.to_file(filepath, driver=driver)

    def to_geojson(self) -> Dict[str, Any]:
        return json.loads(self._gdf.to_json())

    def __repr__(self) -> str:
        return f"<geoflow.FeatureCollection ({len(self._gdf)} features, cols: {list(self._gdf.columns)}) CRS:{self.crs}>"


# Aliases
Vector = FeatureCollection
