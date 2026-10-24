"""
Spatial and research sampling designs for remote sensing and GIS.
Implements stratified, balanced, minimum-distance, and sampling bias diagnostics.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import warnings
import numpy as np
import scipy.spatial
import geopandas as gpd
import shapely.geometry

from geoflow.core.crs import CRS
from geoflow.geometry.base import Geometry
from geoflow.vector.collection import FeatureCollection


def stratified(
    region: Union[Geometry, Any],
    strata: Any = None,
    n_per_stratum: int = 100,
    random_state: int = 42,
) -> FeatureCollection:
    """
    Spatially stratified sampling across raster or vector strata.
    Supports Image categorical rasters, FeatureCollections, or explicit stratum lists.
    """
    from geoflow.raster.image import Image
    rng = np.random.RandomState(random_state)
    geom = region if isinstance(region, Geometry) else Geometry(region)
    minx, miny, maxx, maxy = geom.bounds

    points = []

    if isinstance(strata, Image):
        # Sample pixel centres of each class, restricted to the region before sampling
        from rasterio.features import geometry_mask
        if geom.crs != strata.crs:
            geom = geom.reproject(strata.crs)
        data = strata.data[0]
        in_region = geometry_mask([geom.shapely], out_shape=data.shape, transform=strata.transform, invert=True)
        valid_mask = ~np.isnan(data) & in_region
        unique_classes = np.unique(data[valid_mask])

        for cls in unique_classes:
            rows, cols = np.where(valid_mask & (data == cls))
            if len(rows) < n_per_stratum:
                warnings.warn(f"Stratum {cls} has only {len(rows)} pixels in the region (requested {n_per_stratum}); taking all of them.")
            chosen = rng.choice(len(rows), size=min(n_per_stratum, len(rows)), replace=False)
            for idx in chosen:
                r, c = rows[idx], cols[idx]
                x, y = strata.transform @ (c + 0.5, r + 0.5)
                points.append({"stratum": int(cls) if float(cls).is_integer() else cls, "geometry": shapely.geometry.Point(x, y)})

    elif hasattr(strata, "gdf") or isinstance(strata, gpd.GeoDataFrame):
        gdf_strata = strata.gdf if hasattr(strata, "gdf") else strata
        if gdf_strata.crs is not None and geom.crs != CRS(gdf_strata.crs):
            geom = geom.reproject(CRS(gdf_strata.crs))
        col = next((c for c in ["stratum", "class", "landcover", "label"] if c in gdf_strata.columns), gdf_strata.columns[0])
        unique_classes = gdf_strata[col].unique()
        for cls in unique_classes:
            sub = gdf_strata[gdf_strata[col] == cls]
            collected = 0
            sub_union = sub.unary_union
            s_minx, s_miny, s_maxx, s_maxy = sub.total_bounds
            attempts = 0
            while collected < n_per_stratum and attempts < n_per_stratum * 100:
                attempts += 1
                rx = rng.uniform(s_minx, s_maxx)
                ry = rng.uniform(s_miny, s_maxy)
                pt = shapely.geometry.Point(rx, ry)
                if sub_union.contains(pt) and geom.contains(pt):
                    points.append({"stratum": cls, "geometry": pt})
                    collected += 1
    else:
        # Strata is a list of classes or None
        classes = list(strata) if strata is not None and hasattr(strata, "__iter__") else [1, 2, 3]
        for cls in classes:
            collected = 0
            attempts = 0
            while collected < n_per_stratum and attempts < n_per_stratum * 100:
                attempts += 1
                rx = rng.uniform(minx, maxx)
                ry = rng.uniform(miny, maxy)
                pt = shapely.geometry.Point(rx, ry)
                if geom.contains(pt):
                    points.append({
                        "stratum": cls,
                        "geometry": pt,
                    })
                    collected += 1

    if not points:
        warnings.warn("Stratified sampling found no points inside the region; returning an empty FeatureCollection.")
        gdf = gpd.GeoDataFrame({"stratum": []}, geometry=[], crs=geom.crs.to_string())
        return FeatureCollection(gdf, crs=geom.crs)

    gdf = gpd.GeoDataFrame(points, crs=geom.crs.to_string())
    return FeatureCollection(gdf, crs=geom.crs)


def min_distance(
    candidates: Union[FeatureCollection, gpd.GeoDataFrame],
    n: int = 500,
    distance: float = 1000.0,
    unit: str = "meters",
    random_state: int = 42,
) -> FeatureCollection:
    """
    Minimum-distance sampling to suppress spatial clustering in field samples.
    """
    rng = np.random.RandomState(random_state)
    gdf = candidates.gdf if hasattr(candidates, "gdf") else candidates.copy()

    # Measure in metres: project geographic data to its local UTM zone
    work = gdf
    if gdf.crs is not None and gdf.crs.is_geographic and unit.lower() in ("meter", "meters", "metres", "m"):
        work = gdf.to_crs(gdf.estimate_utm_crs())
    eff_distance = distance

    coords = np.array([[g.centroid.x, g.centroid.y] for g in work.geometry])
    indices = np.arange(len(coords))
    rng.shuffle(indices)

    selected_indices = []
    selected_coords = []

    for idx in indices:
        pt = coords[idx]
        if not selected_coords:
            selected_indices.append(idx)
            selected_coords.append(pt)
        else:
            dists = np.sqrt(np.sum((np.array(selected_coords) - pt)**2, axis=1))
            if np.all(dists >= eff_distance):
                selected_indices.append(idx)
                selected_coords.append(pt)
                if len(selected_indices) >= n:
                    break

    selected_gdf = gdf.iloc[selected_indices].copy()
    return FeatureCollection(selected_gdf, crs=selected_gdf.crs)


def bias_report(samples: Union[FeatureCollection, gpd.GeoDataFrame]) -> Dict[str, Any]:
    """
    Analyze sampling spatial clustering, edge effects, and density distribution.
    """
    gdf = samples.gdf if hasattr(samples, "gdf") else samples
    coords = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry])
    n = len(coords)

    # Nearest neighbor distance analysis (Clark-Evans R)
    tree = scipy.spatial.KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    mean_nn_dist = float(np.mean(dists[:, 1]))

    minx, miny, maxx, maxy = gdf.total_bounds
    area = max(1e-5, (maxx - minx) * (maxy - miny))
    expected_nn = 0.5 / np.sqrt(n / area)
    clark_evans_r = mean_nn_dist / expected_nn if expected_nn > 0 else 1.0

    clustering = "Clustered" if clark_evans_r < 0.9 else ("Dispersed" if clark_evans_r > 1.1 else "Random")

    return {
        "sample_size": n,
        "mean_nn_distance": mean_nn_dist,
        "clark_evans_r": clark_evans_r,
        "spatial_pattern": clustering,
    }
