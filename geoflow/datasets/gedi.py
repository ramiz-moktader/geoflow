"""
GEDI (Global Ecosystem Dynamics Investigation) L2A/L2B LiDAR adapter.
Extracts georeferenced footprint shots, relative height metrics (RH98), and quality flags.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import datetime
import numpy as np
import geopandas as gpd

from geoflow.core.base import EarthObject
from geoflow.geometry.base import Geometry
from geoflow.vector.collection import FeatureCollection


class GEDI(EarthObject):
    """GEE-like GEDI LiDAR dataset object."""

    def __init__(self, level: str = "L2A", files: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(metadata=metadata)
        self.level = level.upper()
        self.files = files or []
        self._bounds_filter: Optional[Geometry] = None
        self._date_filter: Optional[Tuple[str, str]] = None
        self._quality_filter_active: bool = False
        self._selected_variables: List[str] = ["rh98", "sensitivity", "quality_flag"]

    @classmethod
    def load(
        cls,
        level: str = "L2A",
        region: Optional[Union[Geometry, Any]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10,
        output_dir: str = "./data",
        **kwargs: Any,
    ) -> GEDI:
        """
        Search NASA Earthdata (using earthaccess backend) for GEDI granules,
        download them to output_dir, and return an initialized GEDI object.
        """
        from geoflow.earthdata.search import search_data
        from geoflow.earthdata.downloader import download

        dataset_id = f"GEDI/{level.upper()}"
        results = search_data(
            dataset=dataset_id,
            region=region,
            start=start,
            end=end,
            limit=limit,
            **kwargs,
        )
        if not results:
            raise ValueError(f"No NASA GEDI granules found for {dataset_id}")

        downloaded_files = download(results, output_dir=output_dir)
        h5_files = [str(f) for f in downloaded_files if str(f).lower().endswith((".h5", ".he5", ".hdf5"))]
        if not h5_files:
            raise RuntimeError(
                f"Found {len(results)} {dataset_id} granules but no HDF5 files were downloaded. "
                "Run `geoflow.login()` and check your NASA Earthdata credentials."
            )
        gedi_obj = cls(level=level, files=h5_files)
        if region is not None:
            gedi_obj = gedi_obj.filterBounds(region)
        if start and end:
            gedi_obj = gedi_obj.filterDate(start, end)
        return gedi_obj

    def filterBounds(self, aoi: Union[Geometry, Any]) -> GEDI:
        clone = self._clone()
        clone._bounds_filter = aoi if isinstance(aoi, Geometry) else Geometry(aoi)
        return clone

    def filterDate(self, start: str, end: str) -> GEDI:
        clone = self._clone()
        clone._date_filter = (start, end)
        return clone

    def filterQuality(self) -> GEDI:
        clone = self._clone()
        clone._quality_filter_active = True
        return clone

    def select(self, variables: Union[str, List[str]]) -> GEDI:
        clone = self._clone()
        clone._selected_variables = [variables] if isinstance(variables, str) else list(variables)
        return clone

    def _clone(self) -> GEDI:
        g = GEDI(self.level, files=list(self.files), metadata=dict(self._metadata))
        g._bounds_filter = self._bounds_filter
        g._date_filter = self._date_filter
        g._quality_filter_active = self._quality_filter_active
        g._selected_variables = list(self._selected_variables)
        return g

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Extract footprint shots from the GEDI HDF5 files into a GeoPandas GeoDataFrame,
        applying spatial and quality filters. Returns an empty GeoDataFrame if no shots
        fall inside the filters.
        """
        if not self.files:
            raise ValueError(
                "This GEDI object has no data files. Use GEDI.load(...) or pass files=[...]."
            )
        import h5py
        import pandas as pd

        # Spatial filter as a WGS84 bounding box
        bbox = None
        if self._bounds_filter:
            bbox = self._bounds_filter.bounds
            if str(self._bounds_filter.crs) != "EPSG:4326":
                from geoflow.core.crs import reproject_bounds
                bbox = reproject_bounds(bbox, self._bounds_filter.crs, "EPSG:4326")

        # Temporal filter as [start, end] inclusive, in seconds since the GEDI epoch (delta_time)
        time_window = None
        if self._date_filter:
            from geoflow.core.filter import _parse_date
            start, end = (_parse_date(d) for d in self._date_filter)
            if start is None or end is None:
                raise ValueError(f"Could not parse GEDI date filter {self._date_filter!r}")
            epoch = datetime.date(2018, 1, 1)
            time_window = ((start - epoch).days * 86400.0, ((end - epoch).days + 1) * 86400.0)

        frames = []
        read_errors = []
        for f in self.files:
            try:
                with h5py.File(f, "r") as h5:
                    for beam in [k for k in h5.keys() if k.startswith("BEAM")]:
                        frame = self._read_beam(h5[beam], beam, bbox, time_window)
                        if frame is not None:
                            frames.append(frame)
            except Exception as e:
                read_errors.append(f"{f}: {e}")

        if read_errors and len(read_errors) == len(self.files):
            raise RuntimeError(
                "Could not read any GEDI file (possibly corrupt or incomplete downloads). "
                "First error: " + read_errors[0]
            )
        if read_errors:
            import warnings
            warnings.warn(f"Skipped {len(read_errors)} unreadable GEDI file(s): " + "; ".join(read_errors[:3]))

        columns = [
            "shot_number", "beam", "latitude", "longitude", "elevation", "quality_flag",
            "sensitivity", "rh50", "rh98", "rh100", "agbd", "delta_time",
        ]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({c: [] for c in columns})
        gdf = gpd.GeoDataFrame(
            df[columns],
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )
        if gdf.empty:
            import warnings
            warnings.warn("No GEDI shots matched the region/quality filters.")
        # Keep selected variables
        keep_cols = ["geometry", "latitude", "longitude"] + [v for v in self._selected_variables if v in gdf.columns]
        return gdf[keep_cols]

    def _read_beam(self, b_group: Any, beam: str, bbox: Optional[Tuple[float, ...]], time_window: Optional[Tuple[float, float]]):
        """
        Read one BEAM group into a DataFrame of filtered shots, or None if nothing matches.
        Handles L2A/L4A (variables at the beam root) and L2B (geolocation under BEAMxxxx/geolocation,
        rh100 in centimetres, l2b_quality_flag).
        """
        import pandas as pd

        def find(*names):
            for name in names:
                if name in b_group:
                    return b_group[name]
            return None

        lat_ds = find("lat_lowestmode", "geolocation/lat_lowestmode")
        lon_ds = find("lon_lowestmode", "geolocation/lon_lowestmode")
        if lat_ds is None or lon_ds is None:
            return None
        lats = np.asarray(lat_ds[:], dtype=np.float64)
        lons = np.asarray(lon_ds[:], dtype=np.float64)
        n_points = len(lats)
        if n_points == 0:
            return None

        def read(ds, default, dtype=np.float64):
            return np.asarray(ds[:], dtype=dtype) if ds is not None else np.full(n_points, default, dtype=dtype)

        qual = read(find("quality_flag", "l2b_quality_flag", "l4_quality_flag"), 1, np.int32)
        degrade = read(find("degrade_flag", "geolocation/degrade_flag"), 0, np.int32)
        sens = read(find("sensitivity", "geolocation/sensitivity"), 1.0)
        delta_time = read(find("delta_time", "geolocation/delta_time"), np.nan)

        mask = np.isfinite(lats) & np.isfinite(lons)
        if self._quality_filter_active:
            mask &= (qual == 1) & (degrade == 0) & (sens >= 0.9)
        if bbox is not None:
            mask &= (lons >= bbox[0]) & (lons <= bbox[2]) & (lats >= bbox[1]) & (lats <= bbox[3])
        if time_window is not None:
            mask &= (delta_time >= time_window[0]) & (delta_time < time_window[1])
        if not np.any(mask):
            return None
        idx = np.where(mask)[0]

        shot_ds = find("shot_number", "geolocation/shot_number")
        shot_nums = np.asarray(shot_ds[:])[idx] if shot_ds is not None else idx
        elevs = read(find("elev_lowestmode", "geolocation/elev_lowestmode"), np.nan)[idx]

        nan = np.full(len(idx), np.nan)
        rh50 = rh98 = rh100 = nan
        rh_ds = find("rh")
        if rh_ds is not None and rh_ds.ndim > 1:  # L2A: rh[:, 0..100] in metres
            rh = np.asarray(rh_ds[:], dtype=np.float64)[idx]
            rh50, rh98, rh100 = rh[:, 50], rh[:, 98], rh[:, 100]
        elif find("rh100") is not None:  # L2B: rh100 in centimetres
            rh100 = read(find("rh100"), np.nan)[idx] / 100.0
        agbd = read(find("agbd"), np.nan)[idx]  # L4A

        return pd.DataFrame({
            "shot_number": shot_nums,
            "beam": beam,
            "latitude": lats[idx],
            "longitude": lons[idx],
            "elevation": elevs,
            "quality_flag": qual[idx],
            "sensitivity": sens[idx],
            "rh50": rh50,
            "rh98": rh98,
            "rh100": rh100,
            "agbd": agbd,
            "delta_time": delta_time[idx],
        })

    def to_feature_collection(self) -> FeatureCollection:
        return FeatureCollection(self.to_geodataframe())

    to_fc = to_feature_collection

    def __repr__(self) -> str:
        return f"<geoflow.GEDI Level:{self.level} variables={self._selected_variables}>"
