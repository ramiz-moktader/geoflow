"""
GEDI (Global Ecosystem Dynamics Investigation) L2A/L2B LiDAR adapter.
Extracts georeferenced footprint shots, relative height metrics (RH98), and quality flags.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import geopandas as gpd
import shapely.geometry

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

        shots = []
        read_errors = []
        for f in self.files:
            try:
                    with h5py.File(f, "r") as h5:
                        for beam in [k for k in h5.keys() if k.startswith("BEAM")]:
                            b_group = h5[beam]
                            if "lat_lowestmode" not in b_group or "lon_lowestmode" not in b_group:
                                continue
                            lats = np.asarray(b_group["lat_lowestmode"][:], dtype=np.float64)
                            lons = np.asarray(b_group["lon_lowestmode"][:], dtype=np.float64)
                            n_points = len(lats)
                            if n_points == 0:
                                continue

                            qual = b_group["quality_flag"][:] if "quality_flag" in b_group else np.ones(n_points, dtype=np.int32)
                            degrade = b_group["degrade_flag"][:] if "degrade_flag" in b_group else np.zeros(n_points, dtype=np.int32)
                            sens = b_group["sensitivity"][:] if "sensitivity" in b_group else np.ones(n_points, dtype=np.float32)

                            mask = np.ones(n_points, dtype=bool)
                            if self._quality_filter_active:
                                mask &= (qual == 1) & (degrade == 0) & (sens >= 0.9)

                            if self._bounds_filter:
                                b = self._bounds_filter.bounds
                                if hasattr(self._bounds_filter, "crs") and str(self._bounds_filter.crs) != "EPSG:4326":
                                    from geoflow.core.crs import reproject_bounds
                                    b = reproject_bounds(b, self._bounds_filter.crs, "EPSG:4326")
                                mask &= (lons >= b[0]) & (lons <= b[2]) & (lats >= b[1]) & (lats <= b[3])

                            if not np.any(mask):
                                continue

                            indices = np.where(mask)[0]
                            val_lats = lats[indices]
                            val_lons = lons[indices]
                            val_qual = qual[indices]
                            val_sens = sens[indices]

                            shot_nums = b_group["shot_number"][:][indices] if "shot_number" in b_group else indices
                            elevs = b_group["elev_lowestmode"][:][indices] if "elev_lowestmode" in b_group else np.zeros(len(indices))

                            # L2A Relative Heights
                            rh_data = b_group["rh"][:][indices] if "rh" in b_group else None
                            rh98_vals = rh_data[:, 98] if (rh_data is not None and rh_data.ndim > 1) else (rh_data if rh_data is not None else np.zeros(len(indices)))
                            rh100_vals = rh_data[:, 100] if (rh_data is not None and rh_data.ndim > 1) else rh98_vals
                            rh50_vals = rh_data[:, 50] if (rh_data is not None and rh_data.ndim > 1) else np.zeros(len(indices))

                            # L4A Aboveground Biomass
                            agbd_vals = b_group["agbd"][:][indices] if "agbd" in b_group else np.zeros(len(indices))

                            for idx_in_sub in range(len(indices)):
                                rec = {
                                    "shot_number": int(shot_nums[idx_in_sub]),
                                    "beam": beam,
                                    "latitude": float(val_lats[idx_in_sub]),
                                    "longitude": float(val_lons[idx_in_sub]),
                                    "elevation": float(elevs[idx_in_sub]),
                                    "quality_flag": int(val_qual[idx_in_sub]),
                                    "sensitivity": float(val_sens[idx_in_sub]),
                                    "rh50": float(rh50_vals[idx_in_sub]),
                                    "rh98": float(rh98_vals[idx_in_sub]),
                                    "rh100": float(rh100_vals[idx_in_sub]),
                                    "agbd": float(agbd_vals[idx_in_sub]),
                                    "geometry": shapely.geometry.Point(float(val_lons[idx_in_sub]), float(val_lats[idx_in_sub])),
                                }
                                shots.append(rec)
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
            "sensitivity", "rh50", "rh98", "rh100", "agbd", "geometry",
        ]
        gdf = gpd.GeoDataFrame(shots, columns=columns, geometry="geometry", crs="EPSG:4326")
        if gdf.empty:
            import warnings
            warnings.warn("No GEDI shots matched the region/quality filters.")
        # Keep selected variables
        keep_cols = ["geometry", "latitude", "longitude"] + [v for v in self._selected_variables if v in gdf.columns]
        return gdf[keep_cols]

    def to_feature_collection(self) -> FeatureCollection:
        return FeatureCollection(self.to_geodataframe())

    to_fc = to_feature_collection

    def __repr__(self) -> str:
        return f"<geoflow.GEDI Level:{self.level} variables={self._selected_variables}>"
