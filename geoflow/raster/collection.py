"""
GEE-compatible ImageCollection implementation for GeoFlow.
Provides temporal stacking, spatiotemporal filtering, compositing, and mosaicking.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import numpy as np

from geoflow.core.base import EarthObject
from geoflow.core.filter import Filter
from geoflow.core.reducer import Reducer
from geoflow.geometry.base import Geometry
from geoflow.raster.image import Image


class ImageCollection(EarthObject):
    """GEE-like ImageCollection holding multiple Image granules across space and time."""

    def __init__(self, images: Optional[List[Image]] = None, dataset_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(metadata=metadata)
        self.dataset_name = dataset_name or "CustomCollection"
        self._images: List[Image] = list(images) if images is not None else []

    def __len__(self) -> int:
        return len(self._images)

    def size(self) -> int:
        return len(self._images)

    def __iter__(self) -> Iterator[Image]:
        return iter(self._images)

    def __getitem__(self, index: int) -> Image:
        return self._images[index]

    def first(self) -> Optional[Image]:
        return self._images[0] if self._images else None

    def to_list(self) -> List[Image]:
        return list(self._images)

    # -------------------------------------------------------------
    # GEE-Style Filtering
    # -------------------------------------------------------------
    def filterBounds(self, geometry: Union[Geometry, Any]) -> Any:
        """Filter images whose spatial bounds intersect the given geometry."""
        f = Filter.bounds(geometry)
        filtered = [img for img in self._images if f(img)]
        return self.__class__(filtered, dataset_name=self.dataset_name, metadata=self._metadata)

    def filterDate(self, start: str, end: str) -> Any:
        """Filter images within date window [start, end]."""
        f = Filter.date(start, end)
        filtered = [img for img in self._images if f(img)]
        return self.__class__(filtered, dataset_name=self.dataset_name, metadata=self._metadata)

    def filter(self, filter_obj: Filter) -> Any:
        """Filter images using arbitrary Filter expression."""
        filtered = [img for img in self._images if filter_obj(img)]
        return self.__class__(filtered, dataset_name=self.dataset_name, metadata=self._metadata)

    def filterMetadata(self, name: str, operator: str, value: Any) -> Any:
        """Filter images by metadata property and operator."""
        op_map = {
            "equals": Filter.eq,
            "=": Filter.eq,
            "==": Filter.eq,
            "not_equals": Filter.neq,
            "!=": Filter.neq,
            "greater_than": Filter.gt,
            ">": Filter.gt,
            "greater_than_or_equals": Filter.gte,
            ">=": Filter.gte,
            "less_than": Filter.lt,
            "<": Filter.lt,
            "less_than_or_equals": Filter.lte,
            "<=": Filter.lte,
            "contains": Filter.stringContains,
            "starts_with": Filter.stringStartsWith,
        }
        fn = op_map.get(operator.lower())
        if not fn:
            raise ValueError(f"Unsupported operator: {operator}")
        return self.filter(fn(name, value))

    def filterQuality(self) -> Any:
        """Filter images based on quality flag or cloud cover threshold."""
        # By default, retain images with CLOUDY_PIXEL_PERCENTAGE <= 30 or quality_flag == 1
        def is_good(img: Image):
            cloud = img.get("CLOUDY_PIXEL_PERCENTAGE") or img.get("cloud_cover")
            if cloud is not None and cloud > 30.0:
                return False
            qf = img.get("quality_flag")
            if qf is not None and qf == 0:
                return False
            return True

        filtered = [img for img in self._images if is_good(img)]
        return self.__class__(filtered, dataset_name=self.dataset_name, metadata=self._metadata)

    def select(self, bands: Union[str, List[str]]) -> Any:
        """Select specific bands across all images in the collection."""
        return self.__class__([img.select(bands) for img in self._images], dataset_name=self.dataset_name)

    def sort(self, property_name: str, ascending: bool = True) -> Any:
        """Sort images by a metadata property."""
        sorted_imgs = sorted(
            self._images,
            key=lambda img: img.get(property_name, 0),
            reverse=not ascending,
        )
        return self.__class__(sorted_imgs, dataset_name=self.dataset_name, metadata=self._metadata)

    def limit(self, count: int) -> Any:
        """Limit collection to first N images."""
        return self.__class__(self._images[:count], dataset_name=self.dataset_name, metadata=self._metadata)

    def map(self, func: Callable[[Image], Image]) -> Any:
        """Apply a function to every image in collection."""
        return self.__class__([func(img) for img in self._images], dataset_name=self.dataset_name)

    # -------------------------------------------------------------
    # GEE-Style Compositing & Reducers
    # -------------------------------------------------------------
    def _reduce_stack(self, reducer: Reducer) -> Image:
        """Stack compatible images along time axis and apply reducer."""
        if not self._images:
            raise ValueError("Cannot reduce empty ImageCollection")

        ref = self._images[0]
        # Align/resample if dimensions differ
        arrays = []
        for img in self._images:
            if (img.height, img.width) != (ref.height, ref.width):
                # Simple resize match to reference
                resampled = img.resample(ref.width / img.width)
                arrays.append(resampled._data)
            else:
                arrays.append(img._data)

        # Shape: (N, Bands, H, W)
        stack = np.stack(arrays, axis=0)
        # Reduce across N (axis 0)
        reduced_arr = reducer.reduce(stack, axis=0)

        return Image(
            reduced_arr,
            transform=ref.transform,
            crs=ref.crs,
            band_names=ref.bands,
            nodata=ref.nodata,
        )

    def mean(self) -> Image:
        """Calculate pixel-wise mean composite over time."""
        return self._reduce_stack(Reducer.mean())

    def median(self) -> Image:
        """Calculate pixel-wise median composite over time."""
        return self._reduce_stack(Reducer.median())

    def min(self) -> Image:
        """Calculate pixel-wise minimum composite over time."""
        return self._reduce_stack(Reducer.min())

    def max(self) -> Image:
        """Calculate pixel-wise maximum composite over time."""
        return self._reduce_stack(Reducer.max())

    def sum(self) -> Image:
        """Calculate pixel-wise sum composite over time."""
        return self._reduce_stack(Reducer.sum())

    def mosaic(self) -> Image:
        """
        GEE-like mosaic: stitches overlapping images, with later images on top.
        """
        if not self._images:
            raise ValueError("Cannot mosaic empty ImageCollection")

        ref = self._images[0]
        # Composite last valid pixel on top
        arrays = [img._data for img in self._images]
        stack = np.stack(arrays, axis=0)
        # Nan-safe last valid value
        out = np.full_like(ref._data, np.nan)
        for i in range(len(self._images)):
            valid = ~np.isnan(arrays[i])
            out[valid] = arrays[i][valid]

        return Image(out, transform=ref.transform, crs=ref.crs, band_names=ref.bands)

    def qualityMosaic(self, quality_band: str) -> Image:
        """
        Select pixels from images where quality_band is maximum (e.g. greenest pixel NDVI mosaic).
        """
        if not self._images:
            raise ValueError("Cannot perform qualityMosaic on empty collection")

        ref = self._images[0]
        if quality_band not in ref.bands:
            raise KeyError(f"Quality band '{quality_band}' not found in image bands: {ref.bands}")

        q_idx = ref.bands.index(quality_band)
        q_stack = np.stack([img._data[q_idx] for img in self._images], axis=0)  # (N, H, W)
        best_indices = np.nanargmax(q_stack, axis=0)  # (H, W)

        out_data = np.zeros_like(ref._data)
        for b in range(ref.count):
            band_stack = np.stack([img._data[b] for img in self._images], axis=0)
            out_data[b] = np.take_along_axis(band_stack, best_indices[np.newaxis, :, :], axis=0)[0]

        return Image(out_data, transform=ref.transform, crs=ref.crs, band_names=ref.bands)

    def to_xarray(self) -> Any:
        """
        Convert ImageCollection into a 4D xarray.DataArray with dimensions ('time', 'band', 'y', 'x').
        """
        import xarray as xr
        import pandas as pd
        if not self._images:
            raise ValueError("Cannot convert empty ImageCollection to xarray")

        das = []
        times = []
        for i, img in enumerate(self._images):
            da = img.to_xarray()
            t = img.get("date") or img.get("datetime") or img.get("timestamp") or f"T{i}"
            times.append(str(t))
            das.append(da)

        combined = xr.concat(das, dim="time")
        combined = combined.assign_coords(time=times)
        return combined

    def to_dataset(self) -> Any:
        """
        Convert ImageCollection into an xarray.Dataset where each band is a distinct data variable with dimensions ('time', 'y', 'x').
        """
        da = self.to_xarray()
        return da.to_dataset(dim="band")

    def __repr__(self) -> str:
        return f"<geoflow.ImageCollection '{self.dataset_name}' ({len(self)} images)>"

    @classmethod
    def from_stac(
        cls,
        collection: str,
        catalog: str = "planetary-computer",
        geometry: Optional[Union[Geometry, Any]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100,
        bands: Optional[List[str]] = None,
        resolution: Optional[Union[int, float]] = None,
    ) -> ImageCollection:
        """
        Lazily load an ImageCollection from a STAC catalog (Cloud-Native Streaming).
        """
        from geoflow.earthdata.stac import search_stac, load_stac_collection
        
        items = search_stac(
            collection=collection,
            catalog=catalog,
            geometry=geometry,
            start=start,
            end=end,
            limit=limit,
        )
        
        if not items:
            raise ValueError(f"No STAC items found for {collection} in {catalog}")
            
        bbox = geometry.bounds if geometry and hasattr(geometry, "bounds") else None
        
        patch_url = None
        if "planetarycomputer" in catalog:
            try:
                import planetary_computer
                patch_url = planetary_computer.sign_inplace
            except ImportError:
                pass
        
        return load_stac_collection(
            items=items,
            bands=bands,
            resolution=resolution,
            bbox=bbox,
            patch_url=patch_url
        )
    @classmethod
    def from_earthdata(
        cls,
        dataset: str,
        region: Optional[Union[Geometry, Any]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10,
        bands: Optional[List[str]] = None,
    ) -> ImageCollection:
        """
        GEE-like abstraction that securely searches NASA Earthdata (via earthaccess),
        downloads the raw files (HLS/GEDI), and loads them directly into an ImageCollection.
        """
        from geoflow.earthdata.search import search_data
        from geoflow.earthdata.downloader import download
        
        # 1. Search
        results = search_data(
            dataset=dataset,
            region=region,
            start=start,
            end=end,
            limit=limit,
        )
        
        if not results:
            raise ValueError(f"No NASA Earthdata found for {dataset}")
            
        # 2. Download
        downloaded_files = download(results, output_dir="./data")
        
        # 3. Group files by granule/date (heuristic based on filename prefix)
        from collections import defaultdict
        from pathlib import Path
        
        granule_groups = defaultdict(list)
        for f in downloaded_files:
            fp = Path(f)
            # HLS filenames usually start with HLS.L30.T46QCK.2023001T042149...
            # We group by the first 4 parts of the dot-separated name
            parts = fp.name.split(".")
            group_key = ".".join(parts[:4]) if len(parts) >= 4 else fp.stem
            
            # Filter to requested bands if specified
            if bands:
                if not any(b in fp.name for b in bands):
                    continue
            granule_groups[group_key].append(fp)
            
        # 4. Load each group into an Image
        images = []
        for key, files in granule_groups.items():
            if not files:
                continue
            img = Image.from_files(files)
            # Try to set date from key (e.g. 2023001T042149)
            date_str = key.split(".")[-1] if "." in key else key
            img.set("date", date_str)
            images.append(img)
            
        return cls(images, dataset_name=dataset)


RasterCollection = ImageCollection
