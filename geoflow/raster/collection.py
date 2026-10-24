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

    def __init__(self, images: Optional[Union[List[Image], Any]] = None, dataset_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if isinstance(images, (str, bytes)):
            raise TypeError(
                f"ImageCollection expects a list of Image objects, but received a string {images!r}. "
                f"To load collections by dataset name or STAC, use `ImageCollection.from_earthdata(...)` "
                f"or `ImageCollection.from_stac(...)`."
            )
        super().__init__(metadata=metadata)
        self.dataset_name = dataset_name or "CustomCollection"
        self._images: List[Image] = list(images) if images is not None else []

    def mask_clouds(self, qa_band: Optional[str] = None, method: str = "auto", keep_clear: bool = True) -> Any:
        """Apply pixel QA cloud masking to every Image in the collection."""
        return self.map(lambda img: img.mask_clouds(qa_band=qa_band, method=method, keep_clear=keep_clear))

    def auto_scale(
        self,
        scale: Optional[float] = None,
        offset: Optional[float] = None,
        dataset: Optional[str] = None,
        exclude_bands: Optional[List[str]] = None,
    ) -> Any:
        """Apply radiometric scaling and offset to every Image in the collection."""
        return self.map(lambda img: img.auto_scale(scale=scale, offset=offset, dataset=dataset, exclude_bands=exclude_bands))

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
    def _common_grid(self):
        """
        Return (transform, width, height) of the grid shared by all images.
        If every image already sits on the first image's grid, that grid is reused; otherwise the
        grid covers the union of all image extents in the first image's CRS at its resolution.
        """
        from affine import Affine
        from geoflow.core.crs import reproject_bounds

        ref = self._images[0]
        if all(img.shape[1:] == ref.shape[1:] and img.transform == ref.transform and img.crs == ref.crs
               for img in self._images):
            return ref.transform, ref.width, ref.height

        all_bounds = [reproject_bounds(img.bounds, img.crs, ref.crs) for img in self._images]
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        res_x = abs(ref.transform.a)
        res_y = abs(ref.transform.e)
        width = max(1, int(np.ceil((max_x - min_x) / res_x - 1e-9)))
        height = max(1, int(np.ceil((max_y - min_y) / res_y - 1e-9)))
        return Affine(res_x, 0.0, min_x, 0.0, -res_y, max_y), width, height

    def _aligned_arrays(self, resampling: str = "nearest"):
        """Warp every image onto the common grid. Returns (list of (B, H, W) arrays, transform)."""
        from rasterio.warp import reproject, Resampling

        ref = self._images[0]
        dst_transform, width, height = self._common_grid()
        resamp = getattr(Resampling, resampling)
        arrays = []
        for img in self._images:
            if img.shape[1:] == (height, width) and img.transform == dst_transform and img.crs == ref.crs:
                arrays.append(img._data)
                continue
            # Each image gets its own fresh destination, so no warp overwrites another image's pixels
            out = np.full((img.count, height, width), np.nan, dtype=np.float32)
            for b in range(img.count):
                reproject(
                    source=img._data[b],
                    destination=out[b],
                    src_transform=img.transform,
                    src_crs=img.crs.to_string(),
                    dst_transform=dst_transform,
                    dst_crs=ref.crs.to_string(),
                    resampling=resamp,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )
            arrays.append(out)
        return arrays, dst_transform

    def _reduce_stack(self, reducer: Reducer) -> Image:
        """Align images on a common grid, stack them along time and apply the reducer."""
        if not self._images:
            raise ValueError("Cannot reduce empty ImageCollection")

        ref = self._images[0]
        if any(img.count != ref.count for img in self._images):
            raise ValueError(
                f"Cannot composite images with different band counts: {sorted({img.count for img in self._images})}. "
                "Use .select([...]) to choose common bands first."
            )
        arrays, dst_transform = self._aligned_arrays()

        # Shape: (N, Bands, H, W)
        stack = np.stack(arrays, axis=0)
        # Reduce across N (axis 0); pixels with no valid observation stay NaN
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reduced_arr = reducer.reduce(stack, axis=0)

        return ref._clone(
            np.asarray(reduced_arr, dtype=np.float32),
            transform=dst_transform,
            crs=ref.crs,
            band_names=ref.bands,
            nodata=np.nan,
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
        GEE-like mosaic: stitches overlapping images across space, with later images on top.
        Seamlessly handles both identical grids and differing spatial extents.
        """
        if not self._images:
            raise ValueError("Cannot mosaic empty ImageCollection")

        ref = self._images[0]
        arrays, dst_transform = self._aligned_arrays()
        dst_data = np.full((ref.count,) + arrays[0].shape[1:], np.nan, dtype=np.float32)
        for arr in arrays:
            n = min(ref.count, arr.shape[0])
            valid = ~np.isnan(arr[:n])
            dst_data[:n][valid] = arr[:n][valid]

        return ref._clone(
            dst_data,
            transform=dst_transform,
            crs=ref.crs,
            band_names=ref.bands,
            nodata=np.nan,
        )

    def qualityMosaic(self, quality_band: str) -> Image:
        """
        Select pixels from images where quality_band is maximum (e.g. greenest pixel NDVI mosaic).
        """
        if not self._images:
            raise ValueError("Cannot perform qualityMosaic on empty collection")

        ref = self._images[0]
        if quality_band not in ref.bands:
            raise KeyError(f"Quality band '{quality_band}' not found in image bands: {ref.bands}")

        if any(img.bands != ref.bands for img in self._images):
            raise ValueError("qualityMosaic requires every image to have the same bands in the same order.")

        q_idx = ref.bands.index(quality_band)
        arrays, dst_transform = self._aligned_arrays()
        stack = np.stack(arrays, axis=0)  # (N, B, H, W)
        q_stack = stack[:, q_idx]  # (N, H, W)
        no_data = np.all(np.isnan(q_stack), axis=0)  # pixels with no valid quality value in any image
        best_indices = np.argmax(np.where(np.isnan(q_stack), -np.inf, q_stack), axis=0)  # (H, W)

        out_data = np.take_along_axis(stack, best_indices[np.newaxis, np.newaxis, :, :], axis=0)[0]
        out_data[:, no_data] = np.nan

        return ref._clone(out_data, transform=dst_transform, crs=ref.crs, band_names=ref.bands, nodata=np.nan)

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
            
        from geoflow.earthdata.stac import _wgs84_bbox
        bbox = _wgs84_bbox(geometry)
        
        from geoflow.earthdata.stac import CATALOGS
        patch_url = None
        if "planetarycomputer" in CATALOGS.get(catalog, catalog):
            # Planetary Computer assets need SAS-signed URLs; unsigned reads fail with HTTP 409
            try:
                import planetary_computer
            except ImportError:
                raise ImportError(
                    "Reading from Microsoft Planetary Computer requires the 'planetary-computer' package "
                    "to sign asset URLs. Install it with: pip install \"geoflow[stac]\""
                ) from None
            patch_url = planetary_computer.sign_inplace
        
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
        cloud_cover: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> ImageCollection:
        """
        GEE-like abstraction that securely searches NASA Earthdata (via earthaccess),
        downloads the raw files (HLS/GEDI), and loads them directly into an ImageCollection.
        Supports advanced earthaccess kwargs (e.g. cloud_cover=(0, 20)).
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
            cloud_cover=cloud_cover,
            **kwargs
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
