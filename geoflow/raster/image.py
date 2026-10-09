"""
GEE-compatible Image and Raster implementation for GeoFlow.
Full support for band math, expressions, indices, focal filters, terrain,
masking, clipping, zonal statistics, reprojection, and visualization.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import math
import numpy as np
import rasterio
from rasterio.transform import from_bounds, Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import geometry_mask
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from geoflow.core.base import EarthObject
from geoflow.core.crs import CRS, check_crs_compatibility
from geoflow.core.reducer import Reducer
from geoflow.geometry.base import Geometry


class Image(EarthObject):
    """GEE-like Image object wrapping multi-band geospatial raster data."""

    def __init__(
        self,
        source: Union[str, Path, np.ndarray, Image],
        transform: Optional[Affine] = None,
        crs: Union[str, int, CRS] = "EPSG:4326",
        band_names: Optional[List[str]] = None,
        nodata: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(metadata=metadata)

        if isinstance(source, Image):
            self._data = source._data.copy()
            self._transform = source._transform
            self._crs = source._crs
            self._band_names = list(source._band_names)
            self._nodata = source._nodata
            return

        if isinstance(source, (str, Path)):
            # Load from raster file
            with rasterio.open(source) as src:
                self._data = src.read().astype(np.float32)
                self._transform = src.transform
                self._crs = CRS(src.crs.to_string() if src.crs else "EPSG:4326")
                self._nodata = src.nodata if src.nodata is not None else np.nan
                descriptions = [d for d in src.descriptions if d]
                if band_names:
                    self._band_names = band_names
                elif descriptions and len(descriptions) == src.count:
                    self._band_names = list(descriptions)
                else:
                    self._band_names = [f"B{i+1}" for i in range(src.count)]
        elif isinstance(source, np.ndarray):
            # Array shape: either (H, W) or (B, H, W)
            arr = source.astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            elif arr.ndim != 3:
                raise ValueError(f"Array must be 2D or 3D, got ndim={arr.ndim}")

            self._data = arr
            self._transform = transform or Affine.identity()
            self._crs = crs if isinstance(crs, CRS) else CRS(crs)
            self._nodata = nodata if nodata is not None else np.nan

            if band_names:
                if len(band_names) != self._data.shape[0]:
                    raise ValueError(f"Band names length ({len(band_names)}) does not match band count ({self._data.shape[0]})")
                self._band_names = list(band_names)
            else:
                self._band_names = [f"B{i+1}" for i in range(self._data.shape[0])]
        else:
            raise TypeError(f"Unsupported source type for Image: {type(source)}")

    @classmethod
    def from_files(cls, filepaths: List[Union[str, Path]], band_names: Optional[List[str]] = None) -> Image:
        """
        Create a multi-band Image by stacking multiple single-band raster files.
        If band_names is not provided, it infers names from the filenames (e.g. 'B04' from 'HLS...B04.tif').
        """
        import rasterio
        
        if not filepaths:
            raise ValueError("Filepaths list cannot be empty.")
            
        # Read the first file to get transform and shape
        with rasterio.open(filepaths[0]) as src:
            transform = src.transform
            crs = src.crs.to_string() if src.crs else "EPSG:4326"
            nodata = src.nodata
            height, width = src.shape
            
        data = np.zeros((len(filepaths), height, width), dtype=np.float32)
        inferred_names = []
        
        for i, fp in enumerate(filepaths):
            with rasterio.open(fp) as src:
                data[i] = src.read(1).astype(np.float32)
                
            # Attempt to infer band name from filename
            name = Path(fp).stem
            # e.g., HLS.L30.T46QCK.2023001T042149.v2.0.B04
            parts = name.split(".")
            inferred_names.append(parts[-1] if len(parts) > 1 else name)
            
        b_names = band_names if band_names else inferred_names
        return cls(data, transform=transform, crs=crs, band_names=b_names, nodata=nodata)

    # -------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._data.shape

    @property
    def bands(self) -> List[str]:
        return list(self._band_names)

    @property
    def count(self) -> int:
        return self._data.shape[0]

    @property
    def height(self) -> int:
        return self._data.shape[1]

    @property
    def width(self) -> int:
        return self._data.shape[2]

    @property
    def transform(self) -> Affine:
        return self._transform

    @property
    def crs(self) -> CRS:
        return self._crs

    @property
    def nodata(self) -> float:
        return self._nodata

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns (minx, miny, maxx, maxy)."""
        minx, maxy = self._transform @ (0, 0)
        maxx, miny = self._transform @ (self.width, self.height)
        return (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))

    @property
    def geometry(self) -> Geometry:
        return Geometry.BBox(*self.bounds, crs=self._crs)

    def _clone(self, data: np.ndarray, band_names: Optional[List[str]] = None) -> Any:
        names = band_names if band_names is not None else self._band_names
        new_obj = self.__class__(
            data,
            transform=self._transform,
            crs=self._crs,
            band_names=names,
            nodata=self._nodata,
            metadata=dict(self._metadata),
        )
        # Propagate custom subclass attributes
        for k, v in self.__dict__.items():
            if k not in ("_data", "_transform", "_crs", "_band_names", "_nodata", "_metadata"):
                new_obj.__dict__[k] = v
        return new_obj

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """NumPy array protocol support (np.asarray(img))."""
        return self._data.astype(dtype) if dtype else self._data

    # -------------------------------------------------------------
    # Band Selection & Renaming
    # -------------------------------------------------------------
    def select(self, bands: Union[str, int, List[Union[str, int]]], new_names: Optional[List[str]] = None) -> Image:
        """Select specific bands by name or 0-indexed integer."""
        if isinstance(bands, (str, int)):
            bands = [bands]

        indices = []
        out_names = []
        for b in bands:
            if isinstance(b, int):
                if 0 <= b < self.count:
                    indices.append(b)
                    out_names.append(self._band_names[b])
                else:
                    raise IndexError(f"Band index {b} out of range [0, {self.count})")
            elif isinstance(b, str):
                if b in self._band_names:
                    idx = self._band_names.index(b)
                    indices.append(idx)
                    out_names.append(b)
                else:
                    raise KeyError(f"Band '{b}' not found. Available: {self._band_names}")

        selected_data = self._data[indices, :, :]
        final_names = new_names if new_names and len(new_names) == len(indices) else out_names
        return self._clone(selected_data, band_names=final_names)

    def rename(self, new_names: List[str]) -> Image:
        if len(new_names) != self.count:
            raise ValueError(f"Expected {self.count} names, got {len(new_names)}")
        return self._clone(self._data.copy(), band_names=new_names)

    # -------------------------------------------------------------
    # Arithmetic & Band Math
    # -------------------------------------------------------------
    def _apply_binary_op(self, other: Any, op_fn: Any) -> Image:
        if isinstance(other, (int, float, np.number)):
            out_data = op_fn(self._data, other)
            return self._clone(out_data)
        elif isinstance(other, Image):
            check_crs_compatibility(self._crs, other._crs, "in band math operation")
            if (self.height, self.width) != (other.height, other.width):
                raise ValueError(f"Image dimensions mismatch: {(self.height, self.width)} vs {(other.height, other.width)}")
            out_data = op_fn(self._data, other._data)
            return self._clone(out_data)
        else:
            return NotImplemented

    def add(self, other: Any) -> Image:
        return self._apply_binary_op(other, np.add)

    def subtract(self, other: Any) -> Image:
        return self._apply_binary_op(other, np.subtract)

    def multiply(self, other: Any) -> Image:
        return self._apply_binary_op(other, np.multiply)

    def divide(self, other: Any) -> Image:
        def safe_div(a, b):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.divide(a, b)
        return self._apply_binary_op(other, safe_div)

    def pow(self, power: float) -> Image:
        return self._clone(np.power(self._data, power))

    def abs(self) -> Image:
        return self._clone(np.abs(self._data))

    def sqrt(self) -> Image:
        with np.errstate(invalid="ignore"):
            return self._clone(np.sqrt(self._data))

    def log(self) -> Image:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._clone(np.log(self._data))

    def log10(self) -> Image:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._clone(np.log10(self._data))

    def exp(self) -> Image:
        return self._clone(np.exp(self._data))

    def sin(self) -> Image:
        return self._clone(np.sin(self._data))

    def cos(self) -> Image:
        return self._clone(np.cos(self._data))

    def tan(self) -> Image:
        return self._clone(np.tan(self._data))

    def floor(self) -> Image:
        return self._clone(np.floor(self._data))

    def ceil(self) -> Image:
        return self._clone(np.ceil(self._data))

    def round(self) -> Image:
        return self._clone(np.round(self._data))

    # Python operator overloading
    def __add__(self, other: Any) -> Image:
        return self.add(other)

    def __radd__(self, other: Any) -> Image:
        return self.add(other)

    def __sub__(self, other: Any) -> Image:
        return self.subtract(other)

    def __rsub__(self, other: Any) -> Image:
        if isinstance(other, (int, float, np.number)):
            return self._clone(other - self._data)
        return NotImplemented

    def __mul__(self, other: Any) -> Image:
        return self.multiply(other)

    def __rmul__(self, other: Any) -> Image:
        return self.multiply(other)

    def __truediv__(self, other: Any) -> Image:
        return self.divide(other)

    def __rtruediv__(self, other: Any) -> Image:
        if isinstance(other, (int, float, np.number)):
            with np.errstate(divide="ignore", invalid="ignore"):
                return self._clone(np.divide(other, self._data))
        return NotImplemented

    def __pow__(self, power: float) -> Image:
        return self.pow(power)

    def __neg__(self) -> Image:
        return self._clone(-self._data)

    def __gt__(self, other: Any) -> Image:
        return self._apply_binary_op(other, lambda a, b: (a > b).astype(np.float32))

    def __gte__(self, other: Any) -> Image:
        return self._apply_binary_op(other, lambda a, b: (a >= b).astype(np.float32))

    def __lt__(self, other: Any) -> Image:
        return self._apply_binary_op(other, lambda a, b: (a < b).astype(np.float32))

    def __lte__(self, other: Any) -> Image:
        return self._apply_binary_op(other, lambda a, b: (a <= b).astype(np.float32))

    def __eq__(self, other: Any) -> Image:
        return self._apply_binary_op(other, lambda a, b: (a == b).astype(np.float32))

    def __ne__(self, other: Any) -> Image:
        return self._apply_binary_op(other, lambda a, b: (a != b).astype(np.float32))

    # -------------------------------------------------------------
    # Expressions & Map Calculations
    # -------------------------------------------------------------
    def expression(self, expr_str: str, band_dict: Optional[Dict[str, Union[Image, float]]] = None) -> Image:
        """
        Evaluate a mathematical expression.
        e.g., image.expression("((NIR - RED) / (NIR + RED))", {"NIR": img.select("B8"), "RED": img.select("B4")})
        """
        import ast

        band_dict = band_dict or {}
        # Also auto-populate available band names from this image
        local_scope = {}
        for b_name in self._band_names:
            idx = self._band_names.index(b_name)
            local_scope[b_name] = self._data[idx]

        for k, v in band_dict.items():
            if isinstance(v, Image):
                local_scope[k] = v._data[0] if v.count == 1 else v._data
            elif isinstance(v, (int, float, np.ndarray)):
                local_scope[k] = v

        # Add math functions
        local_scope["sin"] = np.sin
        local_scope["cos"] = np.cos
        local_scope["tan"] = np.tan
        local_scope["log"] = np.log
        local_scope["exp"] = np.exp
        local_scope["sqrt"] = np.sqrt
        local_scope["abs"] = np.abs

        result_arr = eval(expr_str, {"__builtins__": {}}, local_scope)
        if isinstance(result_arr, (int, float)):
            result_arr = np.full((1, self.height, self.width), result_arr, dtype=np.float32)
        elif isinstance(result_arr, np.ndarray):
            if result_arr.ndim == 2:
                result_arr = result_arr[np.newaxis, :, :]

        return self._clone(result_arr.astype(np.float32), band_names=["expression"])

    # -------------------------------------------------------------
    # Remote Sensing Indices (with auto-detection)
    # -------------------------------------------------------------
    def _find_band(self, options: List[str]) -> str:
        """Helper to auto-detect band names from common aliases."""
        for opt in options:
            if opt in self._band_names:
                return opt
        # Fallback to fuzzy match (e.g. "B04" if "B4" requested)
        for opt in options:
            for b in self._band_names:
                if opt.lower() in b.lower() or opt.replace("B", "B0") == b:
                    return b
        return options[0] # will raise KeyError downstream if not found

    def normalizedDifference(self, bands: List[str]) -> Image:
        """Normalized difference: (band1 - band2) / (band1 + band2)."""
        if len(bands) != 2:
            raise ValueError("normalizedDifference requires exactly 2 band names")
        b1 = self.select(bands[0])._data[0]
        b2 = self.select(bands[1])._data[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = b1 + b2
            nd = np.where(denom != 0, (b1 - b2) / denom, np.nan)
        return self._clone(nd[np.newaxis, :, :], band_names=["nd"])

    def ndvi(self, nir: Optional[str] = None, red: Optional[str] = None) -> Image:
        n = nir or self._find_band(["B8", "B8A", "B08", "NIR"])
        r = red or self._find_band(["B4", "B04", "RED"])
        return self.normalizedDifference([n, r]).rename(["NDVI"])

    def ndwi(self, green: str = "B3", nir: str = "B8") -> Image:
        return self.normalizedDifference([green, nir]).rename(["NDWI"])

    def ndbi(self, swir: str = "B11", nir: str = "B8") -> Image:
        return self.normalizedDifference([swir, nir]).rename(["NDBI"])

    def nbr(self, nir: str = "B8", swir2: str = "B12") -> Image:
        return self.normalizedDifference([nir, swir2]).rename(["NBR"])

    def evi(self, nir: str = "B8", red: str = "B4", blue: str = "B2") -> Image:
        n = self.select(nir)._data[0]
        r = self.select(red)._data[0]
        b = self.select(blue)._data[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = n + 6.0 * r - 7.5 * b + 1.0
            evi_arr = np.where(denom != 0, 2.5 * ((n - r) / denom), np.nan)
        return self._clone(evi_arr[np.newaxis, :, :], band_names=["EVI"])

    def savi(self, nir: str = "B8", red: str = "B4", L: float = 0.5) -> Image:
        n = self.select(nir)._data[0]
        r = self.select(red)._data[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = n + r + L
            savi_arr = np.where(denom != 0, ((n - r) / denom) * (1.0 + L), np.nan)
        return self._clone(savi_arr[np.newaxis, :, :], band_names=["SAVI"])

    # -------------------------------------------------------------
    # Masking, Clipping & Normalization
    # -------------------------------------------------------------
    def normalize(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Image:
        """Min-max normalize image values to [0, 1]."""
        out = self._data.copy()
        for i in range(self.count):
            band = out[i]
            valid = band[~np.isnan(band)]
            b_min = np.nanmin(valid) if min_val is None else min_val
            b_max = np.nanmax(valid) if max_val is None else max_val
            rng = b_max - b_min
            if rng > 0:
                out[i] = np.clip((band - b_min) / rng, 0.0, 1.0)
        return self._clone(out)

    def clamp(self, low: float, high: float) -> Image:
        return self._clone(np.clip(self._data, low, high))

    def clip(self, geometry: Union[Geometry, Any]) -> Image:
        """Clip raster to geometry polygon mask."""
        geom_shapely = geometry.to_shapely() if hasattr(geometry, "to_shapely") else geometry
        # Generate boolean mask where False is inside polygon, True is outside
        mask = geometry_mask(
            [geom_shapely],
            transform=self._transform,
            invert=True,  # True inside geometry
            out_shape=(self.height, self.width),
        )
        out_data = self._data.copy()
        out_data[:, ~mask] = np.nan
        return self._clone(out_data)

    def updateMask(self, mask: Union[Image, np.ndarray]) -> Image:
        """Retain pixels where mask is non-zero / True; others set to NaN."""
        m = mask._data[0] if isinstance(mask, Image) else mask
        if m.ndim == 3:
            m = m[0]
        out_data = self._data.copy()
        out_data[:, m == 0] = np.nan
        return self._clone(out_data)

    def unmask(self, value: float = 0.0) -> Image:
        """Replace nodata / NaN with constant value."""
        out_data = np.nan_to_num(self._data, nan=value)
        return self._clone(out_data)

    def where(self, condition: Union[Image, np.ndarray], value: Union[float, Image]) -> Image:
        """Replace pixel with value where condition is True/non-zero."""
        cond_arr = condition._data[0] if isinstance(condition, Image) else condition
        if cond_arr.ndim == 3:
            cond_arr = cond_arr[0]
        val_arr = value._data if isinstance(value, Image) else value
        out_data = self._data.copy()
        if isinstance(val_arr, np.ndarray):
            out_data[:, cond_arr != 0] = val_arr[:, cond_arr != 0]
        else:
            out_data[:, cond_arr != 0] = val_arr
        return self._clone(out_data)

    def mask(self) -> Image:
        """Return 1-band boolean mask (1 where valid, 0 where NaN)."""
        valid = (~np.isnan(self._data[0])).astype(np.float32)
        return self._clone(valid[np.newaxis, :, :], band_names=["mask"])

    def fill_nodata(self, value: float = 0.0) -> Image:
        return self.unmask(value)

    # -------------------------------------------------------------
    # Reprojection & Resampling
    # -------------------------------------------------------------
    def reproject(self, dst_crs: Union[str, int, CRS], scale: Optional[float] = None, resampling: str = "bilinear") -> Image:
        """Reproject image to new CRS with optional pixel resolution scale."""
        target_crs = dst_crs if isinstance(dst_crs, CRS) else CRS(dst_crs)
        resamp_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }
        resamp = resamp_map.get(resampling.lower(), Resampling.bilinear)

        dst_transform, dst_width, dst_height = calculate_default_transform(
            self._crs.pyproj_crs,
            target_crs.pyproj_crs,
            self.width,
            self.height,
            *self.bounds,
            resolution=scale,
        )

        dst_data = np.full((self.count, dst_height, dst_width), np.nan, dtype=np.float32)

        for i in range(self.count):
            reproject(
                source=self._data[i],
                destination=dst_data[i],
                src_transform=self._transform,
                src_crs=self._crs.to_string(),
                dst_transform=dst_transform,
                dst_crs=target_crs.to_string(),
                resampling=resamp,
                src_nodata=self._nodata,
                dst_nodata=np.nan,
            )

        return Image(
            dst_data,
            transform=dst_transform,
            crs=target_crs,
            band_names=self._band_names,
            nodata=np.nan,
            metadata=dict(self._metadata),
        )

    def resample(self, scale_factor: float = 1.0, method: str = "bilinear") -> Image:
        """Resample image by a scale factor."""
        if scale_factor == 1.0:
            return self
        new_width = max(1, int(self.width * scale_factor))
        new_height = max(1, int(self.height * scale_factor))
        new_transform = self._transform @ Affine.scale(1.0 / scale_factor, 1.0 / scale_factor)

        dst_data = np.full((self.count, new_height, new_width), np.nan, dtype=np.float32)
        resamp = Resampling.bilinear if method == "bilinear" else Resampling.nearest

        for i in range(self.count):
            reproject(
                source=self._data[i],
                destination=dst_data[i],
                src_transform=self._transform,
                src_crs=self._crs.to_string(),
                dst_transform=new_transform,
                dst_crs=self._crs.to_string(),
                resampling=resamp,
            )

        return Image(dst_data, transform=new_transform, crs=self._crs, band_names=self._band_names)

    # -------------------------------------------------------------
    # Focal / Neighborhood Filters
    # -------------------------------------------------------------
    def _get_kernel(self, radius: int, kernel_type: str = "circle") -> np.ndarray:
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        if kernel_type == "circle":
            return (x*x + y*y <= radius*radius).astype(float)
        elif kernel_type == "gaussian":
            sigma = radius / 2.0
            return np.exp(-(x*x + y*y) / (2.0 * sigma * sigma))
        else:  # square
            return np.ones((size, size), dtype=float)

    def focal_mean(self, radius: int = 3, kernel: str = "circle") -> Image:
        k = self._get_kernel(radius, kernel)
        k /= np.sum(k)
        out = np.zeros_like(self._data)
        for i in range(self.count):
            filled = np.nan_to_num(self._data[i], nan=np.nanmean(self._data[i]))
            out[i] = ndi.convolve(filled, k, mode="reflect")
        return self._clone(out)

    def focal_median(self, radius: int = 3) -> Image:
        size = 2 * radius + 1
        out = np.zeros_like(self._data)
        for i in range(self.count):
            filled = np.nan_to_num(self._data[i], nan=0.0)
            out[i] = ndi.median_filter(filled, size=size)
        return self._clone(out)

    def focal_min(self, radius: int = 3) -> Image:
        size = 2 * radius + 1
        out = np.zeros_like(self._data)
        for i in range(self.count):
            filled = np.nan_to_num(self._data[i], nan=np.nanmax(self._data[i]))
            out[i] = ndi.minimum_filter(filled, size=size)
        return self._clone(out)

    def focal_max(self, radius: int = 3) -> Image:
        size = 2 * radius + 1
        out = np.zeros_like(self._data)
        for i in range(self.count):
            filled = np.nan_to_num(self._data[i], nan=np.nanmin(self._data[i]))
            out[i] = ndi.maximum_filter(filled, size=size)
        return self._clone(out)

    def focal_std(self, radius: int = 3) -> Image:
        mean = self.focal_mean(radius)
        mean_sq = (self * self).focal_mean(radius)
        variance = (mean_sq - (mean * mean)).clamp(0, float("inf"))
        return variance.sqrt().rename([f"{b}_std" for b in self._band_names])

    # -------------------------------------------------------------
    # Terrain Analysis (Elevation DEM)
    # -------------------------------------------------------------
    def slope(self) -> Image:
        """Calculate terrain slope in degrees."""
        dem = self._data[0]
        # pixel sizes
        dx = abs(self._transform.a)
        dy = abs(self._transform.e)
        gy, gx = np.gradient(dem, dy, dx)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_deg = np.degrees(slope_rad)
        return self._clone(slope_deg[np.newaxis, :, :], band_names=["slope"])

    def aspect(self) -> Image:
        """Calculate terrain aspect in degrees (0 to 360 clockwise from North)."""
        dem = self._data[0]
        dx = abs(self._transform.a)
        dy = abs(self._transform.e)
        gy, gx = np.gradient(dem, dy, dx)
        aspect_rad = np.arctan2(-gx, gy)
        aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
        return self._clone(aspect_deg[np.newaxis, :, :], band_names=["aspect"])

    def hillshade(self, azimuth: float = 315.0, elevation: float = 45.0) -> Image:
        """Compute hillshade illumination."""
        dem = self._data[0]
        dx = abs(self._transform.a)
        dy = abs(self._transform.e)
        gy, gx = np.gradient(dem, dy, dx)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        aspect_rad = np.arctan2(-gx, gy)

        az_rad = np.radians(azimuth)
        alt_rad = np.radians(elevation)

        shaded = (
            np.sin(alt_rad) * np.cos(slope_rad)
            + np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
        )
        hs = np.clip(255.0 * np.maximum(0.0, shaded), 0.0, 255.0)
        return self._clone(hs[np.newaxis, :, :], band_names=["hillshade"])

    # -------------------------------------------------------------
    # Zonal Statistics
    # -------------------------------------------------------------
    def reduceRegion(self, reducer: Reducer, geometry: Optional[Geometry] = None, scale: Optional[float] = None) -> Dict[str, Any]:
        """Aggregate image band values within a spatial geometry region."""
        img = self.clip(geometry) if geometry else self
        results = {}
        for i, b_name in enumerate(img._band_names):
            band_vals = img._data[i]
            val = reducer.reduce(band_vals)
            results[b_name] = float(val) if np.isscalar(val) else val
        return results

    # -------------------------------------------------------------
    # Visualization & Plotting
    # -------------------------------------------------------------
    def visualize(self, vis_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        GEE-like visualization rendering: returns 8-bit RGB array (H, W, 3).
        vis_params supports:
            - bands: List[str] e.g. ["B4", "B3", "B2"]
            - min: float or List[float]
            - max: float or List[float]
            - gamma: float
            - palette: List[str] or str colormap for single-band
        """
        vis_params = vis_params or {}
        bands = vis_params.get("bands", self._band_names[:3] if self.count >= 3 else [self._band_names[0]])
        if isinstance(bands, str):
            bands = [bands]

        vmin = vis_params.get("min", None)
        vmax = vis_params.get("max", None)
        gamma = vis_params.get("gamma", 1.0)
        palette = vis_params.get("palette", None)

        if len(bands) >= 3:
            # Multi-band RGB
            r = self.select(bands[0])._data[0]
            g = self.select(bands[1])._data[0]
            b = self.select(bands[2])._data[0]

            def scale_band(arr, mn, mx):
                val_min = np.nanpercentile(arr, 2) if mn is None else mn
                val_max = np.nanpercentile(arr, 98) if mx is None else mx
                scaled = np.clip((arr - val_min) / max(1e-5, (val_max - val_min)), 0.0, 1.0)
                if gamma != 1.0:
                    scaled = np.power(scaled, 1.0 / gamma)
                return (scaled * 255).astype(np.uint8)

            r_scaled = scale_band(r, vmin if not isinstance(vmin, list) else vmin[0], vmax if not isinstance(vmax, list) else vmax[0])
            g_scaled = scale_band(g, vmin if not isinstance(vmin, list) else vmin[1], vmax if not isinstance(vmax, list) else vmax[1])
            b_scaled = scale_band(b, vmin if not isinstance(vmin, list) else vmin[2], vmax if not isinstance(vmax, list) else vmax[2])
            return np.dstack([r_scaled, g_scaled, b_scaled])

        else:
            # Single-band with Palette / Colormap
            single = self.select(bands[0])._data[0]
            val_min = np.nanpercentile(single, 2) if vmin is None else vmin
            val_max = np.nanpercentile(single, 98) if vmax is None else vmax
            scaled = np.clip((single - val_min) / max(1e-5, (val_max - val_min)), 0.0, 1.0)
            if gamma != 1.0:
                scaled = np.power(scaled, 1.0 / gamma)

            if isinstance(palette, str):
                cmap = plt.get_cmap(palette)
                rgba = cmap(scaled)
                return (rgba[:, :, :3] * 255).astype(np.uint8)
            elif isinstance(palette, list):
                from matplotlib.colors import LinearSegmentedColormap
                cmap = LinearSegmentedColormap.from_list("custom_pal", palette)
                rgba = cmap(scaled)
                return (rgba[:, :, :3] * 255).astype(np.uint8)
            else:
                # Default grayscale
                gray = (scaled * 255).astype(np.uint8)
                return np.dstack([gray, gray, gray])

    def plot(
        self,
        bands: Optional[List[str]] = None,
        stretch: str = "percentile",
        percentile: Tuple[float, float] = (2, 98),
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
        ax: Any = None,
        title: Optional[str] = None,
    ):
        """Plot image using Matplotlib."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        bands_to_plot = bands or (self._band_names[:3] if self.count >= 3 else [self._band_names[0]])
        if isinstance(bands_to_plot, str):
            bands_to_plot = [bands_to_plot]

        if len(bands_to_plot) >= 3:
            rgb_arr = self.visualize({"bands": bands_to_plot})
            im = ax.imshow(rgb_arr, extent=[self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]])
            ax.set_title(title or f"RGB ({', '.join(bands_to_plot[:3])})")
        else:
            single = self.select(bands_to_plot[0])._data[0]
            mn = np.nanpercentile(single, percentile[0]) if (vmin is None and stretch == "percentile") else vmin
            mx = np.nanpercentile(single, percentile[1]) if (vmax is None and stretch == "percentile") else vmax
            im = ax.imshow(
                single,
                extent=[self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]],
                cmap=cmap,
                vmin=mn,
                vmax=mx,
            )
            ax.set_title(title or f"Band: {bands_to_plot[0]}")
            if colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xlabel(f"Longitude ({self._crs.to_string()})")
        ax.set_ylabel(f"Latitude ({self._crs.to_string()})")
        return ax

    def plot_carto_map(self, **kwargs) -> Any:
        """
        Render a publication-ready cartographic map complete with North Arrow,
        Scale Bar, degree-formatted Lat/Long ticks, and Legend.
        Delegates directly to geoflow.viz.cartography.plot_carto_map.
        """
        from geoflow.viz.cartography import plot_carto_map
        kwargs["image_or_gdf"] = self
        return plot_carto_map(**kwargs)

    def rgb(self, bands: Optional[List[str]] = None) -> Image:
        """Convenience to select RGB bands."""
        b = bands or ["B4", "B3", "B2"]
        return self.select(b)

    def false_color(self, bands: Optional[List[str]] = None) -> Image:
        """Convenience to select False Color NIR/Red/Green."""
        b = bands or ["B8", "B4", "B3"]
        return self.select(b)

    def hist(self, bands: Optional[List[str]] = None, bins: int = 50, ax: Any = None):
        """Plot spectral band histograms."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        target_bands = bands or self._band_names
        if isinstance(target_bands, str):
            target_bands = [target_bands]

        for b in target_bands:
            arr = self.select(b)._data[0]
            valid = arr[~np.isnan(arr)]
            ax.hist(valid.flatten(), bins=bins, alpha=0.5, label=b)

        ax.set_title("Band Distribution Histogram")
        ax.set_xlabel("Reflectance / Value")
        ax.set_ylabel("Pixel Count")
        ax.legend()
        return ax

    # -------------------------------------------------------------
    # I/O & Export
    # -------------------------------------------------------------
    def save(self, filepath: Union[str, Path], driver: str = "GTiff", compress: str = "lzw"):
        """Save image to raster file (e.g. GeoTIFF)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            filepath,
            "w",
            driver=driver,
            height=self.height,
            width=self.width,
            count=self.count,
            dtype=self._data.dtype,
            crs=self._crs.to_string(),
            transform=self._transform,
            nodata=self._nodata,
            compress=compress,
        ) as dst:
            dst.write(self._data)
            dst.descriptions = tuple(self._band_names)

    def to_numpy(self) -> np.ndarray:
        return self._data.copy()

    def to_xarray(self, name: Optional[str] = None) -> Any:
        """
        Convert to xarray.DataArray with spatial coordinates and rioxarray CRS attributes.
        """
        import xarray as xr
        xs = np.array([self._transform.c + (c + 0.5) * self._transform.a for c in range(self.width)])
        ys = np.array([self._transform.f + (r + 0.5) * self._transform.e for r in range(self.height)])

        coords = {
            "band": list(self._band_names),
            "y": ys,
            "x": xs,
        }
        da = xr.DataArray(
            self._data,
            dims=("band", "y", "x"),
            coords=coords,
            name=name or "raster",
            attrs={
                "crs": self._crs.to_string(),
                "transform": tuple(self._transform)[:6],
                "nodata": float(self._nodata) if not np.isnan(self._nodata) else None,
            },
        )
        try:
            da.rio.write_crs(self._crs.to_string(), inplace=True)
            da.rio.write_transform(self._transform, inplace=True)
            if self._nodata is not None and not np.isnan(self._nodata):
                da.rio.write_nodata(self._nodata, inplace=True)
        except Exception:
            pass
        return da

    @classmethod
    def from_xarray(cls, da: Any) -> Image:
        """
        Create a geoflow.Image from an xarray.DataArray.
        """
        arr = da.values.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

        # Extract band names
        if "band" in da.coords:
            band_names = [str(b) for b in da.coords["band"].values]
        else:
            band_names = [f"B{i+1}" for i in range(arr.shape[0])]

        crs_str = "EPSG:4326"
        transform = None
        nodata = np.nan

        if hasattr(da, "rio"):
            try:
                if da.rio.crs:
                    crs_str = da.rio.crs.to_string()
                transform = da.rio.transform()
                if da.rio.nodata is not None:
                    nodata = float(da.rio.nodata)
            except Exception:
                pass

        if transform is None and "transform" in da.attrs:
            transform = Affine(*da.attrs["transform"][:6])
        if "crs" in da.attrs and crs_str == "EPSG:4326":
            crs_str = da.attrs["crs"]

        # Derive transform from coordinate vectors if still missing
        if transform is None and "x" in da.coords and "y" in da.coords:
            xs = da.coords["x"].values
            ys = da.coords["y"].values
            dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
            dy = float(ys[1] - ys[0]) if len(ys) > 1 else -1.0
            minx = float(xs[0] - dx / 2.0)
            maxy = float(ys[0] - dy / 2.0) if dy < 0 else float(ys[-1] + dy / 2.0)
            transform = Affine.translation(minx, maxy) @ Affine.scale(dx, dy)

        return cls(arr, transform=transform, crs=crs_str, band_names=band_names, nodata=nodata)

    def plot_map(self, **kwargs: Any) -> Any:
        """
        Render a publication-ready cartographic map complete with
        North Arrow, dynamic Scale Bar, degree-formatted Lat/Long ticks, and Legend.
        """
        from geoflow.viz.cartography import plot_carto_map
        return plot_carto_map(self, **kwargs)

    def __repr__(self) -> str:
        return f"<geoflow.Image ({self.count} bands: {', '.join(self._band_names)}) [H:{self.height}, W:{self.width}] CRS:{self._crs}>"


# Alias for Image
Raster = Image
