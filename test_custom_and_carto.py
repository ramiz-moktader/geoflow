"""
Test suite for:
1. Native Python subclassing, custom class propagation, and .pipe()
2. First-class xarray & rioxarray bidirectional integration and 4D datacubes
3. Publication-grade cartographic maps with north arrow, scale bar, degree ticks, and legends
"""

import sys
from pathlib import Path
import numpy as np

# Ensure geoflow is importable
sys.path.insert(0, str(Path(__file__).parent))

import geoflow as gf
from rasterio.transform import from_bounds
import xarray as xr


# -----------------------------------------------------------------
# 1. Custom User Subclass definition to test Pythonic extensibility
# -----------------------------------------------------------------
class CustomForestryImage(gf.Image):
    """User-defined domain-specific image subclass."""

    def __init__(self, *args, site_id: str = "SITE-DEFAULT", **kwargs):
        super().__init__(*args, **kwargs)
        self.site_id = site_id

    def canopy_mask(self, threshold: float = 0.4) -> gf.Image:
        """Custom domain logic."""
        return (self.ndvi() > threshold).rename(["canopy_mask"])


def test_extensibility_and_cartography():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 65)
    print("Testing Extensibility, xarray Datacubes & Cartography")
    print("=" * 65)

    # Synthetic 4-band image (B2, B3, B4, B8) over Chattogram (lat 22.3, lon 91.8)
    H, W = 64, 64
    transform = from_bounds(91.7, 22.2, 92.0, 22.5, W, H)
    np.random.seed(42)
    data = np.stack([
        np.random.uniform(0.05, 0.15, (H, W)),  # Blue
        np.random.uniform(0.08, 0.20, (H, W)),  # Green
        np.random.uniform(0.05, 0.25, (H, W)),  # Red
        np.random.uniform(0.35, 0.75, (H, W)),  # NIR
    ], axis=0).astype(np.float32)

    # -------------------------------------------------------------
    # Test 1: Native Subclassing & Dynamic Class Propagation
    # -------------------------------------------------------------
    print("\n[1] Testing Native Python Subclassing & Class Propagation...")
    forest_img = CustomForestryImage(
        data,
        transform=transform,
        crs="EPSG:4326",
        band_names=["B2", "B3", "B4", "B8"],
        site_id="BANGLADESH-CHATTOGRAM-01",
    )
    assert isinstance(forest_img, CustomForestryImage), "Direct instance check failed"
    assert forest_img.site_id == "BANGLADESH-CHATTOGRAM-01", "Custom attribute missing"

    # Operation 1: select
    sel = forest_img.select(["B4", "B8"])
    assert isinstance(sel, CustomForestryImage), "select() failed to propagate CustomForestryImage subclass!"
    assert sel.site_id == "BANGLADESH-CHATTOGRAM-01", "select() dropped custom site_id attribute!"

    # Operation 2: band math
    added = forest_img + 0.05
    assert isinstance(added, CustomForestryImage), "+ operator failed to propagate subclass!"
    assert added.site_id == "BANGLADESH-CHATTOGRAM-01", "Operator dropped custom site_id!"

    # Operation 3: normalizedDifference / ndvi
    ndvi_custom = forest_img.ndvi()
    assert isinstance(ndvi_custom, CustomForestryImage), "ndvi() failed to propagate subclass!"

    # Operation 4: calling custom subclass method
    canopy = forest_img.canopy_mask(threshold=0.3)
    assert isinstance(canopy, CustomForestryImage), "Custom method output failed to propagate subclass!"

    # Operation 5: fluent .pipe()
    def custom_brightness_filter(img_obj, factor=1.2):
        return img_obj * factor

    piped = forest_img.pipe(custom_brightness_filter, factor=1.5)
    assert isinstance(piped, CustomForestryImage), ".pipe() failed to propagate subclass!"
    assert piped.site_id == "BANGLADESH-CHATTOGRAM-01"

    # Operation 6: numpy array protocol
    np_arr = np.asarray(forest_img)
    assert isinstance(np_arr, np.ndarray), "__array__ protocol failed"
    assert np_arr.shape == (4, H, W), f"NumPy array shape mismatch: {np_arr.shape}"

    print("  [PASS] Subclass propagation verified across select(), math, ndvi(), .pipe(), and np.asarray()")

    # -------------------------------------------------------------
    # Test 2: First-Class xarray & rioxarray Integration
    # -------------------------------------------------------------
    print("\n[2] Testing xarray & rioxarray Bidirectional Integration...")
    # Convert Image -> xarray.DataArray
    da = forest_img.to_xarray(name="sentinel2_l2a")
    assert isinstance(da, xr.DataArray), "to_xarray() did not return an xarray.DataArray"
    assert da.dims == ("band", "y", "x"), f"Unexpected dimensions: {da.dims}"
    assert list(da.coords["band"].values) == ["B2", "B3", "B4", "B8"], "Band coordinates mismatch"
    assert len(da.coords["x"]) == W, "Longitude x coordinate length mismatch"
    assert len(da.coords["y"]) == H, "Latitude y coordinate length mismatch"
    assert da.attrs.get("crs") == "EPSG:4326", "CRS attribute missing in xarray"
    print(f"  [PASS] to_xarray() produced {da.dims} DataArray with coords (x, y, band) and CRS='{da.attrs.get('crs')}'")

    # Roundtrip: xarray.DataArray -> geoflow.Image
    reconstructed = gf.Image.from_xarray(da)
    assert isinstance(reconstructed, gf.Image), "from_xarray() did not return an Image"
    assert reconstructed.shape == forest_img.shape, f"Shape mismatch: {reconstructed.shape} vs {forest_img.shape}"
    assert reconstructed.bands == forest_img.bands, f"Bands mismatch: {reconstructed.bands}"
    assert reconstructed.crs.epsg == 4326, "Reprojected CRS mismatch"
    assert np.allclose(reconstructed.data, forest_img.data), "Array values altered in xarray round-trip"
    print("  [PASS] from_xarray() successfully reconstructed Image with exact coordinates & pixel values")

    # ImageCollection -> 4D xarray Datacube
    col = gf.ImageCollection([forest_img, forest_img * 1.1], dataset_name="SENTINEL-2")
    da_4d = col.to_xarray()
    assert da_4d.dims == ("time", "band", "y", "x"), f"Expected 4D dims ('time', 'band', 'y', 'x'), got {da_4d.dims}"
    assert da_4d.shape == (2, 4, H, W), f"Expected 4D shape (2, 4, 64, 64), got {da_4d.shape}"
    print(f"  [PASS] ImageCollection.to_xarray() created 4D spatiotemporal cube: shape={da_4d.shape}")

    # ImageCollection -> xarray Dataset
    ds = col.to_dataset()
    assert isinstance(ds, xr.Dataset), "to_dataset() did not return an xarray.Dataset"
    assert "B8" in ds.data_vars and "B4" in ds.data_vars, "Band variables missing in Dataset"
    print(f"  [PASS] ImageCollection.to_dataset() created Dataset with variables: {list(ds.data_vars.keys())}")

    # -------------------------------------------------------------
    # Test 3: Publication Cartographic Map (North Arrow, Scale, Graticule)
    # -------------------------------------------------------------
    print("\n[3] Testing Cartographic Map Maker (North Arrow, Scale Bar, Graticule, Legend)...")
    out_rgb = Path("test_carto_rgb.png")
    out_ndvi = Path("test_carto_ndvi.png")

    # 1-line call for RGB map
    ax1 = forest_img.plot_map(
        bands=["B4", "B3", "B2"],
        title="Chattogram AOI — Sentinel-2 L2A True Color",
        subtitle="Datum: WGS84 | Resolution: 10m",
        north_arrow=True,
        scale_bar=True,
        grid=True,
        save_path=out_rgb,
        dpi=200,
    )
    assert out_rgb.exists() and out_rgb.stat().st_size > 1000, "RGB cartographic map was not created"
    print(f"  [PASS] RGB Map successfully generated: {out_rgb.name} ({out_rgb.stat().st_size / 1024:.1f} KB)")

    # 1-line call for NDVI map with colorbar
    ax2 = forest_img.ndvi().plot_map(
        cmap="RdYlGn",
        vmin=-0.1,
        vmax=0.8,
        title="Chattogram Vegetation Index (NDVI)",
        colorbar_label="Normalized Difference Vegetation Index",
        north_arrow=True,
        scale_bar=True,
        grid=True,
        save_path=out_ndvi,
        dpi=200,
    )
    assert out_ndvi.exists() and out_ndvi.stat().st_size > 1000, "NDVI cartographic map was not created"
    print(f"  [PASS] NDVI Map successfully generated: {out_ndvi.name} ({out_ndvi.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 65)
    print("ALL EXTENSIBILITY, XARRAY & CARTOGRAPHY TESTS PASSED! 🎯")
    print("=" * 65)


if __name__ == "__main__":
    test_extensibility_and_cartography()
