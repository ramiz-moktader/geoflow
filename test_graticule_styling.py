"""
Test suite for advanced cartographic graticule customizations:
- Vertical vs horizontal latitude label orientation
- Displaying coordinates on all four sides (bottom, left, top, right)
- Text font size, font family, font weight, color, and degree precision
"""

import sys
from pathlib import Path
import numpy as np

# Ensure geoflow is importable
sys.path.insert(0, str(Path(__file__).parent))

import geoflow as gf
from rasterio.transform import from_bounds


def test_graticule_customization():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 65)
    print("Testing Graticule Orientation, 4-Sided Display & Typography")
    print("=" * 65)

    H, W = 64, 64
    transform = from_bounds(91.7, 22.2, 92.0, 22.5, W, H)
    np.random.seed(42)
    data = np.stack([
        np.random.uniform(0.05, 0.15, (H, W)),
        np.random.uniform(0.08, 0.20, (H, W)),
        np.random.uniform(0.05, 0.25, (H, W)),
        np.random.uniform(0.35, 0.75, (H, W)),
    ], axis=0).astype(np.float32)

    img = gf.Image(data, transform=transform, crs="EPSG:4326", band_names=["B2", "B3", "B4", "B8"])

    # -------------------------------------------------------------
    # Test 1: Vertical Latitude, 4-Sided Display, Custom Fonts
    # -------------------------------------------------------------
    print("\n[1] Testing 4-Sided Coordinates + Vertical Latitude + Serif Font...")
    out_4sided = Path("test_carto_4sided_vertical.png")

    ax = img.plot_map(
        bands=["B4", "B3", "B2"],
        title="Chattogram Region — 4-Sided Graticule with Vertical Latitudes",
        subtitle="Datum: WGS84 | Custom Cartographic Layout",
        
        # 4-Sided display: top, right, bottom, left
        show_top=True,
        show_right=True,
        show_bottom=True,
        show_left=True,
        
        # Orientations: latitude vertical (90 deg), longitude horizontal (0 deg)
        lat_orientation="vertical",
        lon_orientation="horizontal",
        
        # Typography: custom size, serif font, bold weight, navy blue color
        tick_fontsize=9,
        tick_fontfamily="serif",
        tick_fontweight="bold",
        tick_color="#1e3a8a",
        degree_precision=3,
        
        # Cartographic widgets
        north_arrow=True,
        scale_bar=True,
        grid=True,
        grid_color="#94a3b8",
        grid_alpha=0.6,
        
        save_path=out_4sided,
        dpi=200,
    )

    assert out_4sided.exists() and out_4sided.stat().st_size > 1000, "4-sided map failed to save"
    print(f"  [PASS] 4-Sided Map with vertical lat labels generated: {out_4sided.name} ({out_4sided.stat().st_size / 1024:.1f} KB)")

    # -------------------------------------------------------------
    # Test 2: Single-Band Index (NDVI) with Top & Right coordinates
    # -------------------------------------------------------------
    print("\n[2] Testing NDVI with Top/Right Coordinates + Custom Font Size...")
    out_ndvi_styled = Path("test_carto_ndvi_top_right.png")

    ax2 = img.ndvi().plot_map(
        cmap="RdYlGn",
        vmin=-0.1,
        vmax=0.8,
        title="NDVI with Top & Right Coordinates",
        colorbar_label="NDVI Index",
        
        # Top and right enabled
        show_top=True,
        show_right=True,
        show_bottom=True,
        show_left=True,
        
        lat_orientation="vertical",
        tick_fontsize=8,
        tick_fontfamily="sans-serif",
        degree_precision=2,
        
        north_arrow=True,
        scale_bar=True,
        save_path=out_ndvi_styled,
        dpi=200,
    )

    assert out_ndvi_styled.exists() and out_ndvi_styled.stat().st_size > 1000, "NDVI map failed to save"
    print(f"  [PASS] NDVI Map with top/right labels generated: {out_ndvi_styled.name} ({out_ndvi_styled.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 65)
    print("ALL GRATICULE ORIENTATION & 4-SIDED DISPLAY TESTS PASSED! 🚀")
    print("=" * 65)


if __name__ == "__main__":
    test_graticule_customization()
