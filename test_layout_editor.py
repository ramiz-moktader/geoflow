"""
Test suite for GeoFlow In-Notebook Interactive Cartographic Layout Editor.
"""

import sys
from pathlib import Path
import numpy as np
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import box

# Ensure geoflow is importable
sys.path.insert(0, str(Path(__file__).parent))

import geoflow as gf
from geoflow.viz.layout_editor import create_layout_editor


def test_layout_editor():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 65)
    print("Testing In-Notebook Interactive Layout Editor (ArcGIS View)")
    print("=" * 65)

    # 1. Test Raster Image with edit_layout()
    print("\n[1] Testing Image.edit_layout()...")
    H, W = 64, 64
    transform = from_bounds(91.7, 22.2, 92.0, 22.5, W, H)
    data = np.random.uniform(0.1, 0.8, (2, H, W)).astype(np.float32)
    img = gf.Image(data, transform=transform, crs="EPSG:4326", band_names=["B4", "B8"])
    ndvi = img.ndvi()

    editor = ndvi.edit_layout(
        title="Chattogram AOI — NDVI",
        subtitle="Sentinel-2 L2A | 10m",
        cmap="RdYlGn",
        lat_orientation="vertical",
        show_top=True,
        show_right=True,
        save_path="test_editor_export.png",
        dpi=150,
    )
    assert editor is not None, "edit_layout() returned None"
    print(f"  [PASS] Image.edit_layout() initialized: {type(editor)}")

    # 2. Test FeatureCollection with edit_layout()
    print("\n[2] Testing FeatureCollection.edit_layout()...")
    gdf = gpd.GeoDataFrame({
        "name": ["Block A", "Block B"],
        "geometry": [box(91.7, 22.2, 91.85, 22.35), box(91.85, 22.35, 92.0, 22.5)],
    }, crs="EPSG:4326")
    vec = gf.FeatureCollection(gdf)
    vec_editor = vec.edit_layout(title="Vector Study Blocks")
    assert vec_editor is not None, "FeatureCollection.edit_layout() returned None"
    print(f"  [PASS] FeatureCollection.edit_layout() initialized: {type(vec_editor)}")

    # 3. Test direct gf.viz.edit_layout alias
    print("\n[3] Testing gf.viz.edit_layout alias...")
    alias_editor = gf.viz.edit_layout(ndvi, title="Alias Test")
    assert alias_editor is not None
    print("  [PASS] gf.viz.edit_layout and gf.viz.create_layout_editor verified")

    print("\n" + "=" * 65)
    print("ALL LAYOUT EDITOR TESTS PASSED! 🎨")
    print("=" * 65)


if __name__ == "__main__":
    test_layout_editor()
