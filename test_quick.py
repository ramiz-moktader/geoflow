"""
Quick end-to-end integration test for GeoFlow (gf).
Tests:
1. GEE-like Filter & Reducer
2. Geometry (Point, Polygon, Rectangle, buffer, bounds)
3. Raster & Image (band math, expressions, NDVI, focal filters, clip, reduceRegion, visualize)
4. ImageCollection (filtering, sorting, temporal median composite)
5. Vector & FeatureCollection (spatial ops, sample_raster)
6. Interactive Map (addLayer, centerObject, to_html)
7. GEDI LiDAR adapter
8. Spatial statistics (Global Moran's I, Local Moran's I / LISA clusters)
9. Spatial blocking & Spatial Cross-Validation (SpatialCV, BufferedSpatialCV)
10. Leakage diagnostics (leakage_report)
11. ML PatchDataset & mini-batch dataloader
12. Statistical inference & Spatial Effective Sample Size (ESS)
"""

import sys
from pathlib import Path
import numpy as np

# Ensure geoflow is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

import geoflow as gf
from rasterio.transform import from_bounds

def test_all():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print(f"Testing GeoFlow Framework v{gf.__version__}")
    print("=" * 60)

    # 1. Geometry
    print("\n[1] Testing Geometry engine...")
    rect = gf.Rectangle([91.7, 22.2, 92.0, 22.5], crs="EPSG:4326")
    pt = gf.Point([91.85, 22.35], crs="EPSG:4326")
    buffered = pt.buffer(0.05)
    assert rect.contains(pt), "Rectangle should contain point"
    assert rect.area > 0, "Area should be positive"
    print(f"  [PASS] Geometry created: BBox={rect.bounds}, pt inside={rect.contains(pt)}")

    # 2. Filter & Reducer
    print("\n[2] Testing GEE-like Filter & Reducer...")
    f_eq = gf.Filter.eq("platform", "Sentinel-2A")
    f_date = gf.Filter.date("2023-01-01", "2023-06-01")
    f_combined = f_eq & f_date
    assert f_combined({"platform": "Sentinel-2A", "date": "2023-03-15"}), "Filter should match record"
    assert not f_combined({"platform": "Landsat-8", "date": "2023-03-15"}), "Filter should reject record"
    
    red_mean = gf.Reducer.mean()
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert red_mean.reduce(arr) == 3.0, "Reducer.mean failed"
    print("  [PASS] Filter & Reducer evaluation passed")

    # 3. Raster Image & GEE Processing
    print("\n[3] Testing Image band math, expressions, indices, and focal filters...")
    H, W = 64, 64
    transform = from_bounds(91.7, 22.2, 92.0, 22.5, W, H)
    np.random.seed(42)
    # 4 bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
    data = np.stack([
        np.random.uniform(0.05, 0.15, (H, W)),  # Blue
        np.random.uniform(0.08, 0.20, (H, W)),  # Green
        np.random.uniform(0.05, 0.25, (H, W)),  # Red
        np.random.uniform(0.30, 0.70, (H, W)),  # NIR
    ], axis=0).astype(np.float32)

    img = gf.Image(data, transform=transform, crs="EPSG:4326", band_names=["B2", "B3", "B4", "B8"])
    assert img.count == 4, "Image must have 4 bands"

    # Band math
    img_double = img * 2.0
    assert np.isclose(img_double.data[0, 0, 0], img.data[0, 0, 0] * 2.0), "Band multiply failed"

    # Expression (EVI)
    evi = img.expression("2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))", {
        "NIR": img.select("B8"),
        "RED": img.select("B4"),
        "BLUE": img.select("B2"),
    })
    assert evi.count == 1, "Expression output should have 1 band"

    # Normalized Difference (NDVI)
    ndvi = img.ndvi(nir="B8", red="B4")
    assert ndvi.count == 1
    assert -1.0 <= np.nanmean(ndvi.data) <= 1.0, "NDVI mean must be in [-1, 1]"

    # Focal filter
    focal = img.focal_mean(radius=2)
    assert focal.shape == img.shape, "Focal filter shape mismatch"

    # Zonal reduceRegion
    stats = img.reduceRegion(gf.Reducer.mean(), geometry=rect)
    assert "B8" in stats, "reduceRegion missing B8"

    # Visualize 8-bit RGB
    rgb_vis = img.visualize({"bands": ["B4", "B3", "B2"]})
    assert rgb_vis.shape == (H, W, 3), f"RGB shape must be (64, 64, 3), got {rgb_vis.shape}"
    assert rgb_vis.dtype == np.uint8, "RGB array must be uint8"
    print(f"  [PASS] Image operations passed: NDVI mean={float(np.nanmean(ndvi.data)):.3f}, EVI mean={float(np.nanmean(evi.data)):.3f}")

    # 4. ImageCollection
    print("\n[4] Testing ImageCollection & Temporal Compositing...")
    img2 = img * 1.1
    img2.set("CLOUDY_PIXEL_PERCENTAGE", 5.0).set("date", "2023-02-01")
    img.set("CLOUDY_PIXEL_PERCENTAGE", 15.0).set("date", "2023-03-01")

    col = gf.ImageCollection([img, img2], dataset_name="SENTINEL-2")
    assert len(col) == 2, "Collection size must be 2"

    filtered_col = col.filterBounds(rect).filterQuality().sort("CLOUDY_PIXEL_PERCENTAGE")
    assert len(filtered_col) == 2, "Filtered collection mismatch"

    median_comp = col.median()
    assert median_comp.count == 4, "Median composite must have 4 bands"
    print("  [PASS] ImageCollection filtering and median compositing passed")

    # 5. Vector & FeatureCollection
    print("\n[5] Testing FeatureCollection & Raster Sampling...")
    features = [
        gf.Feature(gf.Point([91.75, 22.25]), {"id": 1, "type": "forest"}),
        gf.Feature(gf.Point([91.85, 22.35]), {"id": 2, "type": "cropland"}),
        gf.Feature(gf.Point([91.95, 22.45]), {"id": 3, "type": "urban"}),
    ]
    fc = gf.FeatureCollection(features)
    assert len(fc) == 3, "FeatureCollection size must be 3"

    sampled_fc = fc.sample_raster(img, bands=["B4", "B8"])
    assert "B4" in sampled_fc.columns and "B8" in sampled_fc.columns, "sample_raster failed to attach columns"
    print(f"  [PASS] FeatureCollection raster sampling passed: sampled B8 values={sampled_fc.gdf['B8'].tolist()}")

    # 6. Interactive Map
    print("\n[6] Testing Interactive Map...")
    m = gf.Map()
    m.centerObject(rect)
    m.addLayer(img, vis_params={"bands": ["B4", "B3", "B2"]}, name="Sentinel-2 RGB")
    m.addLayer(fc, name="Samples")
    html = m.to_html()
    assert "<!DOCTYPE html>" in html, "Map HTML generation failed"
    assert "L.map" in html, "Map Leaflet initialization missing"
    print("  [PASS] Map generation passed: clean standalone HTML produced")

    # 7. GEDI LiDAR Adapter
    print("\n[7] Testing GEDI LiDAR footprint extraction...")
    gedi = gf.GEDI("L2A").filterBounds(rect).filterQuality().select(["rh98", "sensitivity"])
    gedi_gdf = gedi.to_geodataframe()
    assert len(gedi_gdf) > 0, "GEDI footprints empty"
    assert "rh98" in gedi_gdf.columns, "rh98 metric missing"
    print(f"  [PASS] GEDI footprints extracted: {len(gedi_gdf)} shots, mean RH98={gedi_gdf['rh98'].mean():.2f}m")

    # 8. Spatial Statistics & Autocorrelation
    print("\n[8] Testing Spatial Autocorrelation (Moran's I & Local LISA)...")
    moran = gf.spatial.morans_i(gedi_gdf, value="rh98", k=6)
    assert -1.0 <= moran.I <= 1.0, f"Moran's I out of bounds: {moran.I}"

    local_moran = gf.spatial.local_morans_i(gedi_gdf, value="rh98", k=6)
    assert "cluster" in local_moran.columns, "LISA cluster column missing"
    cluster_counts = local_moran["cluster"].value_counts().to_dict()
    print(f"  [PASS] Moran's I={moran.I:.4f} (z={moran.z_score:.2f}, p={moran.p_value:.4f})")
    print(f"  [PASS] LISA Clusters: {cluster_counts}")

    # 9. Spatial Blocking & Cross-Validation
    print("\n[9] Testing Spatial Blocking & BufferedSpatialCV...")
    blocks = gf.spatial.block(rect, size=8000, unit="meters", shape="square")
    assert len(blocks) > 0, "Spatial blocks generation failed"

    cv = gf.validation.SpatialCV(blocks=blocks, n_splits=3)
    splits = list(cv.split(gedi_gdf))
    assert len(splits) == 3, "SpatialCV must generate 3 splits"

    bcv = gf.validation.BufferedSpatialCV(buffer_distance=0.01, blocks=blocks, n_splits=3)
    b_splits = list(bcv.split(gedi_gdf))
    train_idx, val_idx = b_splits[0]
    print(f"  [PASS] Spatial blocks: {len(blocks)} blocks created")
    print(f"  [PASS] BufferedSpatialCV split 0: Train={len(train_idx)}, Val={len(val_idx)}")

    # 10. Leakage Diagnostics
    print("\n[10] Testing ML Leakage Diagnostics...")
    train_gdf = gedi_gdf.iloc[train_idx]
    val_gdf = gedi_gdf.iloc[val_idx]
    leakage = gf.validation.leakage_report(train_gdf, val_gdf, buffer_threshold=0.005)
    print(f"  [PASS] Leakage report: Min separation distance = {leakage.min_distance:.4f} deg")

    # 11. ML PatchDataset
    print("\n[11] Testing ML PatchDataset & DataLoader...")
    dataset = gf.ml.PatchDataset(image=img, patch_size=16, stride=16)
    assert len(dataset) > 0, "Patch dataset is empty"
    first_patch = dataset[0]
    assert first_patch.shape == (4, 16, 16), f"Patch shape mismatch: {first_patch.shape}"
    loader = dataset.dataloader(batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch.shape == (4, 4, 16, 16), f"Batch shape mismatch: {batch.shape}"
    print(f"  [PASS] PatchDataset created {len(dataset)} patches; batch shape={batch.shape}")

    # 12. Statistical Inference & Spatial ESS
    print("\n[12] Testing Statistical Inference & Spatial Effective Sample Size...")
    # Add dummy group
    gedi_gdf["stratum"] = np.where(gedi_gdf["rh98"] > gedi_gdf["rh98"].median(), "High", "Low")
    test_result = gf.stats.test(gedi_gdf, value="rh98", group="stratum")
    assert test_result.p_value <= 1.0, "Invalid p-value"

    ess = gf.stats.effective_sample_size(gedi_gdf, value="rh98")
    print(f"  [PASS] Stat test: {test_result.test_name} p={test_result.p_value:.4e}, effect={test_result.effect_size:.2f}")
    print(f"  [PASS] Spatial ESS: Nominal N={ess['nominal_n']} -> Effective N={ess['effective_n']} (Design Effect={ess['design_effect']:.2f})")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY! 🚀")
    print("=" * 60)

if __name__ == "__main__":
    test_all()
