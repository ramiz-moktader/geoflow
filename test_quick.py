"""
Quick end-to-end integration tests for GeoFlow (gf), one test per subsystem:
1. Geometry (Point, Polygon, Rectangle, buffer, bounds)
2. GEE-like Filter & Reducer
3. Raster & Image (band math, expressions, NDVI, focal filters, clip, reduceRegion, visualize)
4. ImageCollection (filtering, sorting, temporal median composite)
5. Vector & FeatureCollection (sample_raster)
6. Interactive Map (addLayer, centerObject, to_html)
7. Spatial statistics (Global Moran's I, Local Moran's I / LISA clusters)
8. Spatial blocking & Spatial Cross-Validation (SpatialCV, BufferedSpatialCV)
9. Leakage diagnostics (leakage_report)
10. ML PatchDataset & mini-batch dataloader
11. Statistical inference & Spatial Effective Sample Size (ESS)
GEDI file reading is covered separately in test_gedi.py.
"""

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_bounds

import geoflow as gf


@pytest.fixture
def rect():
    return gf.Rectangle([91.7, 22.2, 92.0, 22.5], crs="EPSG:4326")


@pytest.fixture
def img():
    H, W = 64, 64
    transform = from_bounds(91.7, 22.2, 92.0, 22.5, W, H)
    rng = np.random.RandomState(42)
    # 4 bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
    data = np.stack([
        rng.uniform(0.05, 0.15, (H, W)),
        rng.uniform(0.08, 0.20, (H, W)),
        rng.uniform(0.05, 0.25, (H, W)),
        rng.uniform(0.30, 0.70, (H, W)),
    ], axis=0).astype(np.float32)
    return gf.Image(data, transform=transform, crs="EPSG:4326", band_names=["B2", "B3", "B4", "B8"])


@pytest.fixture
def shots():
    """Synthetic GEDI-like footprints with a spatially smooth canopy height (rh98) field."""
    rng = np.random.RandomState(0)
    lon = rng.uniform(91.7, 92.0, 300)
    lat = rng.uniform(22.2, 22.5, 300)
    rh98 = 20 + 10 * np.sin((lon - 91.7) * 20) + rng.normal(0, 1.0, 300)
    return gpd.GeoDataFrame({"rh98": rh98}, geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")


def test_geometry(rect):
    pt = gf.Point([91.85, 22.35], crs="EPSG:4326")
    assert rect.contains(pt)
    assert rect.area > 0
    assert pt.buffer(0.05).contains(pt)


def test_filter_and_reducer():
    f_combined = gf.Filter.eq("platform", "Sentinel-2A") & gf.Filter.date("2023-01-01", "2023-06-01")
    assert f_combined({"platform": "Sentinel-2A", "date": "2023-03-15"})
    assert not f_combined({"platform": "Landsat-8", "date": "2023-03-15"})
    assert gf.Reducer.mean().reduce(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 3.0


def test_image_processing(img, rect):
    assert img.count == 4
    assert np.isclose((img * 2.0).data[0, 0, 0], img.data[0, 0, 0] * 2.0)

    evi = img.expression("2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))", {
        "NIR": img.select("B8"),
        "RED": img.select("B4"),
        "BLUE": img.select("B2"),
    })
    assert evi.count == 1

    ndvi = img.ndvi(nir="B8", red="B4")
    assert ndvi.count == 1
    assert -1.0 <= np.nanmean(ndvi.data) <= 1.0

    assert img.focal_mean(radius=2).shape == img.shape
    assert "B8" in img.reduceRegion(gf.Reducer.mean(), geometry=rect)

    rgb_vis = img.visualize({"bands": ["B4", "B3", "B2"]})
    assert rgb_vis.shape == (64, 64, 3)
    assert rgb_vis.dtype == np.uint8


def test_image_collection(img, rect):
    img2 = img * 1.1
    img2.set("CLOUDY_PIXEL_PERCENTAGE", 5.0).set("date", "2023-02-01")
    img.set("CLOUDY_PIXEL_PERCENTAGE", 15.0).set("date", "2023-03-01")

    col = gf.ImageCollection([img, img2], dataset_name="SENTINEL-2")
    assert len(col) == 2
    assert len(col.filterBounds(rect).filterQuality().sort("CLOUDY_PIXEL_PERCENTAGE")) == 2
    assert len(col.filterDate("2023-02-15", "2023-03-31")) == 1
    assert col.median().count == 4


def test_feature_collection_sampling(img):
    fc = gf.FeatureCollection([
        gf.Feature(gf.Point([91.75, 22.25]), {"id": 1, "type": "forest"}),
        gf.Feature(gf.Point([91.85, 22.35]), {"id": 2, "type": "cropland"}),
        gf.Feature(gf.Point([91.95, 22.45]), {"id": 3, "type": "urban"}),
    ])
    assert len(fc) == 3
    sampled = fc.sample_raster(img, bands=["B4", "B8"])
    assert "B4" in sampled.columns and "B8" in sampled.columns
    assert not np.isnan(sampled.gdf["B8"]).any()


def test_map(img, rect):
    fc = gf.FeatureCollection([gf.Feature(gf.Point([91.75, 22.25]), {"id": 1})])
    m = gf.Map()
    m.centerObject(rect)
    m.addLayer(img, vis_params={"bands": ["B4", "B3", "B2"]}, name="Sentinel-2 RGB")
    m.addLayer(fc, name="Samples")
    html = m.to_html()
    assert "<!DOCTYPE html>" in html
    assert "L.map" in html


def test_spatial_statistics(shots):
    moran = gf.spatial.morans_i(shots, value="rh98", k=6)
    assert -1.0 <= moran.I <= 1.0
    assert moran.I > 0.3 and moran.p_value < 0.05  # the synthetic field is strongly autocorrelated

    lisa = gf.spatial.local_morans_i(shots, value="rh98", k=6)
    assert "cluster" in lisa.columns
    assert {"HH", "LL"} & set(lisa["cluster"])


def test_spatial_blocking_and_cv(rect, shots):
    blocks = gf.spatial.block(rect, size=8000, unit="meters", shape="square")
    assert len(blocks) > 0

    splits = list(gf.validation.SpatialCV(blocks=blocks, n_splits=3).split(shots))
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(train_idx) + len(val_idx) == len(shots)

    bcv = gf.validation.BufferedSpatialCV(buffer_distance=2000.0, blocks=blocks, n_splits=3)
    for train_idx, val_idx in bcv.split(shots):
        report = gf.validation.leakage_report(shots.iloc[train_idx], shots.iloc[val_idx], buffer_threshold=2000.0)
        assert not report.buffer_violation


def test_leakage_report(shots):
    train, test = shots.iloc[:200], shots.iloc[200:]
    report = gf.validation.leakage_report(train, test, buffer_threshold=500.0)
    assert report.min_distance < 500.0  # random split: neighbours are close
    assert report.has_leakage

    dup = gf.validation.leakage_report(train, shots.iloc[:5])
    assert dup.duplicate_count == 5 and dup.has_leakage


def test_patch_dataset(img):
    dataset = gf.ml.PatchDataset(image=img, patch_size=16, stride=16)
    assert len(dataset) > 0
    assert dataset[0].shape == (4, 16, 16)
    batch = next(iter(dataset.dataloader(batch_size=4, shuffle=False)))
    assert tuple(batch.shape) == (4, 4, 16, 16)


def test_inference_and_ess(shots):
    shots = shots.copy()
    shots["stratum"] = np.where(shots["rh98"] > shots["rh98"].median(), "High", "Low")
    result = gf.stats.test(shots, value="rh98", group="stratum")
    assert 0.0 <= result.p_value <= 1.0
    assert result.effect_size != 0

    ess = gf.stats.effective_sample_size(shots, value="rh98")
    assert ess["nominal_n"] == 300
    assert ess["effective_n"] < ess["nominal_n"]
