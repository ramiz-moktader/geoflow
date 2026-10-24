"""
Regression tests for correctness bugs found in the code audit. Each test reproduces the
original failure scenario and asserts the fixed behaviour.
"""

import math
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
import shapely.geometry as sg
from pyproj import Transformer
from rasterio.transform import from_origin

import geoflow as gf
from geoflow.core.crs import reproject_bounds
from geoflow.viz.cartography import draw_scale_bar


# ---------------------------------------------------------------- raster

def test_nodata_is_nan_after_loading(tmp_path):
    path = tmp_path / "nd.tif"
    with rasterio.open(path, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(0, 20, 10, 10), nodata=-9999) as dst:
        dst.write(np.array([[[1, 2], [3, -9999]]], dtype=np.float32))
    img = gf.Image(path)
    assert np.isnan(img.data[0, 1, 1])
    assert img.reduceRegion(gf.Reducer.mean())["B1"] == pytest.approx(2.0)


def test_array_nodata_sentinel_becomes_nan():
    img = gf.Image(np.array([[1.0, 0.0]]), nodata=0)
    assert np.isnan(img.data[0, 0, 1])


def test_from_files_rejects_mismatched_grids(tmp_path):
    def write(name, transform):
        path = tmp_path / name
        with rasterio.open(path, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32",
                           crs="EPSG:32633", transform=transform) as dst:
            dst.write(np.ones((1, 2, 2), dtype=np.float32))
        return path
    a = write("a.B04.tif", from_origin(0, 20, 10, 10))
    b = write("a.B08.tif", from_origin(1000, 20, 10, 10))
    with pytest.raises(ValueError, match="differs"):
        gf.Image.from_files([a, b])


def tile(value, x0, size=10, crs="EPSG:32633"):
    return gf.Image(np.full((1, size, size), value), transform=from_origin(x0, 10, 1, 1), crs=crs)


def test_composite_aligns_tiles_by_location():
    composite = gf.ImageCollection([tile(1.0, 0), tile(100.0, 900000)]).mean()
    assert composite.data[0, 0, 0] == 1.0      # tile A location keeps tile A's value
    assert composite.data[0, 0, -1] == 100.0   # tile B location keeps tile B's value


def test_mosaic_keeps_every_tile():
    mosaic = gf.ImageCollection([tile(1.0, 0), tile(2.0, 10)]).mosaic()
    assert mosaic.width == 20
    assert np.all(mosaic.data[0, :, :10] == 1.0)
    assert np.all(mosaic.data[0, :, 10:] == 2.0)


def test_mosaic_mixed_crs_uses_reference_units():
    utm = tile(1.0, 500000, crs="EPSG:32633")
    b = reproject_bounds(utm.bounds, utm.crs, "EPSG:4326")
    geo = gf.Image(np.full((1, 4, 4), 2.0), transform=from_origin(b[0], b[3], (b[2] - b[0]) / 4, (b[3] - b[1]) / 4), crs="EPSG:4326")
    mosaic = gf.ImageCollection([utm, geo]).mosaic()
    assert mosaic.crs == utm.crs
    assert mosaic.width < 50  # not a union of metres and degrees


def test_quality_mosaic_handles_all_nan_pixels():
    q1 = gf.Image(np.array([[[np.nan, 1.0]]]), band_names=["q"])
    q2 = gf.Image(np.array([[[np.nan, 2.0]]]), band_names=["q"])
    out = gf.ImageCollection([q1, q2]).qualityMosaic("q")
    assert np.isnan(out.data[0, 0, 0])
    assert out.data[0, 0, 1] == 2.0


def test_slope_on_geographic_dem_uses_metres():
    dem = np.tile(np.arange(10, dtype=float), (10, 1))  # +1 m per pixel eastward
    img = gf.Image(dem, transform=from_origin(90, 23, 0.001, 0.001), crs="EPSG:4326")
    m_per_px = 0.001 * 111412.84 * math.cos(math.radians(22.995))
    expected = math.degrees(math.atan(1.0 / m_per_px))
    assert np.nanmean(img.slope().data) == pytest.approx(expected, rel=0.01)


def test_clip_reprojects_geometry():
    img = gf.Image(np.ones((1, 10, 10)), transform=from_origin(500000, 2500000, 100, 100), crs="EPSG:32646")
    footprint = gf.Rectangle(list(reproject_bounds(img.bounds, img.crs, "EPSG:4326")))
    assert not np.isnan(img.clip(footprint).data).any()


def test_focal_mean_ignores_nan_and_keeps_mask():
    data = np.ones((1, 5, 5))
    data[0, 2, 2] = np.nan
    out = gf.Image(data).focal_mean(radius=1)
    assert np.isnan(out.data[0, 2, 2])
    assert np.allclose(out.data[0][~np.isnan(out.data[0])], 1.0)


def test_mask_clouds_qa60_bits():
    qa = np.array([[0, 1 << 10, 1 << 11, np.nan]], dtype=float)
    img = gf.Image(np.stack([np.ones((1, 4)), qa]), band_names=["B4", "QA60"])
    out = img.mask_clouds()
    assert out.data[0, 0, 0] == 1.0
    assert np.isnan(out.data[0, 0, 1:]).all()


# ---------------------------------------------------------------- filters

def test_filter_date_parses_hls_ordinal_dates():
    img = gf.Image(np.ones((1, 2, 2)))
    img.set("date", "2023001T042149")
    col = gf.ImageCollection([img])
    assert len(col.filterDate("2022-12-31", "2023-01-02")) == 1
    assert len(col.filterDate("2023-02-01", "2023-03-01")) == 0


def test_filter_on_columns_named_like_pandas_members():
    gdf = gpd.GeoDataFrame({"name": ["a", "b"], "count": [10, 1]},
                           geometry=[sg.Point(0, 0), sg.Point(1, 1)], crs="EPSG:4326")
    fc = gf.FeatureCollection(gdf)
    assert len(fc.filter(gf.Filter.eq("name", "a"))) == 1
    assert len(fc.filter(gf.Filter.gt("count", 5))) == 1


# ---------------------------------------------------------------- cartography

def test_web_mercator_scale_bar_length():
    tf = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = tf.transform(0, 60)
    x1, y1 = tf.transform(1, 61)
    fig, ax = plt.subplots()
    draw_scale_bar(ax, (x0, y0, x1, y1), crs="EPSG:3857")
    drawn_units = ax.patches[0].get_width() * 2
    label = ax.texts[-1].get_text()
    labelled_m = float(label.split()[0]) * (1000 if "km" in label else 1)
    true_m = drawn_units * math.cos(math.radians(60.5))
    assert true_m == pytest.approx(labelled_m, rel=0.02)


# ---------------------------------------------------------------- spatial / validation / ML

def pairs_overlap(gdf):
    geoms = list(gdf.geometry)
    return sum(geoms[i].intersection(geoms[j]).area > 1e-6
               for i in range(len(geoms)) for j in range(i + 1, len(geoms)))


def test_hexagonal_blocks_tile_without_overlap():
    aoi = gf.Rectangle([0, 0, 10000, 10000], crs="EPSG:32633")
    blocks = gf.spatial.block(aoi, size=2000, shape="hexagonal").gdf
    assert pairs_overlap(blocks) == 0
    assert aoi.shapely.difference(blocks.union_all()).area < 1e-3  # full coverage up to float rounding
    assert blocks.geometry.area.iloc[0] == pytest.approx(2000**2 * math.sqrt(3) / 2)  # hexagon of width 2000


def test_spatial_cv_assigns_outside_samples_to_nearest_block():
    blocks = gpd.GeoDataFrame({"block_id": [0, 1, 2]},
                              geometry=[sg.box(0, 0, 10, 10), sg.box(10, 0, 20, 10), sg.box(20, 0, 30, 10)],
                              crs="EPSG:32633")
    # last sample lies just outside block 2; non-unique index must not matter
    pts = gpd.GeoDataFrame(geometry=[sg.Point(5, 5), sg.Point(15, 5), sg.Point(25, 5), sg.Point(31, 5)],
                           index=[7, 7, 3, 1], crs="EPSG:32633")
    from geoflow.validation.spatial_cv import assign_blocks
    assert list(assign_blocks(pts, blocks)) == [0, 1, 2, 2]


def test_spatial_cv_reprojects_blocks():
    blocks = gf.spatial.block(gf.Rectangle([91.7, 22.2, 92.0, 22.5]), size=8000).gdf.to_crs("EPSG:32646")
    rng = np.random.RandomState(0)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(rng.uniform(91.7, 92.0, 100), rng.uniform(22.2, 22.5, 100)),
                           crs="EPSG:4326")
    folds = list(gf.validation.SpatialCV(blocks=blocks, n_splits=3).split(pts))
    assert all(len(val) > 0 for _, val in folds)


def test_buffered_cv_buffers_in_metres_at_high_latitude():
    lon = np.linspace(10, 12, 40)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, np.full(40, 70.0)), crs="EPSG:4326")
    cv = gf.validation.BufferedSpatialCV(buffer_distance=5000.0, n_splits=4)
    utm = pts.to_crs(pts.estimate_utm_crs())
    for train_idx, val_idx in cv.split(pts):
        if len(train_idx) == 0:
            continue
        d = utm.geometry.iloc[train_idx].apply(lambda g: utm.geometry.iloc[val_idx].distance(g).min())
        assert d.min() >= 5000.0


def test_leakage_report_overlap_alone_is_not_leakage():
    # interleaved checkerboard blocks: hulls overlap, but nearest pairs are far apart
    train = gpd.GeoDataFrame(geometry=[sg.Point(x, y) for x in (0, 2000) for y in (0, 2000)], crs="EPSG:32633")
    test = gpd.GeoDataFrame(geometry=[sg.Point(1000, 1000)], crs="EPSG:32633")
    report = gf.validation.leakage_report(train, test, buffer_threshold=500.0)
    assert report.has_overlap
    assert not report.has_leakage
    assert "NOT CHECKED" in gf.validation.leakage_report(train, test).summary()


def test_patch_split_never_shares_pixels():
    img = gf.Image(np.random.rand(1, 512, 512), transform=from_origin(0, 512, 1, 1), crs="EPSG:32633")
    ds = gf.ml.PatchDataset(img, patch_size=64, stride=32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for strategy in ("spatial", "random"):
            subsets = ds.split(strategy=strategy)
            assert all(len(s) > 0 for s in subsets), strategy
            windows = [{(p["row"], p["col"]) for p in s.patches} for s in subsets]
            for a in range(3):
                for b in range(a + 1, 3):
                    for (r1, c1) in windows[a]:
                        for (r2, c2) in windows[b]:
                            assert abs(r1 - r2) >= 64 or abs(c1 - c2) >= 64, strategy


def test_stratified_sampling_respects_region_without_duplicates():
    strata = gf.Image(np.tile(np.repeat([1.0, 2.0], 10), (20, 1)), transform=from_origin(0, 20, 1, 1), crs="EPSG:32633")
    region = gf.Rectangle([0, 0, 15, 20], crs="EPSG:32633")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = gf.sampling.stratified(region, strata=strata, n_per_stratum=150).gdf
    assert (samples["stratum"] == 1).sum() == 150
    assert (samples["stratum"] == 2).sum() == 100  # only 5 x 20 pixels of class 2 inside the region
    assert not samples.geometry.duplicated().any()
    assert samples.geometry.x.max() < 15


def test_stratified_sampling_does_not_invent_points():
    strata = gf.Image(np.ones((1, 5, 5)), transform=from_origin(0, 5, 1, 1), crs="EPSG:32633")
    with pytest.warns(UserWarning, match="no points"):
        samples = gf.sampling.stratified(gf.Rectangle([100, 100, 110, 110], crs="EPSG:32633"), strata=strata)
    assert len(samples) == 0


# ---------------------------------------------------------------- stats

def test_morans_i_rejects_nan():
    gdf = gpd.GeoDataFrame({"v": [1.0, np.nan, 3.0, 4.0, 5.0]},
                           geometry=gpd.points_from_xy(range(5), range(5)))
    with pytest.raises(ValueError, match="missing values"):
        gf.spatial.morans_i(gdf, "v", k=2)


def test_rank_biserial_sign_matches_cohens_d():
    rng = np.random.RandomState(1)
    gdf = gpd.GeoDataFrame({"v": np.r_[rng.normal(10, 1, 30), rng.normal(0, 1, 30)], "g": ["a"] * 30 + ["b"] * 30},
                           geometry=gpd.points_from_xy(rng.uniform(0, 100, 60), rng.uniform(0, 100, 60)))
    d = gf.stats.test(gdf, "v", "g")
    r = gf.stats.test(gdf, "v", "g", method="mann_whitney")
    assert d.effect_size > 0 and r.effect_size > 0


def test_version_matches_package_metadata():
    import tomllib
    from pathlib import Path
    pyproject = tomllib.loads((Path(__file__).parent / "pyproject.toml").read_text())
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "geoflow.__version__"
