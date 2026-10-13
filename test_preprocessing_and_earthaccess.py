"""
Unit tests for data preprocessing and Earthdata access facilities in GeoFlow:
- Image comparison operators (__ge__, __le__, __gte__, __lte__)
- Spectral indices with fuzzy band detection (ndvi, ndwi, ndbi, nbr, evi, savi)
- Radiometric auto-scaling (auto_scale) and QA preservation
- Cloud, shadow, and cirrus masking (mask_clouds) across Fmask, SCL, and QA_PIXEL
- Subclass persistence in reproject and resample via _clone
- ImageCollection string guard and multi-tile mosaic alignment
- Stratified sampling with raster Image and GeoDataFrame strata
- BufferedSpatialCV geographic CRS meter-to-degree handling
- Leakage diagnostic convex hull overlap
- Earthdata S3 credentials and search_collections utilities
"""

import pytest
import numpy as np
import geopandas as gpd
import shapely.geometry
from affine import Affine

import geoflow as gf
from geoflow.raster.image import Image
from geoflow.raster.collection import ImageCollection
from geoflow.geometry.base import Geometry
from geoflow.core.crs import CRS


class CustomResearchRaster(Image):
    """Custom subclass to verify persistence across transformations."""
    def __init__(self, *args, sensor_model: str = "SensorX", **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_model = sensor_model


def create_synthetic_image(band_names, data_shape=(4, 20, 20), crs="EPSG:32631", fill_value=1000.0):
    data = np.full(data_shape, fill_value, dtype=np.float32)
    transform = Affine.translation(500000.0, 4000000.0) * Affine.scale(10.0, -10.0)
    return Image(data, transform=transform, crs=crs, band_names=band_names)


def test_comparison_operators():
    img = create_synthetic_image(["B1"], data_shape=(1, 5, 5), fill_value=10.0)
    
    # Standard python comparison operators
    gt_res = img > 5.0
    ge_res = img >= 10.0
    lt_res = img < 15.0
    le_res = img <= 10.0
    
    assert np.all(gt_res.data == 1.0)
    assert np.all(ge_res.data == 1.0)
    assert np.all(lt_res.data == 1.0)
    assert np.all(le_res.data == 1.0)
    
    # Backwards compatibility aliases
    assert np.all((img.__gte__(10.0)).data == 1.0)
    assert np.all((img.__lte__(10.0)).data == 1.0)


def test_spectral_indices_fuzzy_matching():
    # Test with standard HLS/Sentinel-2 2-digit band names: B02, B03, B04, B08, B11, B12
    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
    data = np.zeros((6, 10, 10), dtype=np.float32)
    # NIR (B08) = 0.8, RED (B04) = 0.2, GREEN (B03) = 0.3, BLUE (B02) = 0.1, SWIR (B11) = 0.4, SWIR2 (B12) = 0.5
    data[0] = 0.1  # B02
    data[1] = 0.3  # B03
    data[2] = 0.2  # B04
    data[3] = 0.8  # B08
    data[4] = 0.4  # B11
    data[5] = 0.5  # B12

    img = Image(data, transform=Affine.identity(), crs="EPSG:4326", band_names=bands)

    ndvi = img.ndvi()
    assert ndvi.bands == ["NDVI"]
    # (0.8 - 0.2) / (0.8 + 0.2) = 0.6
    np.testing.assert_allclose(ndvi.data[0, 0, 0], 0.6, rtol=1e-4)

    ndwi = img.ndwi()
    assert ndwi.bands == ["NDWI"]
    # (0.3 - 0.8) / (0.3 + 0.8) = -0.4545
    np.testing.assert_allclose(ndwi.data[0, 0, 0], (0.3 - 0.8) / (0.3 + 0.8), rtol=1e-4)

    ndbi = img.ndbi()
    assert ndbi.bands == ["NDBI"]
    # (0.4 - 0.8) / (0.4 + 0.8) = -0.3333
    np.testing.assert_allclose(ndbi.data[0, 0, 0], (0.4 - 0.8) / (0.4 + 0.8), rtol=1e-4)

    nbr = img.nbr()
    assert nbr.bands == ["NBR"]
    # (0.8 - 0.5) / (0.8 + 0.5) = 0.2307
    np.testing.assert_allclose(nbr.data[0, 0, 0], (0.8 - 0.5) / (0.8 + 0.5), rtol=1e-4)

    evi = img.evi()
    assert evi.bands == ["EVI"]
    assert not np.isnan(evi.data[0, 0, 0])

    savi = img.savi()
    assert savi.bands == ["SAVI"]
    assert not np.isnan(savi.data[0, 0, 0])


def test_auto_scale_and_qa_preservation():
    bands = ["B2", "B3", "B4", "Fmask"]
    data = np.full((4, 10, 10), 5000.0, dtype=np.float32)
    data[3] = 2.0  # QA band value
    img = Image(data, transform=Affine.identity(), crs="EPSG:4326", band_names=bands)

    scaled = img.auto_scale(dataset="HLS")
    # Reflectance scaled by 0.0001 -> 0.5
    np.testing.assert_allclose(scaled.data[0, 0, 0], 0.5, rtol=1e-4)
    np.testing.assert_allclose(scaled.data[1, 0, 0], 0.5, rtol=1e-4)
    np.testing.assert_allclose(scaled.data[2, 0, 0], 0.5, rtol=1e-4)
    # QA band should NOT be scaled
    np.testing.assert_allclose(scaled.data[3, 0, 0], 2.0, rtol=1e-4)


def test_mask_clouds_fmask():
    bands = ["B2", "B3", "B4", "Fmask"]
    data = np.full((4, 4, 4), 100.0, dtype=np.float32)
    # Set Fmask: pixel (0,0) is clear (0), pixel (0,1) is cloud (bit 1 = 2)
    data[3, 0, 0] = 0.0
    data[3, 0, 1] = 2.0  # cloud
    data[3, 1, 0] = 4.0  # shadow (bit 2 = 4)
    data[3, 1, 1] = 0.0  # clear

    img = Image(data, transform=Affine.identity(), crs="EPSG:4326", band_names=bands)
    masked = img.mask_clouds()

    assert not np.isnan(masked.data[0, 0, 0])
    assert np.isnan(masked.data[0, 0, 1])  # cloudy pixel masked to NaN
    assert np.isnan(masked.data[0, 1, 0])  # shadow pixel masked to NaN
    assert not np.isnan(masked.data[0, 1, 1])


def test_subclass_persistence_in_transformations():
    data = np.ones((1, 10, 10), dtype=np.float32)
    transform = Affine.translation(100.0, 200.0) * Affine.scale(10.0, -10.0)
    custom_img = CustomResearchRaster(data, transform=transform, crs="EPSG:32631", sensor_model="ModelAlpha")

    reproj = custom_img.reproject(dst_crs="EPSG:4326")
    assert isinstance(reproj, CustomResearchRaster)
    assert reproj.sensor_model == "ModelAlpha"

    resamp = custom_img.resample(scale_factor=0.5)
    assert isinstance(resamp, CustomResearchRaster)
    assert resamp.sensor_model == "ModelAlpha"


def test_image_collection_string_validation():
    with pytest.raises(TypeError, match="expects a list of Image objects, but received a string"):
        ImageCollection("SENTINEL-2")


def test_image_collection_mask_and_mosaic():
    bands = ["B2", "Fmask"]
    d1 = np.full((2, 5, 5), 10.0, dtype=np.float32)
    d1[1, 0, 0] = 2.0  # cloud
    d2 = np.full((2, 5, 5), 20.0, dtype=np.float32)
    d2[1, 0, 0] = 0.0  # clear in second image

    t = Affine.identity()
    img1 = Image(d1, transform=t, crs="EPSG:4326", band_names=bands)
    img2 = Image(d2, transform=t, crs="EPSG:4326", band_names=bands)

    col = ImageCollection([img1, img2])
    masked_col = col.mask_clouds()
    
    # First image has NaN at (0,0), second image has 20.0 at (0,0)
    mosaic = masked_col.mosaic()
    assert mosaic.data[0, 0, 0] == 20.0  # Valid pixel from img2 used in mosaic


def test_stratified_sampling_with_image_strata():
    # 20x20 raster with 2 distinct strata classes (1 and 2)
    strata_data = np.zeros((1, 20, 20), dtype=np.float32)
    strata_data[0, :, :10] = 1.0
    strata_data[0, :, 10:] = 2.0
    t = Affine.translation(0.0, 20.0) * Affine.scale(1.0, -1.0)
    strata_img = Image(strata_data, transform=t, crs="EPSG:4326")
    region = Geometry.BBox(0.0, 0.0, 20.0, 20.0, crs="EPSG:4326")

    samples = gf.sampling.stratified(region=region, strata=strata_img, n_per_stratum=10)
    gdf = samples.gdf
    assert len(gdf) == 20
    assert set(gdf["stratum"].unique()) == {1, 2}


def test_buffered_spatial_cv_geographic_crs():
    # Points in EPSG:4326
    points = [shapely.geometry.Point(x, y) for x in np.linspace(0, 1, 10) for y in np.linspace(0, 1, 10)]
    gdf = gpd.GeoDataFrame({"geometry": points, "val": np.arange(len(points))}, crs="EPSG:4326")

    # buffer_distance = 1000 meters; should be converted to approx degrees without wiping train set
    cv = gf.validation.BufferedSpatialCV(buffer_distance=1000.0, n_splits=3)
    train_idx, val_idx = next(cv.split(gdf))
    
    assert len(train_idx) > 0, "Training set was unexpectedly wiped by buffer"
    assert len(val_idx) > 0


def test_leakage_report_convex_hull():
    # Disjoint diagonal clusters: Box envelopes overlap, but convex hulls do NOT overlap
    # Cluster 1: [0, 2] x [8, 10] (NW)
    pts1 = [shapely.geometry.Point(x, y) for x in [0.0, 2.0] for y in [8.0, 10.0]]
    # Cluster 2: [8, 10] x [0, 2] (SE)
    pts2 = [shapely.geometry.Point(x, y) for x in [8.0, 10.0] for y in [0.0, 2.0]]

    gdf1 = gpd.GeoDataFrame({"geometry": pts1}, crs="EPSG:4326")
    gdf2 = gpd.GeoDataFrame({"geometry": pts2}, crs="EPSG:4326")

    report = gf.validation.leakage_report(gdf1, gdf2, buffer_threshold=0.0)
    assert not report.has_overlap, "Disjoint diagonal clusters should not report geometry overlap"
    assert not report.has_leakage


def test_earthdata_search_collections():
    cols = gf.earthdata.search_collections(keyword="GEDI", limit=2)
    assert isinstance(cols, list)
    if cols:
        assert "title" in cols[0]
