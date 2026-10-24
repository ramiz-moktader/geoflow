"""
Tests for the GEDI HDF5 reader using small synthetic L2A and L2B granules.
"""

import warnings

import h5py
import numpy as np
import pytest

import geoflow as gf

# delta_time is seconds since the GEDI epoch 2018-01-01T00:00:00Z
DAY = 86400.0
T_2020_01_01 = 730 * DAY  # 2018 (365) + 2019 (365) days


def write_l2a(path, n=20):
    lats = np.linspace(22.20, 22.49, n)
    lons = np.linspace(91.70, 91.99, n)
    with h5py.File(path, "w") as h5:
        for beam in ("BEAM0000", "BEAM0101"):
            g = h5.create_group(beam)
            g["lat_lowestmode"] = lats
            g["lon_lowestmode"] = lons
            g["shot_number"] = np.arange(n, dtype=np.uint64) + (0 if beam == "BEAM0000" else 1000)
            g["quality_flag"] = np.array([1] * (n - 2) + [0, 0], dtype=np.uint8)
            g["degrade_flag"] = np.zeros(n, dtype=np.uint8)
            g["sensitivity"] = np.full(n, 0.95)
            g["elev_lowestmode"] = np.full(n, 10.0)
            g["delta_time"] = T_2020_01_01 + np.arange(n) * DAY  # one shot per day from 2020-01-01
            rh = np.tile(np.linspace(0, 30, 101), (n, 1))
            g["rh"] = rh


def write_l2b(path, n=10):
    with h5py.File(path, "w") as h5:
        g = h5.create_group("BEAM0000")
        geo = g.create_group("geolocation")
        geo["lat_lowestmode"] = np.linspace(22.2, 22.4, n)
        geo["lon_lowestmode"] = np.linspace(91.7, 91.9, n)
        geo["shot_number"] = np.arange(n, dtype=np.uint64)
        geo["degrade_flag"] = np.zeros(n, dtype=np.uint8)
        geo["delta_time"] = T_2020_01_01 + np.arange(n) * DAY
        g["l2b_quality_flag"] = np.ones(n, dtype=np.uint8)
        g["sensitivity"] = np.full(n, 0.95)
        g["rh100"] = np.full(n, 2500, dtype=np.int16)  # centimetres


def test_l2a_reads_all_shots(tmp_path):
    f = tmp_path / "l2a.h5"
    write_l2a(f)
    gdf = gf.GEDI("L2A", files=[str(f)]).select(["rh98", "rh100", "shot_number"]).to_geodataframe()
    assert len(gdf) == 40
    assert gdf.crs.to_epsg() == 4326
    assert np.allclose(gdf["rh100"], 30.0)
    assert np.allclose(gdf["rh98"], 30.0 * 0.98)


def test_l2a_quality_and_bounds_filters(tmp_path):
    f = tmp_path / "l2a.h5"
    write_l2a(f)
    gedi = gf.GEDI("L2A", files=[str(f)])
    assert len(gedi.filterQuality().to_geodataframe()) == 36  # 2 low-quality shots per beam dropped

    aoi = gf.Rectangle([91.70, 22.20, 91.80, 22.30])
    gdf = gedi.filterBounds(aoi).to_geodataframe()
    assert len(gdf) > 0
    assert gdf["longitude"].max() <= 91.80 and gdf["latitude"].max() <= 22.30


def test_l2a_date_filter_is_applied(tmp_path):
    f = tmp_path / "l2a.h5"
    write_l2a(f)
    gdf = gf.GEDI("L2A", files=[str(f)]).filterDate("2020-01-03", "2020-01-05").to_geodataframe()
    # Shots on Jan 3, 4 and 5 (end date inclusive), for each of the two beams
    assert len(gdf) == 6


def test_l2b_geolocation_layout_and_units(tmp_path):
    f = tmp_path / "l2b.h5"
    write_l2b(f)
    gdf = gf.GEDI("L2B", files=[str(f)]).select(["rh100"]).filterQuality().to_geodataframe()
    assert len(gdf) == 10
    assert np.allclose(gdf["rh100"], 25.0)  # converted from cm to m


def test_no_matching_shots_returns_empty_frame(tmp_path):
    f = tmp_path / "l2a.h5"
    write_l2a(f)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gdf = gf.GEDI("L2A", files=[str(f)]).filterBounds(gf.Rectangle([0, 0, 1, 1])).to_geodataframe()
    assert gdf.empty
    assert any("No GEDI shots" in str(w.message) for w in caught)


def test_unreadable_files_raise(tmp_path):
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not an hdf5 file")
    with pytest.raises(RuntimeError, match="Could not read any GEDI file"):
        gf.GEDI("L2A", files=[str(bad)]).to_geodataframe()


def _fake_search_and_download(monkeypatch, tmp_path, calls):
    """Replace NASA search/download with fakes that write one synthetic L2A granule."""
    import sys
    import geoflow.earthdata  # noqa: F401  (the package re-exports search(), shadowing the module name)
    search_mod = sys.modules["geoflow.earthdata.search"]
    dl_mod = sys.modules["geoflow.earthdata.downloader"]

    def fake_search(dataset, region=None, start=None, end=None, limit=10, **kw):
        calls.append({"dataset": dataset, "region": region, "start": start, "end": end, "limit": limit})
        return [{"urls": ["https://example.invalid/g.h5"]}]

    def fake_download(results, output_dir="./data"):
        path = tmp_path / "g.h5"
        write_l2a(path)
        return [path]

    monkeypatch.setattr(search_mod, "search_data", fake_search)
    monkeypatch.setattr(dl_mod, "download", fake_download)


def test_load_on_filtered_object_uses_its_filters(monkeypatch, tmp_path):
    calls = []
    _fake_search_and_download(monkeypatch, tmp_path, calls)
    aoi = gf.Rectangle([91.70, 22.20, 91.85, 22.35])
    gedi = (
        gf.GEDI("L2B")
        .filterBounds(aoi)
        .filterDate("2020-01-01", "2020-01-05")
        .filterQuality()
        .select(["rh98"])
        .load(limit=1)
    )
    assert calls == [{"dataset": "GEDI/L2B", "region": aoi, "start": "2020-01-01", "end": "2020-01-05", "limit": 1}]
    assert gedi.files and gedi._quality_filter_active and gedi._selected_variables == ["rh98"]


def test_classmethod_load_still_works(monkeypatch, tmp_path):
    calls = []
    _fake_search_and_download(monkeypatch, tmp_path, calls)
    aoi = gf.Rectangle([91.70, 22.20, 91.85, 22.35])
    gedi = gf.GEDI.load("L2A", region=aoi, start="2020-01-01", end="2020-01-05", limit=1)
    assert calls[0]["dataset"] == "GEDI/L2A" and calls[0]["region"] is aoi
    assert len(gedi.to_geodataframe()) > 0


def test_load_without_region_refuses_global_download(monkeypatch, tmp_path):
    calls = []
    _fake_search_and_download(monkeypatch, tmp_path, calls)
    with pytest.raises(ValueError, match="needs a region"):
        gf.GEDI("L2A").filterQuality().load()
    with pytest.raises(ValueError, match="needs a region"):
        gf.GEDI.load("L2A")
    assert calls == []
