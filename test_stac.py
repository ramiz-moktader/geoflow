"""
Live STAC streaming test against Microsoft Planetary Computer.
Needs network access and the optional `stac` extra; deselect with `pytest -m "not network"`.
"""

import pytest

pytest.importorskip("pystac_client")
pytest.importorskip("odc.stac")

import geoflow as gf


@pytest.mark.network
def test_from_stac_sentinel2():
    aoi = gf.Rectangle([91.7, 22.2, 91.8, 22.3])  # very small area
    collection = gf.ImageCollection.from_stac(
        collection="sentinel-2-l2a",
        catalog="planetary-computer",
        geometry=aoi,
        start="2023-01-01",
        end="2023-01-10",
        limit=5,
        bands=["B04", "B03", "B02"],
    )
    assert len(collection) > 0
    for img in collection:
        assert img.bands == ["B04", "B03", "B02"]
        assert img.get("stac_item_ids")
