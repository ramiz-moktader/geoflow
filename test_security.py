"""
Security regression tests: safe band-math expressions, HTML escaping in Map, and
Earthdata credential scoping (credentials only ever sent to urs.earthdata.nasa.gov).
"""

import geopandas as gpd
import numpy as np
import pytest
import requests
import shapely.geometry

import geoflow as gf
from geoflow.earthdata.downloader import EarthdataSession, URS_HOST


# ---------------------------------------------------------------- expressions

def make_image():
    return gf.Image(np.array([[[1.0, 2.0], [3.0, 4.0]]]), band_names=["NIR"])


def test_expression_band_math_still_works():
    img = make_image()
    out = img.expression("(NIR - 1) / (NIR + 1) if NIR > 1 else -1")
    assert np.allclose(out.data[0], [[-1, 1 / 3], [0.5, 0.6]])
    assert np.allclose(img.expression("sqrt(NIR) * 2").data[0], 2 * np.sqrt(img.data[0]))
    assert np.allclose(img.expression("RED * 2", {"RED": 5.0}).data, 10.0)


@pytest.mark.parametrize("expr", [
    "().__class__.__base__.__subclasses__()",
    "NIR.__class__",
    "__import__('os').system('echo hi')",
    "open('x')",
    "[NIR][0]",
    "lambda: 1",
])
def test_expression_rejects_non_math_syntax(expr):
    with pytest.raises((ValueError, NameError)):
        make_image().expression(expr)


def test_image_truthiness_is_ambiguous():
    a, b = make_image(), make_image() + 1
    with pytest.raises(TypeError):
        bool(a == b)
    assert a in {a}  # identity hashing still works


# ---------------------------------------------------------------- Map HTML

def test_map_html_escapes_feature_properties_and_legends():
    payload = "</script><script>alert(1)</script>"
    fc = gf.FeatureCollection(gpd.GeoDataFrame({"n": [payload]}, geometry=[shapely.geometry.Point(0, 0)], crs="EPSG:4326"))
    m = gf.Map().addLayer(fc, name=payload).add_legend(title=payload, colors=["red"], labels=[payload])
    html = m.to_html()
    assert payload not in html
    assert "<script>alert(1)" not in html
    assert "OpenStreetMap" in gf.Map(basemap="OpenStreetMap").to_html()  # attribution present


# ---------------------------------------------------------------- credentials

class FakeAdapter(requests.adapters.BaseAdapter):
    """data host -> 302 to URS -> 302 back to data host -> 200; records Authorization per request."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def send(self, request, **kwargs):
        host = requests.utils.urlparse(request.url).hostname
        self.seen.append((host, request.headers.get("Authorization")))
        resp = requests.Response()
        resp.request = request
        resp.url = request.url
        if host == "data.example.org" and "code=" not in request.url:
            resp.status_code, resp.headers["Location"] = 302, f"https://{URS_HOST}/oauth/authorize"
        elif host == URS_HOST:
            resp.status_code, resp.headers["Location"] = 302, "https://data.example.org/file.h5?code=abc"
        else:
            resp.status_code = 200
            resp._content = b"data"
        return resp

    def close(self):
        pass


def test_credentials_only_sent_to_urs():
    session = EarthdataSession("user", "secret")
    session.trust_env = False  # ignore any real ~/.netrc on the test machine
    adapter = FakeAdapter()
    session.mount("https://", adapter)

    r = session.get("https://data.example.org/file.h5")
    assert r.status_code == 200
    auth_by_host = {}
    for host, auth in adapter.seen:
        auth_by_host.setdefault(host, []).append(auth)
    assert all(a is not None for a in auth_by_host[URS_HOST])
    assert all(a is None for a in auth_by_host["data.example.org"])

    adapter.seen.clear()
    session.get("https://other-host.example.com/doc.html")
    assert adapter.seen == [("other-host.example.com", None)]


def test_download_stops_after_first_login_failure(tmp_path, monkeypatch):
    """A 401 on the first file must not fire off parallel requests for the rest."""
    import geoflow.earthdata.downloader as dl

    seen = []

    def fake_download_file(url, dest, resume=True, timeout=60, session=None):
        seen.append(url)
        raise PermissionError("NASA Earthdata refused access (HTTP 401)")

    monkeypatch.setattr(dl, "_download_file", fake_download_file)
    urls = [f"https://data.example.invalid/f{i}.h5" for i in range(5)]
    with pytest.raises(PermissionError, match="401"):
        dl.download(urls, output_dir=tmp_path)
    assert seen == urls[:1]


def test_download_fetches_first_file_before_the_rest(tmp_path, monkeypatch):
    import threading
    import geoflow.earthdata.downloader as dl

    order = []
    first_done = threading.Event()

    def fake_download_file(url, dest, resume=True, timeout=60, session=None):
        if url.endswith("f0.h5"):
            first_done.set()
        else:
            assert first_done.is_set(), "parallel download started before the first one finished"
        order.append(url)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(dl, "_download_file", fake_download_file)
    urls = [f"https://data.example.invalid/f{i}.h5" for i in range(4)]
    paths = dl.download(urls, output_dir=tmp_path)
    assert order[0] == urls[0] and len(paths) == 4


def test_login_force_prompts_even_with_saved_credentials(tmp_path, monkeypatch):
    """force=True must ignore a saved (possibly wrong) password and ask again, then replace it."""
    import builtins
    import getpass
    import pathlib
    import sys
    from geoflow.auth import session

    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    netrc = tmp_path / (".netrc" if session.os.name != "nt" else "_netrc")
    netrc.write_text("machine urs.earthdata.nasa.gov login olduser password typo\n")
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setattr(session, "ensure_earthaccess_login", lambda: False)
    monkeypatch.setitem(sys.modules, "earthaccess", None)  # behave as if not installed
    monkeypatch.setattr(builtins, "input", lambda prompt="": "newuser")
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "right")
    session.logout()

    assert session.login() is True                      # saved entry is accepted without prompting
    assert session.get_credentials() == (None, None)
    assert session.login(force=True) is True            # force asks again
    assert session.get_credentials() == ("newuser", "right")
    assert netrc.read_text().count("urs.earthdata.nasa.gov") == 1
    assert "newuser" in netrc.read_text() and "typo" not in netrc.read_text()
    session.logout()
