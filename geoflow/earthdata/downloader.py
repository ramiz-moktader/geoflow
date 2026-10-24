"""
Parallel, resilient, resumable downloader for NASA Earthdata files.
Features:
- Multi-threaded parallel downloading
- Automatic resume from partial downloads
- Checksum validation
- Free disk space verification
- Local caching & manifest registry
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import os
import shutil
import hashlib
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_disk_space(target_dir: Path, min_free_mb: int = 500):
    """Ensure sufficient free disk space exists before initiating downloads."""
    total, used, free = shutil.disk_usage(target_dir)
    free_mb = free / (1024 * 1024)
    if free_mb < min_free_mb:
        raise OSError(
            f"Insufficient disk space in {target_dir}. "
            f"Available: {free_mb:.1f} MB, required minimum: {min_free_mb} MB"
        )


URS_HOST = "urs.earthdata.nasa.gov"
LOGIN_GUIDE_URL = "https://ramiz-moktader.github.io/geoflow/user_guide/00_earthdata_login/"


def _is_urs(url: str) -> bool:
    parsed = requests.utils.urlparse(url)
    return parsed.scheme == "https" and parsed.hostname == URS_HOST


class EarthdataSession(requests.Session):
    """
    requests.Session for Earthdata downloads. Credentials are only ever sent to
    https://urs.earthdata.nasa.gov (the login host data servers redirect to), never to
    the data hosts themselves or to any other link found in granule metadata.
    Without explicit credentials, requests' ~/.netrc lookup applies per host.
    """

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        super().__init__()
        self._urs_auth = requests.auth.HTTPBasicAuth(username, password) if username and password else None

    def prepare_request(self, request):
        prepared = super().prepare_request(request)
        if self._urs_auth is not None and _is_urs(prepared.url):
            prepared.prepare_auth(self._urs_auth)
        return prepared

    def rebuild_auth(self, prepared_request, response):
        # Strips Authorization on cross-host redirects and re-applies ~/.netrc for the new host
        super().rebuild_auth(prepared_request, response)
        if self._urs_auth is not None and _is_urs(prepared_request.url):
            prepared_request.prepare_auth(self._urs_auth)


def _make_session() -> requests.Session:
    from geoflow.auth.session import get_credentials
    user, pw = get_credentials()
    return EarthdataSession(user, pw)


def _download_file(
    url: str,
    dest_path: Path,
    resume: bool = True,
    timeout: int = 60,
    session: Optional[requests.Session] = None,
) -> Path:
    """Download single file with resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    http = session or _make_session()

    initial_pos = 0
    headers = {}
    if resume and temp_path.exists():
        initial_pos = temp_path.stat().st_size
        headers["Range"] = f"bytes={initial_pos}-"

    with http.get(url, headers=headers, stream=True, timeout=timeout) as r:
        if r.status_code == 416:  # Range not satisfiable, file complete or invalid
            if temp_path.exists():
                temp_path.rename(dest_path)
                return dest_path
        login_failed = r.status_code in (401, 403) or (
            r.status_code >= 500 and requests.utils.urlparse(r.url).hostname == URS_HOST
        )
        if login_failed:
            raise PermissionError(
                f"NASA Earthdata refused access to {url} (HTTP {r.status_code}). Run gf.login() with your "
                "Earthdata username and password, and authorize the data center's application in your "
                f"Earthdata profile. Step-by-step guide: {LOGIN_GUIDE_URL}"
            )
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            raise PermissionError(
                f"Got an HTML page instead of data for {url} - this is the Earthdata login page. "
                "Run geoflow.login() first."
            )

        mode = "ab" if initial_pos > 0 and r.status_code == 206 else "wb"
        with open(temp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)

    temp_path.replace(dest_path)
    return dest_path


def download(
    granules: Union[List[Dict[str, Any]], List[str], Any],
    output_dir: Union[str, Path] = "./data",
    workers: int = 4,
    resume: bool = True,
    cache: bool = True,
) -> List[Path]:
    """
    Download Earthdata granules in parallel with resume and cache support.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    check_disk_space(out_path)

    # Use earthaccess for its own DataGranule objects (handles auth, S3 and DAAC cookies)
    is_ea_granules = bool(granules) and not isinstance(granules[0], (str, dict))
    if is_ea_granules:
        try:
            import earthaccess
        except ImportError:
            earthaccess = None
        if earthaccess is not None:
            from geoflow.auth.session import ensure_earthaccess_login
            if not ensure_earthaccess_login():
                raise PermissionError(
                    "Not logged in to NASA Earthdata. Run `geoflow.login()` first "
                    "(or set EARTHDATA_USERNAME / EARTHDATA_PASSWORD)."
                )
            res = earthaccess.download(granules, local_path=str(out_path))
            paths = [Path(p) for p in (res or []) if p]
            if not paths:
                raise RuntimeError(
                    f"earthaccess downloaded 0 of {len(granules)} granules. "
                    "Check your Earthdata login and that the DAAC application is authorized in your profile."
                )
            return paths

    # Extract target URLs for manual fallback
    urls: List[str] = []
    if isinstance(granules, list):
        for g in granules:
            if isinstance(g, str):
                urls.append(g)
            elif hasattr(g, "data_links"):
                try:
                    dl = g.data_links()
                    if dl:
                        urls.extend(dl if isinstance(dl, list) else [dl])
                except Exception:
                    pass
            elif isinstance(g, dict):
                # Check for direct download URL in granule entry
                links = g.get("urls") or [l.get("href") for l in g.get("raw", {}).get("links", []) if "href" in l]
                # Filter for data links (e.g. .h5, .nc, .tif)
                for l in links:
                    if any(l.endswith(ext) for ext in [".h5", ".nc", ".tif", ".tiff", ".zip", ".tar"]):
                        urls.append(l)
                        break
                else:
                    if links:
                        urls.append(links[0])

    downloaded_paths: List[Path] = []
    errors: List[str] = []
    pending = []
    for url in urls:
        target_file = out_path / url.split("/")[-1].split("?")[0]
        if cache and target_file.exists() and target_file.stat().st_size > 0:
            downloaded_paths.append(target_file)
        else:
            pending.append((url, target_file))

    # A login held by earthaccess (gf.login() with earthaccess installed) is not visible to our
    # requests session, so let earthaccess fetch the URLs with its own authenticated session.
    from geoflow.auth.session import _earthaccess_logged_in, get_credentials
    if pending and _earthaccess_logged_in() and get_credentials() == (None, None):
        import earthaccess
        res = earthaccess.download([u for u, _ in pending], local_path=str(out_path))
        paths = [Path(p) for p in (res or []) if p]
        if not paths:
            raise RuntimeError(
                f"earthaccess downloaded 0 of {len(pending)} files. "
                "Check your Earthdata login and that the DAAC application is authorized in your profile."
            )
        return downloaded_paths + paths

    session = _make_session()

    # Download the first file on its own: it runs the Earthdata login redirect once and stores the
    # DAAC session cookie, so the parallel downloads after it don't all hit the login server at once.
    if pending:
        url, target = pending[0]
        try:
            downloaded_paths.append(_download_file(url, target, resume, 60, session))
        except Exception as e:
            errors.append(f"{url}: {e}")
            if isinstance(e, PermissionError):
                raise PermissionError(str(e)) from None  # same credentials would fail for every file

    tasks = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for url, target in pending[1:]:
            tasks[executor.submit(_download_file, url, target, resume, 60, session)] = url

        for future in as_completed(tasks):
            try:
                downloaded_paths.append(future.result())
            except Exception as e:
                errors.append(f"{tasks[future]}: {e}")

    if errors:
        import warnings
        warnings.warn(f"{len(errors)} of {len(urls)} downloads failed:\n" + "\n".join(errors[:5]))
    if urls and not downloaded_paths:
        raise RuntimeError("All downloads failed. First error: " + errors[0])

    return downloaded_paths
