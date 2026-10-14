"""
Authentication and session management for NASA Earthdata.
"""

from __future__ import annotations
import os
from typing import Optional

_SESSION = None


def _earthaccess_logged_in() -> bool:
    """True if earthaccess is installed and holds an authenticated session."""
    try:
        import earthaccess
    except ImportError:
        return False
    auth = getattr(earthaccess, "__auth__", None)
    store = getattr(earthaccess, "__store__", None)
    return bool(store is not None and getattr(auth, "authenticated", False))


def ensure_earthaccess_login() -> bool:
    """
    Make sure earthaccess is authenticated, without prompting.
    Uses EARTHDATA_USERNAME/EARTHDATA_PASSWORD or ~/.netrc.
    Returns False if earthaccess is not installed or no credentials are available.
    """
    global _SESSION
    try:
        import earthaccess
    except ImportError:
        return False
    if _earthaccess_logged_in():
        return True
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy, persist=False)
            if getattr(auth, "authenticated", False):
                _SESSION = auth
                return True
        except Exception:
            continue
    return False


def login(strategy: str = "interactive", persist: bool = True) -> bool:
    """
    Authenticate with NASA Earthdata.
    Uses earthaccess if available, otherwise falls back to environment variables, .netrc, or interactive prompt.
    """
    global _SESSION
    try:
        import earthaccess  # noqa: F401
        has_earthaccess = True
    except ImportError:
        has_earthaccess = False

    if is_authenticated() and (not has_earthaccess or _earthaccess_logged_in()):
        return True

    # 1. earthaccess backend: try stored credentials first, then the requested strategy
    if has_earthaccess:
        if ensure_earthaccess_login():
            return True
        try:
            import earthaccess
            auth = earthaccess.login(strategy=strategy, persist=persist)
            if getattr(auth, "authenticated", False):
                _SESSION = auth
                return True
        except Exception as e:
            import warnings
            warnings.warn(f"earthaccess login failed: {e}")

    # 2. Check environment variables
    user = os.environ.get("EARTHDATA_USERNAME")
    pw = os.environ.get("EARTHDATA_PASSWORD")
    if user and pw:
        _SESSION = {"username": user, "authenticated": True}
        return True

    # 3. Check existing ~/.netrc
    from pathlib import Path
    netrc_path = Path.home() / (".netrc" if os.name != "nt" else "_netrc")
    if netrc_path.exists():
        try:
            content = netrc_path.read_text()
            if "urs.earthdata.nasa.gov" in content:
                _SESSION = {"authenticated": True, "source": "netrc"}
                return True
        except Exception:
            pass

    # 4. Fallback: Prompt interactively in Colab / terminal
    if strategy in ("interactive", "all"):
        import getpass
        print("NASA Earthdata Login (enter your credentials below):")
        try:
            username = input("Enter NASA Earthdata Username: ").strip()
            password = getpass.getpass("Enter NASA Earthdata Password: ").strip()
        except (EOFError, KeyboardInterrupt):
            return False

        if username and password:
            if persist:
                try:
                    mode = "a" if netrc_path.exists() else "w"
                    with open(netrc_path, mode) as f:
                        f.write(f"\nmachine urs.earthdata.nasa.gov login {username} password {password}\n")
                    if os.name != "nt":
                        os.chmod(netrc_path, 0o600)
                except Exception:
                    pass
            _SESSION = {"username": username, "authenticated": True}
            os.environ["EARTHDATA_USERNAME"] = username
            os.environ["EARTHDATA_PASSWORD"] = password
            ensure_earthaccess_login()
            print("Successfully authenticated with NASA Earthdata!")
            return True

    return False


def logout():
    """Clear active Earthdata session."""
    global _SESSION
    _SESSION = None


def get_session():
    """Retrieve active session."""
    return _SESSION


def is_authenticated() -> bool:
    """Check if session is currently authenticated."""
    if _SESSION is None:
        return False
    if hasattr(_SESSION, "authenticated"):
        return bool(_SESSION.authenticated)
    if isinstance(_SESSION, dict):
        return bool(_SESSION.get("authenticated", False))
    return False


def get_s3_credentials(daac: str = "LPDAAC") -> dict:
    """
    Retrieve temporary direct-access AWS S3 credentials for in-region NASA cloud DAACs (us-west-2).
    Enables zero-egress, high-throughput streaming directly from NASA S3 buckets.

    Parameters
    ----------
    daac : str, default 'LPDAAC'
        NASA DAAC name (e.g. 'LPDAAC', 'NSIDC', 'PO.DAAC', 'GES_DISC', 'ORNL_DAAC', 'SEDAC').

    Returns
    -------
    dict
        Dictionary containing 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', and 'expiration'.
    """
    try:
        import earthaccess
        return earthaccess.get_s3_credentials(daac=daac)
    except ImportError:
        pass
    except Exception as e:
        import warnings
        warnings.warn(f"earthaccess.get_s3_credentials failed: {e}")

    # Fallback to AWS environment variables if running in AWS us-west-2
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    token = os.environ.get("AWS_SESSION_TOKEN")
    if ak and sk:
        return {
            "aws_access_key_id": ak,
            "aws_secret_access_key": sk,
            "aws_session_token": token,
        }

    raise RuntimeError(
        f"Unable to retrieve S3 credentials for DAAC '{daac}'. Ensure you are logged into NASA Earthdata "
        f"(`geoflow.auth.login()`), have `earthaccess` installed, and/or are running in an AWS us-west-2 instance."
    )
