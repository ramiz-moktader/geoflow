"""
Authentication and session management for NASA Earthdata.
"""

from __future__ import annotations
import os
from typing import Optional

_SESSION = None


def login(strategy: str = "interactive", persist: bool = True) -> bool:
    """
    Authenticate with NASA Earthdata.
    Uses earthaccess if available, otherwise falls back to environment variables or .netrc.
    """
    global _SESSION
    if is_authenticated():
        return True
    try:
        import earthaccess
        auth = earthaccess.login(strategy=strategy, persist=persist)
        _SESSION = auth
        return bool(auth.authenticated)
    except ImportError:
        # Fallback to checking environment variables
        user = os.environ.get("EARTHDATA_USERNAME")
        pw = os.environ.get("EARTHDATA_PASSWORD")
        if user and pw:
            _SESSION = {"username": user, "authenticated": True}
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
    return bool(_SESSION.get("authenticated", False))
