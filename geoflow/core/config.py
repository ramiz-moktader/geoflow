"""
Global configuration for GeoFlow.
"""

from __future__ import annotations
from pathlib import Path
import tempfile


class Config:
    def __init__(self):
        self.default_crs = "EPSG:4326"
        self.cache_dir = Path.home() / ".geoflow" / "cache"
        self.download_dir = Path.cwd() / "data"
        self.backend = "auto"  # "auto", "cpu", "cuda"
        self.workers = 4
        self.strict_crs = False
        self.verbose = True

        # Ensure cache dir exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def set_cache_dir(self, path: str | Path):
        self.cache_dir = Path(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def set_backend(self, backend: str):
        if backend.lower() in ("auto", "cpu", "cuda", "mps"):
            self.backend = backend.lower()
        else:
            raise ValueError(f"Unsupported backend: {backend}")


config = Config()
