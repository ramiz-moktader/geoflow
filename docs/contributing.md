# Contributor & Developer Guide

Welcome to the **GeoFlow** contributor guide! This document explains GeoFlow's software architecture, engineering patterns, development setup, testing protocols, and known roadmap areas.

---

## 1. Architectural Philosophy

GeoFlow is designed around three foundational principles:
1. **Google Earth Engine (GEE) Familiarity**: Provide the intuitive, chainable API that remote sensing researchers love (`Image`, `ImageCollection`, `FeatureCollection`, `Filter`, `Reducer`, `Map.addLayer`).
2. **Native Python Execution**: Execute computations locally using industry-standard Python libraries (**NumPy, Rasterio, GeoPandas, SciPy, PySAL, PyTorch, Xarray**) without requiring proprietary cloud runtimes.
3. **Scientific Rigor**: Treat spatial autocorrelation, pseudoreplication, and spatial ML data leakage as first-class architectural concerns rather than afterthoughts.

---

## 2. Repository & Package Layout

```text
geoflow/
├── __init__.py               # Top-level API exports (gf, Filter, Reducer, Image, etc.)
├── auth/                     # NASA Earthdata session & token authentication
│   └── session.py
├── cli/                      # Click-based CLI (geoflow search, info, gpu)
│   └── main.py
├── core/                     # Foundational abstractions
│   ├── base.py               # EarthObject base class with provenance & .pipe()
│   ├── config.py             # Global runtime configuration
│   ├── crs.py                # CRS wrapper & compatibility validation
│   ├── filter.py             # Declarative Filter engine
│   └── reducer.py            # Reducer engine
├── datasets/                 # Mission-specific dataset adapters
│   └── gedi.py               # NASA GEDI L2A/L2B LiDAR parser
├── earthdata/                # NASA CMR discovery, downloading & STAC streaming
│   ├── downloader.py         # Parallel resumable downloader
│   ├── search.py             # CMR search API client
│   └── stac.py               # STAC cloud-native streaming (odc-stac + Dask)
├── geometry/                 # Vector geometries wrapping Shapely
│   ├── base.py               # Geometry base class
│   ├── io.py                 # Vector file & GeoJSON deserializers
│   ├── point.py              # Point & MultiPoint
│   └── polygon.py            # Polygon, MultiPolygon & Rectangle
├── ml/                       # Geospatial machine learning
│   └── datasets.py           # PatchDataset & PyTorch batch generator
├── raster/                   # Raster data structures
│   ├── collection.py         # ImageCollection & compositing
│   └── image.py              # Image, band math, indices & terrain
├── sampling/                 # Spatial research sampling designs
│   └── spatial_sampling.py   # Stratified, min_distance & bias diagnostics
├── spatial/                  # Spatial statistics & geostatistics
│   ├── autocorrelation.py    # Global Moran's I & Geary's C
│   ├── blocks.py             # Spatial blocking (square & hexagonal)
│   ├── local_stats.py        # LISA / Local Moran's I
│   ├── variogram.py          # Empirical semivariograms
│   └── weights.py            # KNN & distance spatial weights
├── stats/                    # Statistical inference & research design
│   └── inference.py          # Spatial Effective Sample Size (ESS) & hypothesis tests
├── validation/               # Spatial cross-validation & data leakage
│   ├── diagnostics.py        # SpatialEvalReport & Residual Moran's I
│   ├── leakage.py            # Spatial leakage auditor (leakage_report)
│   └── spatial_cv.py         # SpatialCV & BufferedSpatialCV
├── visualization/            # Web visualization
│   └── map.py                # Interactive Leaflet Map
└── viz/                      # Publication cartography & interactive layout
    ├── cartography.py        # plot_carto_map, north arrow, scale bar, graticules
    ├── layout_editor.py      # In-notebook ArcGIS-style Layout Editor
    └── publication.py        # Journal themes (Nature/Science) & LaTeX tables
```

---

## 3. Development Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Initializing Environment

```bash
# Clone the repository
git clone https://github.com/ramiz-moktader/geoflow.git
cd geoflow

# Create and activate a virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install in editable mode with all optional dependencies
pip install -e ".[all]"

# Install documentation & testing tools
pip install pytest pytest-cov mkdocs-material
```

---

## 4. Key Engineering & Design Patterns

### The `_clone()` Pattern & Subclass Propagation
GeoFlow enables users to subclass `Image`, `ImageCollection`, or `FeatureCollection` to add domain methods. All internal operations (such as `.select()`, `.add()`, `.ndvi()`, `.clip()`) must preserve the custom subclass and its custom instance attributes:

```python
# From geoflow/raster/image.py:
def _clone(self, data: np.ndarray, band_names: Optional[List[str]] = None) -> Any:
    names = band_names if band_names is not None else self._band_names
    new_obj = self.__class__(
        data,
        transform=self._transform,
        crs=self._crs,
        band_names=names,
        nodata=self._nodata,
        metadata=dict(self._metadata),
    )
    # Propagate custom subclass attributes
    for k, v in self.__dict__.items():
        if k not in ("_data", "_transform", "_crs", "_band_names", "_nodata", "_metadata"):
            new_obj.__dict__[k] = v
    return new_obj
```
When implementing new raster methods, **always use `self._clone(new_data, ...)`** rather than instantiating `Image(...)` directly.

### Strict CRS Awareness
GeoFlow methods must never silently mix or corrupt Coordinate Reference Systems.
- Use `check_crs_compatibility(self.crs, other.crs, context="...")` before binary raster math or geometric intersections.
- When sampling rasters at point coordinates (e.g. `fc.sample_raster(image)`), reproject the coordinates to the image's coordinate space on-the-fly rather than reprojecting the entire raster grid.

### Fluent Functional Composition (`.pipe()`)
All `EarthObject` instances support `.pipe(func, *args, **kwargs)`:
```python
def mask_clouds(img, threshold=20):
    return img.updateMask(img.select("cloud_mask") < threshold)

processed = img.pipe(mask_clouds, threshold=15).pipe(calc_indices)
```

---

## 5. Testing Protocols

GeoFlow uses `pytest`. All Matplotlib figures created during tests must be cleaned up to avoid resource leaks. `conftest.py` handles this automatically via the `cleanup_figures` fixture.

### Running Test Suites

```bash
# Run all tests
pytest -v

# Run with coverage report
pytest --cov=geoflow --cov-report=term-missing

# Run quick end-to-end integration test
python test_quick.py
```

### Adding New Tests
- Unit tests belong in dedicated test files prefixed with `test_` at the repository root or in a `tests/` directory.
- Test both NumPy array inputs and rasterio-backed datasets.
- Test that custom subclasses persist through your new method chain.

---

## 6. Known Limitations & Roadmap

Issues from the codebase audit have been fixed and are covered by `test_audit_regressions.py`. The following limitations remain; contributions are welcome:

| Area | Current Behavior | Recommended Improvement |
|---|---|---|
| **Basemap terms of use** | The default `HYBRID`/`TERRAIN` basemaps load Google tiles directly, which Google's terms do not permit outside its APIs. | Switch the default to an open basemap (e.g. Esri `SATELLITE` or OpenStreetMap) and keep Google as opt-in. |
| **`login()` verification** | Credentials from environment variables or `~/.netrc` are reported as authenticated without contacting Earthdata. | Validate with a lightweight authenticated request (e.g. URS profile endpoint). |
| **Variogram memory** | `variogram()` builds all point pairs: memory grows as $O(n^2)$. | Subsample pairs or bin with a KD-tree for large point sets. |
| **LISA inference** | Local Moran's I uses analytical p-values without multiple-testing correction, so many clusters can be false positives. | Add conditional permutation inference and FDR / Bonferroni correction. |
| **Spatial ESS** | `effective_sample_size` uses the AR(1) formula with Moran's I as $\rho$, a screening heuristic. | Offer variogram- or model-based effective sample sizes. |
| **`CRS` hashing** | `CRS` defines `__eq__` without `__hash__`, so CRS objects can't be used in sets or as dict keys. | Implement `__hash__` from the normalised WKT / EPSG code. |
| **`Filter` errors** | `Filter.__call__` treats any exception inside a predicate as "no match". | Raise or warn on predicate errors other than missing properties. |
| **`focal_median` with NaN** | Falls back to a slow `generic_filter` when the band contains NaN. | Use a vectorized NaN-aware median filter. |
| **Map image overlays** | Rasters are reprojected to EPSG:4326 and drawn on Web Mercator, which stretches large extents in latitude. | Reproject overlays to EPSG:3857 before rendering. |
| **Downloader filenames** | Files are named by the URL's last path segment; two granules with the same name overwrite each other. | Namespace downloads by collection/granule ID. |


---

## 7. Documentation Workflow

The documentation is generated using **MkDocs** with the **Material theme**:

```bash
# Serve documentation locally with live reload
mkdocs serve

# Build production static bundle
mkdocs build
```

When contributing code:
1. Update docstrings following Google / NumPy style.
2. Update the corresponding markdown file in `docs/api_reference/`.
3. Add tutorials or practical examples in `docs/user_guide/` if introducing user-facing capabilities.

---

## 8. Pull Request Guidelines

1. **Branch Naming**: Use descriptive branch names (e.g. `fix/image-comparison-operators`, `feat/modis-adapter`, `docs/api-overhaul`).
2. **Clean Commits**: Write clear, descriptive commit messages.
3. **Pass Tests**: Ensure `pytest` passes with zero failures before opening a PR.
4. **Documentation**: Any new feature or bug fix must include corresponding documentation and test coverage.
5. **Backwards Compatibility**: Do not remove public methods from `gf.__all__` without a deprecation warning cycle.
