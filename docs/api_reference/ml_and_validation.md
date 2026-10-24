# Machine Learning, Validation & Sampling API Reference (`geoflow.validation`, `geoflow.ml`, `geoflow.sampling`)

Geospatial machine learning models frequently suffer from severe over-optimistic evaluation due to spatial autocorrelation between adjacent training and validation samples. GeoFlow provides spatial cross-validation splitters, spatial buffer dead-zones, automated data leakage diagnostics, spatial error diagnostics, and patch dataset generators for deep learning.

| Module | Source Files | Purpose |
|---|---|---|
| [`geoflow.validation`](#validation-geoflowvalidation) | `spatial_cv.py`, `leakage.py`, `diagnostics.py` | Spatial blocked CV, buffered exclusion zones, leakage audits, and residual Moran's I diagnostics |
| [`geoflow.ml`](#machine-learning--patch-datasets-geoflowml) | `datasets.py` | Geospatial patch extraction for CNNs, ViTs, PyTorch `DataLoader`, and NumPy pipelines |
| [`geoflow.sampling`](#sampling-designs--bias-diagnostics-geoflowsampling) | `spatial_sampling.py` | Stratified point sampling, minimum distance dispersion, and Clark-Evans clustering reports |

---

## Validation (`geoflow.validation`)

```python
import geoflow.validation as val
```

### Spatially Blocked Cross-Validation (`SpatialCV`)

```python
cv = gf.validation.SpatialCV(
    blocks: FeatureCollection | gpd.GeoDataFrame | int = 5,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
)
```

Partitions observations such that points located within the same spatial polygon block are never split across training and validation partitions.

- **Block Assignment**:
  - If `blocks` is a [`FeatureCollection`](vector.md#featurecollection) or `GeoDataFrame`, performs a spatial join (`sjoin`) to assign points to block IDs.
  - If `blocks` is an `int`, clusters coordinates into $k$ geographic spatial groups using $k$-means clustering.
- **Methods**:
  - `split(data, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]`: Yields `(train_indices, val_indices)` arrays.
  - `get_n_splits() -> int`: Returns `n_splits`.

---

### Buffered Spatial Cross-Validation (`BufferedSpatialCV`)

```python
cv = gf.validation.BufferedSpatialCV(
    buffer_distance: float = 1000.0,
    n_splits: int = 5,
    blocks: FeatureCollection | gpd.GeoDataFrame | None = None,
    random_state: int = 42,
    unit: str = "meters"   # or "crs" to buffer in the data's own CRS units
)
```

Enforces a spatial "dead-zone" exclusion buffer around validation blocks. Any training sample that falls within `buffer_distance` of any validation point is discarded from that fold's training set, eradicating spatial leakage.

```python
# Create spatial blocks in meters
blocks = gf.spatial.block(aoi, size=5000, unit="meters", shape="hexagonal")

# Set up 2000m buffer zone between train and test
cv = gf.validation.BufferedSpatialCV(buffer_distance=2000.0, blocks=blocks, n_splits=5)

for train_idx, val_idx in cv.split(samples_gdf):
    train_data = samples_gdf.iloc[train_idx]
    val_data = samples_gdf.iloc[val_idx]
    # Train scikit-learn or PyTorch model
```

!!! important "Coordinate Units Notice"
    With the default `unit="meters"`, `buffer_distance` is in metres. Data in a geographic CRS (e.g. `EPSG:4326`) is projected to its local UTM zone for buffering, so the buffer is correct at any latitude. Pass `unit="crs"` to buffer in the dataset's own CRS units instead.

---

### Spatial Leakage Diagnostics (`leakage_report`)

```python
report = gf.validation.leakage_report(
    train: Any,
    test: Any,
    buffer_threshold: float = 0.0,
    unit: str = "meters"
) -> LeakageReport
```

Audits two partitions (training vs validation, or train vs test) for duplicate locations and proximity violations. Distances are in metres (geographic data is projected to its local UTM zone); pass `unit="crs"` for CRS units. Set `buffer_threshold` to the range of spatial autocorrelation: with the default `0.0` proximity is not checked.

#### `LeakageReport`
- `.has_leakage` (`bool`): `True` if duplicate locations exist or `min_distance < buffer_threshold`.
- `.has_overlap` (`bool`): Convex hulls of train and test intersect. Informational only: interleaved spatial blocks always overlap at hull level without leaking.
- `.proximity_checked` (`bool`): `True` when a `buffer_threshold > 0` was given.
- `.duplicate_count` (`int`): Count of samples sharing identical coordinates ($< 10^{-7}$ separation).
- `.min_distance` (`float`): Euclidean distance between closest train and test sample points.
- `.buffer_violation` (`bool`): `True` if `min_distance < buffer_threshold`.
- `.train_count`, `.test_count`: Sample sizes.
- `.summary()` (`str`): Multi-line diagnostic report and recommendations.

---

### Spatial Model Diagnostics (`evaluate_spatial`)

```python
report = gf.validation.evaluate_spatial(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    geometry: Any | None = None,
    k: int = 8
) -> SpatialEvalReport
```

Evaluates regression predictions using standard numerical metrics alongside **Residual Moran's I** to detect whether model errors cluster geographically.

#### `SpatialEvalReport`
- `.rmse`: Root Mean Squared Error.
- `.mae`: Mean Absolute Error.
- `.r2`: Coefficient of determination ($R^2$).
- `.residuals`: 1D array of residuals ($y_{\text{true}} - y_{\text{pred}}$).
- `.moran`: [`MoranResult`](spatial.md#moranresult) of the residuals (tests if errors violate independence).
- `.summary()`: Diagnostic readout interpreting whether the model suffers from omitted spatial covariates.
- `.residual_map(ax=None)` (alias: `.error_map`): Plots residuals spatially on Matplotlib axes with a diverging colormap.

---

## Machine Learning & Patch Datasets (`geoflow.ml`)

```python
from geoflow.ml import PatchDataset
```

### `PatchDataset`

```python
dataset = gf.ml.PatchDataset(
    image: Image,
    labels: Image | str | np.ndarray | None = None,
    region: Geometry | None = None,
    patch_size: int = 64,
    stride: int = 32,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    metadata: dict[str, Any] | None = None
)
```

Extracts 2D/3D spatial image patches from multi-band rasters, tracking georeferenced bounding boxes and handling continuous or categorical raster labels.

- Filters out patches with $> 80\%$ `nan` or missing pixels.
- Indexing `dataset[i]` returns `patch_x` of shape `(bands, patch_size, patch_size)` or `(patch_x, patch_y)`.

#### Methods
- `.to_geodataframe() -> gpd.GeoDataFrame`: Returns polygon footprint geometries for each patch.
- `.split(strategy="spatial", blocks=None, buffer_distance=0.0, ratios=(0.7, 0.15, 0.15), block_size=None, random_state=42)`: Splits the patch dataset into `(train_ds, val_ds, test_ds)`.
  - `strategy="spatial"` assigns whole blocks to a subset: the given `blocks` polygons, or square blocks of `block_size` pixels (default `2 * patch_size`). `strategy="random"` assigns individual patches.
  - With either strategy, train patches whose pixel window overlaps (or lies within `buffer_distance` CRS units of) a val/test patch are dropped, and val patches overlapping test patches are dropped, so subsets never share pixels even when `stride < patch_size`. A warning is issued if more than 25% of patches are dropped.
- `.dataloader(batch_size=32, shuffle=True, device="cpu")`:
  - If **PyTorch** is installed, returns a native `torch.utils.data.DataLoader` yielding float32 tensors with pinned memory.
  - If PyTorch is not installed, returns a pure **NumPy** batch iterator yielding stacked mini-batch arrays.

```python
import geoflow as gf

img = gf.Image("sentinel2_rgbnir.tif")
dataset = gf.ml.PatchDataset(img, patch_size=128, stride=64)

# Create PyTorch-compatible batch generator
train_loader = dataset.dataloader(batch_size=16, shuffle=True)
for batch in train_loader:
    # batch shape: (16, 4, 128, 128)
    pass
```

---

## Sampling Designs & Bias Diagnostics (`geoflow.sampling`)

```python
from geoflow.sampling import stratified, min_distance, bias_report
```

### Spatially Stratified Sampling (`stratified`)

```python
stratified(
    region: Geometry | Any,
    strata: Any,
    n_per_stratum: int = 100,
    random_state: int = 42
) -> FeatureCollection
```

Generates balanced spatial point samples partitioned across strata classes within an area of interest polygon.

---

### Minimum Distance Dispersion (`min_distance`)

```python
min_distance(
    candidates: FeatureCollection | gpd.GeoDataFrame,
    n: int = 500,
    distance: float = 1000.0,
    unit: str = "meters",
    random_state: int = 42
) -> FeatureCollection
```

Filters candidate field locations or LiDAR shots by enforcing a minimum spatial distance threshold between points, preventing spatial over-sampling in clusters.

---

### Sampling Spatial Bias Diagnostics (`bias_report`)

```python
bias_report(samples: FeatureCollection | gpd.GeoDataFrame) -> dict[str, Any]
```

Evaluates spatial clustering in field observation coordinates using the **Clark-Evans Nearest Neighbor Ratio ($R$)**:
- $R < 1.0$: Spatially clustered distribution (high pseudoreplication risk).
- $R \approx 1.0$: Spatially random Poisson process.
- $R > 1.0$: Spatially dispersed / regular grid sampling.

Returns a dictionary with `"sample_size"`, `"mean_nn_distance"`, `"clark_evans_r"`, and `"spatial_pattern"`.
