# Spatial Statistics & Inference API Reference (`geoflow.spatial` & `geoflow.stats`)

GeoFlow provides first-class spatial econometrics, geostatistics, and research design inference tools built specifically to detect spatial autocorrelation, model variograms, generate spatial blocking, and calculate Spatial Effective Sample Size (ESS) to prevent pseudoreplication.

| Module | Source Files | Purpose |
|---|---|---|
| [`geoflow.spatial`](#spatial-statistics-geoflowspatial) | `autocorrelation.py`, `local_stats.py`, `weights.py`, `blocks.py`, `variogram.py` | Spatial weights, Moran's I, LISA clusters, spatial blocking, and semivariograms |
| [`geoflow.stats`](#statistical-inference--ess-geoflowstats) | `inference.py` | Spatial Effective Sample Size, hypothesis testing with spatial warnings, and power analysis |

---

## Spatial Statistics (`geoflow.spatial`)

```python
import geoflow.spatial as spatial
```

### Spatial Weights (`SpatialWeights`)

```python
from geoflow.spatial import SpatialWeights, knn, distance
```

Constructs spatial neighborhood weight matrices with automatic row standardization.

| Function | Signature | Description |
|---|---|---|
| `knn` | `knn(data: Any, k: int = 8) -> SpatialWeights` | Constructs K-Nearest Neighbor spatial weights from point coordinates or polygon centroids. |
| `distance` | `distance(data: Any, threshold: float | None = None) -> SpatialWeights` | Constructs distance-band threshold spatial weights. Defaults to $1.5 \times$ median nearest-neighbor distance. |

#### `SpatialWeights` Class Methods
- `.row_standardize() -> SpatialWeights`: Scales neighbor weights so that every row sums to $1.0$ ($\sum_j w_{ij} = 1$).
- `.to_dense() -> np.ndarray`: Converts sparse dictionary weights into a dense $N \times N$ NumPy matrix.

---

### Global Spatial Autocorrelation: Moran's I & Geary's C

```python
moran = gf.spatial.morans_i(data, value="rh98", weights=None, k=8)
c_stat = gf.spatial.gearys_c(data, value="rh98", weights=None, k=8)
```

#### `morans_i`
Computes Global Moran's I against an analytical randomization null hypothesis.

- **Parameters**:
  - `data`: A `GeoDataFrame` or [`FeatureCollection`](vector.md#featurecollection).
  - `value`: Attribute column name (or 1D NumPy array).
  - `weights`: Optional `SpatialWeights` matrix. If `None`, builds a $k$-nearest neighbor matrix.
  - `k`: Neighbor count for automatic KNN weights (default: `8`).
- **Returns**: A [`MoranResult`](#moranresult) object.

#### `MoranResult`
Encapsulates test statistics and hypothesis evaluation:

| Attribute | Type | Description |
|---|---|---|
| `.I` | `float` | Observed Global Moran's I statistic ($-1.0$ to $+1.0$). Positive indicates spatial clustering; negative indicates spatial dispersion. |
| `.expected` | `float` | Analytical expected value under spatial randomness: $-1 / (N - 1)$. |
| `.variance` | `float` | Analytical variance under the randomization assumption. |
| `.z_score` | `float` | Standardized test score: $(I - E[I]) / \sqrt{\text{Var}(I)}$. |
| `.p_value` | `float` | Two-tailed $p$-value from standard normal distribution. |
| `.n` | `int` | Sample size. |
| `.summary()` | `str` | Formatted multi-line text report of the statistical test. |

!!! warning "Documentation Errata"
    Older tutorials referenced `.p_sim` and `.z_sim`. The true attributes on `MoranResult` are `.p_value` and `.z_score`.

---

### Local Spatial Autocorrelation: LISA (`local_morans_i`)

```python
lisa_gdf = gf.spatial.local_morans_i(data, value, weights=None, k=8, p_threshold=0.05)
```

Computes Anselin's Local Indicators of Spatial Association (LISA) for each individual spatial observation.

- **Parameters**:
  - `p_threshold`: Alpha level to assign significant clusters (default: `0.05`).
- **Returns**: A copy of the input GeoDataFrame with 5 newly appended diagnostic columns:
  - `value_std`: Standardized deviation from the mean ($z_i = y_i - \bar{y}$).
  - `I`: Local Moran's statistic $I_i$.
  - `z_score`: Standardized local score.
  - `p_value`: Local significance $p$-value.
  - `cluster`: Categorical classification string:
    - `"HH"`: High-High spatial cluster (hotspot: high value surrounded by high values).
    - `"LL"`: Low-Low spatial cluster (coldspot: low value surrounded by low values).
    - `"HL"`: High-Low spatial outlier (high value surrounded by low values).
    - `"LH"`: Low-High spatial outlier (low value surrounded by high values).
    - `"not significant"`: Observation does not deviate from spatial randomness ($p > p_{\text{threshold}}$).

---

### Spatial Blocking (`block`)

```python
blocks = gf.spatial.block(
    aoi: Geometry | Any,
    size: float = 5000.0,
    unit: str = "meters",
    shape: str = "square"  # "square", "grid", or "hexagonal"
) -> FeatureCollection
```

Partitions an area of interest into spatial blocks to support leakage-safe machine learning cross-validation and spatially stratified sampling.

- **Automatic Metric Reprojection**: If `aoi` is geographic (e.g. `EPSG:4326`) and `unit="meters"`, `block()` dynamically determines the local UTM zone, reprojects the geometry to metric coordinates, tessellates the grid, and projects the resulting block boundaries back to the source CRS.
- **Shapes**:
  - `"square"` / `"grid"`: Regular square grid blocks of dimension `size x size`.
  - `"hexagonal"`: Regular hexagonal tessellation with radius $R = \text{size} / \sqrt{3}$.
- **Output**: [`FeatureCollection`](vector.md#featurecollection) containing `block_id` and polygon geometries.

---

### Empirical Semivariograms (`variogram`)

```python
var_result = gf.spatial.variogram(data, value, n_lags=15, max_dist=None)
```

Models spatial autocorrelation decay as a function of Euclidean separation distance:
$$\gamma(h) = \frac{1}{2 N(h)} \sum_{i=1}^{N(h)} (z(x_i) - z(x_i + h))^2$$

- **Returns**: A `VariogramResult` object.
  - `.lag_distances`: 1D array of mean bin separation distances $h$.
  - `.semivariances`: 1D array of semivariances $\gamma(h)$.
  - `.npairs`: Number of observation pairs falling into each distance bin.
  - `.nugget`: Estimated measurement noise / micro-scale variance at $h \approx 0$.
  - `.sill`: Estimated total plateau variance.
  - `.plot(ax=None, title="Experimental Semivariogram")`: Matplotlib scatter & line plot of the experimental variogram.

---

## Statistical Inference & ESS (`geoflow.stats`)

```python
import geoflow.stats as stats
```

### Spatial Effective Sample Size (`effective_sample_size`)

```python
ess_info = gf.stats.effective_sample_size(data, value="rh98", geometry=None)
```

In satellite and GIS datasets, observations violate independent and identically distributed (i.i.d.) assumptions due to Tobler's First Law. Traditional statistical tests experience severe Type I error inflation (pseudoreplication).

`effective_sample_size` adjusts the nominal sample size using the spatial autocorrelation parameter $\rho$:
$$N_{\text{eff}} = N \cdot \frac{1 - \rho}{1 + \rho}$$

- **Returns a `dict`**:
  - `"nominal_n"`: Total count of rows ($N$).
  - `"effective_n"`: True number of statistically independent observations ($N_{\text{eff}}$).
  - `"morans_i"`: Observed Moran's I.
  - `"p_value"`: Significance of spatial autocorrelation.
  - `"design_effect"`: Ratio $N / N_{\text{eff}}$.
  - `"pseudoreplication_risk"`: Boolean alert (`True` when design effect $> 2.0$ and $p < 0.05$).

---

### Spatial Hypothesis Testing (`test`)

```python
result = gf.stats.test(data, value="rh98", group="landcover_class", method=None)
```

Performs two-sample or multi-sample hypothesis tests with effect sizes and **automated spatial autocorrelation alerts**.

- **Two Groups**:
  - `method=None`: Welch's $t$-test (robust to unequal variances) with **Cohen's $d$** effect size and 95% confidence intervals.
  - `method="mann_whitney"`: Nonparametric Mann-Whitney U test with **Rank-Biserial correlation** effect size.
- **Three or More Groups**:
  - One-way ANOVA with **Eta-squared ($\eta^2$)** effect size.
- **Spatial Warning**: Automatically checks if `value` exhibits pseudoreplication risk. If detected, appends a warning to `result.spatial_warning`.

#### `StatTestResult`
- `.test_name`: Name of test performed.
- `.statistic`: Test statistic ($t$, $U$, or $F$).
- `.p_value`: Test $p$-value.
- `.effect_size`: Standardized effect size magnitude.
- `.effect_type`: `"cohens_d"`, `"rank_biserial"`, or `"eta_squared"`.
- `.ci_95`: 95% Confidence interval tuple.
- `.sample_sizes`: Dictionary of counts per group.
- `.spatial_warning`: Spatial pseudoreplication advisory string (or `None`).
- `.summary()`: Formatted text printout.

---

### Sample Size & Power Analysis (`power`)

```python
gf.stats.power.sample_size(effect_size=0.5, alpha=0.05, power=0.8) -> int
```

Calculates required sample size per experimental group for a two-sample $t$-test given Cohen's $d$ effect size, significance level $\alpha$, and statistical power ($1 - \beta$).
