# Spatial Statistics API Reference

The `geoflow.spatial` module provides native spatial statistics algorithms for geospatial analysis. 

## Autocorrelation

### `gf.spatial.morans_i(gdf, value, k=8, weight_type="knn")`
Calculates Global Moran's I for spatial autocorrelation.
- **Parameters**: 
  - `gdf` (GeoDataFrame or FeatureCollection)
  - `value` (str): Column name to analyze
  - `k` (int): Number of nearest neighbors
- **Returns**: A Moran's I results object with `.I`, `.p_sim`, and `.z_sim` attributes.

### `gf.spatial.local_morans_i(gdf, value, k=8)`
Calculates Local Indicators of Spatial Association (LISA).
- **Returns**: Adds columns to the GeoDataFrame for local I statistics, p-values, and cluster classifications (`HH`, `LL`, `HL`, `LH`, `not significant`).

### `gf.spatial.gearys_c(gdf, value, k=8)`
Calculates Geary's C statistic.

## Interpolation & Variograms

### `gf.spatial.variogram(gdf, value, bins=15)`
Calculates the empirical semivariogram for modeling spatial variance over distance.

## Spatial Blocking & Weights

### `gf.spatial.block(geometry, size, unit="meters", shape="square")`
Generates spatial blocks (square or hexagonal) over an area of interest for cross-validation or stratified sampling.

### `gf.spatial.knn(gdf, k=8)`
Generates K-Nearest Neighbor spatial weights matrices.

### `gf.spatial.distance(gdf, threshold)`
Generates distance-band spatial weights matrices.
