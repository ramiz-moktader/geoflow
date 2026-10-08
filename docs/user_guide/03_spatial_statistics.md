# Spatial Statistics User Guide

GeoFlow brings advanced spatial statistics directly into the geospatial processing pipeline, bridging the gap between raw data manipulation and spatial econometrics/ecology.

## Spatial Autocorrelation

Spatial autocorrelation measures the degree to which a set of spatial features and their associated data values tend to be clustered together in space.

### Global Moran's I

Global Moran's I indicates the overall clustering of data in your study area.

```python
import geoflow as gf

# Assuming you have a FeatureCollection with some value like canopy height
gedi_features = gf.GEDI("L2A").filterBounds(aoi).to_geodataframe()

moran = gf.spatial.morans_i(gedi_features, value="rh98", k=8)
print(f"Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")
```

### Local Indicators of Spatial Association (LISA)

LISA calculates the local Moran's I for each individual feature, identifying spatial clusters (`HH`, `LL`) and spatial outliers (`HL`, `LH`).

```python
lisa_df = gf.spatial.local_morans_i(gedi_features, value="rh98", k=8)
# lisa_df now contains a 'cluster' column you can use for mapping
print(lisa_df['cluster'].value_counts())
```

## Spatial Effective Sample Size (ESS)

When samples are spatially autocorrelated (Tobler's First Law), traditional statistical tests assume independent observations, which leads to artificially low p-values (pseudoreplication). 

GeoFlow can calculate the **Spatial Effective Sample Size** to determine the *true* number of independent samples you actually have.

```python
ess = gf.stats.effective_sample_size(gedi_features, value="rh98")
print(f"Nominal N: {len(gedi_features)}")
print(f"Effective N: {ess}")
```
You can use this reduced `N` when performing ANOVA or t-tests to avoid Type I errors.
