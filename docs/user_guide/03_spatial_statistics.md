# Spatial Statistics User Guide

GeoFlow brings advanced spatial statistics directly into the geospatial processing pipeline, bridging the gap between raw data manipulation and spatial econometrics/ecology.

## Spatial Autocorrelation

Spatial autocorrelation measures the degree to which a set of spatial features and their associated data values tend to be clustered together in space.

### Global Moran's I

Global Moran's I indicates the overall clustering of data in your study area.

```python
import geoflow as gf

# GEDI canopy heights for the area (needs gf.login(); see the Earthdata login guide)
gedi_features = (
    gf.GEDI("L2A").filterBounds(aoi).filterDate("2022-01-01", "2022-03-31")
    .filterQuality().load(limit=1).to_geodataframe()
)

moran = gf.spatial.morans_i(gedi_features, value="rh98", k=8)
print(f"Moran's I: {moran.I:.4f}, p-value: {moran.p_value:.4f}")
print(moran.summary())
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
print(f"Nominal N: {ess['nominal_n']}")
print(f"Effective N: {ess['effective_n']}")
print(f"Design Effect: {ess['design_effect']:.2f}")
print(f"Pseudoreplication Alert: {ess['pseudoreplication_risk']}")
```
You can use this reduced `N` when performing ANOVA or t-tests to avoid Type I errors.
