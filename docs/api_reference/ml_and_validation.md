# Machine Learning and Validation API Reference

GeoFlow integrates closely with machine learning workflows, offering spatial splitters, leakage diagnostics, and PyTorch dataset adapters.

## `gf.validation.BufferedSpatialCV`

A spatial cross-validator that employs exclusionary buffer zones (dead-zones) between training and validation blocks to prevent spatial data leakage.

```python
cv = gf.validation.BufferedSpatialCV(
    buffer_distance=1000, # 1km buffer
    blocks=spatial_blocks,
    n_splits=5
)
for train_idx, val_idx in cv.split(gdf):
    # Train model
    pass
```

## `gf.validation.leakage_report(train_df, val_df, buffer_threshold)`

Audits training and validation splits for spatial data leakage by calculating minimum separation distances.

## `gf.ml.PatchDataset`

A PyTorch-compatible `Dataset` that extracts multi-channel image patches from a `gf.Image`.

```python
dataset = gf.ml.PatchDataset(
    image=img, 
    patch_size=64, 
    stride=32
)
loader = dataset.dataloader(batch_size=16, shuffle=True)

for batch in loader:
    # batch shape: (16, bands, 64, 64)
    pass
```

## `gf.stats.effective_sample_size(gdf, value, k=8)`

Calculates the Spatial Effective Sample Size (ESS) to adjust statistical significance by accounting for pseudoreplication caused by spatial autocorrelation.
