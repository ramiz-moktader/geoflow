# Machine Learning and Validation User Guide

GeoFlow is built to safely interface with machine learning frameworks like `scikit-learn` and `PyTorch` while rigorously protecting against spatial data leakage.

## The Problem: Spatial Data Leakage

In spatial datasets, nearby samples are highly correlated. If you randomly split points into a training set and a validation set, you will likely have validation points that are just a few meters away from training points. Your model will "memorize" the local spatial structure rather than learning generalizable features, leading to severely over-optimistic performance metrics.

## Solution: Spatial Blocking & Buffered Cross-Validation

GeoFlow solves this with **Spatial Blocking** and **Buffered Spatial Cross Validation**.

### 1. Create Spatial Blocks
First, divide your study area into spatial blocks (squares or hexagons).

```python
import geoflow as gf

aoi = gf.Rectangle([91.7, 22.2, 92.0, 22.5])
blocks = gf.spatial.block(aoi, size=5000, unit="meters", shape="hexagonal")
```

### 2. Buffered Spatial CV
Next, use `BufferedSpatialCV`. This splitter ensures that an entire block is held out for validation, AND it enforces a "dead-zone" buffer between the training data and validation block so no training data is physically adjacent to the validation data.

```python
cv = gf.validation.BufferedSpatialCV(
    buffer_distance=2000, # 2km dead-zone
    blocks=blocks,
    n_splits=5
)

for train_idx, val_idx in cv.split(samples_gdf):
    train_data = samples_gdf.iloc[train_idx]
    val_data = samples_gdf.iloc[val_idx]
    # Train your model here...
```

### 3. Leakage Diagnostics
You can audit any split using the `leakage_report` tool:

```python
report = gf.validation.leakage_report(train_data, val_data, buffer_threshold=2000)
print(report.summary())
```

## Deep Learning with PyTorch

GeoFlow provides a native `PatchDataset` that acts as a bridge to deep learning frameworks by generating spatial image patches.

```python
# Create a patch generator from a GeoFlow Image
dataset = gf.ml.PatchDataset(
    image=img,
    patch_size=64, # 64x64 pixel patches
    stride=32      # Overlapping patches
)

# Convert to a PyTorch DataLoader
train_loader = dataset.dataloader(batch_size=32, shuffle=True)

for batch in train_loader:
    # batch is a tensor of shape (32, bands, 64, 64)
    # Feed to your CNN / ViT
    pass
```
