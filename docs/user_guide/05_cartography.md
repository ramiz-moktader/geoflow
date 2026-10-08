# Cartography and Visualization

GeoFlow provides tools to generate professional, publication-ready cartographic maps in a single line of code.

## Plotting Maps

The `plot_map()` method handles scaling, North Arrows, Scale Bars, and complex graticule formatting.

```python
import geoflow as gf

img = gf.Image("data.tif")

ax = img.plot_map(
    bands=["B4", "B3", "B2"], # True color composite
    title="Study Area",
    north_arrow=True,
    scale_bar=True,
    grid=True,
    save_path="map.png"
)
```

## Graticule and Corner Collision Pruning

When plotting maps, coordinates at the corners of the bounding box often collide (e.g., the lowest latitude label overlapping the leftmost longitude label). GeoFlow automatically handles this using cartographic best practices.

### Four-Sided Graticules and Vertical Latitudes

You can enable coordinate ticks on all four sides and rotate the latitude labels vertically:

```python
img.ndvi().plot_map(
    cmap="RdYlGn",
    title="Vegetation Index",
    show_top=True,
    show_right=True,
    show_bottom=True,
    show_left=True,
    lat_orientation="vertical", # Rotates latitude labels 90°
    prune_corners=True,         # Automatically hides colliding corner labels
    degree_precision=3
)
```

- **`prune_corners`**: Set to `True` (default) to suppress corner labels that collide. You can also specify `"x"`, `"y"`, or `"both"` to enforce which axes have corner labels pruned.
- **`lat_orientation`**: Use `"vertical"` to align latitude text cleanly alongside the left/right neatlines.
- **`tick_fontfamily`** and **`tick_fontsize`**: Style the coordinates for publication (e.g., `tick_fontfamily="serif"`).

## Interactive Maps

For exploratory analysis, use the interactive `Map` class:

```python
m = gf.Map()
m.centerObject(aoi)
m.addLayer(img, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}, "Sentinel-2")
m.save("interactive.html")
```
