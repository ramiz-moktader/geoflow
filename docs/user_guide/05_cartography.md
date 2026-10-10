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

## Interactive Layout Editor (ArcGIS View in Notebooks)

GeoFlow provides an in-notebook layout designer similar to the Print Layout view in ArcGIS Pro or QGIS. It runs directly inside **Google Colab, JupyterLab, and VS Code**:

```python
# Launch the interactive Layout Editor
editor = img.ndvi().edit_layout(
    title="Study Area — NDVI",
    subtitle="Sentinel-2 MSI Level-2A",
    cmap="RdYlGn",
)

# Display the editor widget in notebook
editor
```

### What You Can Customize Interactively:
- **📄 Canvas**: Choose standard print sizes (Square, Landscape, Nature Single/Double Column) and custom titles.
- **🧭 Carto**: Toggle & reposition the North Arrow and dynamically scaled metric Scale Bar.
- **🌐 Graticule**: Control 4-sided coordinate labels, vertical/horizontal orientations, decimal precision, and grid styles.
- **🎨 Colors**: Switch between 15+ remote sensing palettes, continuous colorbars, and stretch limits.
- **💾 Export**: Export directly to 300 / 600 DPI, trigger a 1-click Google Colab download, or auto-generate the exact Python code snippet.

## Interactive Leaflet Maps

For web tile exploratory analysis, use the interactive `Map` class:

```python
m = gf.Map()
m.centerObject(aoi)
m.addLayer(img, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}, "Sentinel-2")
m.save("interactive.html")
```
