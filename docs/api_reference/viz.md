# Visualization API Reference

The `geoflow.viz` module contains robust cartographic mapping and publication-ready rendering functions.

## Cartography

### `plot_carto_map(image_or_gdf, ...)`

Renders a publication-ready map with a north arrow, scale bar, and geographic graticule.

**Parameters**:
- `image_or_gdf`: The `Image`, `FeatureCollection`, or spatial object to plot.
- `bands` (List[str]): Bands to plot (e.g. `["B4", "B3", "B2"]`).
- `cmap` (str): Colormap to use for single-band images (e.g. `"viridis"`).
- `vmin`, `vmax` (float): Value ranges.
- `title`, `subtitle` (str): Map titles.
- `north_arrow` (bool): Draw a north arrow. Default is `True`.
- `scale_bar` (bool): Draw a dynamic scale bar in kilometers/meters. Default is `True`.
- `grid` (bool): Draw a grid graticule. Default is `True`.
- `colorbar` (bool): Draw a colorbar for continuous rasters. Default is `True`.
- `show_top`, `show_bottom`, `show_left`, `show_right` (bool): Enable coordinate ticks on the respective sides.
- `lat_orientation`, `lon_orientation` (str): Text orientation (`"horizontal"` or `"vertical"`).
- `tick_fontsize`, `tick_fontfamily`, `tick_color`: Typography controls.
- `prune_corners` (bool | str): Prevent corner tick collisions by pruning. Default is `True`. (Options: `True`/`"auto"`, `"x"`, `"y"`, `"both"`, `False`).

## Interactive Maps

### `gf.Map()`

Initializes an interactive Leaflet-based map.

**Methods**:
- `.addLayer(ee_object, vis_params, name)`: Adds an Image, FeatureCollection, or Geometry.
- `.centerObject(obj, zoom)`: Centers the map on the object.
- `.save(filepath)`: Saves the interactive map to an HTML file.
