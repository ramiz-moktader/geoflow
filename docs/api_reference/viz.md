# Visualization & Cartography API Reference (`geoflow.viz` & `geoflow.visualization`)

GeoFlow contains a two-tiered visualization engine:
1. **Interactive Web Maps** (`geoflow.visualization.Map`): Smooth, responsive Leaflet map overlays with zero mandatory third-party web dependencies.
2. **Publication-Grade Cartography** (`geoflow.viz`): Matplotlib rendering with dynamic metric scale bars, 3D north arrows, 4-sided graticule degree notation, corner collision pruning, and an in-notebook **ArcGIS-style Layout Editor**.

| Module | Source Files | Features |
|---|---|---|
| [`geoflow.viz.cartography`](#publication-cartography-engine-geoflowvizcartography) | `cartography.py` | `plot_carto_map`, `draw_north_arrow`, `draw_scale_bar`, `format_degree_ticks` |
| [`geoflow.viz.layout_editor`](#in-notebook-layout-editor-geoflowvizlayout_editor) | `layout_editor.py` | `create_layout_editor`, `edit_layout` for Jupyter & Colab |
| [`geoflow.visualization.map`](#interactive-leaflet-maps-geoflowvisualizationmap) | `map.py` | `Map` for interactive web tiles, GeoJSON, and raster layer overlays |
| [`geoflow.viz.publication`](#scientific-publishing-styles-geoflowvizpublication) | `publication.py` | `theme`, `figure`, `Table`, LaTeX/Markdown export |

---

## Publication Cartography Engine (`geoflow.viz.cartography`)

```python
from geoflow.viz.cartography import (
    plot_carto_map,
    draw_north_arrow,
    draw_scale_bar,
    format_degree_ticks,
)
```

### `plot_carto_map`

Generates ready-to-publish map figures with dynamic scale bars, north arrows, graticules, and legends. Available directly as `.plot_map()` on [`Image`](raster.md#image) and [`FeatureCollection`](vector.md#featurecollection).

```python
ax = plot_carto_map(
    image_or_gdf: Any,
    bands: list[str] | None = None,
    active_band: str | None = None,
    display_mode: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    title_fontsize: int | None = None,
    subtitle_fontsize: int | None = None,
    north_arrow: bool = True,
    north_loc: str = "top-right",
    north_pos: tuple[float, float] | None = None,
    north_size: float = 0.07,
    north_color: str = "#111827",
    scale_bar: bool = True,
    scale_loc: str = "bottom-left",
    scale_pos: tuple[float, float] | None = None,
    scale_fraction: float = 0.25,
    scale_custom_length_m: float | None = None,
    grid: bool = True,
    grid_style: str = ":",
    grid_color: str = "#6b7280",
    grid_alpha: float = 0.5,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    legend: bool = False,
    legend_title: str | None = None,
    legend_labels: dict[str, str] | None = None,
    legend_loc: str = "lower right",
    legend_pos: tuple[float, float] | None = None,
    discrete_bins: list[float] | None = None,
    discrete_colors: list[str] | None = None,
    interpolation: str = "bilinear",
    show_bottom: bool = True,
    show_left: bool = True,
    show_top: bool = False,
    show_right: bool = False,
    lat_orientation: str | int | float = "horizontal",
    lon_orientation: str | int | float = "horizontal",
    tick_fontsize: int = 8,
    tick_fontfamily: str | None = None,
    tick_fontweight: str = "normal",
    tick_color: str = "#111827",
    degree_precision: int | None = None,
    coord_format: str = "DD",
    graticule_interval: float | None = None,
    extent: tuple[float, float, float, float] | None = None,
    prune_corners: bool | str = True,
    corner_threshold: float = 0.025,
    tight_bbox: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Axes
```

#### Parameter Reference

| Category | Parameter | Default | Description |
|---|---|---|---|
| **Data & Stretch** | `bands` | `None` | List of bands to render (e.g. `["B4", "B3", "B2"]` for RGB). |
| | `cmap` | `"viridis"` | Matplotlib colormap for single-band continuous rasters. |
| | `vmin`, `vmax` | 2% / 98% | Numerical display limits (auto-calculated from percentiles if omitted). |
| | `discrete_bins` | `None` | Class break values for classified rasters (e.g. `[0.0, 0.2, 0.5, 0.8]`). |
| | `discrete_colors` | `None` | Color hex codes corresponding to discrete bins. |
| **Titles** | `title` | `None` | Main map title. |
| | `subtitle` | `None` | Subtitle displayed below the title with clean offset. |
| **North Arrow** | `north_arrow` | `True` | Draw cartographic 3D North Arrow. |
| | `north_loc` | `"top-right"` | Corner preset: `"top-right"`, `"top-left"`, `"bottom-right"`, `"bottom-left"`. |
| | `north_pos` | `None` | Freeform `(x, y)` position in axes fractions ($0.0 - 1.0$). |
| | `north_size` | `0.07` | Fraction of axes height. |
| **Scale Bar** | `scale_bar` | `True` | Draw dynamic segmented scale bar. |
| | `scale_loc` | `"bottom-left"` | Corner preset. |
| | `scale_pos` | `None` | Freeform `(x, y)` position in axes fractions. |
| | `scale_fraction` | `0.25` | Maximum width of scale bar relative to total map width. |
| | `scale_custom_length_m` | `None` | Explicit metric length in meters (e.g., `5000` for 5 km). |
| **Graticules & Ticks** | `show_bottom`, `show_left` | `True` | Display coordinates on bottom (X) and left (Y) neatlines. |
| | `show_top`, `show_right` | `False` | Display coordinates on top (X) and right (Y) neatlines. |
| | `lat_orientation` | `"horizontal"` | `"vertical"` rotates latitude labels $90^\circ$ to align with neatlines. |
| | `lon_orientation` | `"horizontal"` | Longitude text rotation. |
| | `coord_format` | `"DD"` | Format: `"DD"` (decimal degrees), `"DM"` (degrees-minutes), `"DMS"` (degrees-minutes-seconds). |
| | `graticule_interval` | `None` | Explicit coordinate grid step in degrees (e.g., `0.25`). |
| | `prune_corners` | `True` | Suppresses corner labels that collide with perpendicular axes text (`True`, `"x"`, `"y"`, `"both"`, `False`). |
| **Export** | `save_path` | `None` | Output file path (PNG, PDF, SVG, TIFF). |
| | `dpi` | `300` | Export resolution ($300$ or $600$ DPI for print). |

---

### Low-Level Cartographic Primitives

- `draw_north_arrow(ax, loc="top-right", pos=None, size=0.07, color="#111827")`: Draws standalone 3D arrow patch.
- `draw_scale_bar(ax, bounds, crs="EPSG:4326", loc="bottom-left", pos=None, max_fraction=0.25, custom_length_m=None)`: Computes curvature-corrected scale bar accounting for Earth ellipsoid and Web Mercator distortion.
- `format_degree_ticks(ax, bounds=None, crs="EPSG:4326", ...)`: Configures custom degree formatters with cardinal notation (`N`/`S`/`E`/`W`).

---

## In-Notebook Layout Editor (`geoflow.viz.layout_editor`)

```python
from geoflow.viz import create_layout_editor, edit_layout

editor = img.ndvi().edit_layout(
    title="Chittagong Vegetation",
    cmap="RdYlGn",
    dpi=300
)
display(editor)
```

Provides a live, interactive UI resembling **ArcGIS Pro Print Layout** or **QGIS Print Layout** directly inside **Google Colab, JupyterLab, and VS Code Notebooks**.

### Interactive Tabs
1. **📄 Page Setup**: Choose standard ISO print sizes (`A4`, `A3`, `A2`, `A1`, `A0`, `US Letter`, `US Tabloid`) or journal formats (`Nature Single Column`, `Nature Double Column`, `Science`), portrait/landscape orientation, and title typography.
2. **🧭 Carto**: Toggle, resize, and freely reposition the North Arrow and dynamically segmented metric Scale Bar.
3. **🗺️ Frame**: Interactive pan and zoom controls to frame study areas without re-running code cells.
4. **🌐 Graticule**: Control 4-sided coordinate tick visibility, vertical/horizontal label orientations, DD/DM/DMS notation, font families (e.g. Serif), and corner collision pruning.
5. **🎨 Colors**: Switch between 15+ remote sensing palettes (`RdYlGn`, `viridis`, `plasma`, `Spectral`, etc.), continuous stretch limits, or discrete thematic land-cover classification presets.
6. **💾 Export Canvas**: 1-click export to high-resolution PNG/PDF ($300 - 600$ DPI), instant Google Colab browser download button, and auto-generation of the exact reproducible Python code snippet.

---

## Interactive Leaflet Maps (`geoflow.visualization.map`)

```python
gf.Map(
    center: tuple[float, float] = (0.0, 0.0),
    zoom: int = 2,
    height: str = "600px",
    basemap: str = "HYBRID"
)
```

Renders an interactive browser map. Emits native HTML/JavaScript using Leaflet.js with responsive controls.

### Basemap Options
- `"HYBRID"`: Google Satellite imagery with roads and labels.
- `"SATELLITE"`: Esri World Imagery.
- `"ROADMAP"` / `"OpenStreetMap"`: OpenStreetMap standard.
- `"TERRAIN"`: Google Physical Terrain.
- `"Positron"` / `"Dark_Matter"`: CartoDB minimalist basemaps.

### Methods

| Method | Signature | Description |
|---|---|---|
| `.setCenter(lon, lat, zoom=10)` | `setCenter(...) -> Map` | Center map on coordinate. |
| `.centerObject(obj, zoom=None)` | `centerObject(...) -> Map` | Centers map on bounds of `Image`, `Geometry`, or `FeatureCollection`. |
| `.addLayer(ee_obj, vis_params=None, name=None, shown=True, opacity=1.0)` | `addLayer(...) -> Map` | Overlays a raster `Image`, `Geometry`, or `FeatureCollection` onto the map. Reprojects to EPSG:4326 automatically. |
| `.add_legend(title="Legend", colors=None, labels=None)` | `add_legend(...) -> Map` | Adds floating categorical legend box. |
| `.add_colorbar(vis_params, label="Value")` | `add_colorbar(...) -> Map` | Adds floating continuous gradient colorbar. |
| `.to_html(filepath=None)` / `.save(filepath)` | `to_html(...) -> str` | Generates self-contained HTML file. |

```python
m = gf.Map(center=[22.35, 91.85], zoom=11, basemap="HYBRID")
m.addLayer(sentinel_img.ndvi(), {"min": 0, "max": 0.8, "palette": "RdYlGn"}, "NDVI")
m.addLayer(gedi_points, name="GEDI LiDAR Shots")
m.save("interactive_report.html")
```

---

## Scientific Publishing Styles (`geoflow.viz.publication`)

```python
from geoflow.viz.publication import theme, figure, table, Table
```

### Journal Stylesheet Presets (`theme`)

```python
gf.viz.theme("nature")  # "nature", "science", "minimal", or "dark"
```

Configures Matplotlib global `rcParams` for peer-reviewed journal submission:
- Eliminates unnecessary top/right spines.
- Sets publication-standard font sizes ($8\text{pt} - 10\text{pt}$ sans-serif).
- Configures default figure DPI to 300 and savefig DPI to 600.

### Exact Journal Column Figures (`figure`)

```python
fig, ax = gf.viz.figure(width="single_column", aspect_ratio=0.75, dpi=300)
```
- `"single_column"`: Exactly 89 mm (~3.5 inches).
- `"double_column"`: Exactly 183 mm (~7.2 inches).

### Scientific Publication Tables (`Table`)

```python
tbl = gf.viz.table(df)
print(tbl.to_latex(caption="Model Accuracy Comparison", label="tab:accuracy"))
tbl.save("table1.tex")
```

Formats summary statistics into publication-ready LaTeX (`booktabs`-compatible), GitHub-flavored Markdown, or HTML.
