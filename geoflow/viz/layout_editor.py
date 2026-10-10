"""
In-Notebook Interactive Cartographic Layout Editor for GeoFlow.
Provides an ArcGIS/QGIS-style Layout View inside Jupyter and Google Colab notebooks.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import math
from pathlib import Path
import matplotlib.pyplot as plt

from geoflow.viz.cartography import plot_carto_map


# Built-in Discrete Classification Presets
DISCRETE_PRESETS: Dict[str, Dict[str, str]] = {
    "NDVI 4-Class (Water, Soil, Moderate, Dense)": {
        "Water / Cloud (<0.0)": "#2563eb",
        "Barren / Soil (0.0 - 0.2)": "#d97706",
        "Moderate Veg (0.2 - 0.5)": "#84cc16",
        "Dense Canopy (>0.5)": "#15803d",
    },
    "NDVI 5-Class (Detailed)": {
        "Water (<0.0)": "#1d4ed8",
        "Built-up / Bare (0.0 - 0.15)": "#b45309",
        "Sparse Veg (0.15 - 0.35)": "#eab308",
        "Moderate Veg (0.35 - 0.60)": "#65a30d",
        "Dense Forest (>0.60)": "#14532d",
    },
    "Land Cover 5-Class": {
        "Water Body": "#0284c7",
        "Forest / Woodland": "#16a34a",
        "Agricultural Field": "#ca8a04",
        "Urban / Built-up": "#dc2626",
        "Barren Land": "#a8a29e",
    },
}

# Standard Paper Sizes (Width, Height in inches for Landscape)
PAPER_SIZES: Dict[str, Tuple[float, float]] = {
    "A4 (11.7 x 8.3 in)": (11.69, 8.27),
    "A3 (16.5 x 11.7 in)": (16.54, 11.69),
    "A5 (8.3 x 5.8 in)": (8.27, 5.83),
    "US Letter (11.0 x 8.5 in)": (11.0, 8.5),
    "US Legal (14.0 x 8.5 in)": (14.0, 8.5),
    "Tabloid / Ledger (17.0 x 11.0 in)": (17.0, 11.0),
    "Square (8.0 x 8.0 in)": (8.0, 8.0),
    "Nature Double Column (7.2 x 5.5 in)": (7.2, 5.5),
    "Nature Single Column (3.5 x 3.2 in)": (3.5, 3.2),
}

GRATICULE_INTERVALS: Dict[str, Optional[float]] = {
    "Auto (Adaptive)": None,
    "1° 00' (1.0°)": 1.0,
    "0° 30' (30 minutes)": 0.5,
    "0° 15' (15 minutes)": 0.25,
    "0° 10' (10 minutes)": 10.0 / 60.0,
    "0° 05' (5 minutes)": 5.0 / 60.0,
    "0° 02' (2 minutes)": 2.0 / 60.0,
    "0° 01' (1 minute)": 1.0 / 60.0,
    "0° 00' 30\" (30 seconds)": 30.0 / 3600.0,
    "0° 00' 15\" (15 seconds)": 15.0 / 3600.0,
}


def _make_preview_obj(obj: Any) -> Any:
    """Create a lightweight downsampled thumbnail for blazing-fast live preview."""
    if hasattr(obj, "height") and hasattr(obj, "width") and hasattr(obj, "_data"):
        h, w = obj.height, obj.width
        step = max(1, max(h, w) // 384)
        if step > 1:
            try:
                from rasterio.transform import Affine
                sub_data = obj._data[:, ::step, ::step].copy()
                t = obj.transform
                new_t = Affine(t.a * step, t.b, t.c, t.d, t.e * step, t.f)
                return type(obj)(sub_data, transform=new_t, crs=obj.crs, band_names=obj.bands, nodata=obj.nodata)
            except Exception:
                return obj
    return obj


def create_layout_editor(
    image_or_gdf: Any,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    north_arrow: bool = True,
    scale_bar: bool = True,
    grid: bool = True,
    show_bottom: bool = True,
    show_left: bool = True,
    show_top: bool = False,
    show_right: bool = False,
    lat_orientation: str = "vertical",
    lon_orientation: str = "horizontal",
    degree_precision: int = 2,
    coord_format: str = "DD",
    prune_corners: bool = True,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    save_path: str = "publication_map.png",
    dpi: int = 300,
    **kwargs: Any,
) -> Any:
    """
    Launch the interactive ArcGIS-style Layout Editor in a Jupyter or Google Colab notebook.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML
    except ImportError:
        raise ImportError(
            "The interactive Layout Editor requires 'ipywidgets'. "
            "Please install it via: pip install ipywidgets"
        )

    # Cache fast preview object
    preview_obj = _make_preview_obj(image_or_gdf)

    default_cb_label = colorbar_label
    if default_cb_label is None and hasattr(image_or_gdf, "bands") and image_or_gdf.bands:
        default_cb_label = image_or_gdf.bands[0]
    default_cb_label = default_cb_label or "Value"

    # -------------------------------------------------------------
    # 1. UI Control Widgets
    # -------------------------------------------------------------
    # Tab 1: Canvas & Page Layout
    w_title = widgets.Text(
        value=title or "Study Area Map",
        description="Title:",
        layout=widgets.Layout(width="95%"),
    )
    w_subtitle = widgets.Text(
        value=subtitle or "GeoFlow Publication Layout",
        description="Subtitle:",
        layout=widgets.Layout(width="95%"),
    )
    w_paper = widgets.Dropdown(
        options=list(PAPER_SIZES.items()),
        value=PAPER_SIZES["A4 (11.7 x 8.3 in)"],
        description="Paper Size:",
        layout=widgets.Layout(width="95%"),
    )
    w_orient = widgets.Dropdown(
        options=["Landscape", "Portrait"],
        value="Landscape",
        description="Orientation:",
        layout=widgets.Layout(width="95%"),
    )

    # Tab 2: Cartographic Elements (North Arrow & Scale Bar)
    w_north_arrow = widgets.Checkbox(value=north_arrow, description="Show North Arrow")
    w_north_loc = widgets.Dropdown(
        options=["top-right", "top-left", "bottom-right", "bottom-left"],
        value="top-right",
        description="Position:",
        layout=widgets.Layout(width="95%"),
    )
    w_north_size = widgets.FloatSlider(
        value=0.07,
        min=0.03,
        max=0.15,
        step=0.01,
        description="Arrow Size:",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )

    w_scale_bar = widgets.Checkbox(value=scale_bar, description="Show Dynamic Scale Bar")
    w_scale_loc = widgets.Dropdown(
        options=["bottom-left", "bottom-right", "top-left", "top-right"],
        value="bottom-left",
        description="Position:",
        layout=widgets.Layout(width="95%"),
    )
    w_scale_frac = widgets.FloatSlider(
        value=0.25,
        min=0.10,
        max=0.45,
        step=0.05,
        description="Bar Size:",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )

    # Tab 3: Map Frame Navigation (Zoom & Pan)
    w_zoom = widgets.FloatSlider(
        value=1.0,
        min=0.4,
        max=3.5,
        step=0.1,
        description="Zoom:",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    w_pan_x = widgets.FloatSlider(
        value=0.0,
        min=-60.0,
        max=60.0,
        step=5.0,
        description="Pan X (%):",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    w_pan_y = widgets.FloatSlider(
        value=0.0,
        min=-60.0,
        max=60.0,
        step=5.0,
        description="Pan Y (%):",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    btn_reset_frame = widgets.Button(
        description="⟲ Reset Zoom & Pan",
        button_style="warning",
        layout=widgets.Layout(width="95%"),
    )

    # Tab 4: Graticules & Coordinates
    w_show_bottom = widgets.Checkbox(value=show_bottom, description="Bottom (S)")
    w_show_left = widgets.Checkbox(value=show_left, description="Left (W)")
    w_show_top = widgets.Checkbox(value=show_top, description="Top (N)")
    w_show_right = widgets.Checkbox(value=show_right, description="Right (E)")

    w_coord_fmt = widgets.Dropdown(
        options=[
            ("Decimal Degrees (DD: 22.34° N)", "DD"),
            ("Degrees & Minutes (DM: 22° 20.4' N)", "DM"),
            ("Degrees, Min, Sec (DMS: 22° 20' 24\" N)", "DMS"),
        ],
        value=coord_format,
        description="Format:",
        layout=widgets.Layout(width="95%"),
    )
    w_grat_interval = widgets.Dropdown(
        options=list(GRATICULE_INTERVALS.items()),
        value=None,
        description="Interval:",
        layout=widgets.Layout(width="95%"),
    )
    w_lat_orient = widgets.Dropdown(
        options=[("Vertical (90° Rotated)", "vertical"), ("Horizontal (0°)", "horizontal")],
        value=lat_orientation,
        description="Lat Orient:",
        layout=widgets.Layout(width="95%"),
    )
    w_lon_orient = widgets.Dropdown(
        options=[("Horizontal (0°)", "horizontal"), ("Vertical (90°)", "vertical")],
        value=lon_orientation,
        description="Lon Orient:",
        layout=widgets.Layout(width="95%"),
    )
    w_deg_prec = widgets.IntSlider(
        value=degree_precision,
        min=1,
        max=4,
        description="Precision:",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    w_prune = widgets.Checkbox(value=prune_corners, description="Prune Overlapping Corners")

    w_grid = widgets.Checkbox(value=grid, description="Graticule Grid Lines")
    w_grid_style = widgets.Dropdown(
        options=[("Dotted (:)", ":"), ("Dashed (--)", "--"), ("Solid (-)", "-")],
        value=":",
        description="Line Style:",
        layout=widgets.Layout(width="95%"),
    )
    w_grid_alpha = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Opacity:",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    w_fontfamily = widgets.Dropdown(
        options=["sans-serif", "serif", "monospace"],
        value="sans-serif",
        description="Font:",
        layout=widgets.Layout(width="95%"),
    )
    w_fontsize = widgets.IntSlider(
        value=8,
        min=6,
        max=14,
        description="Font Size:",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )

    # Tab 5: Symbology & Classified Legends
    w_symbology = widgets.Dropdown(
        options=[("Continuous Colorbar", "continuous"), ("Discrete Classified Legend", "discrete")],
        value="continuous",
        description="Symbology:",
        layout=widgets.Layout(width="95%"),
    )
    w_cmap = widgets.Dropdown(
        options=[
            "RdYlGn", "viridis", "plasma", "inferno", "magma", "cividis",
            "Spectral", "terrain", "coolwarm", "turbo", "YlGn", "Blues",
            "RdYlGn_r", "viridis_r", "plasma_r", "Spectral_r"
        ],
        value=cmap,
        description="Colormap:",
        layout=widgets.Layout(width="95%"),
    )
    w_colorbar = widgets.Checkbox(value=colorbar, description="Show Continuous Bar")
    w_cb_label = widgets.Text(
        value=default_cb_label,
        description="Bar Label:",
        layout=widgets.Layout(width="95%"),
    )

    w_discrete_preset = widgets.Dropdown(
        options=list(DISCRETE_PRESETS.keys()),
        value="NDVI 4-Class (Water, Soil, Moderate, Dense)",
        description="Classes:",
        layout=widgets.Layout(width="95%"),
    )
    w_discrete_loc = widgets.Dropdown(
        options=["lower right", "upper right", "lower left", "upper left"],
        value="lower right",
        description="Legend Loc:",
        layout=widgets.Layout(width="95%"),
    )
    w_discrete_title = widgets.Text(
        value="Classification",
        description="Legend Title:",
        layout=widgets.Layout(width="95%"),
    )

    # Tab 6: Export & Reproducible Code
    w_out_file = widgets.Text(
        value=save_path,
        description="Filename:",
        layout=widgets.Layout(width="95%"),
    )
    w_dpi = widgets.Dropdown(
        options=[("300 DPI (Publication Standard)", 300), ("600 DPI (Ultra High-Res)", 600), ("150 DPI (Quick Draft)", 150)],
        value=dpi if dpi in (150, 300, 600) else 300,
        description="DPI:",
        layout=widgets.Layout(width="95%"),
    )
    w_auto_refresh = widgets.Checkbox(value=True, description="⚡ Live Auto-Update")

    btn_refresh = widgets.Button(
        description="🔄 Refresh",
        button_style="info",
        icon="refresh",
        layout=widgets.Layout(width="48%"),
    )
    btn_export = widgets.Button(
        description="💾 Export (Disk)",
        button_style="success",
        icon="download",
        layout=widgets.Layout(width="48%"),
    )
    btn_colab_download = widgets.Button(
        description="📥 Download (Colab)",
        button_style="primary",
        icon="cloud-download",
        layout=widgets.Layout(width="48%"),
    )
    btn_copy_code = widgets.Button(
        description="📋 Python Code",
        button_style="",
        icon="code",
        layout=widgets.Layout(width="48%"),
    )

    w_status = widgets.HTML(
        value="<div style='color:#6b7280; font-size:12px;'>Ready. Adjust any parameter on the left.</div>"
    )
    w_code_display = widgets.HTML(value="")

    # The Map Viewport Canvas
    out_map = widgets.Output(layout=widgets.Layout(width="100%", min_height="540px"))

    # -------------------------------------------------------------
    # 2. Geometry & Extent Calculation
    # -------------------------------------------------------------
    def get_current_extent() -> Tuple[float, float, float, float]:
        minx, miny, maxx, maxy = image_or_gdf.bounds
        w = maxx - minx
        h = maxy - miny
        cx = (minx + maxx) / 2.0 + (w_pan_x.value / 100.0) * w
        cy = (miny + maxy) / 2.0 + (w_pan_y.value / 100.0) * h
        z = max(0.2, w_zoom.value)
        nw = w / z
        nh = h / z
        return (cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0)

    def get_figure_dimensions() -> Tuple[float, float]:
        bw, bh = w_paper.value
        if w_orient.value == "Landscape":
            return (max(bw, bh), min(bw, bh))
        else:
            return (min(bw, bh), max(bw, bh))

    # -------------------------------------------------------------
    # 3. Fast Rendering Function
    # -------------------------------------------------------------
    def render_map(export_path: Optional[str] = None, export_dpi: Optional[int] = None):
        out_map.clear_output(wait=True)
        # Use full-res object when exporting, fast preview thumbnail for live UI
        target_obj = image_or_gdf if export_path is not None else preview_obj
        fig_size = get_figure_dimensions()
        cur_extent = get_current_extent()

        is_discrete = (w_symbology.value == "discrete")
        leg_labels = DISCRETE_PRESETS.get(w_discrete_preset.value) if is_discrete else None

        with out_map:
            fig, ax = plt.subplots(figsize=fig_size, dpi=export_dpi or 110)
            try:
                plot_carto_map(
                    target_obj,
                    title=w_title.value,
                    subtitle=w_subtitle.value,
                    cmap=w_cmap.value,
                    vmin=vmin,
                    vmax=vmax,
                    north_arrow=w_north_arrow.value,
                    north_loc=w_north_loc.value,
                    north_size=w_north_size.value,
                    scale_bar=w_scale_bar.value,
                    scale_loc=w_scale_loc.value,
                    scale_fraction=w_scale_frac.value,
                    grid=w_grid.value,
                    grid_style=w_grid_style.value,
                    grid_alpha=w_grid_alpha.value,
                    colorbar=(w_colorbar.value and not is_discrete),
                    colorbar_label=w_cb_label.value,
                    legend=is_discrete,
                    legend_labels=leg_labels,
                    legend_loc=w_discrete_loc.value,
                    legend_title=w_discrete_title.value if is_discrete else None,
                    show_bottom=w_show_bottom.value,
                    show_left=w_show_left.value,
                    show_top=w_show_top.value,
                    show_right=w_show_right.value,
                    lat_orientation=w_lat_orient.value,
                    lon_orientation=w_lon_orient.value,
                    tick_fontsize=w_fontsize.value,
                    tick_fontfamily=w_fontfamily.value,
                    degree_precision=w_deg_prec.value,
                    coord_format=w_coord_fmt.value,
                    graticule_interval=w_grat_interval.value,
                    extent=cur_extent,
                    prune_corners=w_prune.value,
                    ax=ax,
                    save_path=export_path,
                    dpi=export_dpi or 110,
                )
                display(fig)
            finally:
                plt.close(fig)

    # -------------------------------------------------------------
    # 4. Event Handlers
    # -------------------------------------------------------------
    def on_change(change=None):
        if w_auto_refresh.value:
            render_map()

    def on_reset_frame(b):
        w_zoom.value = 1.0
        w_pan_x.value = 0.0
        w_pan_y.value = 0.0
        render_map()

    btn_reset_frame.on_click(on_reset_frame)

    # Interactive observers
    interactive_widgets = [
        w_title, w_subtitle, w_paper, w_orient,
        w_north_arrow, w_north_loc, w_north_size,
        w_scale_bar, w_scale_loc, w_scale_frac,
        w_zoom, w_pan_x, w_pan_y,
        w_show_bottom, w_show_left, w_show_top, w_show_right,
        w_coord_fmt, w_grat_interval, w_lat_orient, w_lon_orient, w_deg_prec,
        w_prune, w_grid, w_grid_style, w_grid_alpha, w_fontfamily, w_fontsize,
        w_symbology, w_cmap, w_colorbar, w_cb_label,
        w_discrete_preset, w_discrete_loc, w_discrete_title
    ]
    for w in interactive_widgets:
        w.observe(on_change, names="value")

    def on_refresh_clicked(b):
        render_map()
        w_status.value = "<div style='color:#10b981; font-size:12px;'>✓ Preview updated.</div>"

    def on_export_clicked(b):
        out_f = w_out_file.value.strip() or "publication_map.png"
        target_dpi = w_dpi.value
        w_status.value = f"<div style='color:#3b82f6; font-size:12px;'>Rendering {target_dpi} DPI full-res map to <b>{out_f}</b>...</div>"
        render_map(export_path=out_f, export_dpi=target_dpi)
        p = Path(out_f).resolve()
        w_status.value = (
            f"<div style='color:#10b981; font-weight:bold; font-size:13px;'>"
            f"✓ Successfully exported publication map to: <code>{p}</code> ({target_dpi} DPI)</div>"
        )

    def on_colab_download_clicked(b):
        out_f = w_out_file.value.strip() or "publication_map.png"
        target_dpi = w_dpi.value
        w_status.value = f"<div style='color:#3b82f6; font-size:12px;'>Generating {target_dpi} DPI map for download...</div>"
        render_map(export_path=out_f, export_dpi=target_dpi)
        p = Path(out_f).resolve()
        if "google.colab" in sys.modules:
            try:
                from google.colab import files as colab_files
                colab_files.download(str(p))
                w_status.value = f"<div style='color:#10b981; font-size:13px;'>✓ Colab download started for <b>{out_f}</b></div>"
                return
            except Exception:
                pass
        w_status.value = f"<div style='color:#10b981; font-size:13px;'>✓ Saved locally: <code>{p}</code></div>"

    def on_copy_code_clicked(b):
        is_discrete = (w_symbology.value == "discrete")
        ext = get_current_extent()
        code_str = (
            f"# Reproduce this exact cartographic layout in Python:\n"
            f"ax = img.plot_map(\n"
            f"    title={repr(w_title.value)},\n"
            f"    subtitle={repr(w_subtitle.value)},\n"
            f"    cmap={repr(w_cmap.value)},\n"
            f"    north_arrow={w_north_arrow.value},\n"
            f"    north_loc={repr(w_north_loc.value)},\n"
            f"    north_size={w_north_size.value},\n"
            f"    scale_bar={w_scale_bar.value},\n"
            f"    scale_loc={repr(w_scale_loc.value)},\n"
            f"    scale_fraction={w_scale_frac.value},\n"
            f"    grid={w_grid.value},\n"
            f"    grid_style={repr(w_grid_style.value)},\n"
            f"    coord_format={repr(w_coord_fmt.value)},\n"
            f"    graticule_interval={w_grat_interval.value},\n"
            f"    extent={tuple(round(x, 5) for x in ext)},\n"
            f"    show_bottom={w_show_bottom.value},\n"
            f"    show_left={w_show_left.value},\n"
            f"    show_top={w_show_top.value},\n"
            f"    show_right={w_show_right.value},\n"
            f"    lat_orientation={repr(w_lat_orient.value)},\n"
            f"    lon_orientation={repr(w_lon_orient.value)},\n"
            f"    tick_fontfamily={repr(w_fontfamily.value)},\n"
            f"    degree_precision={w_deg_prec.value},\n"
            f"    prune_corners={w_prune.value},\n"
            f"    save_path={repr(w_out_file.value)},\n"
            f"    dpi={w_dpi.value},\n"
            f")"
        )
        w_code_display.value = (
            f"<div style='margin-top:8px; padding:10px; background:#1e293b; color:#f8fafc; "
            f"border-radius:6px; font-family:monospace; font-size:11px; white-space:pre-wrap;'>"
            f"{code_str}</div>"
        )

    btn_refresh.on_click(on_refresh_clicked)
    btn_export.on_click(on_export_clicked)
    btn_colab_download.on_click(on_colab_download_clicked)
    btn_copy_code.on_click(on_copy_code_clicked)

    # -------------------------------------------------------------
    # 5. Assemble Tabs
    # -------------------------------------------------------------
    tab_canvas = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>Map Titles & Paper Canvas</b>"),
        w_title, w_subtitle,
        widgets.HTML("<hr style='margin:4px 0;'><b style='color:#1e3a8a;'>Paper Size & Orientation</b>"),
        w_paper, w_orient,
    ])

    tab_carto = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>North Arrow</b>"),
        w_north_arrow, w_north_loc, w_north_size,
        widgets.HTML("<hr style='margin:6px 0;'><b style='color:#1e3a8a;'>Dynamic Scale Bar</b>"),
        w_scale_bar, w_scale_loc, w_scale_frac,
    ])

    tab_frame = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>Map Frame Navigation (Zoom & Pan)</b>"),
        widgets.HTML("<div style='font-size:11px; color:#6b7280; margin-bottom:4px;'>Adjust study area boundaries within the neatline:</div>"),
        w_zoom, w_pan_x, w_pan_y,
        btn_reset_frame,
    ])

    tab_graticule = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>Coordinates & Intervals</b>"),
        w_coord_fmt, w_grat_interval,
        widgets.HTML("<hr style='margin:4px 0;'><b style='color:#1e3a8a;'>Tick Placement & Orientation</b>"),
        widgets.HBox([w_show_bottom, w_show_left]),
        widgets.HBox([w_show_top, w_show_right]),
        w_lat_orient, w_lon_orient, w_deg_prec, w_prune,
        widgets.HTML("<hr style='margin:4px 0;'><b style='color:#1e3a8a;'>Grid Lines & Font</b>"),
        w_grid, w_grid_style, w_grid_alpha, w_fontfamily, w_fontsize,
    ])

    tab_symbology = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>Symbology Type & Palettes</b>"),
        w_symbology,
        widgets.HTML("<hr style='margin:4px 0;'>"),
        w_cmap, w_colorbar, w_cb_label,
        widgets.HTML("<hr style='margin:4px 0;'><b style='color:#1e3a8a;'>Discrete Classified Legend</b>"),
        w_discrete_preset, w_discrete_loc, w_discrete_title,
    ])

    tab_export = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>Publication Export & Code Generation</b>"),
        w_out_file, w_dpi,
        widgets.HBox([btn_export, btn_colab_download]),
        widgets.HBox([btn_refresh, btn_copy_code]),
        w_status,
        w_code_display,
    ])

    tabs = widgets.Tab(
        children=[tab_canvas, tab_carto, tab_frame, tab_graticule, tab_symbology, tab_export],
        layout=widgets.Layout(width="400px")
    )
    tabs.set_title(0, "📄 Canvas")
    tabs.set_title(1, "🧭 Carto")
    tabs.set_title(2, "🗺️ Frame")
    tabs.set_title(3, "🌐 Graticule")
    tabs.set_title(4, "🎨 Colors")
    tabs.set_title(5, "💾 Export")

    header = widgets.HTML(
        "<div style='background:linear-gradient(90deg, #1e3a8a 0%, #0284c7 100%); "
        "color:white; padding:8px 14px; border-radius:6px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;'>"
        "<span style='font-weight:bold; font-size:14px;'>🗺️ GeoFlow Interactive Layout Editor (ArcGIS Pro View)</span>"
        "<span style='font-size:11px; background:rgba(255,255,255,0.2); padding:2px 8px; border-radius:4px;'>Colab & Jupyter</span>"
        "</div>"
    )

    left_panel = widgets.VBox([tabs, w_auto_refresh], layout=widgets.Layout(width="400px", margin="0 15px 0 0"))
    right_panel = widgets.VBox([out_map], layout=widgets.Layout(flex="1 1 auto", min_width="480px"))

    main_view = widgets.VBox([
        header,
        widgets.HBox([left_panel, right_panel], layout=widgets.Layout(width="100%"))
    ], layout=widgets.Layout(padding="10px", border="1px solid #e2e8f0", border_radius="8px", background="#f8fafc"))

    # Initial Render
    render_map()

    return main_view
