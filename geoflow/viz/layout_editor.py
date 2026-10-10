"""
In-Notebook Interactive Cartographic Layout Editor for GeoFlow.
Provides an ArcGIS/QGIS-style Layout View inside Jupyter and Google Colab notebooks.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Union
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from geoflow.viz.cartography import plot_carto_map


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
    prune_corners: bool = True,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    save_path: str = "publication_map.png",
    dpi: int = 300,
    **kwargs: Any,
) -> Any:
    """
    Launch the interactive ArcGIS-style Layout Editor in a Jupyter or Google Colab notebook.

    Parameters:
    - image_or_gdf: A GeoFlow Image, Raster, or FeatureCollection.
    - Initial layout parameters to pre-populate the interactive widgets.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML
    except ImportError:
        raise ImportError(
            "The interactive Layout Editor requires 'ipywidgets'. "
            "Please install it via: pip install ipywidgets"
        )

    # Detect initial bands/label
    default_cb_label = colorbar_label
    if default_cb_label is None and hasattr(image_or_gdf, "bands") and image_or_gdf.bands:
        default_cb_label = image_or_gdf.bands[0]
    default_cb_label = default_cb_label or "Value"

    # -------------------------------------------------------------
    # 1. UI Control Widgets
    # -------------------------------------------------------------
    # Title & Canvas
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
    w_aspect = widgets.Dropdown(
        options=[
            ("Standard (8 x 7 in)", (8, 7)),
            ("Square (8 x 8 in)", (8, 8)),
            ("Landscape (10 x 6 in)", (10, 6)),
            ("Nature Single Col (3.5 x 3.2 in)", (3.5, 3.2)),
            ("Nature Double Col (7.2 x 5.5 in)", (7.2, 5.5)),
        ],
        value=(8, 7),
        description="Page Size:",
        layout=widgets.Layout(width="95%"),
    )

    # North Arrow & Scale Bar
    w_north_arrow = widgets.Checkbox(value=north_arrow, description="Show North Arrow")
    w_north_loc = widgets.Dropdown(
        options=["top-right", "top-left", "bottom-right", "bottom-left"],
        value="top-right",
        description="Position:",
        layout=widgets.Layout(width="95%"),
    )
    w_scale_bar = widgets.Checkbox(value=scale_bar, description="Show Scale Bar")
    w_scale_loc = widgets.Dropdown(
        options=["bottom-left", "bottom-right"],
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
        layout=widgets.Layout(width="95%"),
    )

    # Graticules & Coordinates
    w_show_bottom = widgets.Checkbox(value=show_bottom, description="Bottom (S)")
    w_show_left = widgets.Checkbox(value=show_left, description="Left (W)")
    w_show_top = widgets.Checkbox(value=show_top, description="Top (N)")
    w_show_right = widgets.Checkbox(value=show_right, description="Right (E)")

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
        layout=widgets.Layout(width="95%"),
    )

    # Symbology & Colors
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
    w_colorbar = widgets.Checkbox(value=colorbar, description="Show Colorbar")
    w_cb_label = widgets.Text(
        value=default_cb_label,
        description="Legend Title:",
        layout=widgets.Layout(width="95%"),
    )

    # Export & Code
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
        description="🔄 Refresh Preview",
        button_style="info",
        icon="refresh",
        layout=widgets.Layout(width="48%"),
    )
    btn_export = widgets.Button(
        description="💾 Export to Disk",
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
        description="📋 Show Python Code",
        button_style="",
        icon="code",
        layout=widgets.Layout(width="48%"),
    )

    # Status notification area
    w_status = widgets.HTML(
        value="<div style='color:#6b7280; font-size:12px;'>Ready. Adjust any parameter on the left to customize the layout.</div>"
    )
    w_code_display = widgets.HTML(value="")

    # The Map Viewport Canvas
    out_map = widgets.Output(layout=widgets.Layout(width="100%", min_height="520px"))

    # -------------------------------------------------------------
    # 2. Rendering Function
    # -------------------------------------------------------------
    def render_map(export_path: Optional[str] = None, export_dpi: Optional[int] = None):
        out_map.clear_output(wait=True)
        with out_map:
            fig, ax = plt.subplots(figsize=w_aspect.value, dpi=export_dpi or 120)
            try:
                plot_carto_map(
                    image_or_gdf,
                    title=w_title.value,
                    subtitle=w_subtitle.value,
                    cmap=w_cmap.value,
                    vmin=vmin,
                    vmax=vmax,
                    north_arrow=w_north_arrow.value,
                    scale_bar=w_scale_bar.value,
                    grid=w_grid.value,
                    grid_style=w_grid_style.value,
                    grid_alpha=w_grid_alpha.value,
                    colorbar=w_colorbar.value,
                    colorbar_label=w_cb_label.value,
                    show_bottom=w_show_bottom.value,
                    show_left=w_show_left.value,
                    show_top=w_show_top.value,
                    show_right=w_show_right.value,
                    lat_orientation=w_lat_orient.value,
                    lon_orientation=w_lon_orient.value,
                    tick_fontsize=w_fontsize.value,
                    tick_fontfamily=w_fontfamily.value,
                    degree_precision=w_deg_prec.value,
                    prune_corners=w_prune.value,
                    ax=ax,
                    save_path=export_path,
                    dpi=export_dpi or 120,
                )
                display(fig)
            finally:
                plt.close(fig)

    # -------------------------------------------------------------
    # 3. Event Listeners
    # -------------------------------------------------------------
    def on_change(change=None):
        if w_auto_refresh.value:
            render_map()

    # Bind on_change to interactive widgets
    interactive_widgets = [
        w_title, w_subtitle, w_aspect, w_north_arrow, w_north_loc,
        w_scale_bar, w_scale_loc, w_scale_frac, w_show_bottom, w_show_left,
        w_show_top, w_show_right, w_lat_orient, w_lon_orient, w_deg_prec,
        w_prune, w_grid, w_grid_style, w_grid_alpha, w_fontfamily,
        w_fontsize, w_cmap, w_colorbar, w_cb_label
    ]
    for w in interactive_widgets:
        w.observe(on_change, names="value")

    def on_refresh_clicked(b):
        render_map()
        w_status.value = "<div style='color:#10b981; font-size:12px;'>✓ Preview updated.</div>"

    def on_export_clicked(b):
        out_f = w_out_file.value.strip() or "publication_map.png"
        target_dpi = w_dpi.value
        w_status.value = f"<div style='color:#3b82f6; font-size:12px;'>Rendering {target_dpi} DPI map to <b>{out_f}</b>...</div>"
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
        
        # Check if running in Google Colab
        if "google.colab" in sys.modules:
            try:
                from google.colab import files as colab_files
                colab_files.download(str(p))
                w_status.value = f"<div style='color:#10b981; font-size:13px;'>✓ Colab download started for <b>{out_f}</b></div>"
                return
            except Exception as e:
                pass
        w_status.value = (
            f"<div style='color:#10b981; font-size:13px;'>"
            f"✓ File saved locally at: <code>{p}</code></div>"
        )

    def on_copy_code_clicked(b):
        code_str = (
            f"# Reproduce this exact cartographic layout in Python:\n"
            f"ax = img.plot_map(\n"
            f"    title={repr(w_title.value)},\n"
            f"    subtitle={repr(w_subtitle.value)},\n"
            f"    cmap={repr(w_cmap.value)},\n"
            f"    north_arrow={w_north_arrow.value},\n"
            f"    scale_bar={w_scale_bar.value},\n"
            f"    grid={w_grid.value},\n"
            f"    grid_style={repr(w_grid_style.value)},\n"
            f"    colorbar={w_colorbar.value},\n"
            f"    colorbar_label={repr(w_cb_label.value)},\n"
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
    # 4. Assemble ArcGIS-Style Tabbed Layout
    # -------------------------------------------------------------
    tab_canvas = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>Map Titles & Paper Canvas</b>"),
        w_title, w_subtitle, w_aspect,
    ])

    tab_carto = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>North Arrow & Dynamic Scale Bar</b>"),
        w_north_arrow, w_north_loc,
        widgets.HTML("<hr style='margin:4px 0;'>"),
        w_scale_bar, w_scale_loc, w_scale_frac,
    ])

    tab_graticule = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>4-Sided Graticule Coordinates</b>"),
        widgets.HBox([w_show_bottom, w_show_left]),
        widgets.HBox([w_show_top, w_show_right]),
        w_lat_orient, w_lon_orient, w_deg_prec, w_prune,
        widgets.HTML("<hr style='margin:4px 0;'><b style='color:#1e3a8a;'>Grid Lines & Font</b>"),
        w_grid, w_grid_style, w_grid_alpha, w_fontfamily, w_fontsize,
    ])

    tab_symbology = widgets.VBox([
        widgets.HTML("<b style='color:#1e3a8a;'>Symbology, Palettes & Legend</b>"),
        w_cmap, w_colorbar, w_cb_label,
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
        children=[tab_canvas, tab_carto, tab_graticule, tab_symbology, tab_export],
        layout=widgets.Layout(width="380px")
    )
    tabs.set_title(0, "📄 Canvas")
    tabs.set_title(1, "🧭 Carto")
    tabs.set_title(2, "🌐 Graticule")
    tabs.set_title(3, "🎨 Colors")
    tabs.set_title(4, "💾 Export")

    # Header Bar
    header = widgets.HTML(
        "<div style='background:linear-gradient(90deg, #1e3a8a 0%, #0284c7 100%); "
        "color:white; padding:8px 14px; border-radius:6px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;'>"
        "<span style='font-weight:bold; font-size:14px;'>🗺️ GeoFlow Interactive Layout Editor (ArcGIS View)</span>"
        "<span style='font-size:11px; background:rgba(255,255,255,0.2); padding:2px 8px; border-radius:4px;'>Colab & Jupyter</span>"
        "</div>"
    )

    # Left Control Column + Right Preview Column
    left_panel = widgets.VBox([tabs, w_auto_refresh], layout=widgets.Layout(width="380px", margin="0 15px 0 0"))
    right_panel = widgets.VBox([out_map], layout=widgets.Layout(flex="1 1 auto", min_width="450px"))

    main_view = widgets.VBox([
        header,
        widgets.HBox([left_panel, right_panel], layout=widgets.Layout(width="100%"))
    ], layout=widgets.Layout(padding="10px", border="1px solid #e2e8f0", border_radius="8px", background="#f8fafc"))

    # Initial Render
    render_map()

    return main_view
