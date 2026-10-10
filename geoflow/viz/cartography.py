"""
Publication-Grade Cartographic Map Rendering Engine for GeoFlow.
Provides north arrows, dynamic scale bars, degree-formatted graticules, and legends.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import math
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pyproj import Transformer

from geoflow.core.crs import CRS


def draw_north_arrow(
    ax: plt.Axes,
    loc: str = "top-right",
    pos: Optional[Tuple[float, float]] = None,
    size: float = 0.07,
    color: str = "#111827",
    text_color: str = "#111827",
):
    """
    Draw a professional 3D-styled cartographic North Arrow on Matplotlib Axes.
    loc: 'top-right', 'top-left', 'bottom-right', 'bottom-left'
    pos: Optional (x, y) coordinates in axes fraction (0.0 to 1.0) for arbitrary placement.
    size: fraction of axes height
    """
    if pos is not None:
        cx, cy = pos
    else:
        loc_positions = {
            "top-right": (0.92, 0.88),
            "top-left": (0.08, 0.88),
            "bottom-right": (0.92, 0.15),
            "bottom-left": (0.08, 0.15),
        }
        cx, cy = loc_positions.get(loc, (0.92, 0.88))

    # Width and height in axes fraction
    w = size * 0.35
    h = size

    # Left half (dark)
    left_poly = patches.Polygon(
        [[cx, cy + h], [cx - w, cy], [cx, cy + h * 0.25]],
        closed=True,
        facecolor=color,
        edgecolor=color,
        transform=ax.transAxes,
        zorder=100,
    )
    # Right half (light/white)
    right_poly = patches.Polygon(
        [[cx, cy + h], [cx + w, cy], [cx, cy + h * 0.25]],
        closed=True,
        facecolor="#ffffff",
        edgecolor=color,
        linewidth=1.0,
        transform=ax.transAxes,
        zorder=100,
    )

    ax.add_patch(left_poly)
    ax.add_patch(right_poly)

    # 'N' label on top
    ax.text(
        cx,
        cy + h + 0.015,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=text_color,
        zorder=101,
    )


def draw_scale_bar(
    ax: plt.Axes,
    bounds: Tuple[float, float, float, float],
    crs: Union[str, CRS] = "EPSG:4326",
    loc: str = "bottom-left",
    pos: Optional[Tuple[float, float]] = None,
    max_fraction: float = 0.25,
    custom_length_m: Optional[float] = None,
):
    """
    Draw an accurate segmented scale bar in km or meters dynamically calculated
    based on the map extent and latitude curvature.
    Supports preset corners or freeform (x, y) axes coordinates.
    """
    minx, miny, maxx, maxy = bounds
    crs_obj = crs if isinstance(crs, CRS) else CRS(crs)

    # Calculate span in meters
    center_lat = (miny + maxy) / 2.0
    if crs_obj.is_geographic:
        # 1 deg lon at lat = 111320 * cos(lat)
        deg_to_m = 111320.0 * math.cos(math.radians(center_lat if abs(center_lat) <= 90 else 0))
        total_width_m = (maxx - minx) * deg_to_m
    else:
        total_width_m = (maxx - minx)  # Already in meters
        deg_to_m = 1.0

    if custom_length_m is not None and custom_length_m > 0:
        chosen_m = custom_length_m
    else:
        target_bar_m = total_width_m * max_fraction
        candidates_m = [
            10, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 15000,
            20000, 25000, 50000, 100000, 200000, 500000, 1000000
        ]
        chosen_m = min(candidates_m, key=lambda c: abs(c - target_bar_m))

    bar_width_deg = chosen_m / deg_to_m

    # Label text
    if chosen_m >= 1000:
        km_val = chosen_m / 1000.0
        label = f"{km_val:g} km"
    else:
        label = f"{int(chosen_m)} m"

    bar_h = (maxy - miny) * 0.016

    # Placement
    if pos is not None:
        fx, fy = pos
        loc_x = minx + fx * (maxx - minx)
        loc_y = miny + fy * (maxy - miny)
    else:
        loc_lower = loc.lower()
        if "top" in loc_lower:
            loc_y = maxy - (maxy - miny) * 0.08
        else:
            loc_y = miny + (maxy - miny) * 0.06

        if "right" in loc_lower:
            loc_x = maxx - (maxx - minx) * 0.06 - bar_width_deg
        else:
            loc_x = minx + (maxx - minx) * 0.06

    # Draw two-toned segmented bar (left black, right white)
    half_w = bar_width_deg / 2.0
    seg1 = patches.Rectangle(
        (loc_x, loc_y),
        half_w,
        bar_h,
        facecolor="#111827",
        edgecolor="#111827",
        zorder=100,
    )
    seg2 = patches.Rectangle(
        (loc_x + half_w, loc_y),
        half_w,
        bar_h,
        facecolor="#ffffff",
        edgecolor="#111827",
        linewidth=0.8,
        zorder=100,
    )
    ax.add_patch(seg1)
    ax.add_patch(seg2)

    # Tick labels: "0", label
    text_y = loc_y + bar_h * 1.5
    ax.text(loc_x, text_y, "0", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#111827", zorder=101)
    ax.text(loc_x + bar_width_deg, text_y, label, ha="center", va="bottom", fontsize=8, fontweight="bold", color="#111827", zorder=101)


def format_degree_ticks(
    ax: plt.Axes,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    crs: Union[str, CRS, None] = "EPSG:4326",
    is_geographic: Optional[bool] = None,
    show_bottom: bool = True,
    show_left: bool = True,
    show_top: bool = False,
    show_right: bool = False,
    lat_orientation: Union[str, int, float] = "horizontal",
    lon_orientation: Union[str, int, float] = "horizontal",
    tick_fontsize: int = 8,
    tick_fontfamily: Optional[str] = None,
    tick_fontweight: str = "normal",
    tick_color: str = "#111827",
    degree_precision: Optional[int] = None,
    coord_format: str = "DD",
    interval: Optional[float] = None,
    prune_corners: Union[bool, str] = True,
    corner_threshold: float = 0.025,
):
    """
    Format axes ticks in degree cardinal notation (DD, DM, DMS) with customizable orientation,
    custom intervals, four-sided display, typography styling, and corner collision pruning.
    Automatically handles projected CRS (e.g. UTM) by converting coordinates
    to Lat/Long degrees via pyproj.
    """
    crs_obj = crs if isinstance(crs, CRS) else (CRS(crs) if crs is not None else CRS("EPSG:4326"))
    if is_geographic is None:
        is_geographic = crs_obj.is_geographic

    transformer = None
    if not is_geographic:
        try:
            transformer = Transformer.from_crs(crs_obj.pyproj_crs, "EPSG:4326", always_xy=True)
        except Exception:
            transformer = None

    if bounds is not None:
        minx, miny, maxx, maxy = bounds
    else:
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()

    span_x = abs(maxx - minx)
    span_y = abs(maxy - miny)
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0

    # Auto-adjust precision if degree span is small to prevent duplicate labels
    if degree_precision is None:
        deg_span = 1.0
        if transformer is not None:
            try:
                min_lon, min_lat = transformer.transform(minx, miny)
                max_lon, max_lat = transformer.transform(maxx, maxy)
                deg_span = max(abs(max_lon - min_lon), abs(max_lat - min_lat))
            except Exception:
                deg_span = 1.0
        elif is_geographic:
            deg_span = max(span_x, span_y)

        if deg_span < 0.005:
            prec = 4
        elif deg_span < 0.05:
            prec = 3
        else:
            prec = 2
    else:
        prec = degree_precision

    # Apply custom interval if specified (in degrees)
    if interval is not None and interval > 0:
        try:
            if transformer is not None:
                deg_to_m_x = 111320.0 * math.cos(math.radians(center_y if abs(center_y) <= 90 else 0))
                deg_to_m_y = 111320.0
                ax.xaxis.set_major_locator(MultipleLocator(interval * deg_to_m_x))
                ax.yaxis.set_major_locator(MultipleLocator(interval * deg_to_m_y))
            else:
                ax.xaxis.set_major_locator(MultipleLocator(interval))
                ax.yaxis.set_major_locator(MultipleLocator(interval))
        except Exception:
            pass

    def resolve_rotation(val: Union[str, int, float]) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            if val.lower() == "vertical":
                return 90.0
            if val.lower() == "horizontal":
                return 0.0
        return 0.0

    rot_y = resolve_rotation(lat_orientation)
    rot_x = resolve_rotation(lon_orientation)

    # Vertical text spans greater height/width; adjust threshold accordingly
    y_thresh = max(corner_threshold, 0.04) if rot_y == 90.0 else corner_threshold
    x_thresh = max(corner_threshold, 0.04) if rot_x == 90.0 else corner_threshold

    prune_str = str(prune_corners).strip().lower() if prune_corners is not None else "false"

    def is_y_corner(y: float) -> bool:
        if span_y <= 0:
            return False
        dist_bottom = abs(y - miny) / span_y
        dist_top = abs(y - maxy) / span_y
        if dist_bottom <= y_thresh and (show_bottom or rot_y == 90.0):
            return True
        if dist_top <= y_thresh and (show_top or rot_y == 90.0):
            return True
        return False

    def is_x_corner(x: float) -> bool:
        if span_x <= 0:
            return False
        dist_left = abs(x - minx) / span_x
        dist_right = abs(x - maxx) / span_x
        if dist_left <= x_thresh and (show_left or rot_x == 90.0):
            return True
        if dist_right <= x_thresh and (show_right or rot_x == 90.0):
            return True
        return False

    def format_coord_val(val_deg: float, is_lat: bool) -> str:
        direction = ("N" if val_deg >= 0 else "S") if is_lat else ("E" if val_deg >= 0 else "W")
        abs_deg = abs(val_deg)
        fmt = coord_format.upper()
        if fmt == "DMS":
            d = int(abs_deg)
            m_float = (abs_deg - d) * 60.0
            m = int(m_float)
            s = round((m_float - m) * 60.0)
            if s >= 60:
                s = 0
                m += 1
            if m >= 60:
                m = 0
                d += 1
            return f"{d}° {m:02d}' {s:02d}\" {direction}"
        elif fmt == "DM":
            d = int(abs_deg)
            m_float = (abs_deg - d) * 60.0
            return f"{d}° {m_float:04.1f}' {direction}"
        else:  # DD
            return f"{abs_deg:.{prec}f}° {direction}"

    def fmt_lon(x, pos):
        if prune_corners and prune_str in ("x", "both") and is_x_corner(x):
            return ""
        if transformer is not None:
            try:
                lon_deg, _ = transformer.transform(x, center_y)
                return format_coord_val(lon_deg, is_lat=False)
            except Exception:
                return f"{x:,.0f}"
        elif is_geographic:
            return format_coord_val(x, is_lat=False)
        else:
            return f"{x:,.0f}"

    def fmt_lat(y, pos):
        if prune_corners and prune_str in ("true", "auto", "y", "both") and is_y_corner(y):
            return ""
        if transformer is not None:
            try:
                _, lat_deg = transformer.transform(center_x, y)
                return format_coord_val(lat_deg, is_lat=True)
            except Exception:
                return f"{y:,.0f}"
        elif is_geographic:
            return format_coord_val(y, is_lat=True)
        else:
            return f"{y:,.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    # Configure tick placement on all 4 sides
    ax.tick_params(
        axis="x",
        which="major",
        bottom=show_bottom,
        labelbottom=show_bottom,
        top=show_top,
        labeltop=show_top,
        labelsize=tick_fontsize,
        labelcolor=tick_color,
        color=tick_color,
        labelrotation=rot_x,
    )
    ax.tick_params(
        axis="y",
        which="major",
        left=show_left,
        labelleft=show_left,
        right=show_right,
        labelright=show_right,
        labelsize=tick_fontsize,
        labelcolor=tick_color,
        color=tick_color,
        labelrotation=rot_y,
    )

    # Apply fontfamily and fontweight styling across tick labels
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontsize(tick_fontsize)
        label.set_color(tick_color)
        if tick_fontfamily:
            label.set_fontfamily(tick_fontfamily)
        if tick_fontweight:
            label.set_fontweight(tick_fontweight)


def plot_carto_map(
    image_or_gdf: Any,
    bands: Optional[List[str]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    north_arrow: bool = True,
    north_loc: str = "top-right",
    north_pos: Optional[Tuple[float, float]] = None,
    north_size: float = 0.07,
    north_color: str = "#111827",
    scale_bar: bool = True,
    scale_loc: str = "bottom-left",
    scale_pos: Optional[Tuple[float, float]] = None,
    scale_fraction: float = 0.25,
    scale_custom_length_m: Optional[float] = None,
    grid: bool = True,
    grid_style: str = ":",
    grid_color: str = "#6b7280",
    grid_alpha: float = 0.5,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    legend: bool = False,
    legend_title: Optional[str] = None,
    legend_labels: Optional[Dict[str, str]] = None,
    legend_loc: str = "lower right",
    legend_pos: Optional[Tuple[float, float]] = None,
    discrete_bins: Optional[List[float]] = None,
    discrete_colors: Optional[List[str]] = None,
    interpolation: str = "bilinear",
    show_bottom: bool = True,
    show_left: bool = True,
    show_top: bool = False,
    show_right: bool = False,
    lat_orientation: Union[str, int, float] = "horizontal",
    lon_orientation: Union[str, int, float] = "horizontal",
    tick_fontsize: int = 8,
    tick_fontfamily: Optional[str] = None,
    tick_fontweight: str = "normal",
    tick_color: str = "#111827",
    degree_precision: Optional[int] = None,
    coord_format: str = "DD",
    graticule_interval: Optional[float] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    prune_corners: Union[bool, str] = True,
    corner_threshold: float = 0.025,
    tight_bbox: bool = True,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Axes:
    """
    Render a publication-ready cartographic map complete with North Arrow,
    dynamic Scale Bar, degree-formatted Lat/Long ticks on all four sides,
    configurable label orientation, corner overlap prevention, and Legend/Colorbar.
    """
    if ax is None:
        try:
            fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
        except Exception:
            matplotlib.use("Agg")
            fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    else:
        fig = ax.figure

    bounds = extent if extent is not None else image_or_gdf.bounds
    crs = image_or_gdf.crs

    # Set precise spatial extent on axes
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # Plot data
    if hasattr(image_or_gdf, "visualize") and hasattr(image_or_gdf, "select"):
        # Image object
        target_bands = bands or (image_or_gdf.bands[:3] if image_or_gdf.count >= 3 else [image_or_gdf.bands[0]])
        if len(target_bands) >= 3:
            rgb_arr = image_or_gdf.visualize({"bands": target_bands})
            im = ax.imshow(
                rgb_arr,
                extent=[image_or_gdf.bounds[0], image_or_gdf.bounds[2], image_or_gdf.bounds[1], image_or_gdf.bounds[3]],
                origin="upper",
                interpolation=interpolation,
            )
        else:
            single = image_or_gdf.select(target_bands[0]).data[0]
            if discrete_bins is not None and discrete_colors is not None and len(discrete_bins) >= 2:
                from matplotlib.colors import BoundaryNorm, ListedColormap
                n_cols = len(discrete_colors)
                # Adjust bins if color count requires it
                use_bins = discrete_bins
                if n_cols < len(use_bins) - 1:
                    use_bins = use_bins[: n_cols + 1]
                cmap_obj = ListedColormap(discrete_colors[: len(use_bins) - 1] if len(discrete_colors) >= len(use_bins) - 1 else discrete_colors)
                norm_obj = BoundaryNorm(use_bins, len(cmap_obj.colors), clip=True)
                im = ax.imshow(
                    single,
                    extent=[image_or_gdf.bounds[0], image_or_gdf.bounds[2], image_or_gdf.bounds[1], image_or_gdf.bounds[3]],
                    cmap=cmap_obj,
                    norm=norm_obj,
                    origin="upper",
                    interpolation="nearest",
                )
            else:
                mn = np.nanpercentile(single, 2) if vmin is None else vmin
                mx = np.nanpercentile(single, 98) if vmax is None else vmax
                im = ax.imshow(
                    single,
                    extent=[image_or_gdf.bounds[0], image_or_gdf.bounds[2], image_or_gdf.bounds[1], image_or_gdf.bounds[3]],
                    cmap=cmap,
                    vmin=mn,
                    vmax=mx,
                    origin="upper",
                    interpolation=interpolation,
                )
            if colorbar and (discrete_bins is None or not discrete_colors):
                # Add padding if right-side latitude ticks are visible
                cb_pad = 0.07 if show_right else 0.03
                cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=cb_pad)
                cb.set_label(colorbar_label or target_bands[0], fontsize=9)
                cb.ax.tick_params(labelsize=8)
    else:
        # Vector / FeatureCollection
        gdf = image_or_gdf.gdf if hasattr(image_or_gdf, "gdf") else image_or_gdf
        col = bands[0] if bands and bands[0] in gdf.columns else None
        gdf.plot(column=col, cmap=cmap, ax=ax, legend=legend, alpha=0.9)

    # Custom Discrete Legend
    if legend_labels:
        import matplotlib.patches as mpatches
        legend_handles = []
        for k, v in legend_labels.items():
            if str(v).startswith("#") or v in ("blue", "green", "red", "yellow", "black", "white", "orange", "purple", "cyan", "magenta"):
                patch_color, patch_label = v, k
            else:
                patch_color, patch_label = k, v
            patch = mpatches.Patch(color=patch_color, label=patch_label)
            legend_handles.append(patch)
        
        leg_opts = {
            "handles": legend_handles,
            "title": legend_title,
            "fontsize": 8,
            "title_fontsize": 9,
            "framealpha": 0.9,
            "edgecolor": "#e5e7eb",
        }
        if legend_pos is not None:
            ax.legend(bbox_to_anchor=legend_pos, loc=legend_loc, **leg_opts)
        else:
            ax.legend(loc=legend_loc, **leg_opts)

    # Grid & Graticule lines
    if grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha, color=grid_color, zorder=50)

    # Format ticks with orientation, side placement, corner overlap pruning, and typography
    format_degree_ticks(
        ax,
        bounds=bounds,
        crs=crs,
        show_bottom=show_bottom,
        show_left=show_left,
        show_top=show_top,
        show_right=show_right,
        lat_orientation=lat_orientation,
        lon_orientation=lon_orientation,
        tick_fontsize=tick_fontsize,
        tick_fontfamily=tick_fontfamily,
        tick_fontweight=tick_fontweight,
        tick_color=tick_color,
        degree_precision=degree_precision,
        coord_format=coord_format,
        interval=graticule_interval,
        prune_corners=prune_corners,
        corner_threshold=corner_threshold,
    )

    if scale_bar:
        draw_scale_bar(
            ax, bounds, crs=crs, loc=scale_loc, pos=scale_pos,
            max_fraction=scale_fraction, custom_length_m=scale_custom_length_m
        )

    if north_arrow:
        draw_north_arrow(
            ax, loc=north_loc, pos=north_pos, size=north_size, color=north_color
        )

    # Titles & Labels with clean layered offsets
    if title and subtitle:
        t_pad = 36 if show_top else 20
        sub_pts = 18 if show_top else 5
        ax.set_title(title, fontsize=12, fontweight="bold", pad=t_pad)
        ax.annotate(
            subtitle,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, sub_pts),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#4b5563",
        )
    elif title:
        t_pad = 22 if show_top else 12
        ax.set_title(title, fontsize=12, fontweight="bold", pad=t_pad)
    elif subtitle:
        sub_pts = 16 if show_top else 4
        ax.annotate(
            subtitle,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, sub_pts),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#4b5563",
        )

    crs_obj = crs if isinstance(crs, CRS) else CRS(crs)
    is_degrees = crs_obj.is_geographic or (hasattr(crs_obj, "pyproj_crs") and crs_obj.pyproj_crs is not None)
    if show_bottom:
        ax.set_xlabel("Longitude" if is_degrees else "Easting (m)", fontsize=9, labelpad=8)
    if show_left:
        ax.set_ylabel("Latitude" if is_degrees else "Northing (m)", fontsize=9, labelpad=8)

    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if tight_bbox:
            fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out_p, dpi=dpi)

    return ax
