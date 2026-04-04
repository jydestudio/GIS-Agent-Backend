"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CARTOGRAPHIC MAP RENDERER — AGENT TOOL                        ║
║  Supports: Raster (GeoTIFF), Vector (GeoJSON/Shapefile/GeoDataFrame)       ║
║  Features: Title · Legend · North Arrow · Grid · Scalebar · Coordinates   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import json
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Must be set BEFORE importing pyplot — required for thread safety (FastAPI, Celery, etc.)

BASE_DIR = Path(__file__).resolve().parent
arrow_path = (Path(__file__).resolve().parent / "assets" / "north_arrow.png").resolve()

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

logger.info(arrow_path)
try:
    import rasterio
    from rasterio.plot import show as rasterio_show
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

try:
    from langchain_core.tools import tool
    HAS_LANGCHAIN = True
except ImportError:
    # Fallback decorator: makes the function directly callable AND supports .invoke(input_dict)
    # so the same call pattern works regardless of whether LangChain is installed.
    class _ToolWrapper:
        def __init__(self, func):
            self._func = func
            self.__doc__  = func.__doc__
            self.__name__ = func.__name__

        def __call__(self, *args, **kwargs):
            return self._func(*args, **kwargs)

        def invoke(self, input: dict):
            """LangChain-compatible .invoke() shim."""
            return self._func(**input)

    def tool(func):
        return _ToolWrapper(func)

    HAS_LANGCHAIN = False


# ══════════════════════════════════════════════════════════════════════════════
#  ① DATA CLASSES — Full cartographic configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TitleConfig:
    text: str = ""
    fontsize: int = 14
    fontweight: str = "bold"
    fontfamily: str = "serif"
    color: str = "black"
    pad: int = 20
    location: Literal["center", "left", "right"] = "center"
    subtitle: str = ""
    subtitle_fontsize: int = 10


@dataclass
class GridConfig:
    show: bool = True
    linestyle: str = "--"
    linewidth: float = 0.4
    color: str = "gray"
    alpha: float = 0.5
    n_ticks_x: int = 4
    n_ticks_y: int = 4
    dms_format: bool = True          # Show as 5°30'20"N vs 5.5056°
    label_fontsize: int = 8
    label_fontweight: str = "bold"
    label_color: str = "black"
    label_rotation_y: int = 90
    top_labels: bool = True
    right_labels: bool = True


@dataclass
class NorthArrowConfig:
    show: bool = True
    position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "top-left"
    image_path: str = ""             # Path to custom north arrow PNG
    zoom: float = 0.15
    offset_x: float = 0.09          # Fraction offset from corner
    offset_y: float = 0.05
    fallback_color: str = "black"
    fallback_fontsize: int = 13


@dataclass
class ScaleBarConfig:
    show: bool = True
    position: Literal["bottom-left", "bottom-right", "bottom-center"] = "bottom-center"
    bar_color: str = "black"
    text_color: str = "black"
    fontsize: int = 9
    fontweight: str = "bold"
    fontfamily: str = "serif"
    target_frac: float = 0.28         # Target fraction of map width
    unit: Literal["m", "km"] = "m"


@dataclass
class LegendConfig:
    show: bool = True
    title: str = ""
    title_fontsize: int = 10
    fontsize: int = 9
    fontfamily: str = "serif"
    location: str = "lower right"    # matplotlib loc string
    frameon: bool = True
    framealpha: float = 0.85
    edgecolor: str = "black"
    ncol: int = 1
    custom_handles: list = field(default_factory=list)  # [(label, color, marker), ...]


@dataclass
class VectorStyle:
    """Style settings for a single vector layer."""
    label: str = ""

    # Point settings
    point_color: str = "#FFE600"
    point_size: int = 60
    point_marker: str = "o"
    point_edgecolor: str = "#333333"
    point_edgewidth: float = 0.8

    # Line settings
    line_color: str = "#CC0000"
    line_width: float = 2.0
    line_style: str = "-"

    # Polygon settings
    face_color: str = "#0077b6"
    face_alpha: float = 0.45
    edge_color: str = "#111111"
    edge_width: float = 2.0

    # Choropleth (classify by column)
    classify_by: str = ""            # Column name; empty = single-style
    colormap: str = "YlOrRd"        # Any matplotlib colormap
    n_classes: int = 5
    classification_scheme: str = "quantiles"   # natural_breaks | equal_interval | quantiles

    # Labels
    label_column: str = ""
    label_fontsize: int = 8
    label_color: str = "white"
    label_stroke_color: str = "black"
    label_stroke_width: float = 2.0


@dataclass
class RasterStyle:
    """Style settings for a raster layer."""
    label: str = "Raster Layer"
    colormap: str = "terrain"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    alpha: float = 1.0
    nodata: Optional[float] = None
    band: int = 1                    # Which band to visualise (1-indexed)
    show_colorbar: bool = True
    colorbar_label: str = ""
    colorbar_orientation: Literal["vertical", "horizontal"] = "vertical"
    colorbar_shrink: float = 0.6
    colorbar_pad: float = 0.02


@dataclass
class MapConfig:
    """Master configuration object for a complete map."""

    # ── Output ──────────────────────────────────────────────
    output_path: str = "output_maps/output_map.png"
    dpi: int = 220
    fig_width: float = 10.0
    fig_height: float = 13.0
    tight_layout: bool = True

    # ── Basemap ─────────────────────────────────────────────
    basemap: bool = False
    basemap_url: str = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    basemap_zoom: Optional[int] = None   # None = auto

    # ── View / Extent ────────────────────────────────────────
    pad_frac: float = 0.12
    min_view_degrees: float = 0.0008
    xlim: Optional[tuple[float, float]] = None   # Override auto
    ylim: Optional[tuple[float, float]] = None

    # ── Map Frame ────────────────────────────────────────────
    spine_color: str = "black"
    spine_width: float = 1.5
    background_color: str = "#f0f0f0"

    # ── Sub-configs ──────────────────────────────────────────
    title: TitleConfig = field(default_factory=TitleConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    north_arrow: NorthArrowConfig = field(default_factory=lambda: NorthArrowConfig(image_path=str(arrow_path)))
    scalebar: ScaleBarConfig = field(default_factory=ScaleBarConfig)
    legend: LegendConfig = field(default_factory=LegendConfig)


# ══════════════════════════════════════════════════════════════════════════════
#  ② INTERNAL HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _dms_formatter(axis: str = "lon"):
    """Return a FuncFormatter that shows coordinates in DMS notation."""
    def fmt(val, pos):
        suffix = ("E" if val >= 0 else "W") if axis == "lon" else ("N" if val >= 0 else "S")
        v = abs(val)
        deg = int(v)
        mn = int((v - deg) * 60)
        sec = ((v - deg) * 60 - mn) * 60
        return f"{deg}\u00b0{mn:02d}'{sec:04.1f}\"{suffix}"
    return mticker.FuncFormatter(fmt)


def _decimal_formatter(axis: str = "lon"):
    """Return a FuncFormatter that shows coordinates in decimal degrees."""
    def fmt(val, pos):
        suffix = ("E" if val >= 0 else "W") if axis == "lon" else ("N" if val >= 0 else "S")
        return f"{abs(val):.4f}\u00b0{suffix}"
    return mticker.FuncFormatter(fmt)


def _auto_find_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Auto-detect longitude and latitude column names."""
    lon_candidates = ["longitude", "long", "lon", "x", "easting"]
    lat_candidates = ["latitude",  "lat",  "y",  "northing"]
    cols_lower = {c.lower(): c for c in df.columns}
    f_lon = next((cols_lower[c] for c in lon_candidates if c in cols_lower), None)
    f_lat = next((cols_lower[c] for c in lat_candidates if c in cols_lower), None)
    return f_lon, f_lat


def _north_arrow_position(position: str, offset_x: float, offset_y: float) -> tuple[float, float]:
    """Convert a position string to axes-fraction (x, y) coordinates."""
    mapping = {
        "top-left":     (offset_x,       1.0 - offset_y),
        "top-right":    (1.0 - offset_x, 1.0 - offset_y),
        "bottom-left":  (offset_x,       offset_y),
        "bottom-right": (1.0 - offset_x, offset_y),
    }
    return mapping.get(position, (0.07, 0.90))


def _scalebar_x_anchor(position: str) -> float:
    """Return the x-axis fraction for scale bar alignment."""
    if "right" in position:
        return 0.75
    if "center" in position:
        return 0.5
    return 0.15


# ══════════════════════════════════════════════════════════════════════════════
#  ③ CARTOGRAPHIC ELEMENT DRAWERS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_north_arrow(ax: plt.Axes, cfg: NorthArrowConfig) -> None:
    """Render a north arrow at the configured position."""
    if not cfg.show:
        return
    x, y = _north_arrow_position(cfg.position, cfg.offset_x, cfg.offset_y)

    if cfg.image_path and os.path.exists(cfg.image_path):
        img = mpimg.imread(cfg.image_path)
        ab = AnnotationBbox(
            OffsetImage(img, zoom=cfg.zoom), (x, y),
            xycoords="axes fraction", frameon=False, zorder=10
        )
        ax.add_artist(ab)
    else:
        # Fallback: drawn arrow + "N" label
        arrow_base_y = y - 0.04
        ax.annotate(
            "", xy=(x, y + 0.04), xytext=(x, arrow_base_y),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=cfg.fallback_color, lw=2.5),
            zorder=10
        )
        ax.text(
            x, y + 0.055, "N",
            transform=ax.transAxes,
            color=cfg.fallback_color,
            fontsize=cfg.fallback_fontsize,
            fontweight="bold", ha="center", zorder=10
        )


def _draw_scalebar(
    ax: plt.Axes,
    cfg: ScaleBarConfig,
    xlim: tuple,
    cy: float
) -> None:
    """Draw a scale bar below the map axes."""
    if not cfg.show:
        return

    km_per_deg = 111.0 * abs(np.cos(np.radians(cy)))
    map_width_m = (xlim[1] - xlim[0]) * km_per_deg * 1000

    # Pick a nice round number (expanded for large areas like countries)
    nice_values_m  = [
        1, 2, 5, 10, 20, 50, 100, 200, 500, 
        1000, 2000, 5000, 10000, 20000, 50000, 100000, 
        200000, 500000, 1000000, 2000000, 3000000, 5000000
    ]
    target_m = map_width_m * cfg.target_frac
    bar_m = min(nice_values_m, key=lambda v: abs(v - target_m))

    # Auto-switch to km for large scales
    use_km = bar_m >= 2000 or cfg.unit == "km"

    if use_km:
        bar_label = f"{bar_m / 1000:.0f} km"
    else:
        bar_label = f"{bar_m:.0f} m"

    bar_deg  = bar_m / (km_per_deg * 1000)
    bar_frac = bar_deg / (xlim[1] - xlim[0])

    anchor_x = _scalebar_x_anchor(cfg.position)
    rect_x   = anchor_x - bar_frac / 2
    bar_y    = -0.09 # Moved slightly lower
    text_y   = -0.14 # More space to avoid overlap

    # Alternating black/white segments
    half = bar_frac / 2
    for i, (start, color) in enumerate([(rect_x, "black"), (rect_x + half, "white")]):
        ax.add_patch(plt.Rectangle(
            (start, bar_y), half, 0.012,
            transform=ax.transAxes,
            facecolor=color, edgecolor="black", linewidth=0.8,
            clip_on=False, zorder=8
        ))

    label_kw = dict(
        transform=ax.transAxes,
        fontsize=cfg.fontsize, fontweight=cfg.fontweight,
        fontfamily=cfg.fontfamily, color=cfg.text_color,
        clip_on=False, zorder=8
    )

    # Compute half-bar label
    half_m = bar_m / 2
    if use_km:
        half_label = f"{half_m / 1000:g} km"
    else:
        half_label = f"{half_m:g} m"

    ax.text(rect_x,                text_y, "0",         ha="center", **label_kw)
    ax.text(rect_x + bar_frac / 2, text_y, half_label,  ha="center", **label_kw)
    ax.text(rect_x + bar_frac,     text_y, bar_label,   ha="center", **label_kw)


def _apply_grid(ax: plt.Axes, cfg: GridConfig) -> None:
    """Apply coordinate grid and tick labels."""
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=cfg.n_ticks_x, prune="both"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=cfg.n_ticks_y, prune="both"))

    if cfg.dms_format:
        ax.xaxis.set_major_formatter(_dms_formatter("lon"))
        ax.yaxis.set_major_formatter(_dms_formatter("lat"))
    else:
        ax.xaxis.set_major_formatter(_decimal_formatter("lon"))
        ax.yaxis.set_major_formatter(_decimal_formatter("lat"))

    tick_kw = dict(
        axis="both", labelsize=cfg.label_fontsize,
        direction="in", length=5, width=1.5,
        top=cfg.top_labels, right=cfg.right_labels,
        pad=8, labelcolor=cfg.label_color
    )
    ax.tick_params(**tick_kw)

    for lbl in ax.get_xticklabels():
        lbl.set_fontweight(cfg.label_fontweight)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight(cfg.label_fontweight)
        lbl.set_rotation(cfg.label_rotation_y)

    if cfg.show:
        ax.grid(
            True,
            linestyle=cfg.linestyle, linewidth=cfg.linewidth,
            color=cfg.color, alpha=cfg.alpha, zorder=0
        )


def _apply_title(fig: plt.Figure, ax: plt.Axes, cfg: TitleConfig) -> None:
    """Render main title and optional subtitle."""
    if not cfg.text:
        return
    wrapped = "\n".join(textwrap.wrap(cfg.text, width=50))

    # Combine title + subtitle into a single ax.set_title call so layout is always correct
    if cfg.subtitle:
        subtitle_wrapped = "\n".join(textwrap.wrap(cfg.subtitle, width=60))
        full_title = (
            f"{wrapped}\n"
            + r"$\it{" + subtitle_wrapped.replace(" ", r"\ ") + r"}$"
        )
        # Use two separate text objects via title + suptitle to avoid LaTeX edge-cases
        ax.set_title(
            wrapped,
            fontsize=cfg.fontsize, fontweight=cfg.fontweight,
            fontfamily=cfg.fontfamily, color=cfg.color,
            pad=cfg.pad, loc=cfg.location
        )
        # subtitle sits just above the axes using a second title line
        ax.annotate(
            subtitle_wrapped,
            xy=(0.5, 1.0),
            xytext=(0.5, 1.035 + (cfg.pad / 600)),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center", va="bottom",
            fontsize=cfg.subtitle_fontsize,
            color=cfg.color,
            fontstyle="italic",
            annotation_clip=False,
            zorder=12,
        )
    else:
        ax.set_title(
            wrapped,
            fontsize=cfg.fontsize, fontweight=cfg.fontweight,
            fontfamily=cfg.fontfamily, color=cfg.color,
            pad=cfg.pad, loc=cfg.location
        )


def _auto_extent(
    bounds: np.ndarray,
    pad_frac: float,
    min_view_degrees: float
) -> tuple[tuple, tuple]:
    """Compute padded map extent from layer bounds."""
    dx = max(bounds[2] - bounds[0], min_view_degrees)
    dy = max(bounds[3] - bounds[1], min_view_degrees)

    # Balance aspect (max 2.5:1)
    if dx > dy * 2.5:
        dy = dx / 2.0
    elif dy > dx * 2.5:
        dx = dy / 2.0

    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    z  = 1 + pad_frac * 2

    xlim = (cx - dx * z / 2, cx + dx * z / 2)
    ylim = (cy - dy * z / 2, cy + dy * z / 2)
    return xlim, ylim


# ══════════════════════════════════════════════════════════════════════════════
#  ④ VECTOR LAYER RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _render_vector_layer(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    style: VectorStyle,
    legend_handles: list,
) -> None:
    """Plot a single vector GeoDataFrame on ax with the given style."""

    geom_types = set(gdf.geom_type.unique())

    # ── Choropleth ──────────────────────────────────────────
    if style.classify_by and style.classify_by in gdf.columns:
        cmap = plt.get_cmap(style.colormap, style.n_classes)
        col  = pd.to_numeric(gdf[style.classify_by], errors="coerce")
        vmin, vmax = col.min(), col.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        gdf = gdf.copy()
        gdf["_color"] = [cmap(norm(v)) if not np.isnan(v) else (0.5, 0.5, 0.5, 1) for v in col]

        if any(t in geom_types for t in ["Polygon", "MultiPolygon"]):
            gdf.plot(ax=ax, color=gdf["_color"].tolist(), edgecolor=style.edge_color,
                     linewidth=style.edge_width, zorder=3)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01, label=style.classify_by)
        return

    # ── Single-style per geometry type ──────────────────────
    if any(t in geom_types for t in ["Point", "MultiPoint"]):
        gdf.plot(
            ax=ax,
            color=style.point_color,
            edgecolor=style.point_edgecolor,
            linewidth=style.point_edgewidth,
            markersize=style.point_size,
            marker=style.point_marker,
            zorder=4
        )
        if style.label:
            legend_handles.append(
                mpatches.Patch(facecolor=style.point_color,
                               edgecolor=style.point_edgecolor,
                               label=style.label)
            )

    if any(t in geom_types for t in ["LineString", "MultiLineString"]):
        gdf.plot(
            ax=ax,
            color=style.line_color,
            linewidth=style.line_width,
            linestyle=style.line_style,
            zorder=3
        )
        if style.label:
            legend_handles.append(
                mpatches.Patch(facecolor=style.line_color, label=style.label)
            )

    if any(t in geom_types for t in ["Polygon", "MultiPolygon"]):
        gdf.plot(
            ax=ax,
            facecolor=style.face_color,
            alpha=style.face_alpha,
            edgecolor=style.edge_color,
            linewidth=style.edge_width,
            zorder=3
        )
        if style.label:
            legend_handles.append(
                mpatches.Patch(
                    facecolor=style.face_color,
                    edgecolor=style.edge_color,
                    alpha=style.face_alpha,
                    linewidth=style.edge_width,
                    label=style.label
                )
            )

    # ── Labels ──────────────────────────────────────────────
    if style.label_column and style.label_column in gdf.columns:
        for _, row in gdf.iterrows():
            pt = row.geometry.centroid if hasattr(row.geometry, "centroid") else row.geometry
            ax.text(
                pt.x, pt.y,
                str(row[style.label_column]),
                fontsize=style.label_fontsize,
                color=style.label_color,
                ha="center", va="center", zorder=5,
                path_effects=[pe.withStroke(
                    linewidth=style.label_stroke_width,
                    foreground=style.label_stroke_color
                )]
            )


# ══════════════════════════════════════════════════════════════════════════════
#  ⑤ RASTER LAYER RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _render_raster_layer(
    ax: plt.Axes,
    raster_path: str,
    style: RasterStyle,
) -> np.ndarray:
    """Plot a single raster band on ax. Returns the bounds array."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for raster maps. Install with: pip install rasterio")

    with rasterio.open(raster_path) as src:
        data = src.read(style.band).astype(float)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        if style.nodata is not None:
            data = np.where(data == style.nodata, np.nan, data)

        vmin = style.vmin if style.vmin is not None else np.nanmin(data)
        vmax = style.vmax if style.vmax is not None else np.nanmax(data)

        im = ax.imshow(
            data,
            extent=extent,
            cmap=style.colormap,
            vmin=vmin, vmax=vmax,
            alpha=style.alpha,
            origin="upper",
            zorder=2
        )

        if style.show_colorbar:
            cbar = plt.colorbar(
                im, ax=ax,
                orientation=style.colorbar_orientation,
                shrink=style.colorbar_shrink,
                pad=style.colorbar_pad
            )
            if style.colorbar_label:
                cbar.set_label(style.colorbar_label, fontsize=9)
            cbar.ax.tick_params(labelsize=8)

        return np.array([bounds.left, bounds.bottom, bounds.right, bounds.top])


# ══════════════════════════════════════════════════════════════════════════════
#  ⑥ MASTER MAP RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_map(
    layers: list[dict],
    config: MapConfig,
) -> str:
    """
    Render a cartographic map from one or more data layers.

    Parameters
    ----------
    layers : list of dicts, each with keys:
        - "type"   : "vector" | "raster" | "csv"
        - "data"   : file path (str) or GeoDataFrame
        - "style"  : VectorStyle | RasterStyle instance
        - "crs"    : optional CRS string (default "EPSG:4326")
    config : MapConfig
        Master configuration for the whole map.

    Returns
    -------
    str : Absolute path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
    ax.set_facecolor(config.background_color)

    all_bounds: list[np.ndarray] = []
    legend_handles: list = []

    # ── Process each layer ────────────────────────────────────
    for layer in layers:
        ltype  = layer.get("type", "vector").lower()
        data   = layer.get("data")
        style  = layer.get("style")
        crs    = layer.get("crs", "EPSG:4326")

        if ltype == "raster":
            if style is None:
                style = RasterStyle()
            bounds = _render_raster_layer(ax, data, style)
            all_bounds.append(bounds)

        elif ltype in ("vector", "csv"):
            if isinstance(data, str):
                if data.endswith(".csv"):
                    df = pd.read_csv(data)
                    lon_c, lat_c = _auto_find_columns(df)
                    df[lon_c] = pd.to_numeric(df[lon_c], errors="coerce")
                    df[lat_c] = pd.to_numeric(df[lat_c], errors="coerce")
                    df = df.dropna(subset=[lon_c, lat_c])
                    gdf = gpd.GeoDataFrame(
                        df,
                        geometry=[Point(x, y) for x, y in zip(df[lon_c], df[lat_c])],
                        crs=crs
                    )
                else:
                    gdf = gpd.read_file(data)
            elif isinstance(data, gpd.GeoDataFrame):
                gdf = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            # Reproject to WGS84 for consistent extent calculation
            if gdf.crs and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            if style is None:
                style = VectorStyle()

            _render_vector_layer(ax, gdf, style, legend_handles)
            b = gdf.total_bounds
            all_bounds.append(b)

    if not all_bounds:
        plt.close(fig)
        raise ValueError("No valid layers to render.")

    # ── Compute extent ────────────────────────────────────────
    combined = np.array(all_bounds)
    master_bounds = np.array([
        combined[:, 0].min(),
        combined[:, 1].min(),
        combined[:, 2].max(),
        combined[:, 3].max(),
    ])

    if config.xlim and config.ylim:
        xlim, ylim = config.xlim, config.ylim
    else:
        xlim, ylim = _auto_extent(master_bounds, config.pad_frac, config.min_view_degrees)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # ── Basemap ───────────────────────────────────────────────
    if config.basemap and HAS_CONTEXTILY:
        extent_m = max(master_bounds[2] - master_bounds[0],
                       master_bounds[3] - master_bounds[1]) * 111000
        if config.basemap_zoom is None:
            z = 20 if extent_m < 300 else 18 if extent_m < 800 else 16
        else:
            z = config.basemap_zoom
        ctx.add_basemap(
            ax, crs="EPSG:4326", source=config.basemap_url,
            zoom=z, attribution=False, zorder=1
        )

    # ── Spines ───────────────────────────────────────────────
    for sp in ax.spines.values():
        sp.set_edgecolor(config.spine_color)
        sp.set_linewidth(config.spine_width)

    # ── Grid & Coordinates ───────────────────────────────────
    _apply_grid(ax, config.grid)

    # ── Title ─────────────────────────────────────────────────
    _apply_title(fig, ax, config.title)

    # ── North Arrow ───────────────────────────────────────────
    _draw_north_arrow(ax, config.north_arrow)

    # ── Scale Bar ─────────────────────────────────────────────
    cy = (ylim[0] + ylim[1]) / 2
    _draw_scalebar(ax, config.scalebar, xlim, cy)

    # ── Legend ────────────────────────────────────────────────
    if config.legend.show:
        # Merge user custom handles with auto-collected ones
        extra = [
            mpatches.Patch(facecolor=c, label=lbl, marker=m if m != "patch" else None)
            if m == "patch" else
            plt.Line2D([0], [0], color=c, label=lbl)
            for lbl, c, m in config.legend.custom_handles
        ] if config.legend.custom_handles else []

        all_handles = legend_handles + extra
        if all_handles:
            ax.legend(
                handles=all_handles,
                title=config.legend.title,
                title_fontsize=config.legend.title_fontsize,
                loc=config.legend.location,
                fontsize=config.legend.fontsize,
                frameon=config.legend.frameon,
                framealpha=config.legend.framealpha,
                edgecolor=config.legend.edgecolor,
                ncol=config.legend.ncol,
            )

    # ── Save ──────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(config.output_path))
    os.makedirs(out_dir, exist_ok=True)

    save_kw = {"dpi": config.dpi, "facecolor": fig.get_facecolor()}
    if config.tight_layout:
        save_kw["bbox_inches"] = "tight"

    fig.savefig(config.output_path, **save_kw)
    plt.close(fig)

    return os.path.abspath(config.output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ⑦ LANGCHAIN AGENT TOOL — JSON-driven interface
# ══════════════════════════════════════════════════════════════════════════════

@tool
def create_cartographic_map(
    layers_json: Annotated[str, (
        "JSON array of layer objects. Each object must have: "
        "'type' ('vector'|'raster'|'csv'), "
        "'data' (file path or GeoJSON string), "
        "'style' (optional style dict). "

        "Style Rules:\n"
            "- All colors (face_color, edge_color, point_color, line_color) MUST be HEX format (e.g. '#3A86FF').\n"
            "- Do NOT use named colors like 'red', 'blue', etc.\n"


        "Example: [{\"type\":\"vector\",\"data\":\"/path/to/file.shp\","
        "\"style\":{\"face_color\":\"#3A86FF\",\"label\":\"Study Area\"}}]"
    )],
    config_json: Annotated[str, (
        "JSON object for map configuration. Supports all MapConfig fields plus nested "
        "objects for title, grid, north_arrow, scalebar, legend. "

        "Title Rules:\n"
            "- Title must be uppercase and professional.\n"
            "- Use format: 'STUDY AREA MAP OF <LOCATION>'.\n"
            "- Avoid generic titles like 'Map' or 'My Map'.\n"

        "Example: {\"title\":{\"text\":\"Study Area\",\"fontsize\":16},"
        "\"north_arrow\":{\"position\":\"top-right\"},"
        "\"basemap\":true,"
        "\"output_path\":\"output/my_map.png\"}"
    )] = "{}",
) -> str:
    """
    Create a fully-cartographed PNG map from vector or raster data.

    Styling Rules:
        - All colors MUST be in HEX format (e.g. '#FF5733').
        - Do NOT use named colors.

        Title Rules:
        "- Title is REQUIRED.\n"
        - Titles must be uppercase and professional.
        - Format: 'STUDY AREA MAP OF <LOCATION>'.

        CRITICAL COLOR RULES:
        - For polygons → use "face_color"
        - For points → use "point_color"
        - For lines → use "line_color"
        - "color" alone is NOT reliable


    Supports:
    - Vector layers: GeoJSON, Shapefile, CSV with coordinates
    - Raster layers: GeoTIFF (single-band or multi-band)
    - Satellite/OSM basemap overlay
    - Configurable: title, subtitle, north arrow (position), scale bar,
      coordinate grid (DMS or decimal), legend, color ramps, choropleth,
      font sizes, colors, DPI, figure size, and more.

    Returns a JSON string with status and the output file path.
    """
    try:
        # ── Parse inputs (accept str or already-parsed dict/list) ──
        def _parse(v):
            if isinstance(v, (dict, list)):
                return v
            if isinstance(v, str) and v.strip():
                return json.loads(v)
            return {}

        raw_layers = _parse(layers_json)
        config_dict = _parse(config_json)

        # ── Handle "envelope" format ─────────────────────────
        # Agent sometimes sends ONE dict: {"layers": [...], "title": "...", "subtitle": "..."}
        # as layers_json instead of a bare list.
        if isinstance(raw_layers, dict):
            envelope = raw_layers
            raw_layers  = envelope.pop("layers", [])
            # Merge remaining envelope keys into config_dict (title, subtitle, basemap, etc.)
            for k, v in envelope.items():
                if k not in config_dict:
                    config_dict[k] = v

        # ── Promote flat title / subtitle strings into config ─
        # e.g. {"title": "Map of Nigeria", "subtitle": "Admin Boundary"}
        if "title" in config_dict and isinstance(config_dict["title"], str):
            title_text = config_dict.pop("title")
            subtitle_text = config_dict.pop("subtitle", "")
            config_dict.setdefault("title", {})
            config_dict["title"]["text"]     = title_text
            config_dict["title"]["subtitle"] = subtitle_text
        if "subtitle" in config_dict and isinstance(config_dict.get("subtitle"), str):
            sub = config_dict.pop("subtitle")
            config_dict.setdefault("title", {})
            config_dict["title"]["subtitle"] = sub

        # ── Leaflet → Matplotlib style key translation ────────
        # Accepts both naming conventions so the agent never needs to know the difference.
        _LEAFLET_TO_MPL = {
            "color":        "edge_color",
            "weight":       "edge_width",
            "fillColor":    "face_color",
            "fillOpacity":  "face_alpha",
            "opacity":      "face_alpha",
            "fill_color":   "face_color",
            "fill_opacity": "face_alpha",
            "stroke_color": "edge_color",
            "stroke_width": "edge_width",
            "strokeColor":  "edge_color",
            "strokeWidth":  "edge_width",
        }
        def _normalise_style(raw: dict) -> dict:
            out = {}
            for k, v in raw.items():
                normalised_key = _LEAFLET_TO_MPL.get(k, k)
                out[normalised_key] = v
            return out

        # ── Build MapConfig from dict ────────────────────────
        cfg = MapConfig()
        _nested = {
            "title":       TitleConfig,
            "grid":        GridConfig,
            "north_arrow": NorthArrowConfig,
            "scalebar":    ScaleBarConfig,
            "legend":      LegendConfig,
        }
        for key, val in config_dict.items():
            if key in _nested and isinstance(val, dict):
                sub = getattr(cfg, key)
                for k2, v2 in val.items():
                    if hasattr(sub, k2):
                        setattr(sub, k2, v2)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)

        # ── Ensure a Title exists ────────────────────────────
        if not cfg.title.text:
            first_layer = raw_layers[0] if raw_layers else {}
            cfg.title.text = (
                first_layer.get("name") or 
                first_layer.get("style", {}).get("label") or 
                "Geospatial Map"
            )
            logger.info("Auto-assigned title: %s", cfg.title.text)

        # ── Force output path to static directory ───────────
        from app.agents.tools.gis_tools import TEMP_MAP_DIR
        import uuid
        
        # Take just the filename or generate a new one
        if cfg.output_path and cfg.output_path != "output_maps/output_map.png":
            fname = os.path.basename(cfg.output_path)
        else:
            fname = f"map_{uuid.uuid4().hex[:8]}.png"
            
        cfg.output_path = os.path.join(TEMP_MAP_DIR, fname)

        # ── Build layer list ─────────────────────────────────
        processed_layers = []
        for layer in raw_layers:
            ltype   = layer.get("type", "vector")
            data    = layer.get("data")
            crs     = layer.get("crs", "EPSG:4326")

            # Translate and normalise style dict
            raw_style = _normalise_style(layer.get("style", {}))

            # Map layer-level "name" to style label if label not already set
            if "name" in layer and "label" not in raw_style:
                raw_style["label"] = layer["name"]

            if ltype == "raster":
                style = RasterStyle(**{k: v for k, v in raw_style.items() if hasattr(RasterStyle(), k)})
            else:
                style = VectorStyle(**{k: v for k, v in raw_style.items() if hasattr(VectorStyle(), k)})

            processed_layers.append({"type": ltype, "data": data, "style": style, "crs": crs})

        # ── Render ───────────────────────────────────────────
        output_path = render_map(processed_layers, cfg)
        filename = os.path.basename(output_path)

        return {
            "status":      "success",
            "output_path": output_path,
            "url":         f"/api/v1/static/maps/{filename}",
            "message":     f"Map saved to {output_path}",
        }

    except Exception as e:
        import traceback
        return json.dumps({
            "status":  "error",
            "message": str(e),
            "trace":   traceback.format_exc(),
        })


# ══════════════════════════════════════════════════════════════════════════════
#  ⑧ CONVENIENCE FACTORY — quick_map() for direct Python use
# ══════════════════════════════════════════════════════════════════════════════

def quick_map(
    data,
    output_path: str = "output_map.png",
    title: str = "",
    basemap: bool = False,
    north_arrow_position: str = "top-left",
    north_arrow_image: str = arrow_path,
    colormap: str = "YlOrRd",
    classify_by: str = "",
    face_color: str = "#3A86FF",
    face_alpha: float = 0.35,
    edge_color: str = "#003566",
    point_color: str = "#FFE600",
    point_size: int = 60,
    title_fontsize: int = 14,
    grid_dms: bool = True,
    dpi: int = 220,
    **kwargs
) -> str:
    """
    Convenience wrapper for a single-layer map with the most common options.

    Parameters
    ----------
    data : str or GeoDataFrame
        Path to a CSV/GeoJSON/Shapefile or a GeoDataFrame.
    output_path : str
        Where to save the PNG.
    title : str
        Map title.
    basemap : bool
        Whether to overlay a satellite basemap.
    north_arrow_position : str
        One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
    north_arrow_image : str
        Path to a custom north arrow PNG (leave blank for built-in arrow).
    colormap : str
        Matplotlib colormap for choropleth or raster (e.g. 'terrain', 'RdYlGn').
    classify_by : str
        Column name to use for choropleth classification.
    face_color : str
        Polygon fill colour.
    face_alpha : float
        Polygon fill transparency (0–1).
    edge_color : str
        Polygon / boundary line colour.
    point_color : str
        Point marker fill colour.
    point_size : int
        Point marker size.
    title_fontsize : int
        Title font size.
    grid_dms : bool
        If True, show DMS labels; otherwise decimal degrees.
    dpi : int
        Output resolution.

    Returns
    -------
    str : Absolute path to saved PNG.
    """
    ext = str(data).lower() if isinstance(data, str) else ""
    ltype = "raster" if ext.endswith((".tif", ".tiff")) else "vector"

    if ltype == "raster":
        style = RasterStyle(colormap=colormap, label=title)
    else:
        style = VectorStyle(
            face_color=face_color,
            face_alpha=face_alpha,
            edge_color=edge_color,
            point_color=point_color,
            point_size=point_size,
            colormap=colormap,
            classify_by=classify_by,
            label=title,
        )

    cfg = MapConfig(
        output_path=output_path,
        basemap=basemap,
        dpi=dpi,
        title=TitleConfig(text=title, fontsize=title_fontsize),
        north_arrow=NorthArrowConfig(
            position=north_arrow_position,
            image_path=north_arrow_image
        ),
        grid=GridConfig(dms_format=grid_dms),
    )

    return render_map([{"type": ltype, "data": data, "style": style}], cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  ⑨ EXAMPLE USAGE (runs when executed directly)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("Map Renderer Tool — Example Usage\n" + "=" * 40)

    # ── Example A: Single vector (CSV) map ──────────────────
    # output = quick_map(
    #     data="/content/data/WELL-10 REMEDIATION WORK AREA.csv",
    #     output_path="output_maps/well10.png",
    #     title="Well-10 Remediation Work Area",
    #     basemap=True,
    #     north_arrow_position="top-right",
    #     face_color="#e63946",
    #     edge_color="#1d3557",
    #     point_color="#FFE600",
    # )
    # print(f"Saved: {output}")

    # ── Example B: Choropleth from GeoJSON ──────────────────
    # output = quick_map(
    #     data="/content/data/region.geojson",
    #     output_path="output_maps/choropleth.png",
    #     title="Population Density by District",
    #     classify_by="pop_density",
    #     colormap="YlOrRd",
    # )
    # print(f"Saved: {output}")

    # ── Example C: Via agent tool (JSON interface) ───────────
    result = create_cartographic_map(
        layers_json=json.dumps([
            {
                "type": "vector",
                "data": "sample_points.csv",   # Replace with real path
                "style": {
                    "face_color": "#FF6B6B",
                    "edge_color": "#CC0000",
                    "label": "Work Area",
                }
            }
        ]),
        config_json=json.dumps({
            "output_path": "output_maps/agent_example.png",
            "basemap": False,
            "dpi": 200,
            "title": {
                "text": "REMEDIATION SITE MAP",
                "fontsize": 16,
                "fontweight": "bold",
            },
            "north_arrow": {
                "position": "top-right",
                "fallback_color": "black",
            },
            "grid": {
                "show": True,
                "dms_format": True,
                "n_ticks_x": 4,
                "n_ticks_y": 4,
            },
            "scalebar": {
                "position": "bottom-right",
                "unit": "m",
            },
            "legend": {
                "show": True,
                "title": "Map Legend",
                "location": "lower right",
            }
        }),
    )
    print(result)