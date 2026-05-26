import ee
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

from app.analysis.base import (
    load_boundary,
    get_area_sqkm,
    get_tile_url,
    get_gee_stats,
    calculate_export_options,
    OUTPUT_DIR
)
from app.analysis.export_registry import save_export_record
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Palettes (Color Ramps) ────────────────────────────────────────────────────
COLOR_RAMPS = {
    "terrain": ["#003300", "#006600", "#339900", "#cccc00", "#996633", "#ffffff"],
    "viridis": ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
    "magma":   ["#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f", "#cd4071", "#f1605d", "#fd9668", "#feca8d", "#fcfdbf"],
    "inferno": ["#000004", "#160b39", "#420a68", "#6a1c81", "#932b80", "#bc3754", "#dd513a", "#f37819", "#fca50a", "#f6d746"],
    "plasma":  ["#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"],
    "turbo":   ["#30123b", "#4a429a", "#467cf2", "#2dbdf0", "#21eea4", "#8cfb67", "#dae643", "#fdae34", "#e8590c", "#af1c02", "#7a0403"],
    "slope":   ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"],
    "water":   ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
}

DEM_VIS = {
    "min": 0, "max": 3000,
    "palette": COLOR_RAMPS["terrain"]
}

SLOPE_VIS = {
    "min": 0, "max": 45,  # Fixed range: 0-9 (Gentle) ... 45+ (Very Steep)
    "palette": COLOR_RAMPS["slope"]
}

HILLSHADE_VIS = {
    "min": 0, "max": 255,
    "palette": ["#000000", "#ffffff"]
}

CONTOUR_VIS = {
    "min": 0, "max": 3000,
    "palette": COLOR_RAMPS["terrain"]
}


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _merge_vis_params(
    default_vis: dict,
    stats:       dict,
    params:      dict,
    default_ramp: str
) -> dict:
    """
    Standardises visualization parameter merging.
    Prioritises manual overrides > dynamic stats (for most) > defaults.
    """
    vis = dict(default_vis)

    # 1. Apply Color Ramp
    ramp_name = params.get("color_ramp", default_ramp).lower()
    if ramp_name in COLOR_RAMPS:
        vis["palette"] = COLOR_RAMPS[ramp_name]

    # 2. Range Logic
    is_slope_or_hillshade = default_ramp in ["slope", "hillshade"]
    
    # Manual overrides take highest priority
    manual_min = params.get("min")
    manual_max = params.get("max")

    if manual_min is not None or manual_max is not None:
        if manual_min is not None: vis["min"] = manual_min
        if manual_max is not None: vis["max"] = manual_max
    elif not is_slope_or_hillshade:
        # Auto-stretch for DEM, Contour, etc.
        stat_min = next((v for k, v in stats.items() if k.endswith("_min")), None)
        stat_max = next((v for k, v in stats.items() if k.endswith("_max")), None)
        if stat_min is not None and stat_max is not None:
            vis["min"] = stat_min
            vis["max"] = stat_max
            
    return vis


# ── DEM ───────────────────────────────────────────────────────────────────────

def compute_dem(
    dataset_id:    str,
    boundary_path: str,
    params:        dict = {}
) -> dict:
    """
    Visualise a Digital Elevation Model.
    Returns tile URL for frontend display and export metadata.
    """
    geometry, gdf = load_boundary(boundary_path)
    area_sqkm     = get_area_sqkm(geometry)

    dem = ee.Image(dataset_id).select("elevation").clip(geometry)

    analysis_scale = params.get("scale", 30)
    stats          = get_gee_stats(dem, geometry, scale=analysis_scale)
    vis_params     = _merge_vis_params(DEM_VIS, stats, params, "terrain")

    tile_url       = get_tile_url(dem, vis_params)
    export_options = calculate_export_options(area_sqkm)

    export_id = save_export_record({
        "task":          "dem",
        "dataset_id":    dataset_id,
        "boundary_path": boundary_path,
        "params":        params,
        "tile_url":      tile_url,
        "area_sqkm":     area_sqkm,
        "vis_params":    vis_params,
        "export_options":         export_options["export_options"],
        "recommended_scale":      export_options["recommended_scale"]
    })

    return {
        "status":            "success",
        "tile_url":          tile_url,
        "export_id":         export_id,
        "vis_params":        vis_params,
        "stats":             stats,
        "area_sqkm":         area_sqkm,
        "export_options":    export_options,
        "recommended_scale": export_options["recommended_scale"]
    }


# ── Slope ─────────────────────────────────────────────────────────────────────

def compute_slope(
    dataset_id:    str,
    boundary_path: str,
    params:        dict = {}
) -> dict:
    """
    Compute slope from DEM on GEE cloud.
    Returns tile URL for frontend display and export metadata.
    """
    geometry, gdf = load_boundary(boundary_path)
    area_sqkm     = get_area_sqkm(geometry)

    dem   = ee.Image(dataset_id).select("elevation").clip(geometry)
    slope = ee.Terrain.slope(dem)

    analysis_scale = params.get("scale", 30)
    stats          = get_gee_stats(slope, geometry, scale=analysis_scale)
    vis_params     = _merge_vis_params(SLOPE_VIS, stats, params, "slope")

    tile_url       = get_tile_url(slope, vis_params)
    export_options = calculate_export_options(area_sqkm)

    export_id = save_export_record({
        "task":              "slope",
        "dataset_id":        dataset_id,
        "boundary_path":     boundary_path,
        "params":            params,
        "tile_url":          tile_url,
        "area_sqkm":         area_sqkm,
        "vis_params":        vis_params,
        "export_options":    export_options["export_options"],
        "recommended_scale": export_options["recommended_scale"]
    })

    return {
        "status":            "success",
        "tile_url":          tile_url,
        "export_id":         export_id,
        "vis_params":        vis_params,
        "stats":             stats,
        "area_sqkm":         area_sqkm,
        "export_options":    export_options,
        "recommended_scale": export_options["recommended_scale"]
    }


# ── Hillshade ─────────────────────────────────────────────────────────────────

def compute_hillshade(
    dataset_id:    str,
    boundary_path: str,
    params:        dict = {}
) -> dict:
    """
    Compute hillshade from DEM on GEE cloud.
    Returns tile URL for frontend display and export metadata.
    """
    geometry, gdf = load_boundary(boundary_path)
    area_sqkm     = get_area_sqkm(geometry)

    dem       = ee.Image(dataset_id).select("elevation").clip(geometry)
    hillshade = ee.Terrain.hillshade(dem)

    analysis_scale = params.get("scale", 30)
    stats          = get_gee_stats(hillshade, geometry, scale=analysis_scale)
    vis_params     = _merge_vis_params(HILLSHADE_VIS, stats, params, "hillshade")
    
    tile_url       = get_tile_url(hillshade, vis_params)
    export_options = calculate_export_options(area_sqkm)

    export_id = save_export_record({
        "task":              "hillshade",
        "dataset_id":        dataset_id,
        "boundary_path":     boundary_path,
        "params":            params,
        "tile_url":          tile_url,
        "area_sqkm":         area_sqkm,
        "vis_params":        vis_params,
        "export_options":    export_options["export_options"],
        "recommended_scale": export_options["recommended_scale"]
    })

    return {
        "status":            "success",
        "tile_url":          tile_url,
        "export_id":         export_id,
        "vis_params":        vis_params,
        "stats":             stats,
        "area_sqkm":         area_sqkm,
        "export_options":    export_options,
        "recommended_scale": export_options["recommended_scale"]
    }


# ── Contour ───────────────────────────────────────────────────────────────────

def compute_contour(
    dataset_id:    str,
    boundary_path: str,
    params:        dict = {}
) -> dict:
    """
    Generate contour lines from DEM.
    Contours are computed on GEE as a FeatureCollection
    and returned as a tile URL overlay.
    """
    interval      = params.get("interval", 20)
    geometry, gdf = load_boundary(boundary_path)
    area_sqkm     = get_area_sqkm(geometry)

    dem = ee.Image(dataset_id).select("elevation").clip(geometry)

    # Generate contours as a FeatureCollection on GEE
    contours = dem.reduceToVectors(
        reducer=ee.Reducer.countEvery(),
        geometry=geometry,
        scale=params.get("scale", 30),
        geometryType="polygon",
        eightConnected=False,
        labelProperty="elevation",
        maxPixels=1e10
    )

    # Rasterise contours back for tile display
    contour_image = contours.reduceToImage(
        properties=["elevation"],
        reducer=ee.Reducer.first()
    ).clip(geometry)

    analysis_scale = params.get("scale", 30)
    stats          = get_gee_stats(contour_image, geometry, scale=analysis_scale)
    vis_params     = _merge_vis_params(CONTOUR_VIS, stats, params, "terrain")
        
    tile_url       = get_tile_url(contour_image, vis_params)
    export_options = calculate_export_options(area_sqkm)

    export_id = save_export_record({
        "task":              "contour",
        "dataset_id":        dataset_id,
        "boundary_path":     boundary_path,
        "params":            params,
        "tile_url":          tile_url,
        "area_sqkm":         area_sqkm,
        "vis_params":        vis_params,
        "stats":             stats,
        "export_options":    export_options["export_options"],
        "recommended_scale": export_options["recommended_scale"],
        "interval_m":        interval
    })

    return {
        "status":            "success",
        "tile_url":          tile_url,
        "export_id":         export_id,
        "vis_params":        vis_params,
        "stats":             stats,
        "interval_m":        interval,
        "area_sqkm":         area_sqkm,
        "export_options":    export_options,
        "recommended_scale": export_options["recommended_scale"]
    }