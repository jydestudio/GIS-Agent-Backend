import ee
import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape
from pathlib import Path

from app.analysis.base import (
    load_boundary, download_raster, render_map,
    get_stats, OUTPUT_DIR
)
from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_dem(dataset_id: str, boundary_path: str, params: dict = {}) -> dict:
    """Download and render a Digital Elevation Model."""
    scale = params.get("scale", 30)
    geometry, gdf = load_boundary(boundary_path)

    dem = ee.Image(dataset_id).select("elevation").clip(geometry)
    raster_meta = download_raster(dem, geometry, "dem_raw", scale=scale)

    stats = get_stats(raster_meta["path"])
    png_path = render_map(
        raster_meta["path"],
        title="Digital Elevation Model",
        colormap="terrain",
        label="Elevation (m)",
        boundary_gdf=gdf
    )

    return {
        "status": "success",
        "output_path": png_path,
        "raster_path": raster_meta["path"],
        "resolution_m": scale,
        "stats": stats
    }


def compute_slope(dataset_id: str, boundary_path: str, params: dict = {}) -> dict:
    """Compute and render a classified slope map."""
    scale = params.get("scale", 30)
    geometry, gdf = load_boundary(boundary_path)

    dem = ee.Image(dataset_id).select("elevation").clip(geometry)
    slope = ee.Terrain.slope(dem)
    raster_meta = download_raster(slope, geometry, "slope", scale=scale)

    stats = get_stats(raster_meta["path"])
    png_path = render_map(
        raster_meta["path"],
        title="Slope Map",
        colormap="RdYlGn_r",
        label="Slope (degrees)",
        classified=True,
        class_breaks=[0, 5, 10, 20, 35, 90],
        class_labels=["Flat", "Gentle", "Moderate", "Steep", "Very Steep"],
        boundary_gdf=gdf
    )

    return {
        "status": "success",
        "output_path": png_path,
        "raster_path": raster_meta["path"],
        "resolution_m": scale,
        "stats": stats
    }


def compute_contour(dataset_id: str, boundary_path: str, params: dict = {}) -> dict:
    """Generate contour lines from DEM and render as vector overlay."""
    scale = params.get("scale", 30)
    interval = params.get("interval", 20)
    geometry, gdf = load_boundary(boundary_path)

    dem = ee.Image(dataset_id).select("elevation").clip(geometry)
    raster_meta = download_raster(dem, geometry, "dem_contour", scale=scale)

    with rasterio.open(raster_meta["path"]) as src:
        data = src.read(1).astype(float)
        transform = src.transform

    min_elev = int(np.nanmin(data))
    max_elev = int(np.nanmax(data))
    levels = list(range(min_elev - (min_elev % interval), max_elev + interval, interval))

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    contour = ax.contour(data, levels=levels, colors="cyan", linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=6, colors="white", fmt="%d m")
    gdf.boundary.plot(ax=ax, color="white", linewidth=1.5)

    ax.set_title(f"Contour Map ({interval}m interval)", color="white",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    png_path = str(OUTPUT_DIR / "contour_map.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    return {
        "status": "success",
        "output_path": png_path,
        "interval_m": interval,
        "elevation_range": {"min": min_elev, "max": max_elev}
    }


def compute_hillshade(dataset_id: str, boundary_path: str, params: dict = {}) -> dict:
    """Compute hillshade for 3D terrain visualization."""
    scale = params.get("scale", 30)
    geometry, gdf = load_boundary(boundary_path)

    dem = ee.Image(dataset_id).select("elevation").clip(geometry)
    hillshade = ee.Terrain.hillshade(dem)
    raster_meta = download_raster(hillshade, geometry, "hillshade", scale=scale)

    png_path = render_map(
        raster_meta["path"],
        title="Hillshade",
        colormap="gray",
        label="Hillshade Intensity",
        boundary_gdf=gdf
    )

    return {
        "status": "success",
        "output_path": png_path,
        "raster_path": raster_meta["path"],
        "resolution_m": scale
    }