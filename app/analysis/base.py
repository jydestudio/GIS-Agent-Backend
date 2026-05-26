import ee
import geemap
import geopandas as gpd
import rasterio
import rasterio.warp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from rasterio.enums import Resampling
from pathlib import Path
import json
import os

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Limits ────────────────────────────────────────────────────────────────────
MAX_PIXELS_DIRECT = 100_000_000   # ~100M pixels → direct download
GEE_EXPORT_FOLDER = "GeoAI_Exports"


# ── GEE Init ──────────────────────────────────────────────────────────────────

def init_gee():
    """
    Initialise GEE using service account key.
    Reads key path from GEE_SERVICE_ACCOUNT_KEY env variable.
    Call once at app startup.
    """
    key_path_env = os.getenv("GEE_SERVICE_ACCOUNT_KEY")

    if key_path_env:
        key_path = Path(key_path_env)
        if not key_path.is_absolute():
            key_path = BASE_DIR / key_path
    else:
        key_path = (
            BASE_DIR
            / "data"
            / "GEE authentication"
            / "gis-agent-492315-960d3e9d0fce.json"
        )

    if not key_path.exists():
        raise FileNotFoundError(
            f"GEE service account key not found at {key_path}. "
            "Set GEE_SERVICE_ACCOUNT_KEY in your .env file."
        )

    with open(key_path) as f:
        key_data = json.load(f)

    service_account_email = key_data.get("client_email")
    if not service_account_email:
        raise ValueError("client_email not found in service account key file.")

    credentials = ee.ServiceAccountCredentials(
        email=service_account_email,
        key_file=str(key_path)
    )
    ee.Initialize(credentials)
    logger.info("GEE initialised with service account: %s", service_account_email)


# ── Boundary ──────────────────────────────────────────────────────────────────

def load_boundary(boundary_path: str) -> tuple:
    """
    Load GeoJSON boundary file.
    Returns (ee.Geometry, geopandas.GeoDataFrame).
    """
    gdf      = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    geometry = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)
    return geometry, gdf


# ── Area ──────────────────────────────────────────────────────────────────────

def get_area_sqkm(geometry: ee.Geometry) -> float:
    """Return area of geometry in square kilometres."""
    return round(
        geometry.area(maxError=1000).divide(1e6).getInfo(), 2
    )


# ── Export Options ────────────────────────────────────────────────────────────

def calculate_export_options(
    area_sqkm:        float,
    n_bands:          int  = 1,
    dtype_bytes:      int  = 4,
    candidate_scales: list = None
) -> dict:
    """
    Calculate feasible download scales based on area size.
    Returns export options with size estimates and recommendation.
    """
    if candidate_scales is None:
        candidate_scales = [10, 30, 90, 250, 500, 1000]

    options     = []
    recommended = None

    for scale in candidate_scales:
        estimated_pixels = (area_sqkm * 1e6) / (scale ** 2)
        estimated_mb     = (estimated_pixels * n_bands * dtype_bytes) / 1e6
        feasible_direct  = estimated_pixels <= MAX_PIXELS_DIRECT
        method           = "direct_download" if feasible_direct else "google_drive_export"

        options.append({
            "scale_m":          scale,
            "estimated_mb":     round(estimated_mb, 1),
            "estimated_pixels": int(estimated_pixels),
            "feasible_direct":  feasible_direct,
            "method":           method,
        })

        if feasible_direct and recommended is None:
            recommended = scale

    # If nothing fits direct download, recommend coarsest scale via Drive
    if recommended is None:
        recommended = candidate_scales[-1]

    return {
        "area_sqkm":        area_sqkm,
        "export_options":   options,
        "recommended_scale": recommended,
        "note": (
            "Scales marked 'google_drive_export' will be sent to your "
            "Google Drive. Direct download available for smaller scales."
        )
    }


# ── Tile URL ──────────────────────────────────────────────────────────────────

def get_tile_url(ee_image: ee.Image, vis_params: dict) -> str:
    """
    Get XYZ tile URL from a GEE image for frontend map display.
    No size limit — works for any area worldwide.
    """
    map_id = ee_image.getMapId(vis_params)
    return map_id["tile_fetcher"].url_format


# ── GEE Cloud Stats ───────────────────────────────────────────────────────────

def get_gee_stats(
    ee_image: ee.Image,
    geometry: ee.Geometry,
    scale:    int = 1000,
    band:     str = None
) -> dict:
    """
    Compute raster statistics entirely on GEE cloud.
    Uses a coarse scale for speed — no download needed.
    """
    image = ee_image.select(band) if band else ee_image

    raw = image.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.minMax(),  sharedInputs=True)
            .combine(ee.Reducer.stdDev(),  sharedInputs=True),
        geometry=geometry,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True
    ).getInfo()

    return {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in raw.items()
    }


# ── Direct Download ───────────────────────────────────────────────────────────

def download_raster(
    ee_image: ee.Image,
    geometry: ee.Geometry,
    filename: str,
    scale:    int = 30,
    crs:      str = "EPSG:4326"
) -> dict:
    """
    Download a GEE image to a local GeoTIFF.
    Only use for small areas — check calculate_export_options first.
    """
    output_path = OUTPUT_DIR / f"{filename}.tif"
    logger.info("Downloading raster: %s at %dm", filename, scale)

    geemap.ee_export_image(
        ee_image,
        filename=str(output_path),
        scale=scale,
        region=geometry,
        crs=crs,
        file_per_band=False
    )

    with rasterio.open(output_path) as src:
        actual_crs = str(src.crs)
        res        = src.res
        bounds     = src.bounds

    return {
        "path":         str(output_path),
        "resolution_m": scale,
        "crs":          actual_crs,
        "bounds":       list(bounds),
        "pixel_size":   list(res)
    }


# ── Drive Export ──────────────────────────────────────────────────────────────

def export_to_drive(
    ee_image: ee.Image,
    geometry: ee.Geometry,
    filename: str,
    scale:    int,
    folder:   str = GEE_EXPORT_FOLDER
) -> dict:
    """
    Export a large GEE image to Google Drive.
    Returns task info for progress monitoring.
    """
    task = ee.batch.Export.image.toDrive(
        image=ee_image,
        description=filename,
        folder=folder,
        scale=scale,
        region=geometry,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        crs="EPSG:4326"
    )
    task.start()
    logger.info("Drive export started: %s (task: %s)", filename, task.id)

    return {
        "status":       "exporting",
        "task_id":      task.id,
        "filename":     f"{filename}.tif",
        "folder":       folder,
        "scale_m":      scale,
        "message": (
            f"Export started to Google Drive folder '{folder}'. "
            "Check progress at https://code.earthengine.google.com/tasks"
        )
    }


# ── Smart Export ──────────────────────────────────────────────────────────────

def smart_export(
    ee_image:   ee.Image,
    geometry:   ee.Geometry,
    filename:   str,
    scale:      int,
    area_sqkm:  float,
    n_bands:    int = 1
) -> dict:
    """
    Automatically choose direct download or Drive export
    based on estimated file size.
    Called by the export API endpoint — not during analysis.
    """
    estimated_pixels = (area_sqkm * 1e6) / (scale ** 2)

    if estimated_pixels <= MAX_PIXELS_DIRECT:
        logger.info("Direct download: %s at %dm", filename, scale)
        raster_meta = download_raster(ee_image, geometry, filename, scale)
        return {
            "method":       "direct_download",
            "raster_path":  raster_meta["path"],
            "scale_m":      scale,
            "status":       "success"
        }
    else:
        logger.info("Too large for direct download, routing to Drive")
        return export_to_drive(ee_image, geometry, filename, scale)


# ── Resampling ────────────────────────────────────────────────────────────────

def resample_raster(
    input_path:        str,
    target_resolution: float,
    target_crs:        str = "EPSG:4326"
) -> str:
    """Reproject and resample a local raster to target resolution and CRS."""
    input_path  = Path(input_path)
    output_path = input_path.parent / f"{input_path.stem}_resampled.tif"

    with rasterio.open(input_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs,
            src.width, src.height,
            *src.bounds,
            resolution=target_resolution
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs":       target_crs,
            "transform": transform,
            "width":     width,
            "height":    height
        })
        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )

    logger.info("Resampled %s → %s", input_path.name, output_path.name)
    return str(output_path)


def harmonise_rasters(
    raster_paths:      list,
    target_resolution: float = 30,
    target_crs:        str   = "EPSG:4326"
) -> tuple:
    """
    Reproject and resample all rasters to the same grid.
    Returns (harmonised_paths, report).
    Used before AHP weighted combination.
    """
    harmonised = []
    report = {
        "target_resolution": f"{target_resolution}m",
        "target_crs":        target_crs,
        "layers_resampled":  [],
        "layers_unchanged":  []
    }

    for path in raster_paths:
        with rasterio.open(path) as src:
            current_res  = round(src.res[0], 4)
            current_epsg = src.crs.to_epsg()
            target_epsg  = int(target_crs.split(":")[1])
            name         = Path(path).stem
            needs_resample = (
                current_epsg != target_epsg or
                abs(current_res - target_resolution) > 1
            )

        if needs_resample:
            new_path = resample_raster(path, target_resolution, target_crs)
            harmonised.append(new_path)
            report["layers_resampled"].append(
                f"{name} ({current_res}m → {target_resolution}m)"
            )
        else:
            harmonised.append(path)
            report["layers_unchanged"].append(name)

    return harmonised, report


# ── Local Raster Utils ────────────────────────────────────────────────────────

def read_raster_array(raster_path: str) -> np.ndarray:
    """Read local raster as numpy array, masking nodata."""
    with rasterio.open(raster_path) as src:
        data   = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
    return data


def get_stats(raster_path: str) -> dict:
    """Compute statistics from a local raster file."""
    data = read_raster_array(raster_path)
    return {
        "min":  round(float(np.nanmin(data)),  4),
        "max":  round(float(np.nanmax(data)),  4),
        "mean": round(float(np.nanmean(data)), 4),
        "std":  round(float(np.nanstd(data)),  4)
    }


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_map(
    raster_path:  str,
    title:        str,
    colormap:     str  = "viridis",
    label:        str  = "",
    classified:   bool = False,
    class_breaks: list = None,
    class_labels: list = None,
    boundary_gdf        = None
) -> str:
    """
    Render a locally downloaded GeoTIFF to a styled PNG.
    Only called when raster has been downloaded via smart_export.
    For live display use tile_url instead.
    """
    data     = read_raster_array(raster_path)
    png_path = Path(raster_path).with_suffix(".png")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if classified and class_breaks and class_labels:
        cmap      = plt.get_cmap(colormap, len(class_labels))
        norm      = BoundaryNorm(class_breaks, cmap.N)
        im        = ax.imshow(data, cmap=cmap, norm=norm)
        midpoints = [
            (class_breaks[i] + class_breaks[i + 1]) / 2
            for i in range(len(class_breaks) - 1)
        ]
        cbar = plt.colorbar(im, ax=ax, ticks=midpoints, shrink=0.6)
        cbar.set_ticklabels(class_labels)
        cbar.set_label(label, color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    else:
        im   = ax.imshow(data, cmap=colormap)
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label(label or title, color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="white", linewidth=1.5)

    ax.set_title(title, color="white", fontsize=14,
                 fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    logger.info("Map rendered: %s", png_path)
    return str(png_path)