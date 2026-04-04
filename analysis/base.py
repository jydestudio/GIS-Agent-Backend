import ee
import geemap
import geopandas as gpd
import rasterio
import rasterio.warp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from rasterio.enums import Resampling
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def init_gee(project_id: str = None):
    """Authenticate and initialise GEE. Call once at app startup."""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        logger.info("GEE initialised successfully")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def load_boundary(boundary_path: str) -> tuple:
    """
    Load GeoJSON boundary.
    Returns (ee.Geometry, geopandas.GeoDataFrame)
    """
    gdf = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    geometry = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)
    return geometry, gdf


def download_raster(
    ee_image: ee.Image,
    geometry: ee.Geometry,
    filename: str,
    scale: int = 30,
    crs: str = "EPSG:4326"
) -> dict:
    """
    Download a GEE image to a local GeoTIFF.
    Returns metadata dict including path and resolution.
    """
    output_path = OUTPUT_DIR / f"{filename}.tif"

    logger.info("Downloading raster: %s at %dm resolution", filename, scale)

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
        res = src.res
        bounds = src.bounds

    return {
        "path": str(output_path),
        "resolution_m": scale,
        "crs": actual_crs,
        "bounds": list(bounds),
        "pixel_size": res
    }


def resample_raster(
    input_path: str,
    target_resolution: float,
    target_crs: str = "EPSG:4326"
) -> str:
    """Reproject and resample a raster to target resolution and CRS."""
    input_path = Path(input_path)
    output_path = input_path.parent / f"{input_path.stem}_resampled.tif"

    with rasterio.open(input_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
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

    return str(output_path)


def harmonise_rasters(
    raster_paths: list,
    target_resolution: float = 30,
    target_crs: str = "EPSG:4326"
) -> tuple:
    """
    Resample all rasters to the same grid.
    Returns (harmonised_paths, report)
    """
    harmonised = []
    report = {"target_resolution": f"{target_resolution}m",
              "target_crs": target_crs,
              "layers_resampled": [],
              "layers_unchanged": []}

    for path in raster_paths:
        with rasterio.open(path) as src:
            current_res = round(src.res[0], 4)
            current_crs = src.crs.to_epsg()
            target_epsg = int(target_crs.split(":")[1])
            name = Path(path).stem

            needs_resample = (
                current_crs != target_epsg or
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


def read_raster_array(raster_path: str) -> tuple:
    """Read raster as numpy array, masking nodata."""
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
    return data


def get_stats(raster_path: str) -> dict:
    """Compute basic raster statistics."""
    data = read_raster_array(raster_path)
    return {
        "min": round(float(np.nanmin(data)), 4),
        "max": round(float(np.nanmax(data)), 4),
        "mean": round(float(np.nanmean(data)), 4),
        "std": round(float(np.nanstd(data)), 4)
    }


def render_map(
    raster_path: str,
    title: str,
    colormap: str = "viridis",
    label: str = "",
    classified: bool = False,
    class_breaks: list = None,
    class_labels: list = None,
    boundary_gdf=None
) -> str:
    """
    Render a GeoTIFF to a styled PNG.
    Supports continuous colormaps and classified legend.
    """
    data = read_raster_array(raster_path)
    png_path = Path(raster_path).with_suffix(".png")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if classified and class_breaks and class_labels:
        cmap = plt.get_cmap(colormap, len(class_labels))
        norm = BoundaryNorm(class_breaks, cmap.N)
        im = ax.imshow(data, cmap=cmap, norm=norm)
        cbar = plt.colorbar(im, ax=ax, ticks=class_breaks)
        cbar.set_ticklabels(class_labels)
        cbar.set_label(label, color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    else:
        im = ax.imshow(data, cmap=colormap)
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label(label or title, color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="white", linewidth=1.5)

    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    logger.info("Map rendered: %s", png_path)
    return str(png_path)