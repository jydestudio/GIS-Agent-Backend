import json
from typing import Annotated
from langchain_core.tools import tool
from app.core.logging import get_logger

from app.analysis import terrain

logger = get_logger(__name__)

VALID_TASKS = [
    "dem",
    "slope",
    "contour",
    "hillshade",
]

ROUTER = {
    "dem":       terrain.compute_dem,
    "slope":     terrain.compute_slope,
    "contour":   terrain.compute_contour,
    "hillshade": terrain.compute_hillshade,
}


@tool
def run_gee_analysis(
    task: Annotated[str, (
        f"The analysis task to run. Must be one of: {', '.join(VALID_TASKS)}. "
        "Use 'dem' for elevation model, 'slope' for terrain steepness, "
        "'contour' for contour lines, 'hillshade' for 3D terrain shading."
    )],
    dataset_id: Annotated[str, (
        "The GEE Collection ID from search_gee_catalog result. "
        "Use the exact 'id' field from the catalog search response. "
        "For all terrain tasks (dem, slope, contour, hillshade) "
        "always search for SRTM elevation first."
    )],
    boundary_path: Annotated[str, (
        "The exact 'file_path' value from download_admin_boundary response. "
        "This is a full local path ending in .geojson. "
        "Do NOT use the 'url' field — use 'file_path'."
    )],
    params: Annotated[dict, (
        "Optional parameters. Supported keys: "
        "'scale' (int, resolution in metres, default 30), "
        "'interval' (int, contour interval in metres, default 20), "
        "'color_ramp' (str, palette to use. Options: 'terrain', 'viridis', 'magma', 'inferno', 'plasma', 'turbo', 'slope', 'water'), "
        "'min' (float, minimum value for scaling), "
        "'max' (float, maximum value for scaling), "
        "'date_range' (list, ['YYYY-MM-DD', 'YYYY-MM-DD'] for time-sensitive datasets). "
        "Pass {} if not needed."
    )] = {}
) -> str:
    """
    Run a geospatial analysis using Google Earth Engine cloud computing.
    All computation happens on GEE servers — no file size limits for display.
    
    Returns:
    - tile_url: XYZ tile URL to display on the frontend map
    - export_id: reference ID to trigger download later without rerunning
    - export_options: available scales with estimated file sizes
    - recommended_scale: best resolution for downloading this area
    - stats: min, max, mean computed on GEE cloud
    - area_sqkm: area of the study region

    Strict workflow — always follow this order:
    1. Call download_admin_boundary → get file_path
    2. Call search_gee_catalog → get dataset id
    3. Call run_gee_analysis with task, dataset_id, file_path, params
    """
    try:
        logger.info(
            "run_gee_analysis → task=%s dataset=%s", task, dataset_id
        )

        if task not in ROUTER:
            return json.dumps({
                "status":  "error",
                "message": (
                    f"Unknown task '{task}'. "
                    f"Valid tasks are: {', '.join(VALID_TASKS)}"
                )
            })

        result = ROUTER[task](dataset_id, boundary_path, params)
        result["task"] = task

        logger.info(
            "run_gee_analysis ✓ task=%s export_id=%s",
            task, result.get("export_id")
        )

        return json.dumps(result)

    except Exception as e:
        logger.error("run_gee_analysis failed: task=%s error=%s", task, str(e))
        return json.dumps({
            "status":  "error",
            "task":    task,
            "message": str(e)
        })