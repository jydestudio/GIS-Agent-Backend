import argparse
import sys
import logging
import json
from pathlib import Path

# ── Path Handling ─────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ── Load Environment ──────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# ── Imports ──────────────────────────────────────────
from app.analysis.base import init_gee
from app.analysis.terrain import compute_slope, compute_dem
from app.core.logging import setup_logging, get_logger

# ── Setup Logging ────────────────────────────────────
setup_logging(level="INFO")
logger = get_logger(__name__)

def run_test(mode: str, scale: int, boundary: str = None):
    """Execute a sample terrain computation."""
    
    # 1. Initialize Google Earth Engine
    logger.info("Initializing GEE...")
    try:
        init_gee()
    except Exception as e:
        logger.error("GEE initialization failed: %s", str(e))
        return

    # 2. Select study area
    maps_dir = ROOT_DIR / "output_maps"
    if boundary:
        sample_boundary = str(maps_dir / boundary)
    else:
        geojson_files = list(maps_dir.glob("study_area_*.geojson"))
        if not geojson_files:
            logger.error("No GeoJSON study area found in 'output_maps/'.")
            return
        sample_boundary = str(geojson_files[0])

    if not Path(sample_boundary).exists():
        logger.error(f"Boundary file not found: {sample_boundary}")
        return

    logger.info(f"Using boundary: {sample_boundary}")
    logger.info(f"Mode: {mode.upper()} | Scale: {scale}m")

    # 3. Compute
    try:
        if mode == "slope":
            result = compute_slope(
                dataset_id="USGS/SRTMGL1_003",
                boundary_path=sample_boundary,
                params={"scale": scale}
            )
        else:
            result = compute_dem(
                dataset_id="USGS/SRTMGL1_003",
                boundary_path=sample_boundary,
                params={"scale": scale}
            )
        
        logger.info("Success! Result:")
        print(json.dumps(result, indent=4))
        logger.info(f"Output saved to: {result.get('output_path')}")
        
    except Exception as e:
        err_msg = str(e)
        if "Total request size" in err_msg:
            logger.error("GEE Export Limit Exceeded (50MB).")
            logger.warning("The area is too large for a direct download at this scale.")
            logger.info(f"Try increasing the --scale (e.g. --scale {scale * 2}) or choosing a smaller boundary.")
        else:
            logger.error(f"Computation failed: {err_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Terrain Analysis Tester")
    parser.add_argument("--mode", choices=["slope", "dem"], default="slope", help="Analysis type")
    parser.add_argument("--scale", type=int, default=30, help="Resolution in meters (default: 30)")
    parser.add_argument("--boundary", type=str, help="Specific GeoJSON filename in output_maps/")
    
    args = parser.parse_args()
    run_test(args.mode, args.scale, args.boundary)
