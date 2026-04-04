import os
import uuid
import json
import tempfile
import osmnx as ox
import geopandas as gpd
from typing import Annotated
from langchain_core.tools import tool
from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)

settings = get_settings()
TEMP_MAP_DIR = os.path.abspath(settings.static_maps_dir)
if not os.path.exists(TEMP_MAP_DIR):
    os.makedirs(TEMP_MAP_DIR)


@tool
def download_admin_boundary(
    location: Annotated[str, (
        "The full administrative name of the study area. "
        "IMPORTANT: To find a boundary (Polygon), always include the State and Country. "
        "Example: Use 'Akure South, Ondo State, Nigeria' instead of just 'Akure'."
    )]
) -> str:
    """
    Download the administrative boundary (GeoJSON) for a specific location.
    This tool fetches the boundary geometry from OSM, simplifies it, and saves it as a GeoJSON file.
    Use this tool FIRST whenever you need a geometry/boundary for a map or spatial analysis.
    """
    try:
        logger.info("Downloading admin boundary for %s", location)
        
        # 1. Geocode the location to a GeoDataFrame
        # We use geocode_to_gdf which specifically looks for boundaries
        gdf = ox.geocode_to_gdf(location)
        
        if gdf.empty:
            return json.dumps({
                "status": "error", 
                "message": f"Could not find any results for '{location}'. Try adding the State and Country."
            })

        # 2. Safety Check: Ensure we got a Polygon/MultiPolygon
        # If OSM returns a 'Point', OSMnx cannot treat it as a 'Study Area' boundary
        geom_type = gdf.geom_type.iloc[0]
        if "Polygon" not in geom_type:
            return json.dumps({
                "status": "error", 
                "message": (
                    f"'{location}' returned a {geom_type} instead of a Boundary (Polygon). "
                    "Please be more specific, e.g., 'Akure South Local Government' or 'Enugu State, Nigeria'."
                )
            })

        # 3. Project to the best UTM CRS (Convert degrees to meters for analysis)
        utm_crs = gdf.estimate_utm_crs()
        gdf_projected = gdf.to_crs(utm_crs)
        
        # 4. Generate Unique ID and File Path
        unique_id = f"study_area_{uuid.uuid4().hex[:8]}"
        file_path = os.path.join(TEMP_MAP_DIR, f"{unique_id}.geojson")
        
        # 5. Calculate Area (in sq km)
        area_sqm = gdf_projected.area.iloc[0]
        area_sqkm = area_sqm / 1_000_000
        
        # 6. Save to Disk (GeoJSON must be in WGS84/EPSG:4326 for Leaflet)
        gdf.to_file(file_path, driver='GeoJSON')
        
        # 7. Metadata for the Frontend/Orchestrator
        # The frontend will use the 'url' to fetch and render the GeoJSON
        filename = os.path.basename(file_path)
        result = {
            "status": "success",
            "map_id": unique_id,
            "location": location,
            "file_path": file_path,
            "url": f"/api/v1/static/maps/{filename}",
            "crs": str(utm_crs),
            "geometry_type": geom_type,
            "area_sqkm": round(area_sqkm, 2),
            "bounds": gdf.total_bounds.tolist() # [minx, miny, maxx, maxy] in WGS84
        }
        
        logger.info("Successfully saved boundary to %s", file_path)
        return json.dumps(result)

    except Exception as e:
        logger.error("Failed to generate study area map: %s", str(e))
        return json.dumps({
            "status": "error", 
            "message": f"GIS Error: {str(e)}"
        })


@tool
def get_slope_map(
    location: Annotated[str, "The name of the study area"],
    resolution: Annotated[int, "Resolution in meters (default 30)"] = 30
) -> str:
    """
    Generate a slope (terrain steepness) map for a specific location using SRTM data.
    Useful for site suitability, flood risk, and construction planning.
    """
    return f"Successfully generated {resolution}m slope map for {location}. [Placeholder Map ID: slope_{location.lower().replace(' ', '_')}]"

@tool
def get_contour_map(
    location: Annotated[str, "The name of the study area"],
    interval: Annotated[int, "Contour interval in meters (default 10)"] = 10
) -> str:
    """
    Generate topographic contour lines for a specific location.
    Useful for visualizing elevation changes and civil engineering projects.
    """
    return f"Successfully generated {interval}m contour map for {location}. [Placeholder Map ID: contour_{location.lower().replace(' ', '_')}]"
