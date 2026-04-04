from app.agents.tools.gis_tools import download_admin_boundary, get_slope_map, get_contour_map
from app.agents.tools.intent_tool import extract_project_intent

from app.agents.tools.map_making import create_cartographic_map

ALL_TOOLS = [
    download_admin_boundary,
    get_slope_map,
    get_contour_map,
    extract_project_intent,
    create_cartographic_map,
]
