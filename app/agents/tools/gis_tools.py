import os
import re
import uuid
import json
import requests
import pycountry
import osmnx as ox
import geopandas as gpd
from typing import Annotated
from langchain_core.tools import tool
from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)

# ── OSMnx Optimization Settings ──────────────────
ox.settings.requests_timeout = 30
ox.settings.use_cache = True
ox.settings.cache_folder = os.path.join(os.getcwd(), "cache", "osmnx")

settings = get_settings()
TEMP_MAP_DIR = os.path.abspath(settings.static_maps_dir)
if not os.path.exists(TEMP_MAP_DIR):
    os.makedirs(TEMP_MAP_DIR)


# ── GADM: local country + level resolution (zero network calls) ──────────────

def _parse_location_parts(location: str) -> list[str]:
    """Split 'Akure South, Ondo State, Nigeria' → ['Akure South', 'Ondo State', 'Nigeria']"""
    return [p.strip() for p in location.split(",") if p.strip()]


def _resolve_country_iso3(parts: list[str]) -> str | None:
    """
    Walk comma-separated parts right-to-left and match against pycountry.
    Entirely offline — no network calls.

    'Akure South, Ondo State, Nigeria' → checks 'Nigeria' first → 'NGA'
    'Lagos, Nigeria'                   → checks 'Nigeria'       → 'NGA'
    'Berlin, Germany'                  → checks 'Germany'       → 'DEU'
    """
    for token in reversed(parts):
        clean = re.sub(
            r"\b(state|province|region|district|republic\s+of|the)\b",
            "", token, flags=re.IGNORECASE
        ).strip(" ,.")

        # 1. Exact match by name / common name / official name
        country = (
            pycountry.countries.get(name=clean)
            or pycountry.countries.get(common_name=clean)
            or pycountry.countries.get(official_name=clean)
        )
        if country:
            return country.alpha_3

        # 2. Fuzzy search (handles abbreviations, alternate spellings, etc.)
        try:
            results = pycountry.countries.search_fuzzy(clean)
            if results:
                return results[0].alpha_3
        except LookupError:
            continue

    return None


def _infer_gadm_level(parts: list[str]) -> int:
    """
    Infer the GADM admin level from how many comma tokens the user gave.

      1 token  → 0 (country)       e.g. 'Nigeria'
      2 tokens → 1 (state/region)  e.g. 'Ondo State, Nigeria'
      3+tokens → 2 (LGA/district)  e.g. 'Akure South, Ondo State, Nigeria'
    """
    n = len(parts)
    if n >= 3:
        return 2
    if n == 2:
        return 1
    return 0


# ── GADM Helper ───────────────────────────────────────────────────────────────
def _fetch_boundary_from_gadm(location: str) -> gpd.GeoDataFrame | None:
    """
    Fetch an admin boundary from GADM with NO network calls for country resolution.

    Strategy:
      1. Parse the location string locally → ISO-3 country code + GADM level.
         Zero HTTP requests at this stage; pycountry works entirely offline.
      2. Download the national GADM GeoPackage once and cache it on disk.
         All subsequent calls for any place in that country skip the download.
      3. Filter the cached layer by name to return a single-feature GeoDataFrame.

    Returns a GeoDataFrame (EPSG:4326) or None so the caller falls back to OSM.
    """
    try:
        parts = _parse_location_parts(location)
        if not parts:
            logger.warning("GADM helper: could not parse location '%s'", location)
            return None

        # Step 1: Resolve country & admin level — 100% offline
        iso3 = _resolve_country_iso3(parts)
        if not iso3:
            logger.warning("GADM helper: could not resolve country from '%s'", location)
            return None

        gadm_level = _infer_gadm_level(parts)
        logger.info(
            "GADM helper: resolved '%s' → iso3=%s, level=%d (offline)",
            location, iso3, gadm_level
        )

        # Step 2: Download & cache GADM GeoPackage
        cache_dir = os.path.join(os.getcwd(), "cache", "gadm")
        os.makedirs(cache_dir, exist_ok=True)

        gpkg_path = os.path.join(cache_dir, f"{iso3}_level{gadm_level}.gpkg")
        national_gpkg = os.path.join(cache_dir, f"{iso3}_full.gpkg")

        if not os.path.exists(gpkg_path):
            if not os.path.exists(national_gpkg):
                gadm_url = (
                    f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/"
                    f"gadm41_{iso3}.gpkg"
                )
                logger.info("GADM helper: downloading %s …", gadm_url)
                dl = requests.get(gadm_url, stream=True, timeout=180)
                dl.raise_for_status()

                with open(national_gpkg, "wb") as fh:
                    for chunk in dl.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                logger.info("GADM helper: download complete → %s", national_gpkg)

            # Extract and cache just the needed admin level
            layer_name = f"ADM_ADM_{gadm_level}"
            logger.info("GADM helper: extracting layer '%s' …", layer_name)
            gdf_full = gpd.read_file(national_gpkg, layer=layer_name)
            gdf_full.to_file(gpkg_path, driver="GPKG")
            logger.info("GADM helper: cached level-%d layer for %s", gadm_level, iso3)
        else:
            gdf_full = gpd.read_file(gpkg_path)
            logger.info("GADM helper: loaded level-%d layer for %s from cache", gadm_level, iso3)

        # Step 3: Filter for the matching feature
        name_col = f"NAME_{gadm_level}"
        if name_col not in gdf_full.columns:
            logger.warning("GADM helper: column '%s' not found. Available: %s",
                           name_col, list(gdf_full.columns))
            return None

        search_term = parts[0]
        logger.info("GADM helper: searching for '%s' in column '%s' (%d features) …",
                    search_term, name_col, len(gdf_full))

        # a) Exact case-insensitive match
        exact = gdf_full[gdf_full[name_col].str.lower() == search_term.lower()]
        if not exact.empty:
            match = exact.iloc[[0]]
        else:
            # b) Contains match (e.g. "Ondo" matches "Ondo" when user typed "Ondo State")
            fuzzy = gdf_full[
                gdf_full[name_col].str.lower().str.contains(search_term.lower(), na=False)
            ]
            if fuzzy.empty:
                # c) Reverse contains (e.g. GADM has "Akure" and user typed "Akure South")
                fuzzy = gdf_full[
                    gdf_full[name_col].apply(
                        lambda n: isinstance(n, str) and n.lower() in search_term.lower()
                    )
                ]
            if fuzzy.empty:
                logger.warning(
                    "GADM helper: no feature matching '%s' in %s level %d",
                    search_term, iso3, gadm_level
                )
                return None
            match = fuzzy.iloc[[0]]

        # Step 4: Normalise CRS to WGS84
        match = match.set_crs("EPSG:4326") if match.crs is None else match.to_crs("EPSG:4326")

        logger.info(
            "GADM helper: matched '%s' → '%s' (level %d, iso3=%s)",
            search_term, match[name_col].iloc[0], gadm_level, iso3
        )
        return match

    except Exception as exc:
        logger.warning("GADM helper failed for '%s': %s", location, exc)
        return None


# ── OSM Helper (original logic, fallback) ─────────────────────────────────────
def _fetch_boundary_from_osm(location: str) -> gpd.GeoDataFrame | None:
    """
    Fetch an administrative boundary using OSMnx / Nominatim.
    Returns a single-row GeoDataFrame in EPSG:4326, or None on failure.
    """
    try:
        gdf = ox.geocode_to_gdf(location)
        if gdf.empty:
            return None
        geom_type = gdf.geom_type.iloc[0]
        if "Polygon" not in geom_type:
            return None
        return gdf
    except Exception as exc:
        logger.warning("OSM helper failed for '%s': %s", location, exc)
        return None


# ── Tools ─────────────────────────────────────────────────────────────────────

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
    This tool fetches the boundary geometry and saves it as a GeoJSON file.

    Source priority:
      1. GADM (Global Administrative Areas) — high-resolution, reliable, cached locally.
      2. OpenStreetMap / OSMnx — fallback if GADM is unavailable or has no match.

    Use this tool FIRST whenever you need a geometry/boundary for a map or spatial analysis.
    """
    try:
        logger.info("Downloading admin boundary for '%s'", location)

        gdf = None
        source_used = None

        # 1. Try GADM first
        logger.info("Attempting GADM source for '%s'", location)
        gdf = _fetch_boundary_from_gadm(location)
        if gdf is not None and not gdf.empty:
            source_used = "GADM"
            logger.info("GADM succeeded for '%s'", location)
        else:
            logger.info("GADM returned nothing for '%s'; falling back to OSM", location)

        # 2. Fall back to OSM
        if gdf is None or gdf.empty:
            logger.info("Attempting OSM source for '%s'", location)
            gdf = _fetch_boundary_from_osm(location)
            if gdf is not None and not gdf.empty:
                source_used = "OSM"
                logger.info("OSM succeeded for '%s'", location)

        # 3. Both sources failed
        if gdf is None or gdf.empty:
            return json.dumps({
                "status": "error",
                "message": (
                    f"Could not find a boundary for '{location}' from either GADM or OSM. "
                    "Try adding the State and Country, e.g. 'Akure South, Ondo State, Nigeria'."
                )
            })

        # 4. Safety Check: must be a Polygon/MultiPolygon
        geom_type = gdf.geom_type.iloc[0]
        if "Polygon" not in geom_type:
            return json.dumps({
                "status": "error",
                "message": (
                    f"'{location}' returned a {geom_type} instead of a Boundary (Polygon). "
                    "Please be more specific, e.g., 'Akure South Local Government' or 'Enugu State, Nigeria'."
                )
            })

        # 5. Project to UTM for area calculation
        utm_crs = gdf.estimate_utm_crs()
        gdf_projected = gdf.to_crs(utm_crs)

        # 6. Generate Unique ID and File Path
        unique_id = f"study_area_{uuid.uuid4().hex[:8]}"
        file_path = os.path.join(TEMP_MAP_DIR, f"{unique_id}.geojson")

        # 7. Calculate Area (sq km)
        area_sqkm = round(gdf_projected.area.iloc[0] / 1_000_000, 2)

        # 8. Save to Disk in WGS84 (required for Leaflet)
        gdf.to_file(file_path, driver="GeoJSON")

        # 9. Return metadata
        filename = os.path.basename(file_path)
        result = {
            "status": "success",
            "map_id": unique_id,
            "location": location,
            "file_path": file_path,
            "url": f"/api/v1/static/maps/{filename}",
            "crs": str(utm_crs),
            "geometry_type": geom_type,
            "area_sqkm": area_sqkm,
            "bounds": gdf.total_bounds.tolist(),  # [minx, miny, maxx, maxy] WGS84
            "source": source_used,                 # "GADM" or "OSM"
        }

        logger.info(
            "Successfully saved boundary (%s) to %s [source: %s]",
            location, file_path, source_used
        )
        return json.dumps(result)

    except requests.exceptions.ReadTimeout:
        logger.error("Geocoding timeout for '%s'", location)
        return json.dumps({
            "status": "error",
            "message": (
                f"Geocoding service timed out for '{location}'. "
                "The service might be busy. "
                "Please try again in a few moments or use a more specific name."
            )
        })

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