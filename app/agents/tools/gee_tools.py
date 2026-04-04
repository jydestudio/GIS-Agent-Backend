import json
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Paths ────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
INDEX_PATH = BASE_DIR / "data" / "vector_stores" / "gee_index"

# ── Lazy Loading ─────────────────────────────────
_vector_db = None

def get_vector_db():
    """Load the FAISS index only when first needed."""
    global _vector_db
    if _vector_db is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"GEE index not found at {INDEX_PATH}. "
                "Run scripts/build_gee_index.py first."
            )

        logger.info("Loading GEE vector index from %s", INDEX_PATH)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vector_db = FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("GEE vector index loaded successfully")

    return _vector_db


# ── Tool ─────────────────────────────────────────

@tool
def search_gee_catalog(
    query: Annotated[str, (
        "Keywords describing the data you need. "
        "Examples: 'elevation', 'rainfall', 'land cover', 'nighttime lights', 'population density'. "
        "Always call this before using any GEE dataset to get the correct Collection ID."
    )]
) -> str:
    """
    Search the Google Earth Engine dataset catalog for the correct Collection ID.
    Returns the top matching datasets with their IDs and asset URLs.
    Always use this to verify a GEE ID before writing any Earth Engine script.
    """
    try:
        db = get_vector_db()
        docs = db.similarity_search(query, k=3)

        if not docs:
            return json.dumps({
                "status": "error",
                "message": f"No datasets found for query: '{query}'"
            })

        results = []
        for doc in docs:
            results.append({
                "id": doc.metadata.get("id"),
                "title": doc.page_content.split(".")[0].replace("Title: ", ""),
                "type": doc.metadata.get("type"),
                "asset_url": doc.metadata.get("asset_url")
            })

        logger.info("GEE search for '%s' returned %d results", query, len(results))
        return json.dumps({"status": "success", "results": results}, indent=2)

    except FileNotFoundError as e:
        logger.error("GEE index missing: %s", str(e))
        return json.dumps({"status": "error", "message": str(e)})

    except Exception as e:
        logger.error("GEE search failed: %s", str(e))
        return json.dumps({"status": "error", "message": f"Search failed: {str(e)}"})