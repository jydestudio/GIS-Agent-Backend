import json
import uuid
from pathlib import Path
from datetime import datetime
from app.core.logging import get_logger

logger = get_logger(__name__)

REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "export_registry.json"
)


def save_export_record(record: dict) -> str:
    """Save analysis result to export registry. Returns export_id."""
    export_id        = f"{record['task']}_{uuid.uuid4().hex[:8]}"
    record["export_id"]   = export_id
    record["created_at"]  = datetime.utcnow().isoformat()

    registry = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)

    registry[export_id] = record

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("Export record saved: %s", export_id)
    return export_id


def get_export_record(export_id: str) -> dict:
    """Retrieve an export record by ID."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError("Export registry not found.")

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    if export_id not in registry:
        raise KeyError(f"Export ID '{export_id}' not found.")

    return registry[export_id]


def list_export_records() -> list:
    """List all saved export records."""
    if not REGISTRY_PATH.exists():
        return []

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    return [
        {
            "export_id":   k,
            "task":        v.get("task"),
            "area_sqkm":   v.get("area_sqkm"),
            "created_at":  v.get("created_at"),
            "tile_url":    v.get("tile_url")
        }
        for k, v in registry.items()
    ]