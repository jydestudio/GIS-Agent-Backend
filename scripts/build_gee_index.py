import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "raw" / "gee_catalog.json"
SAVE_PATH = BASE_DIR / "data" / "vector_stores" / "gee_index"

def build_index():
    if not JSON_PATH.exists():
        print(f"❌ Error: Could not find JSON at {JSON_PATH}")
        return

    print(f"📖 Loading GEE Catalog from {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    documents = []
    skipped = 0

    for entry in data:
        if not entry.get("id") or not entry.get("title"):
            skipped += 1
            continue

        tags_list = entry.get("tags", "").split(",")
        tags_clean = ", ".join(t.strip() for t in tags_list)

        # Richer semantic content — title, ID, provider, tags, type, years
        content = (
            f"{entry.get('title')}. "
            f"Collection ID: {entry.get('id')}. "
            f"Provider: {entry.get('provider', '')}. "
            f"Keywords: {tags_clean}. "
            f"Data type: {entry.get('type', '')}. "
            f"Available from {entry.get('startyear', '')} to {entry.get('endyear', '')}."
        )

        metadata = {
            "id": entry.get("id"),
            "asset_url": entry.get("asset_url", ""),
            "type": entry.get("type", "unknown")
        }
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"📦 Prepared {len(documents)} documents ({skipped} skipped).")

    print("🧠 Initializing Embedding Model (this may take a moment on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("⚡ Creating Vector Store...")
    vector_store = FAISS.from_documents(documents, embeddings)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(SAVE_PATH))

    print(f"✅ Success! GEE Vector Index saved to: {SAVE_PATH}")
    print(f"   Total indexed: {len(documents)} datasets")

if __name__ == "__main__":
    build_index()