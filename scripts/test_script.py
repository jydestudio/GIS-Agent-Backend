import sys
from pathlib import Path

# Add the Backend root to the path so imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Load .env before any app imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.agents.tools.gee_tools import search_gee_catalog

def test_search():
    test_queries = [
        "elevation",
        "population density",
        "land cover",
        "rainfall",
        "nighttime lights",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        result = search_gee_catalog.invoke({"query": query})
        print(result)

if __name__ == "__main__":
    test_search()