import sys
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/PMLS/OneDrive/Desktop/AI Research Stack")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from vector_store import VectorStoreService

def inspect():
    store = VectorStoreService()
    stats = store.get_collection_stats()
    print("=== CHROMADB STATS ===")
    print(f"Total chunks: {stats.get('total_chunks')}")
    print(f"Total papers: {stats.get('total_papers')}")
    print("Papers List:")
    for p in stats.get("papers_list", []):
        print(f" - {p}")
        
    print("\n=== QUERY TEST ===")
    # Let's search for framework chunks specifically
    query = "cybersecurity framework"
    chunks = store.query_similar_chunks(query, limit=5)
    print(f"Searching for '{query}' retrieved {len(chunks)} chunks:")
    for idx, c in enumerate(chunks):
        meta = c.get("metadata", {})
        print(f"\n[{idx+1}] Title: '{meta.get('title')}' (Pages: {meta.get('pages')})")
        print(f"Distance: {c.get('distance')}")
        print(f"Snippet: {c.get('text')[:300]}...")

if __name__ == "__main__":
    inspect()
