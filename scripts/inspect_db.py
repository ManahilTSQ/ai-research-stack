import sys
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/PMLS/OneDrive/Desktop/AI Research Stack")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from vector_store import VectorStoreService

def inspect():
    store = VectorStoreService()
    data = store.collection.get(include=["metadatas"])
    from collections import Counter
    title_counts = Counter(m.get("title", "Unknown") for m in data["metadatas"] if m)
    print("=== CHUNKS PER PAPER ===")
    print(f"Total papers: {len(title_counts)}")
    print(f"Total chunks: {sum(title_counts.values())}")
    print("\nChunks per paper:")
    for title, count in sorted(title_counts.items(), key=lambda x: x[1]):
        status = "⚠️ ABSTRACT ONLY" if count < 5 else "✓ OK"
        print(f"  {count:3d} chunks | {status} | {title[:80]}")

if __name__ == "__main__":
    inspect()
