"""
batch_reingest.py — Force re-ingest all PDF files in papers/ that are
not confirmed to be in ChromaDB, regardless of manifest status.
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path("c:/Users/PMLS/OneDrive/Desktop/AI Research Stack")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config import settings
from pdf_processor import PDFProcessorService
from vector_store import VectorStoreService
from manifest_manager import ManifestManagerService

def run():
    pdf_service   = PDFProcessorService()
    vector_store  = VectorStoreService()
    manifest_svc  = ManifestManagerService()

    stats = vector_store.get_collection_stats()
    in_db = set(t.lower().strip() for t in stats.get("papers_list", []))

    print(f"\n{'='*60}")
    print(f"ChromaDB has {stats['total_chunks']} chunks across {stats['total_papers']} papers.")
    print(f"{'='*60}\n")

    pdf_dir  = settings.PDF_DOWNLOAD_DIR
    manifest = manifest_svc.get_all_entries()

    queued = 0
    success = 0
    failed  = 0

    pdf_files = sorted([p for p in pdf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"], key=lambda p: str(p.relative_to(pdf_dir)))
    for pdf_path in pdf_files:
        filename = str(pdf_path.relative_to(pdf_dir))
        meta     = manifest.get(filename, {})
        title    = meta.get("title", filename.replace("_", " ").replace(".pdf", "").title())
        doi      = meta.get("doi")

        # Skip if truly confirmed in ChromaDB
        if any(title.lower().strip() in t or t in title.lower().strip() for t in in_db):
            print(f"[SKIP] Already in ChromaDB: '{title}'")
            continue

        print(f"\n[INGEST] '{title}' ({filename})")
        queued += 1

        try:
            pages  = pdf_service.extract_text_by_page(pdf_path)
            chunks = pdf_service.chunk_text(pages, chunk_size=1000, chunk_overlap=200)

            if not chunks:
                print(f"  [!] No text extracted from '{filename}'. PDF may be image-only (scanned). Skipping.")
                manifest_svc.mark_as_ingested(filename, title, doi, status="failed",
                                              error="No text extracted — PDF may be image-only.")
                failed += 1
                continue

            print(f"  [+] Extracted {len(chunks)} chunks. Upserting into ChromaDB...")
            ok = vector_store.add_paper_chunks(paper_title=title, doi=doi, chunks=chunks, venue=None)

            if ok:
                manifest_svc.mark_as_ingested(filename, title, doi, status="success")
                print(f"  [OK] SUCCESS - '{title}' ingested with {len(chunks)} chunks.")
                success += 1
            else:
                manifest_svc.mark_as_ingested(filename, title, doi, status="failed",
                                              error="ChromaDB upsert returned False.")
                print(f"  [FAIL] FAILED - ChromaDB upsert returned False for '{title}'.")
                failed += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            manifest_svc.mark_as_ingested(filename, title, doi, status="failed", error=str(e))
            failed += 1

    print(f"\n{'='*60}")
    print(f"Re-ingestion complete: {queued} queued | {success} succeeded | {failed} failed")

    final = vector_store.get_collection_stats()
    print(f"ChromaDB now has {final['total_chunks']} chunks across {final['total_papers']} papers.")
    print(f"Papers in DB:")
    for p in final["papers_list"]:
        print(f"  - {p}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run()
