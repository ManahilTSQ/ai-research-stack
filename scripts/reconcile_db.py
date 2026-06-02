"""
reconcile_db.py — Comprehensive Sync, Clean, and Rebuild Utility for AI Research Stack.

Guarantees 100% alignment between:
  1. The physical PDF files in the papers/ directory.
  2. The ChromaDB persistent vector database.
  3. The output/ingestion_manifest.json file.

Features:
  - Deletes "orphaned" chunks from ChromaDB for papers that no longer exist on disk.
  - Automatically identifies and ingests any new PDFs dropped in papers/ or subdirectories.
  - Retries failed or pending ingestions.
  - Optional --reset flag to wipe the vector DB and rebuild everything fresh from disk.
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reconcile_db")

PROJECT_ROOT = Path("c:/Users/PMLS/OneDrive/Desktop/AI Research Stack")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config import settings
from pdf_processor import PDFProcessorService
from vector_store import VectorStoreService
from manifest_manager import ManifestManagerService


def reconcile(force_reset: bool = False):
    pdf_service = PDFProcessorService()
    vector_store = VectorStoreService()
    manifest_svc = ManifestManagerService()

    pdf_dir = settings.PDF_DOWNLOAD_DIR
    logger.info(f"Target PDF Directory: {pdf_dir.resolve()}")
    logger.info(f"Vector Database Path: {settings.VECTOR_DB_DIR.resolve()}")

    # ── Option: Complete Wipe & Reset ──────────────────────────────────────────
    if force_reset:
        logger.warning("🚨 FORCE RESET ACTIVATED! Wiping ChromaDB collection and manifest...")
        try:
            # Delete the collection from ChromaDB
            vector_store.client.delete_collection("research_papers")
            logger.info("ChromaDB 'research_papers' collection deleted successfully.")
            # Recreate the collection
            vector_store = VectorStoreService()
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB collection: {e}")
        
        # Save an empty manifest
        manifest_svc._save_manifest({})
        logger.info("Ingestion manifest reset to empty.")

    # Get current state from ChromaDB
    stats = vector_store.get_collection_stats()
    db_papers = stats.get("papers_metadata", {})
    logger.info(f"Current ChromaDB State: {stats.get('total_papers')} papers, {stats.get('total_chunks')} chunks.")

    # 1. Scan filesystem for physical PDFs
    physical_files = {}
    pdf_paths = [p for p in pdf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    for path in pdf_paths:
        rel_path = str(path.relative_to(pdf_dir))
        physical_files[rel_path] = path

    logger.info(f"Found {len(physical_files)} physical PDF(s) on disk.")

    # Load latest manifest
    manifest = manifest_svc.get_all_entries()

    # ── Pass 1: Clean up Stale & Orphaned Database Entries ────────────────────
    logger.info("Starting Phase 1: Checking for orphaned database and manifest entries...")
    orphans_removed = 0

    # 1a. Purge manifest entries for files that do not exist physically AND are not in ChromaDB
    for filename in list(manifest.keys()):
        meta = manifest[filename]
        title = meta.get("title", "")
        doi = meta.get("doi")
        
        # Check if the title is in ChromaDB
        is_in_chromadb = False
        for db_title in db_papers.keys():
            if db_title.lower().strip() == title.lower().strip():
                is_in_chromadb = True
                break

        # If the file is missing from disk AND not in the vector DB, delete the manifest entry
        if filename not in physical_files and not is_in_chromadb:
            logger.info(f"Removing dead manifest entry: '{filename}' (not on disk and not in database)")
            del manifest[filename]
            orphans_removed += 1

    # 1b. Delete paper chunks from ChromaDB if the paper was deleted on disk AND isn't a success abstract fallback
    for db_title, db_meta in list(db_papers.items()):
        # Try to find matching filename in manifest
        filename_found = None
        for fn, m in manifest.items():
            if m.get("title", "").lower().strip() == db_title.lower().strip():
                filename_found = fn
                break

        # If it's not in the manifest or the manifest filename isn't physically on disk,
        # AND it's NOT an abstract-only fallback (file size in manifest would be 0 or status success with no file)
        # We check if we should delete it.
        should_delete = False
        if not filename_found:
            # No manifest entry exists for this DB paper
            should_delete = True
        else:
            is_file_on_disk = filename_found in physical_files
            is_abstract_only = manifest[filename_found].get("abstract") and not is_file_on_disk and manifest[filename_found].get("status") == "success"
            
            # If there's no file on disk, and it wasn't an intentional abstract-only success, delete it.
            if not is_file_on_disk and not is_abstract_only:
                should_delete = True

        if should_delete:
            logger.info(f"Purging orphaned vector chunks from ChromaDB for paper: '{db_title}'")
            vector_store.delete_paper(title=db_title, doi=db_meta.get("doi"))
            
            # Clean up the manifest entry too
            if filename_found and filename_found in manifest:
                del manifest[filename_found]
            orphans_removed += 1

    if orphans_removed > 0:
        manifest_svc._save_manifest(manifest)
        logger.info(f"Phase 1 complete. Purged {orphans_removed} orphaned manifest/database record(s).")
    else:
        logger.info("Phase 1 complete. No orphaned database entries found.")

    # ── Pass 2: Ingest Missing or Failed Physical PDFs ────────────────────────
    logger.info("Starting Phase 2: Ingesting new, pending, or failed physical PDFs...")
    ingested_count = 0
    failed_count = 0

    for rel_path, full_path in physical_files.items():
        meta = manifest.get(rel_path, {})
        title_guess = meta.get("title", full_path.stem.replace("_", " ").title())
        doi_guess = meta.get("doi") if meta.get("doi") != "N/A" else None
        status = meta.get("status", "pending")

        # Check if already successfully present in ChromaDB
        is_already_in_db = False
        matched_db_title = None
        for db_title in db_papers.keys():
            if (db_title.lower().strip() in title_guess.lower().strip() or 
                    title_guess.lower().strip() in db_title.lower().strip()):
                is_already_in_db = True
                matched_db_title = db_title
                break

        # If it is in the database and manifest is marked success, we skip it
        if is_already_in_db and status == "success":
            # Just keep manifest in sync
            continue

        logger.info(f"Ingesting: '{title_guess}' ({rel_path})...")
        try:
            # 1. Resolve proper academic metadata via S2
            resolved = manifest_svc.resolve_metadata(full_path, title_guess, doi_guess)
            title = resolved["title"]
            authors = resolved["authors"]
            year_str = resolved["year"]
            doi = resolved["doi"]
            abstract = resolved.get("abstract", "")

            year = int(year_str) if year_str.isdigit() else None

            # 2. Extract and Chunk PDF text
            pages, has_full_text = pdf_service.extract_text_by_page(full_path)
            if not has_full_text:
                logger.warning(f"  [!] Extracted minimal text from '{rel_path}' - likely abstract-only or scanned PDF")
            chunks = pdf_service.chunk_text(pages, chunk_size=1000, chunk_overlap=200)

            if not chunks:
                logger.warning(f"  [!] No text could be extracted from '{rel_path}'. May be a scanned PDF. Skipping.")
                manifest_svc.mark_as_ingested(
                    rel_path, title, doi=doi if doi != "N/A" else None, status="failed",
                    error="No text extracted — may be scanned/image-only PDF.",
                    authors=authors, year=year, abstract=abstract
                )
                failed_count += 1
                continue

            # 3. Embed and upsert into ChromaDB
            ok = vector_store.add_paper_chunks(
                paper_title=title, doi=doi if doi != "N/A" else None, chunks=chunks,
                authors=authors, year=year, venue=None
            )

            if ok:
                manifest_svc.mark_as_ingested(
                    rel_path, title, doi=doi if doi != "N/A" else None, status="success",
                    authors=authors, year=year, abstract=abstract, has_full_text=has_full_text
                )
                logger.info(f"  [✓] SUCCESS: '{title}' ingested with {len(chunks)} chunks.")
                ingested_count += 1
            else:
                manifest_svc.mark_as_ingested(
                    rel_path, title, doi=doi if doi != "N/A" else None, status="failed",
                    error="ChromaDB vector insertion returned False.",
                    authors=authors, year=year, abstract=abstract, has_full_text=has_full_text
                )
                logger.error(f"  [✗] FAILED: ChromaDB write error for '{title}'.")
                failed_count += 1

        except Exception as ex:
            logger.error(f"  [✗] ERROR: Ingestion failed for '{rel_path}': {ex}")
            manifest_svc.mark_as_ingested(
                rel_path, title_guess, doi_guess, status="failed",
                error=str(ex), authors=meta.get("authors", "Unknown Authors"),
                year=meta.get("year")
            )
            failed_count += 1

    # ── Final Report ──────────────────────────────────────────────────────────
    final_stats = vector_store.get_collection_stats()
    print(f"\n{'-'*60}")
    print("RECONCILIATION SUMMARY:")
    print(f"  - PDFs on disk: {len(physical_files)}")
    print(f"  - Newly Ingested: {ingested_count}")
    print(f"  - Failed Ingestions: {failed_count}")
    print(f"  - ChromaDB now has {final_stats['total_chunks']} chunks across {final_stats['total_papers']} papers.")
    print(f"{'-'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconcile and synchronize AI Research Stack database and manifest.")
    parser.add_argument("--reset", action="store_true", help="Wipe ChromaDB and manifest, and rebuild completely fresh from disk.")
    args = parser.parse_args()

    reconcile(force_reset=args.reset)
