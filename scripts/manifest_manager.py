"""
manifest_manager.py — PDF Ingestion Manifest Tracker.

Maintains a JSON file (output/ingestion_manifest.json) that records the
ingestion status of every PDF in the papers/ directory.

This prevents:
  - Re-ingesting already-processed papers on batch operations.
  - Orphaned manifest entries when PDFs are manually deleted.

Each manifest entry records:
  - The paper title and DOI
  - Ingestion status: "success" | "pending" | "failed"
  - Any error message if status is "failed"
  - The timestamp of the last successful ingestion
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from config import settings   # Flat import — scripts/ is on sys.path


# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class ManifestManagerService:
    """
    Service to track the ingestion state of PDF files.

    The manifest is a JSON file stored at output/ingestion_manifest.json.
    It maps PDF filenames → metadata dicts with status, title, doi, and timestamp.

    Example manifest entry:
        "attention_is_all_you_need.pdf": {
            "title": "Attention Is All You Need",
            "doi": "10.48550/arXiv.1706.03762",
            "status": "success",
            "error": "",
            "ingested_at": "2026-05-24T10:00:00.000000"
        }
    """

    def __init__(self):
        """
        Initialise the manifest manager and ensure the manifest file exists.

        Creates the output/ directory and an empty manifest file ({})
        if they do not already exist.
        """
        # Manifest file lives in output/ alongside citation CSV reports
        self.manifest_path = settings.BASE_DIR / "output" / "ingestion_manifest.json"
        # Ensure the output/ directory exists
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Create an empty manifest file if this is the first run
        if not self.manifest_path.exists():
            self._save_manifest({})

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Low-level JSON read/write
    # ──────────────────────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        """
        Load the manifest dict from disk.

        Returns:
            Manifest dict (possibly empty {}) on success.
            Returns {} on any read or JSON parse error to avoid crashing.
        """
        try:
            if self.manifest_path.exists():
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading ingestion manifest: {e}")
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        """
        Save the manifest dict to disk as formatted JSON.

        Args:
            manifest: The full manifest dict to persist.
        """
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                # indent=4 for human-readable output; ensure_ascii=False preserves
                # Unicode characters in non-English paper titles
                json.dump(manifest, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving ingestion manifest: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Read Operations
    # ──────────────────────────────────────────────────────────────────────────

    def get_all_entries(self) -> dict:
        """
        Return all entries in the manifest.

        Returns:
            Full manifest dict mapping filenames → status metadata dicts.
        """
        return self._load_manifest()

    def is_ingested(self, filename: str) -> bool:
        """
        Check whether a specific PDF file is already successfully ingested.

        Args:
            filename: The PDF filename (not full path) to check.

        Returns:
            True only if the file has an entry with status == "success".
            Returns False if the file is absent, pending, or failed.
        """
        manifest = self._load_manifest()
        entry = manifest.get(filename)
        # Only consider "success" as truly ingested — not "pending" or "failed"
        return entry is not None and entry.get("status") == "success"

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Write Operations
    # ──────────────────────────────────────────────────────────────────────────

    def mark_as_ingested(
        self,
        filename: str,
        title: str,
        doi: str | None = None,
        status: str = "success",
        error: str = "",
        authors: str | None = None,
        year: int | str | None = None
    ) -> None:
        """
        Record the ingestion result for a PDF file in the manifest.

        Called after every ingestion attempt — successful or not — to keep
        the manifest in sync with what is actually stored in ChromaDB.

        Args:
            filename: PDF filename (e.g. "attention_is_all_you_need.pdf").
            title: Human-readable paper title.
            doi: DOI of the paper, or None if unknown.
            status: "success", "failed", or "pending".
            error: Error message string if status is "failed", else empty string.
            authors: Authors string.
            year: Year of publication.
        """
        manifest = self._load_manifest()

        # Overwrite or create the entry for this filename
        manifest[filename] = {
            "title": title,
            "doi": doi or "N/A",
            "status": status,
            "error": error,
            "authors": authors or "Unknown Authors",
            "year": str(year) if year else "N/A",
            # ISO-format timestamp for human-readable audit trail
            "ingested_at": datetime.now().isoformat()
        }

        self._save_manifest(manifest)
        logger.info(f"Manifest updated: '{filename}' → status={status}")

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Sync Manifest with Filesystem + Vector Store
    # ──────────────────────────────────────────────────────────────────────────

    def sync_with_vector_store(self, vector_store_service) -> dict:
        """
        Reconcile the manifest against the actual papers/ directory and ChromaDB.

        This sync performs three cleanup operations:
          1. Add any new PDFs in papers/ that are not yet in the manifest
             (marking them as "pending" or "success" if already in ChromaDB).
          2. Backfill missing metadata (authors, year, doi) for existing entries.
          3. Remove manifest entries for PDFs that no longer exist on disk
             (so deleted files don't show up in the UI as pending forever).

        Args:
            vector_store_service: An initialised VectorStoreService instance
                                   used to check which papers are already in ChromaDB.

        Returns:
            The updated manifest dict after reconciliation.
        """
        pdf_dir = settings.PDF_DOWNLOAD_DIR  # papers/ directory
        # Glob all PDF files currently present on disk
        pdf_files = list(pdf_dir.glob("*.pdf"))

        manifest = self._load_manifest()

        # Get the set of paper titles currently indexed in ChromaDB
        stats = vector_store_service.get_collection_stats()
        ingested_titles = set(stats.get("papers_list", []))
        papers_metadata = stats.get("papers_metadata", {})

        logger.info(
            f"Syncing manifest: {len(pdf_files)} PDFs on disk, "
            f"{len(ingested_titles)} papers in ChromaDB."
        )

        updated = False  # Track whether any changes were made

        # ── Pass 1: Add new files to the manifest ─────────────────────────────
        for pdf_path in pdf_files:
            filename = pdf_path.name

            if filename not in manifest:
                # Derive a human-readable title from the filename as a best guess
                title = filename.replace("_", " ").replace(".pdf", "").title()

                # Check if this file's title is already in the ChromaDB collection
                # by looking for a close substring match (handles minor title variations)
                matched_title = None
                for t in ingested_titles:
                    if (t.lower().strip() in title.lower().strip() or
                            title.lower().strip() in t.lower().strip()):
                        matched_title = t
                        break

                if matched_title:
                    # File already ingested — mark it as success with the canonical title
                    logger.info(
                        f"Auto-detected existing ChromaDB paper '{matched_title}' "
                        f"for local file '{filename}'"
                    )
                    paper_meta = papers_metadata.get(matched_title, {})
                    manifest[filename] = {
                        "title": matched_title,
                        "doi": paper_meta.get("doi", "N/A"),
                        "status": "success",
                        "error": "",
                        "authors": paper_meta.get("authors", "Unknown Authors"),
                        "year": paper_meta.get("year", "N/A"),
                        # Use the file's last-modified time as a proxy for ingestion time
                        "ingested_at": datetime.fromtimestamp(
                            pdf_path.stat().st_mtime
                        ).isoformat()
                    }
                else:
                    # File not yet in ChromaDB — mark as pending for the next batch ingest
                    manifest[filename] = {
                        "title": title,
                        "doi": "N/A",
                        "status": "pending",
                        "error": "",
                        "authors": "Unknown Authors",
                        "year": "N/A",
                        "ingested_at": None
                    }

                updated = True

        # ── Pass 1b: Backfill metadata and check for phantom-success entries ──
        for filename, meta in manifest.items():
            title = meta.get("title", "")
            is_verified = False
            matched_title = None
            for t in ingested_titles:
                if (t.lower().strip() in title.lower().strip() or
                        title.lower().strip() in t.lower().strip()):
                    is_verified = True
                    matched_title = t
                    break

            if meta.get("status") == "success":
                if not is_verified:
                    logger.warning(
                        f"Manifest integrity: '{filename}' is marked 'success' but "
                        "its title was not found in ChromaDB. Resetting to 'pending' "
                        "for automatic re-ingestion."
                    )
                    manifest[filename]["status"] = "pending"
                    manifest[filename]["error"] = "Auto-reset: title not found in ChromaDB. Will be re-ingested."
                    manifest[filename]["ingested_at"] = None
                    updated = True
                elif matched_title:
                    # Backfill missing metadata fields
                    paper_meta = papers_metadata.get(matched_title, {})
                    if "authors" not in meta or meta["authors"] == "Unknown Authors":
                        meta["authors"] = paper_meta.get("authors", "Unknown Authors")
                        updated = True
                    if "year" not in meta or meta["year"] in [None, "N/A", "None"]:
                        meta["year"] = paper_meta.get("year", "N/A")
                        updated = True
                    if meta.get("doi") in [None, "N/A", "None"]:
                        meta["doi"] = paper_meta.get("doi", "N/A")
                        updated = True

        # ── Pass 2: Remove stale entries for deleted files ─────────────────────
        existing_filenames = {f.name for f in pdf_files}  # Fast set lookup
        for filename in list(manifest.keys()):
            if filename not in existing_filenames:
                meta = manifest[filename]
                # If this entry is already successfully ingested in ChromaDB (e.g., abstract-only fallback), keep it!
                title = meta.get("title", "")
                is_in_chromadb = False
                if meta.get("status") == "success":
                    for t in ingested_titles:
                        if (t.lower().strip() in title.lower().strip() or
                                title.lower().strip() in t.lower().strip()):
                            is_in_chromadb = True
                            break
                
                if not is_in_chromadb:
                    # The PDF was deleted and is not indexed in ChromaDB — remove its manifest entry
                    del manifest[filename]
                    updated = True
                    logger.info(f"Manifest: removed stale entry for deleted file '{filename}'")

        # Only write to disk if something actually changed (avoid unnecessary I/O)
        if updated:
            self._save_manifest(manifest)

        return manifest
