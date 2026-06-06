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
import threading
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
        # Reentrant lock to coordinate manifest file access
        self.manifest_lock = threading.RLock()
        # Set to track filenames currently being resolved by background S2 thread
        self.resolving_filenames = set()
        self.resolving_lock = threading.Lock()

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
        import re
        with self.manifest_lock:
            try:
                if self.manifest_path.exists():
                    with open(self.manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                        
                        # Normalize title and authors to remove newlines/consecutive spaces
                        changed = False
                        for filename, meta in list(manifest.items()):
                            if not isinstance(meta, dict):
                                continue
                            if "title" in meta:
                                orig_title = meta["title"]
                                if orig_title:
                                    clean_title = re.sub(r"\s+", " ", orig_title.strip())
                                    if clean_title != orig_title:
                                        meta["title"] = clean_title
                                        changed = True
                            if "authors" in meta:
                                orig_authors = meta["authors"]
                                if orig_authors:
                                    clean_authors = re.sub(r"\s+", " ", orig_authors.strip())
                                    if clean_authors != orig_authors:
                                        meta["authors"] = clean_authors
                                        changed = True
                            if "venue" in meta:
                                orig_venue = meta["venue"]
                                if orig_venue:
                                    clean_venue = re.sub(r"\s+", " ", orig_venue.strip())
                                    if clean_venue != orig_venue:
                                        meta["venue"] = clean_venue
                                        changed = True
                                        
                        if changed:
                            # Save back normalized version to disk
                            try:
                                with open(self.manifest_path, "w", encoding="utf-8") as wf:
                                    json.dump(manifest, wf, indent=4, ensure_ascii=False)
                            except Exception as save_err:
                                logger.error(f"Failed to save normalized manifest on load: {save_err}")
                                
                        return manifest
            except Exception as e:
                logger.error(f"Error loading ingestion manifest: {e}")
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        """
        Save the manifest dict to disk as formatted JSON.

        Args:
            manifest: The full manifest dict to persist.
        """
        with self.manifest_lock:
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
        year: int | str | None = None,
        venue: str | None = None,
        abstract: str | None = None,
        paper_id: str | None = None,
        has_full_text: bool = True,
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
            abstract: Paper abstract text.
        """
        manifest = self._load_manifest()

        existing_entry = manifest.get(filename, {})
        existing_abstract = existing_entry.get("abstract")
        final_abstract = abstract or existing_abstract or ""

        import re
        clean_title = re.sub(r"\s+", " ", title.strip())
        clean_authors = re.sub(r"\s+", " ", (authors or "Unknown Authors").strip())
        clean_venue = re.sub(r"\s+", " ", (venue or "N/A").strip())

        # Overwrite or create the entry for this filename
        manifest[filename] = {
            "title": clean_title,
            "doi": doi or "N/A",
            "status": status,
            "error": error,
            "authors": clean_authors,
            "year": str(year) if year else "N/A",
            "venue": clean_venue,
            "abstract": final_abstract,
            # Backwards compatibility and Scholar metadata tracking
            "paper_id": paper_id or existing_entry.get("paper_id") or "",
            "ingested_at": datetime.now().isoformat(),
            "has_full_text": has_full_text if status == "success" else False,
        }

        self._save_manifest(manifest)
        logger.info(f"Manifest updated: '{filename}' → status={status}")

    def resolve_metadata(self, pdf_path: Path, title_guess: str, existing_doi: str | None = None) -> dict:
        """
        Step 5: Cascading resolver for academic metadata.
        Follows the specified cascade order:
          Step 1: Crossref — most authoritative for DOI-based metadata
          Step 2: Semantic Scholar — for abstract and citation data
          Step 3: OpenAlex fallback — for books, global proceedings
          Step 4: Unpaywall — full text PDF recovery
        """
        import re
        from paper_discovery import PaperDiscoveryService
        from pdf_processor import PDFProcessorService

        discovery_service = PaperDiscoveryService()
        pdf_service = PDFProcessorService()

        # Clean up title_guess (strip subfolder prefix and leading dates/digits)
        title_guess = Path(title_guess).name
        title_guess = re.sub(r'^\d{4}[-_]\d{2}[-_]\d{2}[-_]?', '', title_guess)
        title_guess = re.sub(r'^\d{4}\s+\d{2}\s+\d{2}\s+', '', title_guess)
        title_guess = re.sub(r'^\d{4}[-_]?', '', title_guess)
        title_guess = re.sub(r'^\d{4}\s+', '', title_guess)
        title_guess = title_guess.replace("_", " ").replace("-", " ").strip()

        def _format_authors_helper(authors: list) -> str:
            if not authors:
                return "Unknown Authors"
            names = [a.get("name", "") for a in authors if a.get("name")]
            return ", ".join(names)

        resolved = {
            "title": title_guess.title(),
            "authors": "Unknown Authors",
            "year": "N/A",
            "doi": existing_doi or "N/A",
            "venue": "N/A",
            "abstract": "",
            "paper_id": "",
            "pdf_url": ""
        }

        # ── Pre-step: Always extract first-page text & DOI from PDF before lookups ──
        first_page_text = ""
        extracted_doi = None
        if pdf_path.exists():
            try:
                pages, _ = pdf_service.extract_text_by_page(pdf_path)
                if pages:
                    first_page_text = pages[0].get("text", "")
                    doi_match = re.search(r"\b(10\.\d{4,9}/[^\s]+)\b", first_page_text, re.IGNORECASE)
                    if doi_match:
                        extracted_doi = doi_match.group(1).rstrip(".,;()[]{}")
                        logger.info(f"Extracted DOI '{extracted_doi}' from PDF text for '{pdf_path.name}'")
            except Exception as pdf_err:
                logger.warning(f"Failed to extract text from PDF '{pdf_path.name}' for DOI lookup: {pdf_err}")

        # Determine the target DOI to query (existing or extracted)
        target_doi = None
        if existing_doi and existing_doi != "N/A":
            target_doi = existing_doi
        elif extracted_doi:
            target_doi = extracted_doi

        # ── Step 1: Crossref — most authoritative for DOI-based metadata ──
        api_resolved = None
        if target_doi:
            crossref_res = discovery_service.fetch_crossref_metadata(target_doi)
            if crossref_res:
                logger.info(f"Step 1: Resolved metadata via Crossref for DOI '{target_doi}'")
                api_resolved = crossref_res

        # ── Step 2: Semantic Scholar — for abstract and citation data ──
        if not api_resolved:
            # If Crossref failed, try Semantic Scholar directly
            s2_query = target_doi or title_guess
            try:
                logger.info(f"Step 2: Querying Semantic Scholar for: '{s2_query}'")
                s2_res = discovery_service.get_paper_details(s2_query)
                if s2_res:
                    logger.info(f"Step 2: Resolved metadata via Semantic Scholar")
                    # Convert S2 format to standard format
                    api_resolved = {
                        "title": s2_res.get("title"),
                        "authors": s2_res.get("authors", []),
                        "year": s2_res.get("year"),
                        "venue": s2_res.get("venue"),
                        "doi": (s2_res.get("externalIds") or {}).get("DOI") or target_doi or "N/A",
                        "abstract": s2_res.get("abstract", "")
                    }
            except Exception as s2_err:
                logger.warning(f"Step 2: Semantic Scholar lookup failed: {s2_err}")

        # ── Step 3: OpenAlex fallback — for books, global proceedings ──
        if not api_resolved:
            if target_doi:
                openalex_res = discovery_service.fetch_openalex_metadata(target_doi)
                if openalex_res:
                    logger.info(f"Step 3: Resolved metadata via OpenAlex fallback for DOI '{target_doi}'")
                    api_resolved = openalex_res
            else:
                # Try OpenAlex title search
                search_title = title_guess.replace("+", " ").replace("_", " ").replace("-", " ").strip(".")
                openalex_res = discovery_service.fetch_openalex_metadata(search_title)
                if openalex_res:
                    # Basic similarity check
                    c_title = openalex_res.get("title", "").lower().strip()
                    t_clean = search_title.lower().strip()
                    c_title_norm = re.sub(r'[^a-z0-9\s]', '', c_title)
                    t_clean_norm = re.sub(r'[^a-z0-9\s]', '', t_clean)
                    
                    t_words = set(t_clean_norm.split())
                    c_words = set(c_title_norm.split())
                    overlap = len(t_words & c_words) / max(len(t_words), 1) if t_words else 0
                    if overlap >= 0.7:
                        logger.info(f"Step 3: Resolved metadata via OpenAlex title search (overlap={overlap:.0%})")
                        api_resolved = openalex_res

        # ── Populate Resolved dictionary from API results ──
        if api_resolved:
            resolved["title"] = api_resolved.get("title") or resolved["title"]
            resolved["authors"] = _format_authors_helper(api_resolved.get("authors"))
            resolved["year"] = str(api_resolved.get("year") or "N/A")
            resolved["doi"] = api_resolved.get("doi") or target_doi or "N/A"
            resolved["venue"] = api_resolved.get("venue") or "N/A"
            resolved["abstract"] = api_resolved.get("abstract") or ""

        # ── Step 4: Unpaywall — full text PDF recovery ──
        if target_doi and target_doi != "N/A":
            pdf_url = discovery_service.fetch_open_access_pdf_url(target_doi)
            if pdf_url:
                logger.info(f"Step 4: Found open-access PDF via Unpaywall for DOI '{target_doi}'")
                resolved["pdf_url"] = pdf_url

        # ── Fallbacks to fitz & filename metadata (for offline or failed lookups) ──
        # Fill year and authors if still Unknown/N/A
        if pdf_path.exists() and (resolved["title"] == title_guess.title() or resolved["authors"] == "Unknown Authors" or resolved["year"] == "N/A"):
            try:
                import fitz
                with fitz.open(pdf_path) as doc:
                    meta = doc.metadata or {}
                    internal_title = meta.get("title")
                    internal_author = meta.get("author")
                    
                    internal_year = None
                    for date_field in ["creationDate", "modDate"]:
                        date_val = meta.get(date_field)
                        if date_val:
                            match = re.search(r"\b(19|20)\d{2}\b", date_val)
                            if match:
                                internal_year = match.group(0)
                                break
                    
                    if resolved["authors"] == "Unknown Authors" and internal_author and internal_author.strip() and internal_author.lower() not in ["unknown", "none", "null"]:
                        resolved["authors"] = internal_author.strip()
                    if resolved["year"] == "N/A" and internal_year:
                        resolved["year"] = str(internal_year)
                    if resolved["title"] == title_guess.title() and internal_title and internal_title.strip() and len(internal_title.strip()) > 5 and internal_title.lower() not in ["unknown", "none", "null", "untitled"]:
                        resolved["title"] = internal_title.strip()
            except Exception as fitz_err:
                logger.warning(f"Failed to extract internal PDF metadata: {fitz_err}")

        # Hard extract year/authors from filename/text as last resort
        if resolved["year"] == "N/A" or not resolved["year"]:
            year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', pdf_path.name)
            if not year_match:
                year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', title_guess)
            if year_match:
                resolved["year"] = year_match.group(1)
            elif first_page_text:
                year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', first_page_text)
                if year_match:
                    resolved["year"] = year_match.group(1)

        if resolved["authors"] == "Unknown Authors" or not resolved["authors"]:
            fn_clean = pdf_path.stem
            year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', fn_clean)
            if year_match:
                year_idx = year_match.start()
                prefix = fn_clean[:year_idx].strip("-_ ()[]{},")
                if prefix and len(prefix.split()) <= 4:
                    resolved["authors"] = re.sub(r'[-_]', ' ', prefix).strip().title()
            
            if resolved["authors"] == "Unknown Authors" and first_page_text:
                lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
                for line in lines[:5]:
                    if ',' in line and len(line) < 200 and not any(x in line.lower() for x in ['http', 'doi:', 'vol.', 'no.', 'issn', '@', 'university', 'department']):
                        line = re.sub(r'\d+', '', line)
                        if len(line.split(',')) >= 2 and len(line.split(',')) <= 10:
                            resolved["authors"] = line[:200].strip()
                            break

        # Fallback abstract from first page text if still empty
        if not resolved["abstract"] and first_page_text:
            paragraphs = [p.strip() for p in first_page_text.split("\n\n") if p.strip()]
            for p in paragraphs[:5]:
                if len(p) > 100 and not any(x in p.lower() for x in ['http', 'downloaded', 'vol.', 'issn', '@', 'page', 'journal']):
                    resolved["abstract"] = p[:600] + ("..." if len(p) > 600 else "")
                    break
            if not resolved["abstract"] and paragraphs:
                resolved["abstract"] = paragraphs[0][:400] + "..."

        return resolved

    def refresh_metadata_sync(self, vector_store_service, max_entries: int = 15) -> int:
        """
        Synchronously resolve missing author/year for manifest rows (bounded batch).
        Called when the UI loads the paper list so sidebar labels show Author, Year
        instead of long titles while background resolution catches up.
        """
        from paper_labels import UNKNOWN_AUTHORS

        manifest = self._load_manifest()
        pdf_dir = settings.PDF_DOWNLOAD_DIR
        updated = 0

        for filename, meta in list(manifest.items()):
            if updated >= max_entries:
                break
            if meta.get("authors") not in (None, "", UNKNOWN_AUTHORS):
                continue
            if meta.get("status") not in ("success", "pending"):
                continue

            pdf_path = pdf_dir / filename
            if not pdf_path.exists():
                continue

            try:
                resolved = self.resolve_metadata(
                    pdf_path,
                    meta.get("title", pdf_path.stem),
                    meta.get("doi") if meta.get("doi") not in ("N/A", None) else None,
                )
                new_authors = resolved.get("authors") or UNKNOWN_AUTHORS
                new_year = resolved.get("year") or "N/A"
                new_title = resolved.get("title") or meta.get("title", filename)

                if new_authors == UNKNOWN_AUTHORS and new_year in ("N/A", "", None):
                    continue

                self.mark_as_ingested(
                    filename,
                    new_title,
                    doi=resolved.get("doi"),
                    status=meta.get("status", "success"),
                    authors=new_authors,
                    year=new_year,
                    abstract=resolved.get("abstract") or meta.get("abstract"),
                    paper_id=meta.get("paper_id"),
                )

                if meta.get("status") == "success" and new_authors != UNKNOWN_AUTHORS:
                    vector_store_service.update_paper_metadata(
                        title=meta.get("title", new_title),
                        authors=new_authors,
                        year=str(new_year),
                        doi=resolved.get("doi"),
                        venue=resolved.get("venue"),
                        new_title=new_title if new_title != meta.get("title") else None,
                    )

                updated += 1
            except Exception as e:
                logger.warning("Sync metadata refresh failed for '%s': %s", filename, e)

        return updated

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
        with self.manifest_lock:
            pdf_dir = settings.PDF_DOWNLOAD_DIR  # papers/ directory
            # Recursively collect all PDF files in papers/ and subfolders (case-insensitive)
            pdf_files = [p for p in pdf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]

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
                # Use relative path as key to support subfolders without collisions
                # e.g. "subgroup/paper.pdf" instead of bare "paper.pdf"
                filename = str(pdf_path.relative_to(pdf_dir))

                if filename not in manifest:
                    # Derive a clean human-readable title from the filename as a best guess
                    import re
                    stem_clean = pdf_path.stem
                    stem_clean = re.sub(r'^\d{4}[-_]\d{2}[-_]\d{2}[-_]?', '', stem_clean)
                    stem_clean = re.sub(r'^\d{4}[-_]?', '', stem_clean)
                    title = stem_clean.replace("_", " ").replace("-", " ").strip().title()

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
                            "venue": paper_meta.get("venue", "N/A"),
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
            # Perform quick checks first: check which manifest entries need resolving
            entries_to_resolve = []
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
                        # Sync local metadata from ChromaDB cache if it exists and is valid
                        paper_meta = papers_metadata.get(matched_title, {})
                        db_authors = paper_meta.get("authors", "Unknown Authors")
                        db_year = paper_meta.get("year", "N/A")
                        db_doi = paper_meta.get("doi", "N/A")
                        db_venue = paper_meta.get("venue", "N/A")

                        # If the DB has valid metadata but the manifest doesn't, sync it synchronously (very fast)
                        if db_authors != "Unknown Authors" or db_year != "N/A":
                            if "authors" not in meta or meta["authors"] != db_authors:
                                meta["authors"] = db_authors
                                updated = True
                            if "year" not in meta or meta["year"] != db_year:
                                meta["year"] = db_year
                                updated = True
                            if meta.get("doi") in [None, "N/A", "None"] or meta["doi"] != db_doi:
                                meta["doi"] = db_doi
                                updated = True
                            if meta.get("venue") in [None, "N/A", "None"] or meta["venue"] != db_venue:
                                meta["venue"] = db_venue
                                updated = True
                        
                        # If the DB and manifest are missing metadata or abstract, queue it for background resolution
                        if (meta.get("authors") == "Unknown Authors" 
                            or meta.get("year") in [None, "N/A", "None"]
                            or "abstract" not in meta 
                            or meta.get("abstract") in [None, ""]):
                            with self.resolving_lock:
                                if filename not in self.resolving_filenames:
                                    self.resolving_filenames.add(filename)
                                    entries_to_resolve.append((filename, matched_title, title, meta.get("doi")))

            # Spawn a background thread to resolve missing metadata asynchronously from Semantic Scholar
            if entries_to_resolve:
                import threading

                def _bg_resolve_metadata():
                    try:
                        logger.info(f"Background thread starting to resolve metadata for {len(entries_to_resolve)} papers...")
                        for filename, matched_title, title, existing_doi in entries_to_resolve:
                            pdf_path = settings.PDF_DOWNLOAD_DIR / filename
                            
                            try:
                                resolved = self.resolve_metadata(pdf_path, title, existing_doi)
                                s2_title = resolved["title"]
                                s2_authors = resolved["authors"]
                                s2_year = resolved["year"]
                                s2_doi = resolved["doi"]
                                s2_venue = resolved.get("venue", "N/A")
                                s2_abstract = resolved.get("abstract", "")

                                logger.info(f"Resolved S2 Metadata for '{title}': title='{s2_title}', authors='{s2_authors}', year={s2_year}")

                                # Update chunks in ChromaDB (updating both metadata fields AND the title)
                                vector_store_service.update_paper_metadata(
                                    title=matched_title,
                                    authors=s2_authors,
                                    year=str(s2_year),
                                    doi=s2_doi if s2_doi != "N/A" else existing_doi,
                                    venue=s2_venue if s2_venue != "N/A" else None,
                                    new_title=s2_title
                                )

                                # Load latest manifest, write the metadata, and save
                                with self.manifest_lock:
                                    current_manifest = self._load_manifest()
                                    if filename in current_manifest:
                                        current_manifest[filename]["title"] = s2_title
                                        current_manifest[filename]["authors"] = s2_authors
                                        current_manifest[filename]["year"] = str(s2_year)
                                        current_manifest[filename]["venue"] = s2_venue
                                        current_manifest[filename]["abstract"] = s2_abstract
                                        if s2_doi and s2_doi != "N/A":
                                            current_manifest[filename]["doi"] = s2_doi
                                        self._save_manifest(current_manifest)
                            except Exception as save_err:
                                logger.error(f"Failed to save resolved metadata for '{filename}': {save_err}")
                            finally:
                                with self.resolving_lock:
                                    self.resolving_filenames.discard(filename)
                    except Exception as e:
                        logger.error(f"Error in background metadata resolution: {e}")
                        # Cleanup resolving list in case of generic thread-level crash
                        with self.resolving_lock:
                            for filename, _, _, _ in entries_to_resolve:
                                self.resolving_filenames.discard(filename)

                threading.Thread(target=_bg_resolve_metadata, daemon=True).start()

            # ── Pass 2: Remove stale entries for deleted files ─────────────────────
            # Build set of current relative paths for O(1) lookup
            existing_filenames = {str(f.relative_to(pdf_dir)) for f in pdf_files}
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

            # ── Pass 3: Clean up ghost papers from ChromaDB ────────────────────────
            # If a paper is in ChromaDB but not marked as success in the manifest,
            # it means the paper was deleted or is orphaned. Delete it from ChromaDB.
            manifest_success_titles = {
                meta.get("title").lower().strip()
                for meta in manifest.values()
                if meta.get("status") == "success" and meta.get("title")
            }
            
            for t in list(ingested_titles):
                t_lower = t.lower().strip()
                # Check if this ChromaDB title exists as a success entry in our manifest
                if t_lower not in manifest_success_titles:
                    # Double check if there's a close substring match to prevent false deletions
                    has_match = False
                    for mst in manifest_success_titles:
                        if mst in t_lower or t_lower in mst:
                            has_match = True
                            break
                    if not has_match:
                        logger.info(f"Manifest Sync: Deleting ghost paper '{t}' from ChromaDB.")
                        vector_store_service.delete_paper(title=t)
                        updated = True

            # Only write to disk if something actually changed (avoid unnecessary I/O)
            if updated:
                self._save_manifest(manifest)

            return manifest
