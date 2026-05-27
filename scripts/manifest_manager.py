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
        with self.manifest_lock:
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
        abstract: str | None = None
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

        existing_abstract = manifest.get(filename, {}).get("abstract")
        final_abstract = abstract or existing_abstract or ""

        # Overwrite or create the entry for this filename
        manifest[filename] = {
            "title": title,
            "doi": doi or "N/A",
            "status": status,
            "error": error,
            "authors": authors or "Unknown Authors",
            "year": str(year) if year else "N/A",
            "abstract": final_abstract,
            # ISO-format timestamp for human-readable audit trail
            "ingested_at": datetime.now().isoformat()
        }

        self._save_manifest(manifest)
        logger.info(f"Manifest updated: '{filename}' → status={status}")

    def resolve_metadata(self, pdf_path: Path, title_guess: str, existing_doi: str | None = None) -> dict:
        """
        Attempt to resolve academic metadata (title, authors, year, doi, abstract) for a PDF file.
        Tries:
          1. Direct lookup if existing_doi is provided.
          2. Extract DOI from PDF text, then direct lookup.
          3. Search Semantic Scholar by clean title guess.
          4. Extract internal PDF metadata using PyMuPDF (fitz) as fallback.
        """
        import re
        from paper_discovery import PaperDiscoveryService
        from pdf_processor import PDFProcessorService

        discovery_service = PaperDiscoveryService()
        pdf_service = PDFProcessorService()

        def _format_authors_helper(authors: list) -> str:
            if not authors:
                return "Unknown Authors"
            names = [a.get("name", "") for a in authors if a.get("name")]
            if len(names) > 3:
                return ", ".join(names[:3]) + " et al."
            return ", ".join(names)

        resolved = {
            "title": title_guess,
            "authors": "Unknown Authors",
            "year": "N/A",
            "doi": existing_doi or "N/A",
            "abstract": ""
        }

        # ── Inline helper: fill abstract from PDF text if S2 didn't return one ──
        def _fill_abstract(r: dict, text: str) -> None:
            """Populate r['abstract'] from the first meaningful paragraph in text."""
            if r.get("abstract") or not text:
                return
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in paragraphs[:5]:
                if len(p) > 100 and not any(
                    x in p.lower() for x in ['http', 'downloaded', 'vol.', 'issn', '@', 'page', 'journal']
                ):
                    r["abstract"] = p[:600] + ("..." if len(p) > 600 else "")
                    return
            if paragraphs:
                r["abstract"] = paragraphs[0][:400] + "..."

        # ── Pre-step: Always extract first-page text & DOI from PDF before S2 lookups ──
        # Having first_page_text available at every lookup path means we can always
        # use it as an abstract fallback, even when S2 returns metadata without one.
        first_page_text = ""
        extracted_doi = None
        if pdf_path.exists():
            try:
                pages = pdf_service.extract_text_by_page(pdf_path)
                if pages:
                    first_page_text = pages[0].get("text", "")
                    doi_match = re.search(r"\b(10\.\d{4,9}/[^\s]+)\b", first_page_text, re.IGNORECASE)
                    if doi_match:
                        extracted_doi = doi_match.group(1).rstrip(".,;()[]{}")
                        logger.info(f"Extracted DOI '{extracted_doi}' from PDF text for '{pdf_path.name}'")
            except Exception as pdf_err:
                logger.warning(f"Failed to extract text from PDF '{pdf_path.name}' for DOI lookup: {pdf_err}")

        # Step 1: Direct lookup if existing_doi is provided
        if existing_doi and existing_doi != "N/A":
            try:
                cand = discovery_service.get_paper_details(existing_doi)
                if cand:
                    resolved["title"] = cand.get("title") or title_guess
                    resolved["authors"] = _format_authors_helper(cand.get("authors", []))
                    resolved["year"] = str(cand.get("year") or "N/A")
                    resolved["doi"] = (cand.get("externalIds") or {}).get("DOI") or existing_doi
                    resolved["abstract"] = cand.get("abstract") or ""
                    _fill_abstract(resolved, first_page_text)  # use PDF text if S2 gave no abstract
                    return resolved
            except Exception as e:
                logger.warning(f"Metadata lookup failed for DOI '{existing_doi}': {e}")

        # Step 2: DOI already extracted in pre-step above — skip to Step 3

        # Step 3: If DOI extracted from PDF text, do a direct S2 lookup
        if extracted_doi:
            try:
                cand = discovery_service.get_paper_details(extracted_doi)
                if cand:
                    resolved["title"] = cand.get("title") or title_guess
                    resolved["authors"] = _format_authors_helper(cand.get("authors", []))
                    resolved["year"] = str(cand.get("year") or "N/A")
                    resolved["doi"] = (cand.get("externalIds") or {}).get("DOI") or extracted_doi
                    resolved["abstract"] = cand.get("abstract") or ""
                    _fill_abstract(resolved, first_page_text)  # use PDF text if S2 gave no abstract
                    return resolved
            except Exception as doi_err:
                logger.warning(f"Direct S2 DOI lookup failed for '{extracted_doi}': {doi_err}")

        # Step 4: Search Semantic Scholar by clean title guess
        search_title = title_guess.replace("+", " ").replace("_", " ").replace("-", " ").strip(".")
        is_short_code = (
            len(search_title) < 15 or 
            search_title.isdigit() or 
            re.match(r'^[a-zA-Z0-9_\-\s\.]+$', search_title) and any(x in search_title.lower() for x in ['vol', 'no', 'issue', 'page', 'gjcs'])
        )
        if is_short_code and first_page_text:
            lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
            clean_lines = []
            for l in lines[:8]:
                if any(x in l.lower() for x in ['http', 'doi:', 'vol.', 'no.', 'issn', '@', 'page', 'journal']):
                    continue
                clean_lines.append(l)
                if len(clean_lines) >= 2:
                    break
            if clean_lines:
                search_title = " ".join(clean_lines)[:150].strip()

        logger.info(f"Querying Semantic Scholar to resolve metadata for: '{search_title}'")
        try:
            results = discovery_service.search_papers(search_title, limit=5)
            for cand in results:
                c_title = (cand.get("title") or "").lower().strip()
                t_clean = search_title.lower().strip()
                c_title_norm = re.sub(r'[^a-z0-9\s]', '', c_title)
                t_clean_norm = re.sub(r'[^a-z0-9\s]', '', t_clean)
                
                matched = False
                if c_title_norm == t_clean_norm or t_clean_norm in c_title_norm or c_title_norm in t_clean_norm:
                    matched = True
                else:
                    t_words = set(t_clean_norm.split())
                    c_words = set(c_title_norm.split())
                    if t_words and c_words:
                        overlap = len(t_words & c_words) / max(len(t_words), 1)
                        if overlap >= 0.7:
                            matched = True
                
                if matched:
                    resolved["title"] = cand.get("title") or title_guess
                    resolved["authors"] = _format_authors_helper(cand.get("authors", []))
                    resolved["year"] = str(cand.get("year") or "N/A")
                    resolved["doi"] = (cand.get("externalIds") or {}).get("DOI") or resolved["doi"]
                    resolved["abstract"] = cand.get("abstract") or ""
                    _fill_abstract(resolved, first_page_text)  # use PDF text if S2 gave no abstract
                    return resolved
        except Exception as search_err:
            logger.warning(f"S2 search failed for '{search_title}': {search_err}")

        # Step 5: Fallback to internal PDF metadata via fitz
        if pdf_path.exists():
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
                    
                    if internal_author and internal_author.strip() and internal_author.lower() not in ["unknown", "none", "null"]:
                        resolved["authors"] = internal_author.strip()
                    if internal_year:
                        resolved["year"] = str(internal_year)
                    if internal_title and internal_title.strip() and len(internal_title.strip()) > 5 and internal_title.lower() not in ["unknown", "none", "null", "untitled"]:
                        resolved["title"] = internal_title.strip()
            except Exception as e:
                logger.warning(f"Failed to extract internal metadata from '{pdf_path.name}': {e}")

        # Step 6: Extract year and author from filename/title_guess if still Unknown/N/A
        if resolved["year"] == "N/A" or not resolved["year"]:
            year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', pdf_path.name)
            if not year_match:
                year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', title_guess)
            if year_match:
                resolved["year"] = year_match.group(1)

        if resolved["authors"] == "Unknown Authors" or not resolved["authors"]:
            fn_clean = pdf_path.stem
            year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', fn_clean)
            if year_match:
                year_idx = year_match.start()
                prefix = fn_clean[:year_idx].strip("-_ ()[]{},")
                if prefix and len(prefix.split()) <= 4:
                    cleaned_author = re.sub(r'[-_]', ' ', prefix).strip()
                    resolved["authors"] = cleaned_author.title()

        # Step 7: Fallback abstract from first page text if empty
        if not resolved["abstract"] and first_page_text:
            paragraphs = [p.strip() for p in first_page_text.split("\n\n") if p.strip()]
            if paragraphs:
                for p in paragraphs[:5]:
                    if len(p) > 100 and not any(x in p.lower() for x in ['http', 'downloaded', 'vol.', 'issn', '@', 'page', 'journal']):
                        resolved["abstract"] = p[:600] + ("..." if len(p) > 600 else "")
                        break
                if not resolved["abstract"]:
                    resolved["abstract"] = paragraphs[0][:400] + "..."

        return resolved

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
                                s2_abstract = resolved.get("abstract", "")

                                logger.info(f"Resolved S2 Metadata for '{title}': title='{s2_title}', authors='{s2_authors}', year={s2_year}")

                                # Update chunks in ChromaDB (updating both metadata fields AND the title)
                                vector_store_service.update_paper_metadata(
                                    title=matched_title,
                                    authors=s2_authors,
                                    year=str(s2_year),
                                    doi=s2_doi if s2_doi != "N/A" else existing_doi,
                                    new_title=s2_title
                                )

                                # Load latest manifest, write the metadata, and save
                                with self.manifest_lock:
                                    current_manifest = self._load_manifest()
                                    if filename in current_manifest:
                                        current_manifest[filename]["title"] = s2_title
                                        current_manifest[filename]["authors"] = s2_authors
                                        current_manifest[filename]["year"] = str(s2_year)
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

            # Only write to disk if something actually changed (avoid unnecessary I/O)
            if updated:
                self._save_manifest(manifest)

            return manifest
