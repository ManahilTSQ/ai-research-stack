"""
server.py — FastAPI Web Server for the AI Research Stack.

Serves the web interface (web/index.html) and exposes a REST API used by the
browser-based SPA (web/app.js) to perform all research operations locally.

API Endpoints:
  GET  /api/health              → Ollama + ChromaDB status + DB stats
  GET  /api/search              → Search Semantic Scholar
  POST /api/download            → Download + ingest a single paper (background)
  GET  /api/pdfs                → List all PDFs in the papers/ manifest
  POST /api/ingest-pending      → Batch-ingest all pending PDFs in papers/
  POST /api/query-rag           → RAG question answering
  GET  /api/prompts             → List available prompt templates
  POST /api/analyze-citations   → Start a citation analysis background job
  GET  /api/analyze-citations/{run_id} → Poll citation job status
  GET  /api/reports             → List all saved CSV reports in output/
  GET  /api/reports/download/{filename} → Download a specific CSV report

Running the server (from the AI Research Stack/ project root):
  cd scripts
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Then open: http://localhost:8000
"""

import sys
import logging
import threading
import time
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
import requests as _req  # Aliased to avoid collision with FastAPI's own 'requests' concept

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Ensure scripts/ directory is on sys.path ──────────────────────────────────
# This insert guarantees all flat imports resolve correctly regardless of
# the working directory from which uvicorn is started.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Flat imports — all service modules are siblings in scripts/
from config import settings
from paper_discovery import PaperDiscoveryService
from pdf_processor import PDFProcessorService
from vector_store import VectorStoreService
from rag_service import RAGService, check_ollama_health
from manifest_manager import ManifestManagerService
from citation_analyzer import CitationAnalyzerService

# ── Logger setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ai_research_server")

# ──────────────────────────────────────────────────────────────────────────────
# SERVICE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
# All services are instantiated once at startup and shared across all requests.
# This is safe because FastAPI runs in a single-threaded async event loop;
# background tasks use threads but the services themselves are read-mostly.

logger.info("Initialising all backend services...")
discover_service = PaperDiscoveryService()   # Semantic Scholar + Unpaywall + PDF download
pdf_service      = PDFProcessorService()     # PyMuPDF text extraction + chunking
vector_store     = VectorStoreService()      # ChromaDB interface
manifest_service = ManifestManagerService()  # Ingestion manifest tracker
logger.info("All backend services initialised.")

# ──────────────────────────────────────────────────────────────────────────────
# IN-MEMORY CITATION JOB STORE
# ──────────────────────────────────────────────────────────────────────────────
# Citation analysis runs as a background thread (not a FastAPI background task)
# because it can take several minutes. The SPA polls /api/analyze-citations/{run_id}
# to track progress. Job state is stored in this dict keyed by UUID run_id.
#
# Schema of each job entry:
#   status   : "running" | "completed" | "failed"
#   progress : Human-readable progress string for the UI progress bar
#   result   : List of result row dicts (populated on completion)
#   csv_path : Filename of the saved CSV report (populated on completion)
#   error    : Error message string (populated on failure)
citation_jobs: dict[str, dict] = {}

# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Stack API",
    description="Self-hosted academic research assistant — all processing runs locally.",
    version="2.0.0"
)

# ── Static Files (Web UI) ─────────────────────────────────────────────────────
# Mount the web/ directory so the browser can load index.html, styles.css, app.js.
# The SPA root (http://localhost:8000/) serves index.html via the FileResponse below.
WEB_DIR = settings.BASE_DIR / "web"   # AI Research Stack/web/
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# ── Output Directory ──────────────────────────────────────────────────────────
REPORTS_DIR = settings.BASE_DIR / "output"   # AI Research Stack/output/
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompts Directory ─────────────────────────────────────────────────────────
PROMPTS_DIR = settings.BASE_DIR / "prompts"  # AI Research Stack/prompts/

# ──────────────────────────────────────────────────────────────────────────────
# PYDANTIC REQUEST MODELS
# ──────────────────────────────────────────────────────────────────────────────
# These models validate the JSON body of incoming POST requests.

class DownloadRequest(BaseModel):
    """Request body for POST /api/download — download and ingest one paper."""
    title: str
    authors: list[dict]        # e.g. [{"name": "Vaswani"}, ...]
    venue: Optional[str] = None
    year: Optional[int] = None
    externalIds: Optional[dict] = {}   # {"DOI": "...", "ArXiv": "..."}
    abstract: Optional[str] = None
    citationCount: Optional[int] = 0


class RAGQueryRequest(BaseModel):
    """Request body for POST /api/query-rag — ask a grounded research question."""
    query: str
    limit: Optional[int] = 5
    prompt_template: Optional[str] = None   # Name of a prompts/*.txt template


class CitationAnalysisRequest(BaseModel):
    """Request body for POST /api/analyze-citations — start a citation analysis job."""
    paper_id: str   # DOI, arXiv ID, S2 CorpusID, or canonical S2 paper ID
    limit: Optional[int] = 5   # Maximum number of citing papers to analyse

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def sanitize_filename(title: str) -> str:
    """
    Convert a paper title into a filesystem-safe PDF filename.

    Removes all characters that are illegal in Windows/Linux filenames,
    replaces spaces with underscores, and truncates to 60 characters.

    Args:
        title: Raw paper title string.

    Returns:
        Lowercase .pdf filename safe for all operating systems.
    """
    clean = re.sub(r"[^a-zA-Z0-9_\-\s]", "", title)
    clean = clean.replace(" ", "_")
    clean = re.sub(r"_{2,}", "_", clean)
    return clean.strip("_")[:60].lower() + ".pdf"

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_ui():
    """
    Serve the web UI's root HTML page.

    Returns web/index.html when the user navigates to http://localhost:8000/.
    The SPA then loads its CSS and JS from the /static/* paths.
    """
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/health")
async def health_check():
    """
    Report the current health status of Ollama and ChromaDB.

    Returns a JSON object consumed by the SPA's status indicator badges.
    Also returns ChromaDB stats (chunk count, paper count) used by the
    Knowledge Base tab's summary badges.

    Returns:
        JSON: {"ollama": "online"|"offline", "vector_db": "online"|"offline", "db_stats": {...}}
    """
    # Check Ollama by calling its /api/tags endpoint (non-destructive, lightweight)
    ollama_status = "offline"
    try:
        resp = _req.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            ollama_status = "online"
    except Exception:
        pass  # Ollama is not running — status stays "offline"

    # Get ChromaDB collection stats (total chunks + unique papers list)
    db_status = "offline"
    db_stats  = {}
    try:
        db_stats  = vector_store.get_collection_stats()
        db_status = "online"  # If get_collection_stats() succeeds, ChromaDB is running
    except Exception:
        pass  # ChromaDB initialisation failed — status stays "offline"

    return {
        "ollama":    ollama_status,
        "vector_db": db_status,
        "db_stats":  db_stats
    }


@app.get("/api/search")
async def search_papers(q: str, limit: int = 10):
    """
    Search Semantic Scholar for academic papers matching the query string.

    Query Parameters:
        q     : The search keywords (required).
        limit : Maximum number of results to return (default: 10).

    Returns:
        List of paper dicts formatted for the SPA's search results cards.
        Each dict includes: paperId, title, authors, year, venue, doi, arxiv,
        abstract, citationCount, has_pdf.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string 'q' is required.")

    results = discover_service.search_papers(q.strip(), limit=limit)
    formatted = []

    for paper in results:
        external_ids = paper.get("externalIds") or {}
        doi   = external_ids.get("DOI")   or "N/A"
        arxiv = external_ids.get("ArXiv") or "N/A"
        # Check actual open-access availability via Unpaywall
        has_pdf = False
        if arxiv != "N/A":
            has_pdf = True  # arXiv papers are always open access
        elif doi != "N/A":
            pdf_url = discover_service.fetch_open_access_pdf_url(doi)
            has_pdf = pdf_url is not None

        formatted.append({
            "paperId":       paper.get("paperId", ""),
            "title":         paper.get("title", "Untitled"),
            "authors":       paper.get("authors", []),
            "year":          paper.get("year", "N/A"),
            "venue":         paper.get("venue", "N/A"),
            "doi":           doi,
            "arxiv":         arxiv,
            "abstract":      (paper.get("abstract") or "")[:500],  # Trim for transmission
            "citationCount": paper.get("citationCount", 0),
            "has_pdf":       has_pdf
        })

    return formatted


@app.post("/api/download")
async def download_paper(request: DownloadRequest, background_tasks: BackgroundTasks):
    """
    Download the PDF for a paper and ingest it into the ChromaDB vector database.

    Runs the download + extraction + ingestion pipeline as a FastAPI background
    task so the HTTP response is returned immediately (the UI shows a spinner
    while the background task runs).

    The ingestion follows the same 3-tier strategy as main.py:
      Tier 1: Unpaywall OA PDF → full-text ingestion
      Tier 2: arXiv direct PDF → full-text ingestion
      Tier 3: Abstract-only ingestion (last resort)

    Returns:
        {"success": True} immediately. The UI refreshes its manifest view after.
    """
    # Extract the identifiers needed for the download strategies
    ext_ids = request.externalIds or {}
    doi      = ext_ids.get("DOI")
    arxiv_id = ext_ids.get("ArXiv")
    title    = request.title

    def _ingest():
        """
        Internal background function that runs the full ingest pipeline.
        Runs in a separate thread so it doesn't block the event loop.
        """
        logger.info(f"BG Ingest started: '{title}'")
        chunks = []

        # ── Tier 1: Unpaywall open-access PDF ─────────────────────────────────
        pdf_url = None
        if doi:
            pdf_url = discover_service.fetch_open_access_pdf_url(doi)

        # ── Tier 2: arXiv direct PDF fallback ─────────────────────────────────
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        # ── Download and extract text from the PDF ────────────────────────────
        if pdf_url:
            safe_name = sanitize_filename(title)
            pdf_path  = discover_service.download_pdf(pdf_url, safe_name)

            if pdf_path and pdf_path.exists():
                try:
                    pages  = pdf_service.extract_text_by_page(pdf_path)
                    chunks = pdf_service.chunk_text(pages, chunk_size=1000, chunk_overlap=200)
                except Exception as e:
                    logger.error(f"Text extraction failed for '{title}': {e}")

        # ── Tier 3: Abstract-only fallback ────────────────────────────────────
        if not chunks and request.abstract:
            abstract = request.abstract.strip()
            identifier = doi or (f"arXiv:{arxiv_id}" if arxiv_id else title)
            chunks = [{
                "chunk_index": 0,
                "text": abstract,
                "metadata": {
                    "pages": [0], "char_start": 0,
                    "char_end": len(abstract), "length": len(abstract)
                }
            }]
            logger.info(f"Abstract-only fallback for '{title}'")
            vector_store.add_paper_chunks(paper_title=title, doi=identifier, chunks=chunks)
            manifest_service.mark_as_ingested(
                sanitize_filename(title), title, doi, status="success"
            )
            return

        # ── Ingest chunks into ChromaDB ────────────────────────────────────────
        if chunks:
            identifier = doi or (f"arXiv:{arxiv_id}" if arxiv_id else title)
            success = vector_store.add_paper_chunks(
                paper_title=title, doi=identifier, chunks=chunks
            )
            filename = sanitize_filename(title)
            manifest_service.mark_as_ingested(
                filename, title, doi,
                status="success" if success else "failed"
            )
            logger.info(
                f"BG Ingest complete for '{title}': "
                f"{len(chunks)} chunks, success={success}"
            )
        else:
            manifest_service.mark_as_ingested(
                sanitize_filename(title), title, doi, status="failed",
                error="No PDF and no abstract available."
            )
            logger.warning(f"Ingestion failed for '{title}': no content could be obtained.")

    # Schedule the ingest function to run in the background
    background_tasks.add_task(_ingest)
    return {"success": True, "message": f"Ingestion started for: {title}"}


@app.get("/api/pdfs")
async def list_pdfs():
    """
    Return the current ingestion manifest for all PDFs in papers/.

    Syncs the manifest against the actual filesystem and ChromaDB state
    before returning so the UI always reflects reality.

    Returns:
        List of file entry dicts: {title, doi, status, ingested_at, size_bytes}.
    """
    # Sync manifest with actual filesystem + ChromaDB state
    # Sync manifest against filesystem + ChromaDB.
    # Then build the UI list primarily from ChromaDB stats so ChromaDB
    # is the source of truth even if titles differ slightly.
    manifest = manifest_service.sync_with_vector_store(vector_store)
    pdf_dir  = settings.PDF_DOWNLOAD_DIR

    db_stats = vector_store.get_collection_stats()
    chroma_titles = set(t.lower().strip() for t in db_stats.get("papers_list", []))

    def _is_in_chroma(meta_title: str) -> bool:
        if not meta_title:
            return False
        mt = meta_title.lower().strip()
        return any(ct == mt or ct in mt or mt in ct for ct in chroma_titles)

    file_list = []
    for filename, meta in manifest.items():
        # Get the file size from the filesystem (0 if file doesn't exist)
        pdf_path = pdf_dir / filename
        size_bytes = pdf_path.stat().st_size if pdf_path.exists() else 0

        title = meta.get("title", filename)
        status = meta.get("status", "unknown")

        # If ChromaDB contains this paper, force status to success for UI.
        if _is_in_chroma(title):
            status = "success"
            meta_ingested_at = meta.get("ingested_at")
            if not meta_ingested_at:
                # Keep manifest timestamp if present, else leave it blank.
                meta_ingested_at = None
        else:
            meta_ingested_at = meta.get("ingested_at")

        file_list.append({
            "filename":    filename,
            "title":       title,
            "doi":         meta.get("doi", "N/A"),
            "status":      status,
            "ingested_at": meta_ingested_at,
            "size_bytes":  size_bytes
        })

    return file_list



@app.post("/api/ingest-pending")
async def ingest_pending():
    """
    Scan the papers/ directory and ingest all PDFs not yet in ChromaDB.

    Only PDFs with status "pending" or absent from the manifest are processed.
    Already-ingested PDFs (status "success") are skipped to avoid duplication.

    Returns:
        {"processed": N, "succeeded": M} — counts for the UI's alert dialog.
    """
    manifest = manifest_service.sync_with_vector_store(vector_store)
    pdf_dir  = settings.PDF_DOWNLOAD_DIR

    processed = 0
    succeeded = 0

    for filename, meta in manifest.items():
        if meta.get("status") == "success":
            continue  # Skip already-ingested PDFs

        pdf_path = pdf_dir / filename
        if not pdf_path.exists():
            continue  # Skip orphaned manifest entries

        logger.info(f"Ingesting pending PDF: {filename}")
        try:
            pages  = pdf_service.extract_text_by_page(pdf_path)
            chunks = pdf_service.chunk_text(pages, chunk_size=1000, chunk_overlap=200)
            title  = meta.get("title", pdf_path.stem.replace("_", " ").title())
            doi    = meta.get("doi")
            success = vector_store.add_paper_chunks(paper_title=title, doi=doi, chunks=chunks)

            manifest_service.mark_as_ingested(
                filename, title, doi,
                status="success" if success else "failed"
            )
            processed += 1
            if success:
                succeeded += 1

        except Exception as e:
            logger.error(f"Failed to ingest '{filename}': {e}")
            manifest_service.mark_as_ingested(
                filename, meta.get("title", filename), None,
                status="failed", error=str(e)
            )
            processed += 1

    return {"processed": processed, "succeeded": succeeded}


@app.post("/api/query-rag")
async def query_rag(request: RAGQueryRequest):
    """
    Answer a research question grounded in the ingested papers (RAG pipeline).

    Optionally applies a prompt template from prompts/*.txt for structured output
    (e.g. summarization, LinkedIn post, comparative analysis).

    Returns:
        {"answer": str, "sources": [...]} on success.
        Raises HTTP 503 if Ollama is offline, HTTP 422 if the query is empty.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query string must not be empty.")

    # Fail fast with 503 if Ollama is not reachable (before hitting ChromaDB)
    if not check_ollama_health():
        raise HTTPException(
            status_code=503,
            detail="Ollama LLM server is not running. Start with: ollama serve"
        )

    # ── Prompted RAG (template mode) ──────────────────────────────────────────
    if request.prompt_template:
        prompt_path = PROMPTS_DIR / f"{request.prompt_template}.txt"
        if not prompt_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Prompt template '{request.prompt_template}' not found in prompts/."
            )

        # Load and split the template on the divider line
        raw_template = prompt_path.read_text(encoding="utf-8").strip()
        divider      = "## USER PROMPT TEMPLATE"

        if divider in raw_template:
            parts         = raw_template.split(divider, 1)
            system_prompt = parts[0].replace("## SYSTEM PROMPT", "").strip()
            user_template = parts[1].strip()
        else:
            system_prompt = raw_template
            user_template = "{context}"

        # Retrieve relevant context chunks from ChromaDB
        chunks = vector_store.query_similar_chunks(request.query, limit=request.limit)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant papers found in the database. Ingest papers first."
            )

        # Build context string and fill all known template placeholders
        context_blocks = [
            f'[Source {i+1}] "{c["metadata"].get("title", "Untitled")}" '
            f'(Pages: {c["metadata"].get("pages", "N/A")})\nContent: {c["text"]}'
            for i, c in enumerate(chunks)
        ]
        context_str  = "\n\n".join(context_blocks)
        titles       = {c["metadata"].get("title", "") for c in chunks}
        title_str    = " | ".join(sorted(titles))

        user_prompt = (
            user_template
            .replace("{context}",  context_str)
            .replace("{context_a}", context_str)
            .replace("{context_b}", context_str)
            .replace("{title}",   title_str)
            .replace("{title_a}", title_str)
            .replace("{title_b}", title_str)
            .replace("{authors}", "See source metadata")
            .replace("{year}",    "Various")
            .replace("{venue}",   "Various")
        )

        # Send to Ollama
        url     = f"{settings.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "stream":  False,
            "options": {"temperature": 0.3}
        }
        resp = _req.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama error {resp.status_code}: {resp.text[:200]}"
            )

        answer = resp.json()["message"]["content"].strip()
        return {
            "answer":  answer,
            "sources": chunks,
            "template_used": request.prompt_template
        }

    # ── Standard RAG (no template) ────────────────────────────────────────────
    rag_service = RAGService()
    result      = rag_service.generate_answer(request.query, limit=request.limit)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "RAG failed."))

    return {"answer": result["answer"], "sources": result["sources"]}


@app.get("/api/prompts")
async def list_prompts():
    """
    Return metadata for all prompt templates found in the prompts/ directory.

    Each template file is parsed to extract a title (from the first heading line)
    and a short description (from the second non-blank line).

    Returns:
        List of dicts: {name, title, description, content} for the SPA's Prompts tab.
    """
    prompts = []
    if not PROMPTS_DIR.exists():
        return []

    # Iterate all .txt files in the prompts/ directory, sorted alphabetically
    for prompt_file in sorted(PROMPTS_DIR.glob("*.txt")):
        content = prompt_file.read_text(encoding="utf-8").strip()
        lines   = [l for l in content.split("\n") if l.strip()]

        # Extract title: strip leading "##" and "SYSTEM PROMPT — " markers
        raw_title = lines[0] if lines else prompt_file.stem
        title = re.sub(r"^#+\s*", "", raw_title).strip()
        title = re.sub(r"^SYSTEM PROMPT\s*[—\-:]*\s*", "", title, flags=re.IGNORECASE).strip()

        # Extract description: first non-heading, non-empty line after the title
        desc_lines = [
            l for l in lines[1:]
            if l.strip() and not l.startswith("#") and not l.startswith("---")
        ]
        description = desc_lines[0].strip() if desc_lines else f"Prompt template: {prompt_file.stem}"

        prompts.append({
            "name":        prompt_file.stem,   # "summarize", "linkedin_draft", etc.
            "title":       title,
            "description": description[:200],  # Trim for the card preview
            "content":     content             # Full content for the modal viewer
        })

    return prompts


@app.post("/api/analyze-citations")
async def start_citation_analysis(request: CitationAnalysisRequest):
    """
    Start a citation analysis pipeline as a background thread.

    Returns a run_id immediately. The SPA polls
    GET /api/analyze-citations/{run_id} to track progress.

    Returns:
        {"run_id": str} — a UUID the SPA uses to poll for status.
    """
    # Generate a unique job identifier for this analysis run
    run_id = str(uuid.uuid4())

    # Initialise the job entry with "running" status
    citation_jobs[run_id] = {
        "status":   "running",
        "progress": "Initialising citation analysis pipeline...",
        "result":   [],
        "csv_path": None,
        "error":    None
    }

    def _run_analysis():
        """
        Background thread function that executes the citation analysis pipeline.
        Updates citation_jobs[run_id] at each step so the polling endpoint
        can report accurate progress to the UI.
        """
        try:
            citation_jobs[run_id]["progress"] = (
                "Fetching target paper metadata from Semantic Scholar..."
            )
            analyzer = CitationAnalyzerService()

            # Run the full pipeline (can take several minutes for large limits)
            citation_jobs[run_id]["progress"] = (
                f"Analysing up to {request.limit} citing papers — "
                "downloading PDFs, extracting passages, classifying with LLM..."
            )
            csv_path = analyzer.analyze_citations(request.paper_id, limit=request.limit)

            if csv_path and csv_path.exists():
                # Read the CSV to build the result rows for the UI table
                import csv
                rows = []
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append({
                            "citing_title":  row.get("Citing Paper Title", ""),
                            "year":          row.get("Year", "N/A"),
                            "passage":       row.get("Extracted Passage", ""),
                            "classification":row.get("LLM Classification", "extending"),
                            "rationale":     row.get("Rationale", "")
                        })

                # Mark the job as completed
                citation_jobs[run_id].update({
                    "status":   "completed",
                    "progress": f"Analysis complete — {len(rows)} citation(s) classified.",
                    "result":   rows,
                    "csv_path": csv_path.name  # Filename only — used in the download URL
                })
                logger.info(f"Citation job {run_id} completed: {csv_path.name}")

            else:
                citation_jobs[run_id].update({
                    "status":   "failed",
                    "progress": "Analysis finished but no records were generated.",
                    "error":    "No citing papers with accessible PDFs or context snippets found."
                })

        except Exception as e:
            logger.error(f"Citation analysis job {run_id} failed: {e}")
            citation_jobs[run_id].update({
                "status":   "failed",
                "progress": "Analysis pipeline encountered an error.",
                "error":    str(e)
            })

    # Start the analysis in a daemon thread so it doesn't block server shutdown
    t = threading.Thread(target=_run_analysis, daemon=True)
    t.start()

    logger.info(f"Citation analysis job started — run_id: {run_id}")
    return {"run_id": run_id}


@app.get("/api/analyze-citations/{run_id}")
async def get_citation_status(run_id: str):
    """
    Poll the status of a running or completed citation analysis job.

    Called by the SPA every 1.5 seconds while the job is running.

    Returns:
        The job entry dict from citation_jobs for this run_id.
        HTTP 404 if the run_id is not recognised.
    """
    if run_id not in citation_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"No citation analysis job found for run_id: {run_id}"
        )
    return citation_jobs[run_id]


@app.get("/api/reports")
async def list_reports():
    """
    List all CSV citation analysis reports saved in the output/ directory.

    Returns:
        List of report dicts: {filename, size_bytes, created_at (ISO string)}.
        Sorted by creation time — newest first.
    """
    reports = []
    for csv_file in sorted(REPORTS_DIR.glob("*.csv"), reverse=True):
        stat = csv_file.stat()
        reports.append({
            "filename":   csv_file.name,
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
        })
    return reports


@app.get("/api/reports/download/{filename}")
async def download_report(filename: str):
    """
    Stream a specific CSV report file to the browser as a download.

    Args:
        filename: The name of the CSV file in output/ to download.

    Returns:
        FileResponse with Content-Disposition: attachment so the browser
        prompts the user to save the file.
    """
    # Sanitise the filename to prevent path traversal attacks
    # (e.g. a request for "../../.env" should be rejected)
    safe_name = Path(filename).name  # Strips any directory components
    report_path = REPORTS_DIR / safe_name

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report '{safe_name}' not found in output/."
        )

    return FileResponse(
        path=str(report_path),
        filename=safe_name,
        media_type="text/csv"
    )


# ── Server entry point (for direct execution) ─────────────────────────────────
# Allows running the server with:  python scripts/server.py
# (equivalent to: uvicorn server:app --reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)]
    )