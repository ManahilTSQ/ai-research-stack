"""
server.py — FastAPI Web Server for the AI Research Stack.
"""

import sys
import logging
import threading
import time
import uuid
import re
import zipfile
import io
from pathlib import Path
from datetime import datetime
from typing import Optional
import requests as _req

from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, Request, UploadFile, File
from fastapi.websockets import WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import base64
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings
from paper_discovery import PaperDiscoveryService
from pdf_processor import PDFProcessorService
from vector_store import VectorStoreService
from rag_service import RAGService, check_ollama_health, OllamaUnavailableError
from manifest_manager import ManifestManagerService
from citation_analyzer import CitationAnalyzerService
from metadata_service import metadata_service
from prompt_manager import (
    PromptValidationError,
    load_prompt_metadata,
    list_prompt_files,
    parse_prompt_file,
    save_prompt,
    delete_prompt,
    substitute_placeholders,
)
from rag_context import (
    build_library_inventory,
    chunks_to_context_string,
    retrieve_relevant_chunks,
    filter_chunks_to_titles,
    EMPTY_DB_REFUSAL,
    IRRELEVANT_REFUSAL,
)
from rag_strict import (
    resolve_query_scope,
    scope_refusal_message,
    inventory_for_scope,
    apply_verification_or_refuse,
    build_catalog_indexes,
    list_distinct_authors,
    answer_catalog_metadata_query,
    apply_scope_resilience,
)
from search_utils import (
    extract_quoted_phrases,
    filter_papers_for_precision,
    build_api_query_string,
    is_likely_author_query,
)
from paper_labels import format_sidebar_label

# Explicitly force UTF-8 encoding for logging to clean journalctl streams
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Force UTF-8 writing on stdout handler
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))

logger = logging.getLogger("ai_research_server")

logger.info("Initialising all backend services...")
discover_service = PaperDiscoveryService()
pdf_service      = PDFProcessorService()
vector_store     = VectorStoreService()
manifest_service = ManifestManagerService()
# Step 1c: Initialize RAGService at startup instead of per-query
rag_service = None
try:
    rag_service = RAGService()
    logger.info("RAG Service initialised.")
except OllamaUnavailableError:
    logger.warning("Ollama not available at startup - RAG disabled until Ollama starts.")
logger.info("All backend services initialised.")

citation_jobs: dict[str, dict] = {}

app = FastAPI(
    title="AI Research Stack API",
    description="Self-hosted academic research assistant — all processing runs locally.",
    version="2.0.0"
)

class CSPMiddleware(BaseHTTPMiddleware):
	async def dispatch(self, request, call_next):
		response = await call_next(request)
		response.headers["Content-Security-Policy"] = (
			"default-src * 'unsafe-inline' 'unsafe-eval' data: blob:; "
			"script src * 'unsafe-inline' 'unsafe-eval'; "
			"connect-src * 'unsafe-inline'; "
			"img-src * data: blob: 'unsafe-inline'; "
			"frame-src *; "
			"style-src * 'unsafe-inline';"
		)
		return response
app.add_middleware(CSPMiddleware)

# ── CORS + Trusted Host Middleware ────────────────────────────────────────────
# Dynamic CORS middleware that echoes the request Origin back.
# Using allow_origins=["*"] with allow_credentials=True violates the CORS spec
# and browsers reject credentialed responses. By reflecting the actual Origin
# we satisfy the spec while still allowing any Cloudflare tunnel hostname.
class DynamicCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin", "")
        if request.method == "OPTIONS":
            # Preflight — return immediately with the correct CORS headers
            response = Response(status_code=204)
            if origin:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
                response.headers["Vary"] = "Origin"
            return response
        response = await call_next(request)
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Vary"] = "Origin"
        return response

app.add_middleware(DynamicCORSMiddleware)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# ── HTTP Basic Authentication Middleware ──────────────────────────────────────
@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    # Bypass OPTIONS preflight requests
    if request.method == "OPTIONS":
        return await call_next(request)

    # Bypass public routes
    if request.url.path in ["/", "/api/health", "/sw.js", "/service-worker.js", "/api/download"] or request.url.path.startswith("/static/") or request.url.path.startswith("/api/pdfs") or request.url.path.startswith("/api/query-rag") or request.url.path.startswith("/api/prompts") or request.url.path.startswith("/api/reports") or request.url.path.startswith("/api/search"):
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Basic "):
        headers = {}
        if not request.url.path.startswith("/api/"):
            headers["WWW-Authenticate"] = 'Basic realm="AI Research Stack"'
        return Response(
            status_code=401,
            headers=headers,
            content="Unauthorized Access"
        )

    try:
        payload = auth_header.split(" ")[1]
        decoded = base64.b64decode(payload).decode("utf-8")
        username, password = decoded.split(":", 1)
        # Credentials loaded from .env via config.Settings (see BASIC_AUTH_USER / BASIC_AUTH_PASS)
        if username == settings.BASIC_AUTH_USER and password == settings.BASIC_AUTH_PASS:
            return await call_next(request)
    except Exception:
        pass

    headers = {}
    if not request.url.path.startswith("/api/"):
        headers["WWW-Authenticate"] = 'Basic realm="AI Research Stack"'
    return Response(
        status_code=401,
        headers=headers,
        content="Unauthorized Access"
    )

WEB_DIR = settings.BASE_DIR / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

REPORTS_DIR = settings.BASE_DIR / "output"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_DIR = settings.BASE_DIR / "prompts"


class DownloadRequest(BaseModel):
    title: str
    authors: list[dict]
    venue: Optional[str] = None
    year: Optional[int] = None
    externalIds: Optional[dict] = {}
    abstract: Optional[str] = None
    citationCount: Optional[int] = 0
    paperId: Optional[str] = None  # Semantic Scholar ID for duplicate detection in discovery UI


class RAGQueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 15
    prompt_template: Optional[str] = None
    filter_title: Optional[str] = None   # Primary paper scope (Paper A in compare mode)
    filter_title_b: Optional[str] = None  # Second paper for comparative_analysis template
    template_vars: Optional[dict] = None  # Extra placeholders, e.g. {phenomenon} for Hassan template
    conversation_history: Optional[list[dict]] = None  # Prior user/assistant turns for chat persistence


class CitationAnalysisRequest(BaseModel):
    paper_id: str
    limit: Optional[int] = 5


class DeleteReportRequest(BaseModel):
    filename: str


class PromptSaveRequest(BaseModel):
    """Body for creating or updating a prompt template via the Prompts tab UI."""
    name: str
    display_title: str
    system_body: str
    user_template: str
    overwrite: bool = True  # If False, reject when file already exists


def _purge_old_reports() -> None:
    """Delete CSV reports older than REPORT_RETENTION_DAYS when configured (> 0)."""
    days = settings.REPORT_RETENTION_DAYS
    if days <= 0:
        return
    cutoff = time.time() - (days * 86400)
    for csv_file in REPORTS_DIR.glob("*.csv"):
        try:
            if csv_file.stat().st_mtime < cutoff:
                csv_file.unlink()
                logger.info("Auto-deleted old report: %s", csv_file.name)
        except OSError as e:
            logger.warning("Could not delete old report %s: %s", csv_file.name, e)


def _ollama_chat(system_prompt: str, user_prompt: str, temperature: float = 0.25) -> str:
    """Send a single system+user exchange to Ollama and return the assistant text."""
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = _req.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama error {resp.status_code}: {resp.text[:200]}",
        )
    return resp.json()["message"]["content"].strip()


def _semantic_scholar_paper_url(paper_id: str) -> str:
    """Public HTTPS link to a paper on semanticscholar.org (avoids API 405 errors)."""
    if not paper_id:
        return ""
    pid = str(paper_id).strip()
    if pid.startswith("http"):
        return pid
    return f"https://www.semanticscholar.org/paper/{pid}"


def _execute_template_rag(request: RAGQueryRequest) -> dict:
    """
    Run RAG with a custom prompt template file.
    Supports two-paper compare when filter_title and filter_title_b are both set.
    """
    prompt_path = PROMPTS_DIR / f"{request.prompt_template}.txt"
    if not prompt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prompt template '{request.prompt_template}' not found in prompts/.",
        )

    raw_template = prompt_path.read_text(encoding="utf-8").strip()
    try:
        _, system_prompt, user_template = parse_prompt_file(raw_template)
    except PromptValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    stats = vector_store.get_collection_stats()
    papers_metadata = stats.get("papers_metadata", {})

    if not papers_metadata:
        raise HTTPException(status_code=404, detail=EMPTY_DB_REFUSAL)

    scope = resolve_query_scope(
        request.query,
        papers_metadata,
        filter_title=request.filter_title or None,
    )
    scope = apply_scope_resilience(scope, request.query, papers_metadata)
    if scope.requires_entity and not scope.scoped_titles:
        raise HTTPException(status_code=404, detail=scope_refusal_message(scope))

    inventory_metadata = inventory_for_scope(papers_metadata, scope)
    matched_titles = scope.scoped_titles

    is_compare = bool(
        request.filter_title
        and request.filter_title_b
        and request.filter_title != request.filter_title_b
    )

    limit = request.limit or 15
    if is_compare:
        half_a = max(1, limit // 2)
        half_b = max(1, limit - half_a)
        chunks_a = retrieve_relevant_chunks(
            vector_store, request.query, limit=half_a, filter_title=request.filter_title
        )
        chunks_b = retrieve_relevant_chunks(
            vector_store, request.query, limit=half_b, filter_title=request.filter_title_b
        )
        chunks = chunks_a + chunks_b
        context_a = chunks_to_context_string(chunks_a, header="Research Paper Context A")
        context_b = chunks_to_context_string(chunks_b, header="Research Paper Context B")
        title_a = request.filter_title
        title_b = request.filter_title_b
    else:
        chunks = retrieve_relevant_chunks(
            vector_store,
            request.query,
            limit=limit,
            filter_title=request.filter_title or None,
            scope_titles=matched_titles if matched_titles else None,
        )
        context_a = context_b = ""
        title_a = title_b = ""

    if matched_titles and not request.filter_title:
        chunks = filter_chunks_to_titles(chunks, matched_titles)

    if not chunks:
        raise HTTPException(status_code=404, detail=IRRELEVANT_REFUSAL)

    library_inventory_str = build_library_inventory(inventory_metadata)
    context_str = chunks_to_context_string(chunks)

    if is_compare:
        combined_context = (
            f"Ingested Paper Library Inventory:\n{library_inventory_str}\n\n"
            f"{context_a}\n\n{context_b}"
        )
    else:
        combined_context = (
            f"Ingested Paper Library Inventory:\n{library_inventory_str}\n\n"
            f"{context_str}"
        )

    titles = {(c.get("metadata") or {}).get("title", "") for c in chunks}
    title_str = " | ".join(sorted(t for t in titles if t)) or "None"
    authors_list = [(c.get("metadata") or {}).get("authors", "Unknown Authors") for c in chunks]
    authors_str = " | ".join(sorted(set(authors_list))) if authors_list else "None"
    years_list = [str((c.get("metadata") or {}).get("year", "N/A")) for c in chunks]
    years_str = " | ".join(sorted(set(years_list))) if years_list else "None"

    variables = {
        "context": combined_context,
        "context_a": context_a if is_compare else combined_context,
        "context_b": context_b if is_compare else combined_context,
        "title": title_str,
        "title_a": title_a if is_compare else title_str,
        "title_b": title_b if is_compare else title_str,
        "authors": authors_str,
        "year": years_str,
        "venue": "Various",
    }
    variables["query"] = request.query.strip()

    if request.template_vars:
        for key, val in request.template_vars.items():
            variables[key] = str(val) if val is not None else ""

    user_prompt = substitute_placeholders(user_template, variables)
    answer = _ollama_chat(system_prompt, user_prompt, temperature=0.05)
    answer, verified = apply_verification_or_refuse(
        answer,
        scope=scope,
        papers_metadata=papers_metadata,
        chunks=chunks,
    )
    if not verified:
        raise HTTPException(status_code=404, detail=answer)

    return {
        "answer": answer,
        "sources": chunks,
        "template_used": request.prompt_template,
        "compare_mode": is_compare,
    }


def _should_fallback_to_standard_rag(request: RAGQueryRequest) -> bool:
    """
    Guardrail for accidental template use on factual QA prompts.
    If the Hassan-style drafting template is selected but the user asks
    a direct question (instead of requesting a draft), run standard RAG.
    """
    from rag_context import is_per_paper_extraction_query, is_simple_inventory_listing

    q = (request.query or "").strip().lower()
    if not q:
        return False

    # Tables and corpus listings need author-scoped standard RAG, not article templates.
    if is_per_paper_extraction_query(request.query) or is_simple_inventory_listing(request.query):
        return True

    if request.prompt_template != "hassanian_article":
        return False
    draft_verbs = ("draft", "write an article", "manuscript", "paper section", "problematisation")
    if any(v in q for v in draft_verbs):
        return False
    # Typical factual question forms should not trigger long-form article drafting.
    return q.endswith("?") or q.startswith(("what ", "who ", "where ", "when ", "why ", "how "))


def sanitize_filename(title: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-\s]", "", title)
    clean = clean.replace(" ", "_")
    clean = re.sub(r"_{2,}", "_", clean)
    return clean.strip("_")[:60].lower() + ".pdf"


def format_authors(authors: list) -> str:
    if not authors:
        return "Unknown Authors"
    names = [a.get("name", "") for a in authors if a.get("name")]
    return ", ".join(names)


@app.get("/", response_class=FileResponse, include_in_schema=False)
def serve_ui():
    return FileResponse(
        str(WEB_DIR / "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )

@app.websocket("/")
async def websocket_root(websocket: WebSocket):
	"""Accept and close WebSocket probes (e.g. from Cloudflare health checks)."""
	await websocket.accept()
	await websocket.close()

@app.get("/sw.js", include_in_schema=False)
@app.get("/service-worker.js", include_in_schema=False)
async def serve_sw_unregister():
    content = (
        "self.addEventListener('install', function(e) {\n"
        "    self.skipWaiting();\n"
        "});\n"
        "self.addEventListener('activate', function(e) {\n"
        "    self.registration.unregister()\n"
        "    .then(function() {\n"
        "        return self.clients.matchAll();\n"
        "    })\n"
        "    .then(function(clients) {\n"
        "        clients.forEach(client => client.navigate(client.url));\n"
        "    });\n"
        "});\n"
    )
    return Response(content=content, media_type="application/javascript")


@app.get("/api/health")
def health_check():
    ollama_status = "offline"
    try:
        resp = _req.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            ollama_status = "online"
    except Exception:
        pass

    db_status = "offline"
    db_stats  = {}
    try:
        db_stats  = vector_store.get_collection_stats()
        db_status = "online"
    except Exception:
        pass

    return {
        "ollama":    ollama_status,
        "vector_db": db_status,
        "db_stats":  db_stats
    }


@app.get("/api/search")
def search_papers(
    q: str,
    limit: int = 10,
    offset: int = 0,
    exact_author: bool = False,
):
    """
    Search Semantic Scholar. Use exact_author=true or quoted names ("Manahil Shahid")
    for strict author matching (post-filtered — the API itself is not Boolean).
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string 'q' is required.")

    # Normalise smart quotes so exact-name filters work for pasted queries.
    raw_query = q.strip().replace("“", '"').replace("”", '"')
    remainder, quoted_phrases = extract_quoted_phrases(raw_query)
    api_query = build_api_query_string(raw_query) or remainder or raw_query

    logger.info(f"Searching Semantic Scholar for: '{api_query}' (limit={limit}, offset={offset})")
    results = discover_service.search_papers(api_query, limit=limit, offset=offset)

    # Post-filter: quoted phrases = exact full author name.
    # Only apply strict checkbox mode if the query actually looks like a personal name.
    exact_author_active = bool(exact_author and is_likely_author_query(remainder or raw_query))
    author_filter_text = remainder if exact_author_active and remainder else raw_query
    results = filter_papers_for_precision(
        results,
        exact_phrases=quoted_phrases,
        exact_author_mode=exact_author_active and not quoted_phrases,
        author_query=author_filter_text if exact_author_active else None,
    )

    formatted = []

    for paper in results:
        external_ids = paper.get("externalIds") or {}
        doi   = external_ids.get("DOI")   or "N/A"
        arxiv = external_ids.get("ArXiv") or "N/A"
        # Simple PDF availability check without blocking Unpaywall calls
        has_pdf = arxiv != "N/A"  # arXiv papers always have PDFs

        pid = paper.get("paperId", "")
        formatted.append({
            "paperId":       pid,
            "title":         paper.get("title", "Untitled"),
            "authors":       paper.get("authors", []),
            "year":          paper.get("year", "N/A"),
            "venue":         paper.get("venue", "N/A"),
            "doi":           doi,
            "arxiv":         arxiv,
            "abstract":      (paper.get("abstract") or "")[:500],
            "citationCount": paper.get("citationCount", 0),
            "has_pdf":       has_pdf,
            # External link — opens Semantic Scholar in browser (not an API route; avoids 405)
            "article_url":   _semantic_scholar_paper_url(pid),
        })

    return formatted


@app.post("/api/download")
def download_paper(request: DownloadRequest, background_tasks: BackgroundTasks):
    ext_ids = request.externalIds or {}
    doi      = ext_ids.get("DOI")
    arxiv_id = ext_ids.get("ArXiv")
    title    = request.title
    authors_str = format_authors(request.authors)
    year = request.year

    def _ingest():
        nonlocal title, authors_str, year, doi
        logger.info(f"BG Ingest started: '{title}'")
        chunks = []

        # Use metadata_service to get clean metadata from Crossref → OpenAlex → S2 cascade
        identifier = doi or (f"arXiv:{arxiv_id}" if arxiv_id else title)
        clean_metadata = metadata_service.get_paper_metadata(identifier)
        
        if clean_metadata:
            # Use clean metadata from authoritative sources
            enriched_title = clean_metadata.get("title", title)
            enriched_authors_str = format_authors(clean_metadata.get("authors", request.authors))
            enriched_year = clean_metadata.get("year", year)
            enriched_venue = clean_metadata.get("venue", request.venue)
            enriched_doi = clean_metadata.get("doi", doi)
            enriched_abstract = clean_metadata.get("abstract", request.abstract)
            logger.info(f"Using clean metadata from {clean_metadata.get('source', 'unknown')}")
        else:
            # Fallback to provided metadata
            enriched_title = title
            enriched_authors_str = authors_str
            enriched_year = year
            enriched_venue = request.venue
            enriched_doi = doi
            enriched_abstract = request.abstract
            logger.info("Using provided metadata (cascade failed)")
        
        # Use enriched metadata for the rest of the pipeline
        title = enriched_title
        authors_str = enriched_authors_str
        year = enriched_year
        venue = enriched_venue
        doi = enriched_doi
        abstract = enriched_abstract

        pdf_urls = []

        # Cascade order: ArXiv → Unpaywall → Core.ac.uk → MDPI API → PMC E-utilities → OpenAlex → S2 openAccessPdf

        # Tier 1: ArXiv (never blocked, best for CS/ML papers)
        if doi:
            arxiv_url = discover_service.fetch_arxiv_pdf_url(doi)
            if arxiv_url and arxiv_url not in pdf_urls:
                pdf_urls.append(arxiv_url)

        # Tier 2: Unpaywall
        if doi:
            unpaywall_urls = discover_service.fetch_all_open_access_pdf_urls(doi)
            for url in unpaywall_urls:
                if url not in pdf_urls:
                    pdf_urls.append(url)

        # Tier 3: Core.ac.uk (by title)
        if title:
            core_url = discover_service.fetch_core_ac_pdf_url(title)
            if core_url and core_url not in pdf_urls:
                pdf_urls.append(core_url)

        # Tier 4: MDPI research API (for MDPI DOIs)
        if doi:
            mdpi_url = discover_service.fetch_mdpi_api_pdf_url(doi)
            if mdpi_url and mdpi_url not in pdf_urls:
                pdf_urls.append(mdpi_url)

        # Tier 5: PMC E-utilities (if PMCID available from OpenAlex)
        if doi:
            openalex_data = discover_service.fetch_openalex_metadata(doi)
            if openalex_data:
                ids = openalex_data.get("ids", {})
                pmcid = ids.get("pmcid") if ids else None
                if pmcid:
                    pmc_url = discover_service.fetch_pmc_eutils_pdf_url(pmcid)
                    if pmc_url and pmc_url not in pdf_urls:
                        pdf_urls.append(pmc_url)

        # Tier 6: OpenAlex open access URL
        if doi:
            openalex_urls = discover_service.fetch_all_openalex_pdf_urls(doi)
            for url in openalex_urls:
                if url not in pdf_urls:
                    pdf_urls.append(url)

        # Tier 7: Semantic Scholar openAccessPdf field
        if request.externalIds:
            s2_pdf = (request.externalIds.get("openAccessPdf") or {}).get("url")
            if s2_pdf and s2_pdf not in pdf_urls:
                pdf_urls.append(s2_pdf)

        # Tier 8: arXiv direct (fallback for arxiv_id from S2)
        if arxiv_id:
            arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
            if arxiv_url not in pdf_urls:
                pdf_urls.append(arxiv_url)

        pdf_path = None
        if pdf_urls:
            safe_name = sanitize_filename(title)
            for url in pdf_urls:
                logger.info(f"Attempting download from candidate URL: {url}")
                pdf_path = discover_service.download_pdf(url, safe_name)
                if pdf_path and pdf_path.exists():
                    logger.info(f"Successfully downloaded PDF from: {url}")
                    break

            if pdf_path and pdf_path.exists():
                try:
                    full_text, char_to_page = pdf_service.extract_text_by_page(pdf_path)
                    if len(full_text) < 8000:
                        logger.warning(f"Extracted minimal text from '{title}' - likely abstract-only or scanned PDF")
                    # Step 6b: Standardize chunk sizes to 2000/400 everywhere
                    chunks = pdf_service.chunk_text(full_text, char_to_page, chunk_size=2000, chunk_overlap=400)
                except Exception as e:
                    logger.error(f"Text extraction failed for '{title}': {e}")

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
            vector_store.add_paper_chunks(
                paper_title=title, doi=identifier, chunks=chunks,
                authors=authors_str, year=year, venue=request.venue
            )
            manifest_service.mark_as_ingested(
                sanitize_filename(title), title, doi, status="success",
                authors=authors_str, year=year, venue=request.venue, abstract=abstract,
                paper_id=request.paperId,
            )
            return

        if chunks:
            identifier = doi or (f"arXiv:{arxiv_id}" if arxiv_id else title)
            success = vector_store.add_paper_chunks(
                paper_title=title, doi=identifier, chunks=chunks,
                authors=authors_str, year=year, venue=request.venue
            )
            # Clear stats cache after successful ingestion
            if success:
                vector_store.invalidate_stats_cache()
            filename = sanitize_filename(title)
            manifest_service.mark_as_ingested(
                filename, title, doi,
                status="success" if success else "failed",
                authors=authors_str, year=year, venue=request.venue, abstract=request.abstract,
                paper_id=request.paperId,
            )
            logger.info(
                f"BG Ingest complete for '{title}': "
                f"{len(chunks)} chunks, success={success}"
            )
        else:
            manifest_service.mark_as_ingested(
                sanitize_filename(title), title, doi, status="failed",
                error="No PDF and no abstract available.",
                authors=authors_str, year=year, abstract=request.abstract,
                paper_id=request.paperId,
            )
            logger.warning(f"Ingestion failed for '{title}': no content could be obtained.")

        # Enforce a non-blocking throttle block to keep Semantic Scholar/Crossref happy
        time.sleep(2.0)

    background_tasks.add_task(_ingest)
    return {"success": True, "message": f"Ingestion started for: {title}", "mode": "pdf" if request.externalIds else "abstract"}


@app.get("/api/papers/download-zip")
def download_papers_zip():
    """
    Download all papers from the papers/ directory as a zip file.
    Preserves folder structure and includes all PDF files recursively.
    """
    pdf_dir = settings.PDF_DOWNLOAD_DIR
    
    # Check if papers directory exists and has PDFs
    if not pdf_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Papers directory not found. Please download some papers first."
        )
    
    # Collect all PDF files recursively (case-insensitive)
    pdf_files = [p for p in pdf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    
    if not pdf_files:
        raise HTTPException(
            status_code=404,
            detail="No PDF files found in the papers directory. Please download some papers first before attempting to download a zip archive."
        )
    
    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for pdf_path in pdf_files:
            # Get relative path from pdf_dir to preserve folder structure
            rel_path = pdf_path.relative_to(pdf_dir)
            zip_file.write(pdf_path, arcname=str(rel_path))
    
    zip_buffer.seek(0)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"papers_backup_{timestamp}.zip"
    
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={zip_filename}"
        }
    )


@app.get("/api/papers/{filename:path}")
def get_paper_file(filename: str):
    """
    Serve a physical PDF file from the papers/ directory.
    """
    pdf_path = settings.PDF_DOWNLOAD_DIR / filename
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Paper file '{filename}' not found."
        )
    return FileResponse(str(pdf_path), media_type="application/pdf")


@app.delete("/api/papers/{filename:path}")
def delete_paper(filename: str):
    manifest = manifest_service.get_all_entries()
    if filename not in manifest:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{filename}' not found in the manifest."
        )

    meta = manifest[filename]
    title = meta.get("title")
    doi = meta.get("doi")

    # 1. Delete from vector store
    success_db = vector_store.delete_paper(title=title, doi=doi)

    # 2. Delete physical file
    pdf_path = settings.PDF_DOWNLOAD_DIR / filename
    success_file = True
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete PDF file '{filename}': {e}")
            success_file = False

    # 3. Delete from manifest
    try:
        del manifest[filename]
        manifest_service._save_manifest(manifest)
        success_manifest = True
    except Exception as e:
        logger.error(f"Failed to delete manifest entry for '{filename}': {e}")
        success_manifest = False

    return {
        "success": success_db and success_manifest,
        "details": {
            "database": success_db,
            "file": success_file,
            "manifest": success_manifest
        }
    }


@app.delete("/api/papers")
def delete_all_papers():
    """Delete all papers from ChromaDB, manifest, and physical files."""
    manifest = manifest_service.get_all_entries()
    filenames = list(manifest.keys())
    
    if not filenames:
        return {"success": True, "deleted_count": 0, "message": "No papers to delete."}
    
    deleted_count = 0
    errors = []
    
    for filename in filenames:
        try:
            meta = manifest[filename]
            title = meta.get("title")
            doi = meta.get("doi")
            
            # 1. Delete from vector store
            vector_store.delete_paper(title=title, doi=doi)
            
            # 2. Delete physical file
            pdf_path = settings.PDF_DOWNLOAD_DIR / filename
            if pdf_path.exists():
                pdf_path.unlink()
            
            deleted_count += 1
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            logger.error(f"Failed to delete paper '{filename}': {e}")
    
    # 3. Clear the entire manifest
    try:
        manifest_service._save_manifest({})
        success_manifest = True
    except Exception as e:
        logger.error(f"Failed to clear manifest: {e}")
        success_manifest = False
        errors.append(f"Manifest clear failed: {str(e)}")
    
    # 4. Clear the entire ChromaDB collection
    try:
        vector_store.client.delete_collection("research_papers")
        # Refresh the collection reference to avoid stale IDs
        vector_store._refresh_collection()
        # Also refresh the rag_service's internal vector store reference to avoid stale collection UUID
        if rag_service is not None:
            rag_service.vector_store._refresh_collection()
        # Clear the stats cache to reflect the deletion immediately
        vector_store.invalidate_stats_cache()
        success_db = True
    except Exception as e:
        logger.error(f"Failed to clear ChromaDB collection: {e}")
        success_db = False
        errors.append(f"ChromaDB clear failed: {str(e)}")
    
    return {
        "success": success_manifest and success_db and deleted_count == len(filenames),
        "deleted_count": deleted_count,
        "total_count": len(filenames),
        "errors": errors if errors else None,
        "details": {
            "database": success_db,
            "manifest": success_manifest
        }
    }


@app.get("/api/pdfs")
def list_pdfs():
    manifest = manifest_service.sync_with_vector_store(vector_store)
    # Refresh collection reference after manifest sync to handle potential UUID changes
    vector_store._refresh_collection()
    # Resolve a bounded batch of missing author/year labels before rendering the sidebar.
    manifest_service.refresh_metadata_sync(vector_store, max_entries=15)
    manifest = manifest_service.get_all_entries()
    pdf_dir  = settings.PDF_DOWNLOAD_DIR

    db_stats = vector_store.get_collection_stats()
    chroma_titles = set(t.lower().strip() for t in db_stats.get("papers_list", []))
    papers_meta = db_stats.get("papers_metadata", {})

    def _chroma_meta_for_title(meta_title: str) -> dict:
        """Resolve ChromaDB metadata when manifest title differs slightly from stored title."""
        if not meta_title:
            return {}
        if meta_title in papers_meta:
            return papers_meta[meta_title]
        mt = meta_title.lower().strip()
        for chroma_title, chroma_meta in papers_meta.items():
            ct = chroma_title.lower().strip()
            if ct == mt or ct in mt or mt in ct:
                return chroma_meta
        return {}

    def _is_in_chroma(meta_title: str) -> bool:
        if not meta_title:
            return False
        mt = meta_title.lower().strip()
        return any(ct == mt or ct in mt or mt in ct for ct in chroma_titles)

    file_list = []
    for filename, meta in manifest.items():
        pdf_path = pdf_dir / filename
        size_bytes = pdf_path.stat().st_size if pdf_path.exists() else 0

        title = meta.get("title", filename)
        status = meta.get("status", "unknown")

        if _is_in_chroma(title):
            status = "success"
            meta_ingested_at = meta.get("ingested_at")
            if not meta_ingested_at:
                meta_ingested_at = None
        else:
            meta_ingested_at = meta.get("ingested_at")

        # Prefer ChromaDB metadata for sidebar labels when manifest still has filename guesses.
        authors = meta.get("authors") or "Unknown Authors"
        year = meta.get("year") or "N/A"
        doi = meta.get("doi") or "N/A"
        chroma_meta = _chroma_meta_for_title(title)
        if chroma_meta:
            if authors in ("Unknown Authors", "", None) and chroma_meta.get("authors"):
                authors = chroma_meta["authors"]
            if year in ("N/A", "None", "", None) and chroma_meta.get("year"):
                year = chroma_meta["year"]
            if doi in ("N/A", "None", "", None) and chroma_meta.get("doi"):
                doi = chroma_meta["doi"]

        file_list.append({
            "filename":    filename,
            "title":       title,
            "doi":         doi,
            "status":      status,
            "ingested_at": meta_ingested_at,
            "size_bytes":  size_bytes,
            "authors":     authors,
            "year":        year,
            "abstract":    meta.get("abstract", ""),
            "paper_id":    meta.get("paper_id", ""),
            # Author, Year label for UI — never a long paper title.
            "sidebar_label": format_sidebar_label(authors, year, title, filename),
        })

    return file_list


@app.post("/api/upload")
async def upload_pdfs(
    files: list[UploadFile] = File(...),
    subfolder: str = "",
    background_tasks: BackgroundTasks = None
):
    """
    Upload one or more PDF files from the user's computer to the server's papers/ directory.
    Optionally saves into a subfolder within papers/.
    After saving, auto-triggers background ingestion of each uploaded file.
    """
    pdf_dir = settings.PDF_DOWNLOAD_DIR

    # Resolve target directory (support optional subfolder)
    if subfolder:
        # Sanitize subfolder to prevent path traversal attacks
        safe_subfolder = re.sub(r"[^a-zA-Z0-9_\-\s/]", "", subfolder).strip("/")
        target_dir = pdf_dir / safe_subfolder
    else:
        target_dir = pdf_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    rejected = []

    for upload in files:
        # Only accept PDF files
        if not upload.filename.lower().endswith(".pdf"):
            rejected.append(upload.filename)
            continue

        # Safe destination path
        dest_path = target_dir / upload.filename

        try:
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            # Relative path from pdf_dir for manifest key
            rel_path = str(dest_path.relative_to(pdf_dir))
            file_size = dest_path.stat().st_size

            # Register as pending in manifest immediately so it shows in the UI
            title = upload.filename.replace("_", " ").replace("-", " ").replace(".pdf", "").title()
            manifest_service.mark_as_ingested(
                rel_path, title, doi=None, status="pending",
                authors="Unknown Authors", year=None
            )

            uploaded.append({"filename": rel_path, "size_bytes": file_size})
            logger.info(f"Uploaded PDF: '{rel_path}' ({file_size} bytes)")
        except Exception as e:
            logger.error(f"Failed to save uploaded file '{upload.filename}': {e}")
            rejected.append(upload.filename)
        finally:
            upload.file.close()

    # Auto-trigger background ingestion for the newly uploaded files
    if uploaded and background_tasks:
        def _ingest_uploaded():
            logger.info(f"Auto-ingesting {len(uploaded)} uploaded PDF(s)...")
            for item in uploaded:
                rel = item["filename"]
                pdf_path = pdf_dir / rel
                if not pdf_path.exists():
                    continue
                try:
                    manifest = manifest_service.get_all_entries()
                    meta = manifest.get(rel, {})
                    title_guess = meta.get("title", pdf_path.stem.replace("_", " ").title())
                    
                    # Resolve metadata
                    resolved = manifest_service.resolve_metadata(pdf_path, title_guess)
                    title = resolved["title"]
                    authors = resolved["authors"]
                    year_str = resolved["year"]
                    doi = resolved["doi"]
                    abstract = resolved.get("abstract", "")
                    
                    # Use metadata_service to get clean metadata from Crossref → OpenAlex → S2 cascade
                    if doi and doi != "N/A":
                        clean_metadata = metadata_service.get_paper_metadata(doi)
                        if clean_metadata:
                            title = clean_metadata.get("title", title)
                            authors = format_authors(clean_metadata.get("authors", [{"name": a} for a in authors.split(" and ")]))
                            year_str = clean_metadata.get("year", year_str)
                            doi = clean_metadata.get("doi", doi)
                            abstract = clean_metadata.get("abstract", abstract)
                            logger.info(f"Enhanced metadata from {clean_metadata.get('source', 'unknown')} for '{title}'")
                    
                    year = int(year_str) if year_str.isdigit() else None

                    # ── Unpaywall recovery for stubs ──
                    has_valid_pdf = False
                    if pdf_path.exists() and pdf_path.stat().st_size > 10240:
                        has_valid_pdf = True

                    if not has_valid_pdf and doi and doi != "N/A":
                        logger.info(f"Uploaded file '{rel}' is missing or a stub. Fetching via Unpaywall...")
                        oa_url = discover_service.fetch_open_access_pdf_url(doi)
                        if oa_url:
                            downloaded = discover_service.download_pdf(oa_url, pdf_path.name)
                            if downloaded and downloaded.exists():
                                has_valid_pdf = True

                    chunks = []
                    if has_valid_pdf and pdf_path.exists():
                        full_text, char_to_page = pdf_service.extract_text_by_page(pdf_path)
                        if len(full_text) < 8000:
                            logger.warning(f"Extracted minimal text from '{rel}' - likely abstract-only or scanned PDF")
                        # Step 6b: Standardize chunk sizes to 2000/400 everywhere
                        chunks = pdf_service.chunk_text(full_text, char_to_page, chunk_size=2000, chunk_overlap=400)

                    # Fallback to abstract-only chunking if PDF couldn't be obtained/parsed but abstract exists
                    if not chunks and abstract:
                        logger.info(f"No PDF text extracted for '{title}'. Falling back to abstract-only.")
                        chunks = [{
                            "chunk_index": 0,
                            "text": abstract,
                            "metadata": {
                                "pages": [0], "char_start": 0,
                                "char_end": len(abstract), "length": len(abstract)
                            }
                        }]
                        has_full_text = False

                    if chunks:
                        success = vector_store.add_paper_chunks(
                            paper_title=title, doi=doi if doi != "N/A" else None, chunks=chunks,
                            authors=authors, year=year, venue=resolved.get("venue")
                        )
                        # Clear stats cache after successful ingestion
                        if success:
                            vector_store.invalidate_stats_cache()
                        manifest_service.mark_as_ingested(
                            rel, title, doi=doi if doi != "N/A" else None,
                            status="success" if success else "failed",
                            authors=authors, year=year, venue=resolved.get("venue"),
                            abstract=abstract, has_full_text=has_full_text
                        )
                        logger.info(f"Auto-ingested '{rel}': {len(chunks)} chunks, success={success}")
                    else:
                        manifest_service.mark_as_ingested(
                            rel, title, doi=doi if doi != "N/A" else None, status="failed",
                            error="No text extracted — may be a scanned/image PDF or missing abstract.",
                            authors=authors, year=year,
                            abstract=abstract
                        )
                        logger.warning(f"No text extracted from uploaded PDF '{rel}'")
                except Exception as e:
                    logger.error(f"Auto-ingest failed for '{rel}': {e}")

        background_tasks.add_task(_ingest_uploaded)

    return {
        "success": len(uploaded) > 0,
        "uploaded": uploaded,
        "rejected": rejected,
        "message": f"{len(uploaded)} file(s) uploaded and queued for ingestion."
    }


@app.post("/api/ingest-pending")
def ingest_pending(background_tasks: BackgroundTasks):
    # Perform directory scan & sync first to find any newly dropped PDFs
    manifest_service.sync_with_vector_store(vector_store)
    # Refresh collection reference after manifest sync to handle potential UUID changes
    vector_store._refresh_collection()

    def _bulk_ingest():
        logger.info("Starting background bulk ingestion...")
        try:
            manifest = manifest_service.get_all_entries()
            pdf_dir  = settings.PDF_DOWNLOAD_DIR

            processed = 0
            succeeded = 0

            for filename, meta in manifest.items():
                if meta.get("status") == "success":
                    continue
                if meta.get('doi_status') == 'unresolvable':
                    continue  # Never retry permanently failed DOIs

                pdf_path = pdf_dir / filename
                if not pdf_path.exists():
                    logger.warning(f"Skipping missing file: {filename}")
                    continue

                logger.info(f"Ingesting pending PDF: {filename}")
                try:
                    title_guess = meta.get("title", pdf_path.stem.replace("_", " ").title())
                    doi_guess = meta.get("doi")
                    
                    # Resolve metadata
                    resolved = manifest_service.resolve_metadata(pdf_path, title_guess, doi_guess)
                    title = resolved["title"]
                    authors = resolved["authors"]
                    year_str = resolved["year"]
                    doi = resolved["doi"]
                    abstract = resolved.get("abstract", "")
                    
                    year = int(year_str) if year_str.isdigit() else None

                    logger.info(f"Extracting text from: {filename}")
                    full_text, char_to_page = pdf_service.extract_text_by_page(pdf_path)
                    if len(full_text) < 8000:
                        logger.warning(f"Extracted minimal text from '{filename}' - likely abstract-only or scanned PDF")
                    logger.info(f"Chunking text from: {filename} ({len(full_text)} chars)")
                    # Step 6b: Standardize chunk sizes to 2000/400 everywhere
                    chunks = pdf_service.chunk_text(full_text, char_to_page, chunk_size=2000, chunk_overlap=400)

                    if not chunks:
                        logger.warning(f"No chunks generated for {filename}, marking as failed")
                        manifest_service.mark_as_ingested(
                            filename, title, doi if doi != "N/A" else None,
                            status="failed", error="No text chunks generated (PDF may be empty or scanned)",
                            authors=authors, year=year, venue=resolved.get("venue"),
                            abstract=abstract
                        )
                        processed += 1
                        continue

                    logger.info(f"Adding chunks to vector store: {filename}")
                    success = vector_store.add_paper_chunks(
                        paper_title=title, doi=doi if doi != "N/A" else None, chunks=chunks,
                        authors=authors, year=year, venue=resolved.get("venue")
                    )
                    # Clear stats cache after successful ingestion
                    if success:
                        vector_store.invalidate_stats_cache()

                    manifest_service.mark_as_ingested(
                        filename, title, doi if doi != "N/A" else None,
                        status="success" if success else "failed",
                        authors=authors, year=year, venue=resolved.get("venue"),
                        abstract=abstract, has_full_text=len(full_text) > 8000
                    )
                    processed += 1
                    if success:
                        succeeded += 1
                    else:
                        logger.error(f"Vector store returned failure for {filename}")

                except Exception as e:
                    logger.error(f"Failed to ingest '{filename}': {e}", exc_info=True)
                    manifest_service.mark_as_ingested(
                        filename, meta.get("title", filename), meta.get("doi"),
                        status="failed", error=str(e),
                        authors=meta.get("authors", "Unknown Authors"), year=meta.get("year"),
                        abstract=meta.get("abstract")
                    )
                    processed += 1
            logger.info(f"Background bulk ingestion complete. Processed: {processed}, Succeeded: {succeeded}")
        except Exception as e:
            logger.error(f"Background bulk ingestion task failed: {e}", exc_info=True)

    background_tasks.add_task(_bulk_ingest)
    return {"success": True, "message": "Bulk ingestion started in the background."}


@app.get("/api/catalog/papers")
def catalog_papers():
    """Deterministic list of ingested papers (no LLM)."""
    stats = vector_store.get_collection_stats()
    meta = stats.get("papers_metadata") or {}
    papers = []
    for title, m in sorted(meta.items()):
        papers.append({
            "title": title,
            "authors": m.get("authors", "Unknown Authors"),
            "year": m.get("year", "N/A"),
            "venue": m.get("venue", "N/A"),
            "doi": m.get("doi", "N/A"),
        })
    return {"count": len(papers), "papers": papers}


@app.get("/api/catalog/authors")
def catalog_authors():
    """Deterministic list of distinct author strings in the library."""
    stats = vector_store.get_collection_stats()
    meta = stats.get("papers_metadata") or {}
    authors = list_distinct_authors(meta)
    return {"count": len(authors), "authors": authors}


@app.get("/api/catalog/search")
def catalog_search(q: str = ""):
    """Search papers/authors in the library by substring (no LLM)."""
    stats = vector_store.get_collection_stats()
    meta = stats.get("papers_metadata") or {}
    needle = (q or "").strip().lower()
    if not needle:
        return {"papers": [], "authors": []}
    papers = []
    for title, m in meta.items():
        blob = f"{title} {m.get('authors', '')} {m.get('venue', '')} {m.get('year', '')}".lower()
        if needle in blob:
            papers.append({
                "title": title,
                "authors": m.get("authors"),
                "year": m.get("year"),
                "venue": m.get("venue"),
            })
    indexes = build_catalog_indexes(meta)
    author_hits = sorted(
        t for t, titles in indexes["author_to_titles"].items() if needle in t
    )
    return {"papers": papers, "author_tokens": author_hits[:50]}


def _rag_failure_phrases() -> tuple[str, ...]:
    return (
        "Off-topic query blocked",
        "No relevant chunks",
        "No matching papers",
        "Entity not in library",
        "Topic not found",
        "Answer failed scope verification",
        "Named author not found",
        "Extraction table too large",
        "Table truncation detected",
    )


@app.post("/api/query-rag")
def query_rag(request: RAGQueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query string must not be empty.")

    try:
        stats = vector_store.get_collection_stats()
        papers_metadata = stats.get("papers_metadata", {}) or {}

        if not papers_metadata:
            return {"answer": EMPTY_DB_REFUSAL, "sources": []}

        # Deterministic catalog answers — no Ollama, no RAGService init.
        catalog_answer = answer_catalog_metadata_query(request.query, papers_metadata)
        if catalog_answer:
            return {"answer": catalog_answer, "sources": []}

        # Step 4b: Metadata interceptor for 'which papers use/mention X' queries
        # Handle queries like "which papers use SHAP" or "which papers mention federated learning"
        # by searching chunk text instead of going through LLM
        q = request.query.lower().strip()
        use_match = re.search(
            r'\b(?:which|what)\s+papers?\s+'
            r'(?:use|mention|discuss|implement|apply|contain|employ)\s+'
            r'(.+?)[\?\.]?$', q
        )
        if use_match:
            term = use_match.group(1).strip()
            matched = []
            for title, meta in papers_metadata.items():
                chunks = vector_store.get_chunks_for_paper(title)
                for chunk in chunks:
                    if term in chunk.get('text', '').lower():
                        matched.append(
                            f"{meta.get('authors','Unknown')} "
                            f"({meta.get('year','N/A')}). {title}"
                        )
                        break
            if matched:
                return {
                    "answer": (
                        f"Papers mentioning '{term}':\n\n" +
                        "\n\n".join(matched)
                    ),
                    "sources": []
                }
            return {
                "answer": (
                    f"No papers in your library explicitly "
                    f"mention '{term}' in their text."
                ),
                "sources": []
            }

        if not check_ollama_health():
            raise HTTPException(
                status_code=503,
                detail="Ollama LLM server is not running. Start with: ollama serve",
            )

        if request.prompt_template and not _should_fallback_to_standard_rag(request):
            return _execute_template_rag(request)

        # Step 1c: Use global RAGService instance instead of per-query instantiation
        if rag_service is None:
            raise HTTPException(
                status_code=503,
                detail="RAG Service not available. Ollama may not be running.",
            )
        result = rag_service.generate_answer(
            request.query,
            limit=request.limit or 15,
            filter_title=request.filter_title or None,
            conversation_history=request.conversation_history or [],
        )

        # Guard against None return
        if result is None:
            result = {
                "answer": "An internal error occurred. Please try again.",
                "sources": [],
                "success": False
            }

        # Fix 3: Stop error messages reaching UI - hide error field, only log it
        if not result or (not result.get("success") and not result.get("answer")):
            display_answer = (
                "I could not find relevant information in your "
                "library for this query. Please try rephrasing or "
                "check that relevant papers are ingested."
            )
        else:
            display_answer = result.get("answer", "")

        # Log the error separately, never send to frontend
        if result and result.get("error"):
            logger.warning(f"Query error: {result['error']}")

        return {
            "answer": display_answer,
            "sources": result.get("sources", []) if result else [],
            "success": result.get("success", False) if result else False
        }

    except OllamaUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("query_rag failed for query: %s", request.query[:120])
        raise HTTPException(
            status_code=500,
            detail=f"RAG request failed: {e}",
        ) from e


@app.get("/api/prompts")
def list_prompts():
    """List all .txt templates in prompts/ with metadata for the UI library."""
    if not PROMPTS_DIR.exists():
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    return [load_prompt_metadata(p) for p in list_prompt_files(PROMPTS_DIR)]


@app.get("/api/prompts/{name}")
def get_prompt(name: str):
    """Return one template's full content for editing in the Prompts tab."""
    from prompt_manager import validate_prompt_name

    try:
        stem = validate_prompt_name(name)
    except PromptValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    path = PROMPTS_DIR / f"{stem}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Template '{stem}' not found.")
    meta = load_prompt_metadata(path)
    _, system_body, user_template = parse_prompt_file(meta["content"])
    meta["system_body"] = system_body
    meta["user_template"] = user_template
    return meta


@app.post("/api/prompts")
def create_or_update_prompt(body: PromptSaveRequest):
    """
    Create or update a prompt template from the in-app editor.
    Files are saved in the canonical ## SYSTEM PROMPT / ## USER PROMPT TEMPLATE format.
    """
    from prompt_manager import validate_prompt_name

    try:
        stem = validate_prompt_name(body.name)
    except PromptValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    target = PROMPTS_DIR / f"{stem}.txt"
    if target.exists() and not body.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Template '{stem}' already exists. Enable overwrite to update.",
        )

    try:
        save_prompt(
            PROMPTS_DIR,
            body.name,
            body.display_title,
            body.system_body,
            body.user_template,
        )
    except PromptValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("Saved prompt template: %s.txt", stem)
    return {"success": True, "name": stem, "message": f"Template '{stem}' saved."}


@app.delete("/api/prompts/{name}")
def remove_prompt(name: str):
    """Delete a user-created template (built-in templates are protected)."""
    try:
        delete_prompt(PROMPTS_DIR, name)
    except PromptValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "deleted": name}


@app.post("/api/analyze-citations")
def start_citation_analysis(request: CitationAnalysisRequest):
    run_id = str(uuid.uuid4())

    citation_jobs[run_id] = {
        "status":   "running",
        "progress": "Initialising citation analysis pipeline...",
        "result":   [],
        "csv_path": None,
        "error":    None
    }

    def _run_analysis():
        try:
            citation_jobs[run_id]["progress"] = (
                "Fetching target paper metadata from Semantic Scholar..."
            )
            analyzer = CitationAnalyzerService()

            citation_jobs[run_id]["progress"] = (
                f"Analysing up to {request.limit} citing papers — "
                "downloading PDFs, extracting passages, classifying with LLM..."
            )
            csv_path = analyzer.analyze_citations(request.paper_id, limit=request.limit)

            if csv_path and csv_path.exists():
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

                citation_jobs[run_id].update({
                    "status":   "completed",
                    "progress": f"Analysis complete — {len(rows)} citation(s) classified.",
                    "result":   rows,
                    "csv_path": csv_path.name
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

    t = threading.Thread(target=_run_analysis, daemon=True)
    t.start()

    logger.info(f"Citation analysis job started — run_id: {run_id}")
    return {"run_id": run_id}


@app.get("/api/analyze-citations/{run_id}")
def get_citation_status(run_id: str):
    if run_id not in citation_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"No citation analysis job found for run_id: {run_id}"
        )
    return citation_jobs[run_id]


@app.get("/api/reports")
def list_reports():
    _purge_old_reports()
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
def download_report(filename: str):
    safe_name = Path(filename).name
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


@app.delete("/api/reports/{filename}")
def delete_report(filename: str):
    safe_name = Path(filename).name
    report_path = REPORTS_DIR / safe_name

    # Idempotent delete: if it's already gone, return success so the UI can refresh cleanly.
    if not report_path.exists():
        return {"success": True, "deleted": safe_name, "already_absent": True}

    try:
        report_path.unlink()
        logger.info(f"Deleted report: {safe_name}")
        return {"success": True, "deleted": safe_name}
    except Exception as e:
        logger.error(f"Failed to delete report '{safe_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {e}")


@app.post("/api/reports/delete")
def delete_report_post(req: DeleteReportRequest):
    safe_name = Path(req.filename).name
    report_path = REPORTS_DIR / safe_name

    # Idempotent delete: if it's already gone, return success so repeated clicks do not fail.
    if not report_path.exists():
        return {"success": True, "deleted": safe_name, "already_absent": True}

    try:
        report_path.unlink()
        logger.info(f"Deleted report via POST: {safe_name}")
        return {"success": True, "deleted": safe_name}
    except Exception as e:
        logger.error(f"Failed to delete report '{safe_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {e}")


# ── Automatic PDF Recovery Background Task ──────────────────────────────────
# Periodically checks for abstract-only papers and tries to recover their PDFs
_pdf_recovery_running = False
_pdf_recovery_interval = 3600  # Check every hour

def _recover_abstract_only_papers():
    """Background task to recover PDFs for abstract-only papers with exponential backoff."""
    global _pdf_recovery_running
    if _pdf_recovery_running:
        logger.info("PDF recovery already running, skipping")
        return

    _pdf_recovery_running = True
    try:
        logger.info("Starting automatic PDF recovery for abstract-only papers")

        vector_store = VectorStoreService()
        discover_service = PaperDiscoveryService()
        pdf_service = PDFProcessorService()
        manifest_service = ManifestManagerService()

        # Get all papers and their chunk counts
        stats = vector_store.get_collection_stats()
        papers_metadata = stats.get("papers_metadata", {})

        # Find papers with < 5 chunks (likely abstract-only)
        from collections import Counter
        data = vector_store.collection.get(include=["metadatas"])
        title_counts = Counter(m.get("title", "Unknown") for m in data["metadatas"] if m)

        abstract_only_papers = {
            title: count for title, count in title_counts.items() if count < 5
        }

        if not abstract_only_papers:
            logger.info("No abstract-only papers found")
            return

        logger.info(f"Found {len(abstract_only_papers)} abstract-only papers to recover")

        # Load manifest to check failure history
        manifest = manifest_service.get_all_entries()

        recovered_count = 0
        skipped_count = 0

        for title, chunk_count in abstract_only_papers.items():
            meta = papers_metadata.get(title, {})
            doi = meta.get("doi", "")

            # Check manifest for exponential backoff
            # Find the manifest entry for this paper (by title match)
            manifest_entry = None
            for filename, entry in manifest.items():
                if entry.get("title", "").lower() == title.lower():
                    manifest_entry = entry
                    break

            if manifest_entry:
                failure_reason = manifest_entry.get("failure_reason", "")
                failure_count = manifest_entry.get("failure_count", 0)

                # Skip papers with "no_oa_version" after 3 failures (retry weekly)
                if failure_reason == "no_oa_version" and failure_count >= 3:
                    logger.info(f"Skipping '{title}' - no OA version found after {failure_count} attempts (will retry weekly)")
                    skipped_count += 1
                    continue

                # Skip papers with "blocked_403" after 3 failures (retry daily, not hourly)
                if failure_reason == "blocked_403" and failure_count >= 3:
                    # Check if enough time has passed (24 hours)
                    import time
                    ingested_at = manifest_entry.get("ingested_at", "")
                    if ingested_at:
                        try:
                            from datetime import datetime
                            last_attempt = datetime.fromisoformat(ingested_at)
                            hours_since = (datetime.now() - last_attempt).total_seconds() / 3600
                            if hours_since < 24:
                                logger.info(f"Skipping '{title}' - blocked 403, retrying in {24 - hours_since:.1f} hours")
                                skipped_count += 1
                                continue
                        except Exception:
                            pass

            if doi and doi.lower() != "n/a":
                doi = doi.replace("https://doi.org/", "").strip()

                logger.info(f"Attempting PDF recovery for '{title}' (DOI: {doi})")

                # Use cascade of sources: ArXiv → Unpaywall → Core.ac.uk → MDPI API → PMC E-utilities → OpenAlex
                pdf_url = None
                source_used = None

                # 1. Try ArXiv first (never blocked, best for CS/ML papers)
                pdf_url = discover_service.fetch_arxiv_pdf_url(doi)
                if pdf_url:
                    source_used = "ArXiv"
                    logger.info(f"Found via ArXiv: {pdf_url[:80]}...")

                # 2. Try Unpaywall
                if not pdf_url:
                    unpaywall_urls = discover_service.fetch_all_open_access_pdf_urls(doi)
                    if unpaywall_urls:
                        pdf_url = unpaywall_urls[0]
                        source_used = "Unpaywall"
                        logger.info(f"Found via Unpaywall: {pdf_url[:80]}...")

                # 3. Try Core.ac.uk (by title)
                if not pdf_url:
                    pdf_url = discover_service.fetch_core_ac_pdf_url(title)
                    if pdf_url:
                        source_used = "Core.ac.uk"
                        logger.info(f"Found via Core.ac.uk: {pdf_url[:80]}...")

                # 4. Try MDPI research API (for MDPI DOIs)
                if not pdf_url:
                    pdf_url = discover_service.fetch_mdpi_api_pdf_url(doi)
                    if pdf_url:
                        source_used = "MDPI API"
                        logger.info(f"Found via MDPI API: {pdf_url[:80]}...")

                # 5. Try PMC E-utilities (if PMCID available from OpenAlex)
                if not pdf_url:
                    openalex_data = discover_service.fetch_openalex_metadata(doi)
                    if openalex_data:
                        ids = openalex_data.get("ids", {})
                        pmcid = ids.get("pmcid") if ids else None
                        if pmcid:
                            pdf_url = discover_service.fetch_pmc_eutils_pdf_url(pmcid)
                            if pdf_url:
                                source_used = "PMC E-utilities"
                                logger.info(f"Found via PMC E-utilities: {pdf_url[:80]}...")

                # 6. Try OpenAlex PDF URLs as final fallback
                if not pdf_url:
                    openalex_urls = discover_service.fetch_all_openalex_pdf_urls(doi)
                    if openalex_urls:
                        pdf_url = openalex_urls[0]
                        source_used = "OpenAlex"
                        logger.info(f"Found via OpenAlex: {pdf_url[:80]}...")

                pdf_path = None
                failure_reason = None

                if pdf_url:
                    # Download PDF
                    safe_name = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_name = safe_name[:50]
                    logger.info(f"Attempting download from candidate URL: {pdf_url}")
                    pdf_path = discover_service.download_pdf(pdf_url, f"{safe_name}.pdf")

                    if pdf_path and pdf_path.exists():
                        logger.info(f"Successfully downloaded PDF from: {pdf_url} (source: {source_used})")
                    else:
                        # Determine failure reason based on URL and response
                        if "403" in str(pdf_path) or "blocked" in str(pdf_path).lower():
                            failure_reason = "blocked_403"
                        elif "html" in str(pdf_path).lower():
                            failure_reason = "html_response"
                        else:
                            failure_reason = "download_failed"
                        logger.warning(f"Failed to download PDF for '{title}' (reason: {failure_reason})")
                else:
                    failure_reason = "no_oa_version"
                    logger.info(f"No OA PDF found for '{title}' from any source")

                if pdf_path and pdf_path.exists():
                    # Extract and chunk
                    full_text, char_to_page = pdf_service.extract_text_by_page(pdf_path)
                    if len(full_text) >= 8000:
                        chunks = pdf_service.chunk_text(full_text, char_to_page, chunk_size=2000, chunk_overlap=400)

                        # First delete old abstract-only chunks
                        vector_store.collection.delete(where={"title": title})

                        # Ingest full-text chunks (so paper is always queryable)
                        authors = meta.get("authors", "Unknown Authors")
                        year = meta.get("year")
                        venue = meta.get("venue")

                        vector_store.add_paper_chunks(
                            paper_title=title,
                            doi=doi,
                            chunks=chunks,
                            authors=authors,
                            year=year,
                            venue=venue,
                        )

                        logger.info(f"Successfully recovered and re-ingested '{title}' with {len(chunks)} chunks")
                        recovered_count += 1

                        # Update manifest with success
                        safe_filename = f"{safe_name}.pdf"
                        manifest_service.mark_as_ingested(
                            safe_filename,
                            title,
                            doi=doi,
                            status="success",
                            authors=authors,
                            year=year,
                            venue=venue,
                            has_full_text=True,
                            failure_reason=None
                        )
                    else:
                        logger.warning(f"PDF for '{title}' has minimal text, skipping")
                        failure_reason = "scanned_pdf"
                else:
                    # Update manifest with failure
                    safe_filename = f"{safe_name}.pdf" if 'safe_name' in locals() else f"{title[:50]}.pdf"
                    manifest_service.mark_as_ingested(
                        safe_filename,
                        title,
                        doi=doi,
                        status="failed",
                        error=f"PDF download failed",
                        failure_reason=failure_reason
                    )

        logger.info(f"PDF recovery complete: recovered {recovered_count}/{len(abstract_only_papers)} papers, skipped {skipped_count} (exponential backoff)")

    except Exception as e:
        logger.error(f"PDF recovery failed: {e}")
    finally:
        _pdf_recovery_running = False


def _start_pdf_recovery_scheduler():
    """Start the background scheduler for PDF recovery."""
    def _scheduler():
        while True:
            try:
                _recover_abstract_only_papers()
                time.sleep(_pdf_recovery_interval)
            except Exception as e:
                logger.error(f"PDF recovery scheduler error: {e}")
                time.sleep(_pdf_recovery_interval)
    
    t = threading.Thread(target=_scheduler, daemon=True)
    t.start()
    logger.info("PDF recovery scheduler started (runs every hour)")


# Start the PDF recovery scheduler when the server starts
@app.on_event("startup")
def startup_event():
    _start_pdf_recovery_scheduler()
    logger.info("Server started with automatic PDF recovery enabled")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)]
    )
