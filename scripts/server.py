"""
server.py — FastAPI Web Server for the AI Research Stack.
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
import requests as _req

from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, Request, UploadFile, File
from fastapi.websockets import WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
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
from rag_service import RAGService, check_ollama_health
from manifest_manager import ManifestManagerService
from citation_analyzer import CitationAnalyzerService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ai_research_server")

logger.info("Initialising all backend services...")
discover_service = PaperDiscoveryService()
pdf_service      = PDFProcessorService()
vector_store     = VectorStoreService()
manifest_service = ManifestManagerService()
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
# Required for Cloudflare Tunnel access. Cloudflare sends WebSocket probes and
# requests with external hostnames; without these middlewares the app returns
# 403 Forbidden and Cloudflare reports "Unable to connect to origin".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# ── HTTP Basic Authentication Middleware ──────────────────────────────────────
@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    # Bypass public routes
    if request.url.path in ["/api/health", "/sw.js", "/service-worker.js"]:
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Basic "):
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": "Basic realm='AI Research Stack'"},
            content="Unauthorized Access"
        )

    try:
        payload = auth_header.split(" ")[1]
        decoded = base64.b64decode(payload).decode("utf-8")
        username, password = decoded.split(":", 1)
        if username == "admin" and password == "aitawfiq2026":
            return await call_next(request)
    except Exception:
        pass

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": "Basic realm='AI Research Stack'"},
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


class RAGQueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    prompt_template: Optional[str] = None
    filter_title: Optional[str] = None  # If set, restrict RAG to chunks from this paper only


class CitationAnalysisRequest(BaseModel):
    paper_id: str
    limit: Optional[int] = 5


def sanitize_filename(title: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-\s]", "", title)
    clean = clean.replace(" ", "_")
    clean = re.sub(r"_{2,}", "_", clean)
    return clean.strip("_")[:60].lower() + ".pdf"


def format_authors(authors: list) -> str:
    if not authors:
        return "Unknown Authors"
    names = [a.get("name", "") for a in authors if a.get("name")]
    if len(names) > 3:
        return ", ".join(names[:3]) + " et al."
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
def search_papers(q: str, limit: int = 10, offset: int = 0):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string 'q' is required.")

    results = discover_service.search_papers(q.strip(), limit=limit, offset=offset)
    formatted = []

    for paper in results:
        external_ids = paper.get("externalIds") or {}
        doi   = external_ids.get("DOI")   or "N/A"
        arxiv = external_ids.get("ArXiv") or "N/A"
        has_pdf = False
        if arxiv != "N/A":
            has_pdf = True
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
            "abstract":      (paper.get("abstract") or "")[:500],
            "citationCount": paper.get("citationCount", 0),
            "has_pdf":       has_pdf
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
        logger.info(f"BG Ingest started: '{title}'")
        chunks = []

        pdf_url = None
        if doi:
            pdf_url = discover_service.fetch_open_access_pdf_url(doi)

        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        if pdf_url:
            safe_name = sanitize_filename(title)
            pdf_path  = discover_service.download_pdf(pdf_url, safe_name)

            if pdf_path and pdf_path.exists():
                try:
                    pages  = pdf_service.extract_text_by_page(pdf_path)
                    chunks = pdf_service.chunk_text(pages, chunk_size=1000, chunk_overlap=200)
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
                authors=authors_str, year=year
            )
            manifest_service.mark_as_ingested(
                sanitize_filename(title), title, doi, status="success",
                authors=authors_str, year=year, abstract=abstract
            )
            return

        if chunks:
            identifier = doi or (f"arXiv:{arxiv_id}" if arxiv_id else title)
            success = vector_store.add_paper_chunks(
                paper_title=title, doi=identifier, chunks=chunks,
                authors=authors_str, year=year
            )
            filename = sanitize_filename(title)
            manifest_service.mark_as_ingested(
                filename, title, doi,
                status="success" if success else "failed",
                authors=authors_str, year=year, abstract=request.abstract
            )
            logger.info(
                f"BG Ingest complete for '{title}': "
                f"{len(chunks)} chunks, success={success}"
            )
        else:
            manifest_service.mark_as_ingested(
                sanitize_filename(title), title, doi, status="failed",
                error="No PDF and no abstract available.",
                authors=authors_str, year=year, abstract=request.abstract
            )
            logger.warning(f"Ingestion failed for '{title}': no content could be obtained.")

    background_tasks.add_task(_ingest)
    return {"success": True, "message": f"Ingestion started for: {title}", "mode": "pdf" if request.externalIds else "abstract"}


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


@app.get("/api/pdfs")
def list_pdfs():
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

        file_list.append({
            "filename":    filename,
            "title":       title,
            "doi":         meta.get("doi", "N/A"),
            "status":      status,
            "ingested_at": meta_ingested_at,
            "size_bytes":  size_bytes,
            "authors":     meta.get("authors", "Unknown Authors"),
            "year":        meta.get("year", "N/A"),
            "abstract":    meta.get("abstract", "")
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
                    
                    year = int(year_str) if year_str.isdigit() else None
                    
                    pages = pdf_service.extract_text_by_page(pdf_path)
                    chunks = pdf_service.chunk_text(pages, chunk_size=1000, chunk_overlap=200)
                    if chunks:
                        success = vector_store.add_paper_chunks(
                            paper_title=title, doi=doi if doi != "N/A" else None, chunks=chunks,
                            authors=authors, year=year
                        )
                        manifest_service.mark_as_ingested(
                            rel, title, doi=doi if doi != "N/A" else None,
                            status="success" if success else "failed",
                            authors=authors, year=year,
                            abstract=abstract
                        )
                        logger.info(f"Auto-ingested '{rel}': {len(chunks)} chunks, success={success}")
                    else:
                        manifest_service.mark_as_ingested(
                            rel, title, doi=doi if doi != "N/A" else None, status="failed",
                            error="No text extracted — may be a scanned/image PDF.",
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

    def _bulk_ingest():
        logger.info("Starting background bulk ingestion...")
        manifest = manifest_service.get_all_entries()
        pdf_dir  = settings.PDF_DOWNLOAD_DIR

        processed = 0
        succeeded = 0

        for filename, meta in manifest.items():
            if meta.get("status") == "success":
                continue

            pdf_path = pdf_dir / filename
            if not pdf_path.exists():
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
                
                pages  = pdf_service.extract_text_by_page(pdf_path)
                chunks = pdf_service.chunk_text(pages, chunk_size=1000, chunk_overlap=200)

                success = vector_store.add_paper_chunks(
                    paper_title=title, doi=doi if doi != "N/A" else None, chunks=chunks,
                    authors=authors, year=year
                )

                manifest_service.mark_as_ingested(
                    filename, title, doi if doi != "N/A" else None,
                    status="success" if success else "failed",
                    authors=authors, year=year,
                    abstract=abstract
                )
                processed += 1
                if success:
                    succeeded += 1

            except Exception as e:
                logger.error(f"Failed to ingest '{filename}': {e}")
                manifest_service.mark_as_ingested(
                    filename, meta.get("title", filename), meta.get("doi"),
                    status="failed", error=str(e),
                    authors=meta.get("authors", "Unknown Authors"), year=meta.get("year"),
                    abstract=meta.get("abstract")
                )
                processed += 1
        logger.info(f"Background bulk ingestion complete. Processed: {processed}, Succeeded: {succeeded}")

    background_tasks.add_task(_bulk_ingest)
    return {"success": True, "message": "Bulk ingestion started in the background."}


@app.post("/api/query-rag")
def query_rag(request: RAGQueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query string must not be empty.")

    if not check_ollama_health():
        raise HTTPException(
            status_code=503,
            detail="Ollama LLM server is not running. Start with: ollama serve"
        )

    if request.prompt_template:
        prompt_path = PROMPTS_DIR / f"{request.prompt_template}.txt"
        if not prompt_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Prompt template '{request.prompt_template}' not found in prompts/."
            )

        raw_template = prompt_path.read_text(encoding="utf-8").strip()
        divider      = "## USER PROMPT TEMPLATE"

        if divider in raw_template:
            parts         = raw_template.split(divider, 1)
            system_prompt = parts[0].replace("## SYSTEM PROMPT", "").strip()
            user_template = parts[1].strip()
        else:
            system_prompt = raw_template
            user_template = "{context}"

        # Fetch library index to support meta-queries
        stats = vector_store.get_collection_stats()
        papers_metadata = stats.get("papers_metadata", {})

        chunks = vector_store.query_similar_chunks(
            request.query,
            limit=request.limit,
            filter_title=request.filter_title or None
        )
        if not chunks and not papers_metadata:
            raise HTTPException(
                status_code=404,
                detail="No relevant papers found in the database. Ingest papers first."
            )

        # Format Library Inventory string
        library_inventory_blocks = []
        for i, (title, meta) in enumerate(papers_metadata.items()):
            library_inventory_blocks.append(
                f"- {meta.get('authors', 'Unknown Authors')} ({meta.get('year', 'N/A')}). \"{title}\". DOI: {meta.get('doi', 'N/A')}"
            )
        library_inventory_str = "\n".join(library_inventory_blocks) if library_inventory_blocks else "No papers in database library."

        context_blocks = []
        for idx, c in enumerate(chunks):
            meta = c["metadata"]
            title = meta.get("title", "Untitled Paper")
            authors = meta.get("authors", "Unknown Authors")
            year = meta.get("year", "N/A")
            doi = meta.get("doi", "N/A")
            pages = meta.get("pages", "N/A")
            text = c["text"]

            block = (
                f'--- Academic Source ({authors}, {year}) ---\n'
                f'Title: "{title}"\n'
                f'DOI: {doi}\n'
                f'Pages: {pages}\n'
                f'Content: {text}\n'
            )
            context_blocks.append(block)

        # Combine library inventory with passage chunks context
        context_str = f"Ingested Paper Library Inventory:\n{library_inventory_str}\n\n"
        if context_blocks:
            context_str += "Context Chunks:\n" + "\n\n".join(context_blocks)
        else:
            context_str += "No relevant text passage chunks found for this query."

        titles       = {c["metadata"].get("title", "") for c in chunks}
        title_str    = " | ".join(sorted(titles)) if titles else "None"
        authors_list = [c["metadata"].get("authors", "Unknown Authors") for c in chunks]
        authors_str  = " | ".join(sorted(set(authors_list))) if authors_list else "None"
        years_list   = [str(c["metadata"].get("year", "N/A")) for c in chunks]
        years_str    = " | ".join(sorted(set(years_list))) if years_list else "None"

        user_prompt = (
            user_template
            .replace("{context}",  context_str)
            .replace("{context_a}", context_str)
            .replace("{context_b}", context_str)
            .replace("{title}",   title_str)
            .replace("{title_a}", title_str)
            .replace("{title_b}", title_str)
            .replace("{authors}", authors_str)
            .replace("{year}",    years_str)
            .replace("{venue}",   "Various")
        )

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

    rag_service = RAGService()
    result      = rag_service.generate_answer(
        request.query,
        limit=request.limit,
        filter_title=request.filter_title or None
    )

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "RAG failed."))

    return {"answer": result["answer"], "sources": result["sources"]}


@app.get("/api/prompts")
def list_prompts():
    prompts = []
    if not PROMPTS_DIR.exists():
        return []

    for prompt_file in sorted(PROMPTS_DIR.glob("*.txt")):
        content = prompt_file.read_text(encoding="utf-8").strip()
        lines   = [l for l in content.split("\n") if l.strip()]

        raw_title = lines[0] if lines else prompt_file.stem
        title = re.sub(r"^#+\s*", "", raw_title).strip()
        title = re.sub(r"^SYSTEM PROMPT\s*[—\-:]*\s*", "", title, flags=re.IGNORECASE).strip()

        desc_lines = [
            l for l in lines[1:]
            if l.strip() and not l.startswith("#") and not l.startswith("---")
        ]
        description = desc_lines[0].strip() if desc_lines else f"Prompt template: {prompt_file.stem}"

        prompts.append({
            "name":        prompt_file.stem,
            "title":       title,
            "description": description[:200],
            "content":     content
        })

    return prompts


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

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report '{safe_name}' not found in output/."
        )

    try:
        report_path.unlink()
        logger.info(f"Deleted report: {safe_name}")
        return {"success": True, "deleted": safe_name}
    except Exception as e:
        logger.error(f"Failed to delete report '{safe_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)]
    )
