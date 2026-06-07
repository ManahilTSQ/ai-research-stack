"""
main.py — CLI Entry Point for the AI Research Stack.

Provides a command-line interface (CLI) for all core operations:

  --query / -q        : Search Semantic Scholar and auto-ingest the top result
  --interactive / -i  : Launch an interactive paper selection wizard
  --ingest-all / -g   : Batch-ingest all PDFs in the papers/ directory
  --query-rag / -r    : Ask a research question across ingested papers (RAG)
  --prompt / -p       : Apply a prompt template to a RAG query
  --analyze-citations / -a : Run citation intent analysis for a paper DOI / ID

Usage (from the AI Research Stack/ project root):
  python scripts/main.py --interactive
  python scripts/main.py -q "attention mechanism transformers"
  python scripts/main.py -r "What is multi-head attention?" -p summarize
  python scripts/main.py -a "10.48550/arXiv.1706.03762" -l 5
"""

import sys
import argparse
import logging
import re
import requests as _requests  # Aliased to avoid name collision with local 'requests' usage
from pathlib import Path

# ── Ensure scripts/ directory is on sys.path ──────────────────────────────────
# When running as `python scripts/main.py` from the project root, Python
# adds scripts/ to sys.path automatically (it adds the script's directory).
# This explicit insert makes the flat imports below robust for edge cases
# (e.g. being called from a different working directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Flat imports — all service modules are siblings in scripts/
from config import settings
from paper_discovery import PaperDiscoveryService
from pdf_processor import PDFProcessorService
from vector_store import VectorStoreService

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
# PowerShell on Windows defaults to cp1252 encoding which cannot display
# Unicode characters (e.g. emoji like ✓ ✗ ⚠). Reconfigure stdout to UTF-8.
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Python < 3.7 does not have reconfigure()

# ── Logging setup ─────────────────────────────────────────────────────────────
# Logs go to stdout so they are visible in the terminal alongside print() output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ai_research_cli")


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def sanitize_filename(title: str) -> str:
    """
    Convert a paper title into a safe, concise PDF filename.

    Replaces all characters that are illegal on Windows/Linux filesystems
    (anything that isn't alphanumeric, space, dash, or underscore) with
    nothing, then converts spaces to underscores and truncates to 60 chars.

    Args:
        title: Raw paper title string.

    Returns:
        A lowercase .pdf filename safe for all operating systems.
    """
    # Keep only alphanumeric characters, spaces, dashes, underscores
    clean = re.sub(r"[^a-zA-Z0-9_\-\s]", "", title)
    # Replace spaces with underscores for readability
    clean = clean.replace(" ", "_")
    # Collapse multiple consecutive underscores (from removed chars)
    clean = re.sub(r"_{2,}", "_", clean)
    # Truncate to 60 chars and append .pdf extension
    return clean.strip("_")[:60].lower() + ".pdf"


def format_authors(authors: list) -> str:
    """
    Format a list of author dicts into a readable string.

    Shows up to 3 author names; if there are more, appends " et al.".

    Args:
        authors: List of author dicts with a "name" key.

    Returns:
        Formatted author string, e.g. "Vaswani, Shazeer, Parmar et al."
    """
    if not authors:
        return "Unknown Authors"
    names = [a.get("name", "") for a in authors if a.get("name")]
    return ", ".join(names)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE: Download → Extract → Chunk → Ingest
# ──────────────────────────────────────────────────────────────────────────────

def process_paper_pipeline(
    paper: dict,
    discover_service: PaperDiscoveryService,
    pdf_service: PDFProcessorService,
    vector_store: VectorStoreService,
    chunk_size: int,
    chunk_overlap: int
) -> bool:
    """
    Execute the 3-tier download + ingestion pipeline for a single paper.

    Tier 1 — Unpaywall OA lookup (requires DOI):
        Queries Unpaywall for a legal open-access PDF URL.

    Tier 2 — arXiv direct PDF (requires arXiv ID):
        Falls back to the free arXiv PDF if Unpaywall has no record.

    Tier 3 — Abstract-only ingestion (last resort):
        If no PDF can be obtained, ingests the Semantic Scholar abstract
        so the paper is at least searchable in the vector database.

    Args:
        paper: Paper metadata dict from Semantic Scholar.
        discover_service: PaperDiscoveryService instance.
        pdf_service: PDFProcessorService instance.
        vector_store: VectorStoreService instance.
        chunk_size: Character length of each text chunk.
        chunk_overlap: Overlap characters between adjacent chunks.

    Returns:
        True on successful ingestion, False on complete failure.
    """
    # Extract the fields we need from the paper metadata dict
    title = paper.get("title", "Untitled Paper")
    authors_list = paper.get("authors", [])
    authors_str  = format_authors(authors_list)
    year         = paper.get("year")
    venue        = paper.get("venue") or paper.get("publicationVenue", {}).get("name") or None
    external_ids = paper.get("externalIds") or {}
    doi      = external_ids.get("DOI")       # Used for Unpaywall lookup
    arxiv_id = external_ids.get("ArXiv")     # Used for direct arXiv PDF
    abstract = (paper.get("abstract") or "").strip()  # Used as last-resort content

    print("\n" + "=" * 80)
    print(f"STARTING INGESTION PIPELINE: {title}")
    print("=" * 80)

    # ── Tier 1: Unpaywall open-access PDF resolution ───────────────────────────
    pdf_url = None
    if doi:
        print(f"[1/3] Querying Unpaywall for open-access PDF (DOI: {doi})...")
        pdf_url = discover_service.fetch_open_access_pdf_url(doi)
        if pdf_url:
            print(f"[+] Found open-access PDF: {pdf_url}")
        else:
            print(f"[-] No open-access PDF found on Unpaywall for DOI: {doi}")
    else:
        print("[1/3] No DOI available — skipping Unpaywall lookup.")

    # ── Tier 2: arXiv direct PDF download ─────────────────────────────────────
    if not pdf_url and arxiv_id:
        # arXiv provides free PDFs for all its preprints at a predictable URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"[2/3] Falling back to arXiv direct PDF: {pdf_url}")
    elif not pdf_url:
        print("[2/3] No arXiv ID available — skipping direct PDF fallback.")

    # ── Attempt PDF download if we have a URL ─────────────────────────────────
    downloaded_path = None
    if pdf_url:
        safe_filename = sanitize_filename(title)
        print(f"[3/3] Downloading PDF → {safe_filename}...")
        downloaded_path = discover_service.download_pdf(pdf_url, safe_filename)

    # ── Full-text ingestion path (PDF successfully downloaded) ─────────────────
    if downloaded_path and downloaded_path.exists():
        print(f"[+] PDF downloaded successfully: {downloaded_path}")

        print("[4/5] Extracting text with PyMuPDF...")
        try:
            full_text, char_to_page = pdf_service.extract_text_by_page(downloaded_path)
            total_chars = len(full_text)
            total_pages = max(char_to_page) + 1 if char_to_page else 0
            print(f"[+] Extracted text from {total_pages} pages ({total_chars:,} chars)")
            if len(full_text) < 8000:
                print(f"[!] Warning: Extracted minimal text - likely abstract-only or scanned PDF")
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return False

        print(f"[5/5] Chunking text (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        try:
            chunks = pdf_service.chunk_text(full_text, char_to_page, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            print(f"[+] Generated {len(chunks)} chunks.")
            print("[6/6] Ingesting into ChromaDB vector database...")
            success = vector_store.add_paper_chunks(
                paper_title=title,
                doi=doi,
                chunks=chunks,
                authors=authors_str,
                year=year,
                venue=venue,
            )

            if success:
                print("[+] Vector DB ingestion complete!")
            else:
                print("[-] Warning: ChromaDB ingestion encountered an issue.")

            # Determine the PDF source label for the summary display
            if pdf_url and "arxiv.org" in pdf_url:
                source_label = f"arXiv PDF (arXiv:{arxiv_id})"
            else:
                source_label = f"Unpaywall OA PDF (DOI:{doi})"

            # Print a concise pipeline success summary
            print("\n" + "-" * 40 + " PIPELINE SUMMARY " + "-" * 40)
            print(f"  Title:            {title}")
            print(f"  Source:           {source_label}")
            print(f"  Local Path:       {downloaded_path}")
            print(f"  Total Pages:      {total_pages}")
            print(f"  Total Characters: {total_chars:,}")
            print(f"  Total Chunks:     {len(chunks)}")
            print(f"  Ingested:         {'YES ✓' if success else 'NO ✗'}")
            print("-" * 98)

            # Show a preview of the first 2 chunks for verification
            if chunks:
                preview_count = min(2, len(chunks))
                print(f"\n--- FIRST {preview_count} CHUNK PREVIEW ---")
                for i in range(preview_count):
                    chunk = chunks[i]
                    meta = chunk["metadata"]
                    print(
                        f"\n[Chunk {chunk['chunk_index']} | "
                        f"Pages {meta['pages']} | "
                        f"Chars {meta['char_start']}–{meta['char_end']}]"
                    )
                    preview = chunk["text"]
                    if len(preview) > 350:
                        preview = preview[:350] + "\n... [TRUNCATED] ..."
                    print(f'  "{preview}"')
                    print("-" * 50)

            return success

        except Exception as e:
            logger.error(f"Chunking/ingestion failed: {e}")
            return False

    # ── Tier 3: Abstract-only ingestion (last resort) ─────────────────────────
    # If we reach here, no PDF was available. Ingest the paper's abstract
    # so it is at least findable via semantic search in the vector database.
    if abstract:
        print(f"\n⚠  ABSTRACT FALLBACK: No PDF available for '{title}'.")
        print("   Ingesting the Semantic Scholar abstract as a single chunk...")

        # Wrap the abstract in the standard chunk format expected by vector_store
        abstract_chunks = [{
            "chunk_index": 0,
            "text": abstract,
            "metadata": {
                "pages": [0],           # Page 0 signifies abstract-only content
                "char_start": 0,
                "char_end": len(abstract),
                "length": len(abstract)
            }
        }]

        # Use the best available identifier for the vector ID
        identifier = doi or (f"arXiv:{arxiv_id}" if arxiv_id else title)
        success = vector_store.add_paper_chunks(
            paper_title=title,
            doi=identifier,
            chunks=abstract_chunks,
            authors=authors_str,
            year=year,
            venue=venue,
        )

        if success:
            print(f"[+] Abstract ingested ({len(abstract):,} chars). Paper is searchable.")
        else:
            print("[-] Abstract ingestion failed.")
        return success

    # Complete failure — no PDF and no abstract
    print(f"[-] Could not ingest '{title}': no PDF, no arXiv ID, and no abstract available.")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ──────────────────────────────────────────────────────────────────────────────

def run_interactive_mode(
    discover_service: PaperDiscoveryService,
    pdf_service: PDFProcessorService,
    vector_store: VectorStoreService,
    chunk_size: int,
    chunk_overlap: int
):
    """
    Launch an interactive terminal wizard for searching and ingesting papers.

    The wizard loops indefinitely, prompting the user to enter search keywords,
    displaying the top 5 results, and allowing the user to select one to ingest.
    Type 'q' at the search prompt to exit.
    """
    print("\n" + "*" * 80)
    print("  AI Research Stack — Interactive CLI Wizard")
    print("*" * 80)

    while True:
        # Prompt for a search query (or 'q' to quit)
        query = input("\nEnter search keywords (or 'q' to quit): ").strip()
        if not query:
            continue  # Ignore empty input — re-prompt
        if query.lower() == "q":
            print("Exiting interactive mode. Goodbye!")
            break

        print("\nSearching Semantic Scholar...")
        papers = discover_service.search_papers(query, limit=5)

        if not papers:
            print("No papers found. Please try different keywords.")
            continue

        # Display the search results with 1-indexed selection numbers
        print(f"\nFound {len(papers)} matching papers:")
        for idx, paper in enumerate(papers):
            title       = paper.get("title", "Untitled")
            authors_str = format_authors(paper.get("authors", []))
            year        = paper.get("year", "N/A")
            venue       = paper.get("venue", "N/A")
            print(f"\n[{idx + 1}] \"{title}\"")
            print(f"    Authors: {authors_str} | Year: {year} | Venue: {venue}")
            print("-" * 60)

        # Prompt the user to select a paper to ingest
        selection = input("\nSelect a paper number to ingest (or 'c' to cancel): ").strip()
        if selection.lower() == "c":
            continue  # Go back to the search prompt

        try:
            sel_idx = int(selection) - 1  # Convert to 0-indexed
            if 0 <= sel_idx < len(papers):
                selected = papers[sel_idx]
                process_paper_pipeline(
                    selected, discover_service, pdf_service, vector_store,
                    chunk_size, chunk_overlap
                )
            else:
                print("[-] Selection out of range. Please enter a valid number.")
        except ValueError:
            print("[-] Invalid input. Please enter a number or 'c' to cancel.")


# ──────────────────────────────────────────────────────────────────────────────
# BATCH INGESTION
# ──────────────────────────────────────────────────────────────────────────────

def run_batch_ingestion(
    pdf_service: PDFProcessorService,
    vector_store: VectorStoreService,
    chunk_size: int,
    chunk_overlap: int
):
    """
    Scan the papers/ directory and ingest every PDF found into ChromaDB.

    This is used when the user manually drops PDFs into papers/ and wants to
    ingest them all without searching Semantic Scholar first. The paper title
    is derived from the filename (underscores → spaces, title-cased).

    Args:
        pdf_service: PDFProcessorService instance.
        vector_store: VectorStoreService instance.
        chunk_size: Chunk character length.
        chunk_overlap: Chunk overlap in characters.
    """
    pdf_dir = settings.PDF_DOWNLOAD_DIR  # papers/ directory
    print(f"\nScanning papers/ directory: {pdf_dir}")

    pdf_files = [p for p in pdf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdf_files:
        print("[-] No PDF files found. Drop PDFs into papers/ and try again.")
        return

    print(f"[+] Found {len(pdf_files)} PDF(s). Starting batch ingestion...")

    success_count = 0
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        try:
            # Extract text and create chunks from this PDF
            full_text, char_to_page = pdf_service.extract_text_by_page(pdf_path)
            if len(full_text) < 8000:
                print(f"[!] Warning: Extracted minimal text - likely abstract-only or scanned PDF")
            chunks = pdf_service.chunk_text(full_text, char_to_page, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Derive a human-readable title from the filename
            # e.g. "attention_is_all_you_need.pdf" → "Attention Is All You Need"
            title = pdf_path.stem.replace("_", " ").title()
            doi   = None  # No DOI available for manually dropped PDFs
            authors_str = "Unknown Authors"  # No author metadata for manually dropped PDFs
            year = None
            venue = None

            success = vector_store.add_paper_chunks(
                paper_title=title,
                doi=doi,
                chunks=chunks,
                authors=authors_str,
                year=year,
                venue=venue,
            )
            if success:
                success_count += 1
                print(f"[+] Ingested: '{title}'")
        except Exception as e:
            print(f"[-] Failed to process {pdf_path.name}: {e}")

    print(f"\nBatch ingestion complete: {success_count}/{len(pdf_files)} papers ingested.")

    # Show updated database statistics
    stats = vector_store.get_collection_stats()
    print(
        f"Database stats: {stats['total_chunks']} chunks "
        f"from {stats['total_papers']} papers."
    )


# ──────────────────────────────────────────────────────────────────────────────
# RAG QUERY
# ──────────────────────────────────────────────────────────────────────────────

def run_rag_query(query: str):
    """
    Execute a standard RAG query and print the grounded answer with sources.

    Initialises the RAGService (which in turn checks Ollama health),
    queries ChromaDB for relevant chunks, sends them to Ollama, and prints
    the grounded academic answer with inline source citations.

    Args:
        query: The research question to answer.
    """
    from rag_service import RAGService  # Import here to avoid circular dependency at module level

    print(f"\nExecuting RAG Query: '{query}'")
    print("Generating grounded academic answer — please wait...\n")

    try:
        rag_service = RAGService()
        result = rag_service.generate_answer(query)

        if result["success"]:
            print("=" * 45 + " RAG ANSWER " + "=" * 45)
            print(result["answer"])
            print("=" * 102)

            # Print deduplicated source list with page references
            print("\n--- RETRIEVED SOURCES ---")
            unique_sources = {}
            for chunk in result["sources"]:
                meta  = chunk["metadata"]
                title = meta.get("title", "Untitled")
                pages = meta.get("pages", "N/A")
                if title not in unique_sources:
                    unique_sources[title] = set()
                for p in pages.split(","):
                    if p.strip():
                        unique_sources[title].add(p.strip())

            for idx, (title, page_set) in enumerate(unique_sources.items()):
                # Sort page numbers numerically for readability
                pages_sorted = sorted(page_set, key=lambda x: int(x) if x.isdigit() else 999)
                print(f"[{idx + 1}] \"{title}\" (Pages: {', '.join(pages_sorted)})")
            print("-" * 50)

        else:
            print(f"\n[-] RAG Query failed: {result['answer']}")
            if "error" in result:
                print(f"    Details: {result['error']}")

    except Exception as e:
        print(f"\n[-] Unexpected error during RAG query: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PROMPTED RAG (with Template)
# ──────────────────────────────────────────────────────────────────────────────

def run_prompted_rag(query: str, prompt_name: str, limit: int = 6):
    """
    Load a prompt template, retrieve RAG context, and generate structured output.

    Prompt templates live in prompts/<name>.txt and follow the convention:
      - Everything above '## USER PROMPT TEMPLATE' is the system prompt.
      - The user prompt block below uses placeholders: {context}, {title},
        {authors}, {year}, {venue}.

    Args:
        query: The research topic or question to fill the template with.
        prompt_name: Name of the template file (without .txt extension).
        limit: Number of ChromaDB context chunks to retrieve (default: 6).
    """
    from rag_service import check_ollama_health  # Flat import

    # Ensure Ollama is running before doing any work
    if not check_ollama_health():
        sys.exit(1)

    # ── Resolve the prompt template file ──────────────────────────────────────
    prompts_dir = settings.BASE_DIR / "prompts"

    # Support both bare name ("summarize") and explicit filename ("summarize.txt")
    prompt_path = prompts_dir / f"{prompt_name}.txt"
    if not prompt_path.exists():
        prompt_path = prompts_dir / prompt_name  # Try without adding .txt
    if not prompt_path.exists():
        print(f"[-] Prompt template not found: '{prompt_name}'")
        print(f"    Available templates in {prompts_dir}:")
        for p in sorted(prompts_dir.glob("*.txt")):
            print(f"      • {p.stem}")
        sys.exit(1)

    # Read and split the template on the USER PROMPT TEMPLATE divider
    raw_template = prompt_path.read_text(encoding="utf-8").strip()
    divider = "## USER PROMPT TEMPLATE"

    if divider in raw_template:
        parts = raw_template.split(divider, 1)
        # Strip the "## SYSTEM PROMPT — ..." heading from the system prompt section
        system_prompt = parts[0].replace("## SYSTEM PROMPT", "").strip()
        user_template  = parts[1].strip()
    else:
        # Treat the entire file as the system prompt with a generic user message
        system_prompt = raw_template
        user_template  = (
            "Context:\n" + "─" * 80 + "\n{context}\n" + "─" * 80 + "\n\nQuery: " + query
        )

    # ── Retrieve context chunks from ChromaDB ─────────────────────────────────
    vector_store = VectorStoreService()
    chunks = vector_store.query_similar_chunks(query, limit=limit)

    if not chunks:
        print("[-] No relevant context found in the local database for this query.")
        print("    Tip: ingest papers first with:  python scripts/main.py -q '<topic>'")
        sys.exit(0)

    # ── Build context string and collect metadata for placeholders ─────────────
    context_blocks = []
    titles = set()

    for idx, chunk in enumerate(chunks):
        meta  = chunk["metadata"]
        t     = meta.get("title", "Untitled")
        pages = meta.get("pages", "N/A")
        titles.add(t)
        context_blocks.append(f'[Source {idx + 1}] "{t}" (Pages: {pages})\nContent: {chunk["text"]}')

    context_str = "\n\n".join(context_blocks)
    title_str   = " | ".join(sorted(titles))

    # ── Fill the template placeholders ────────────────────────────────────────
    user_prompt = (
        user_template
        .replace("{context}",   context_str)
        .replace("{context_a}", context_str)   # comparative_analysis uses {context_a/b}
        .replace("{context_b}", context_str)
        .replace("{title}",    title_str)
        .replace("{title_a}",  title_str)
        .replace("{title_b}",  title_str)
        .replace("{authors}",  "See source metadata")
        .replace("{year}",     "Various")
        .replace("{venue}",    "Various")
    )

    print(
        f"\nRunning prompted RAG — template: '{prompt_path.stem}' | query: '{query}'\n"
        f"Retrieved {len(chunks)} chunks from {len(titles)} paper(s).\n"
        "Sending to local Ollama — please wait...\n"
    )

    # ── Call Ollama /api/chat ──────────────────────────────────────────────────
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9}
    }

    try:
        resp = _requests.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT)
        if resp.status_code == 200:
            answer = resp.json()["message"]["content"].strip()
            # Build a distinctive banner for the template output
            banner = f" {prompt_path.stem.upper().replace('_', ' ')} "
            print("=" * 40 + banner + "=" * 40)
            print(answer)
            print("=" * (80 + len(banner)))
            print(f"\n--- SOURCES ({len(chunks)} chunks from {len(titles)} paper(s)) ---")
            for idx, t in enumerate(sorted(titles), 1):
                print(f"  [{idx}] {t}")
        else:
            print(f"[-] Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")

    except Exception as e:
        print(f"[-] Error calling Ollama: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CITATION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def run_citation_analysis(paper_id: str, limit: int):
    """
    Run the full citation analysis pipeline for a target paper.

    Checks Ollama health, then delegates to CitationAnalyzerService which:
      1. Fetches citing papers from Semantic Scholar.
      2. Downloads and processes their PDFs (or falls back to API snippets).
      3. Classifies each citation passage via the local LLM.
      4. Saves a CSV report to output/.

    Args:
        paper_id: DOI, arXiv ID, S2 CorpusID, or canonical S2 paper ID.
        limit: Maximum number of citing papers to analyse.
    """
    from rag_service import check_ollama_health
    from citation_analyzer import CitationAnalyzerService

    # Fail fast before any expensive API calls if Ollama is not reachable
    if not check_ollama_health():
        sys.exit(1)

    print(f"\nStarting Citation Analysis for: '{paper_id}'")
    print(f"Analysing up to {limit} citing publications...")

    try:
        analyzer = CitationAnalyzerService()
        csv_path = analyzer.analyze_citations(paper_id, limit=limit)

        if csv_path:
            print(f"\n[+] Citation Analysis complete!")
            print(f"    Report saved to: {csv_path}")
        else:
            print("\n[-] Citation Analysis completed with no records generated.")

    except Exception as e:
        print(f"\n[-] Unexpected error during Citation Analysis: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN: Argument Parsing and Routing
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Parse CLI arguments and route execution to the appropriate handler function.

    All commands are mutually exclusive in the sense that each one performs
    a specific task and exits. The router checks each flag in priority order.
    """
    parser = argparse.ArgumentParser(
        description="AI Research Stack CLI — Search, Ingest, RAG, and Citation Analysis",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ── Search / ingestion arguments ──────────────────────────────────────────
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search Semantic Scholar and ingest the top result (non-interactive)."
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5,
        help="Number of search results / context chunks / citing papers (default: 5)."
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch the interactive paper selection wizard."
    )
    parser.add_argument(
        "--ingest-all", "-g",
        action="store_true",
        help="Batch-ingest all PDF files found in the papers/ directory."
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=1000,
        help="Character length of each text chunk during ingestion (default: 1000)."
    )
    parser.add_argument(
        "--chunk-overlap", "-o",
        type=int,
        default=200,
        help="Overlap in characters between adjacent chunks (default: 200)."
    )

    # ── RAG arguments ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--query-rag", "-r",
        type=str,
        help="Ask a research question grounded in ingested papers (RAG mode)."
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        metavar="TEMPLATE",
        help=(
            "Apply a prompt template to the RAG query.\n"
            "Use with --query-rag. Available: summarize | linkedin_draft | "
            "article_draft | comparative_analysis.\n"
            "Example: python scripts/main.py -r 'BERT' -p summarize"
        )
    )

    # ── Citation analysis argument ─────────────────────────────────────────────
    parser.add_argument(
        "--analyze-citations", "-a",
        type=str,
        metavar="PAPER_ID",
        help=(
            "Run citation intent analysis for a paper.\n"
            "Accepts: DOI, arXiv ID, CorpusID, or canonical S2 paper ID.\n"
            "Example: python scripts/main.py -a 10.48550/arXiv.1706.03762 -l 5"
        )
    )

    args = parser.parse_args()

    # ── Route: RAG query (highest priority — exits after completion) ───────────
    if args.query_rag:
        if args.prompt:
            # Prompted RAG: load a template and fill it with retrieved context
            run_prompted_rag(args.query_rag, args.prompt, limit=args.limit)
        else:
            # Standard RAG: direct question answering with source citations
            run_rag_query(args.query_rag)
        sys.exit(0)

    # ── Route: Citation analysis ───────────────────────────────────────────────
    if args.analyze_citations:
        run_citation_analysis(args.analyze_citations, args.limit)
        sys.exit(0)

    # ── Initialise services needed for search / ingestion ─────────────────────
    # These are only loaded when not doing a pure RAG or citation task.
    discover_service = PaperDiscoveryService()
    pdf_service      = PDFProcessorService()
    vector_store     = VectorStoreService()

    # ── Route: Batch ingestion ─────────────────────────────────────────────────
    if args.ingest_all:
        run_batch_ingestion(pdf_service, vector_store, args.chunk_size, args.chunk_overlap)
        sys.exit(0)

    # ── Route: Non-interactive keyword search + auto-ingest top result ─────────
    if args.query and not args.interactive:
        print(f"\nSearching for: '{args.query}' (auto-ingesting top result)...")
        papers = discover_service.search_papers(args.query, limit=args.limit)

        if not papers:
            print(f"No papers found for query: '{args.query}'")
            sys.exit(0)

        # Automatically process only the first (highest-relevance) result
        success = process_paper_pipeline(
            papers[0],
            discover_service, pdf_service, vector_store,
            args.chunk_size, args.chunk_overlap
        )
        sys.exit(0 if success else 1)

    # ── Default: Interactive wizard ────────────────────────────────────────────
    # Runs when no specific subcommand is given, or when --interactive is set.
    run_interactive_mode(
        discover_service, pdf_service, vector_store,
        args.chunk_size, args.chunk_overlap
    )


# ── Script entry point ────────────────────────────────────────────────────────
# This guard ensures main() only runs when this file is executed directly,
# not when it is imported as a module by another script.
if __name__ == "__main__":
    main()
