"""
rag_context.py — Shared helpers for building RAG context strings and retrieval.

Used by server.py and rag_service.py so template mode and standard chat share
the same chunk formatting, relevance filtering, and two-paper compare logic.
"""

from __future__ import annotations

from typing import Any

from config import settings
import re


# Standard refusal when vector search finds nothing sufficiently similar.
IRRELEVANT_REFUSAL = (
    "I could not find any relevant papers or context in the local database "
    "to answer your question. The retrieved passages are not sufficiently related "
    "to your query. Please ingest more on-topic papers or rephrase your question."
)

EMPTY_DB_REFUSAL = (
    "I could not find any relevant papers or context in the local database "
    "to answer your question. Please ingest papers first."
)

NOT_IN_LIBRARY_REFUSAL = (
    "I could not find any relevant papers in your knowledge base matching that "
    "author or paper name. Please check the spelling or ingest the paper first."
)

TABLE_TRUNCATION_REFUSAL = (
    "The table could not be completed for every paper in scope. "
    "Please try again with a paper filter, a smaller author corpus, or ask for "
    "title/year/venue only (metadata table)."
)

# Listing keywords used across RAG paths.
LISTING_QUERY_KEYWORDS = (
    "list", "table", "tabulate", "extract", "all paper",
    "each paper", "for each", "structured", "enumerate",
    "articles with", "papers by", "authored by",
)

# Signals that the user wants content extracted from PDF text, not just metadata.
CONTENT_EXTRACTION_SIGNALS = (
    "core security", "proposed solution", "framework", "assumption",
    "methodology", "contribution", "finding", "abstract", "threat",
    "mitigation", "challenge", "hypothesis", "objective", "research question",
    "surveillance", "connectivity", "citizens", "problem", "solution",
    "limitation", "open problem", "column", "columns",
)

_AUTHOR_PHRASE_PATTERNS = [
    re.compile(r"corpus of\s+(.+?)(?:'s|\u2019s)\s+(?:articles?|papers?|works?)", re.I),
    re.compile(r"corpus of\s+(.+?)\s+(?:articles?|papers?|works?)\s+on\b", re.I),
    re.compile(r"corpus of\s+(.+?)\s+on\b", re.I),
    re.compile(r"papers?\s+by\s+(.+?)(?:\s+on\b|[\.,;]|$)", re.I),
    re.compile(r"articles?\s+by\s+(.+?)(?:\s+on\b|[\.,;]|$)", re.I),
    re.compile(r"(.+?)(?:'s|\u2019s)\s+(?:articles?|papers?|works?)\s+on\b", re.I),
]

_AUTHOR_SCOPED_PATTERNS = [
    r"corpus of\s+",
    r"papers?\s+by\s+",
    r"articles?\s+by\s+",
    r"works?\s+by\s+",
    r"'s\s+(papers?|articles?|works?)",
    r"\u2019s\s+(papers?|articles?|works?)",
    r"articles?\s+with\s+.+?\s+as\s+(author|co-author)",
    r"list\s+(articles?|papers?)\s+with\s+",
    r"papers?\s+authored?\s+by\s+",
    r".+?\s+authored?\s+(papers?|articles?)",
    r"as\s+(author|co-author)\s+or\s+co-author",
]

TABLE_TRUNCATION_RE = re.compile(
    r"remaining\s+papers?|\.\.\.\s*\(|\.\.\.\s*\||\|\s*\.\.\.\s*\|",
    re.IGNORECASE,
)


def format_chunk_block(chunk: dict[str, Any], index: int | None = None) -> str:
    """
    Format one ChromaDB chunk for the LLM prompt (APA-oriented labels, not 'Source N').
    """
    meta = chunk.get("metadata") or {}
    title = meta.get("title", "Untitled Paper")
    authors = meta.get("authors", "Unknown Authors")
    year = meta.get("year", "N/A")
    doi = meta.get("doi", "N/A")
    pages = meta.get("pages", "N/A")
    text = chunk.get("text", "")
    label = f"({authors}, {year})"
    if index is not None:
        label = f"Passage {index + 1} {label}"
    return (
        f"--- Academic Source {label} ---\n"
        f'Title: "{title}"\n'
        f"DOI: {doi}\n"
        f"Pages: {pages}\n"
        f"Content: {text}\n"
    )


def build_library_inventory(papers_metadata: dict) -> str:
    """Format the full ingested-paper inventory block for prompts."""
    if not papers_metadata:
        return "No papers in database library."
    blocks = []
    for title, meta in papers_metadata.items():
        blocks.append(
            f"- {meta.get('authors', 'Unknown Authors')} ({meta.get('year', 'N/A')}). "
            f"\"{title}\". DOI: {meta.get('doi', 'N/A')}"
        )
    return "\n".join(blocks)


def chunks_to_context_string(chunks: list[dict[str, Any]], *, header: str = "Context Chunks") -> str:
    """Join formatted chunk blocks; empty list yields a clear message."""
    if not chunks:
        return "No relevant text passage chunks found for this query."
    blocks = [format_chunk_block(c, i) for i, c in enumerate(chunks)]
    return f"{header}:\n" + "\n\n".join(blocks)


def filter_chunks_by_relevance(
    chunks: list[dict[str, Any]],
    *,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """
    Drop chunks whose cosine distance exceeds RAG_MAX_DISTANCE in settings.
    Lower distance = more similar in ChromaDB.
    """
    threshold = max_distance if max_distance is not None else settings.RAG_MAX_DISTANCE
    return [c for c in chunks if float(c.get("distance", 0.0)) <= threshold]


def _query_stopwords() -> set[str]:
    return {
        "what", "which", "when", "where", "does", "about", "from", "with",
        "that", "this", "have", "into", "your", "their", "paper", "papers",
        "author", "authors", "say", "says", "line", "summarize", "summary",
        "brief", "explain", "describe", "tell", "give", "please", "would",
        # Generic academic terms and common filler words to prevent false-positive RAG matches
        "detailed", "provide", "provides", "present", "presents", "core", "research",
        "method", "methodology", "findings", "contributions", "study", "studies",
        "result", "results", "analysis", "article", "articles", "chapter", "chapters",
        "book", "books", "discuss", "discusses", "explores", "exploring", "contribution",
        "focus", "focuses", "question", "questions", "concept", "concepts", "theory",
        "theories", "approach", "approaches", "framework", "frameworks", "system",
        "systems", "process", "processes", "perspective", "perspectives", "literature",
        "review", "reviews", "empirical", "evidence", "conclusions"
    }


def _significant_query_tokens(query: str) -> list[str]:
    """Extract lowercased meaningful tokens from user query (incl. author surnames)."""
    stop = _query_stopwords()
    raw = re.findall(r"[a-z0-9]+", (query or "").lower())
    # Author surnames are often 4+ characters — keep them even when other tokens are short.
    tokens = [t for t in raw if len(t) >= 4 and t not in stop]
    if not tokens:
        tokens = [t for t in raw if len(t) >= 3 and t not in stop]
    return tokens


def _chunk_search_haystack(chunk: dict[str, Any]) -> str:
    """Text used for lexical matching — includes metadata authors/title, not just body."""
    meta = chunk.get("metadata") or {}
    parts = [
        chunk.get("text") or "",
        meta.get("title") or "",
        meta.get("authors") or "",
        str(meta.get("year") or ""),
        meta.get("doi") or "",
    ]
    return " ".join(parts).lower()


def _filter_chunks_by_query_term_presence(
    chunks: list[dict[str, Any]],
    query: str,
    *,
    skip_if_empty: bool = False,
) -> list[dict[str, Any]]:
    """
    Keep chunks whose text OR metadata contains at least one significant query token.
    When skip_if_empty is True and filtering would remove everything, return the
    original chunks (used after we already matched a library author/paper).
    """
    tokens = _significant_query_tokens(query)
    if not tokens or not chunks:
        return chunks

    kept: list[dict[str, Any]] = []
    for chunk in chunks:
        haystack = _chunk_search_haystack(chunk)
        if any(t in haystack for t in tokens):
            kept.append(chunk)

    if not kept and skip_if_empty:
        return chunks
    return kept


def is_listing_query(query: str) -> bool:
    q = (query or "").lower()
    return any(kw in q for kw in LISTING_QUERY_KEYWORDS)


def is_content_extraction_query(query: str) -> bool:
    q = (query or "").lower()
    if not is_listing_query(query):
        return False
    if any(sig in q for sig in CONTENT_EXTRACTION_SIGNALS):
        return True
    # Numbered column specs like "(1) title (2) year" imply multi-field extraction.
    return bool(re.search(r"\(\d+\)\s*\w+", q))


def is_simple_inventory_listing(query: str) -> bool:
    """Metadata-only list/table (title, year, venue) — safe to generate without LLM."""
    return is_listing_query(query) and not is_content_extraction_query(query)


def is_per_paper_extraction_query(query: str) -> bool:
    """Structured per-paper table requiring text extraction from each PDF."""
    q = (query or "").lower()
    if "table" not in q and "tabulate" not in q and "for each" not in q:
        return False
    return is_content_extraction_query(query) or "extract" in q


def query_expects_named_author(query: str) -> bool:
    """True when the user clearly names an author corpus (must not answer from other authors)."""
    if extract_author_search_phrase(query):
        return True
    q = (query or "").lower()
    return any(re.search(p, q) for p in _AUTHOR_SCOPED_PATTERNS)


def extract_author_search_phrase(query: str) -> str | None:
    """Pull a human author phrase from queries like 'corpus of Noor Zaman Jhanjhi's articles'."""
    q = (query or "").strip()
    if not q:
        return None
    for pattern in _AUTHOR_PHRASE_PATTERNS:
        m = pattern.search(q)
        if not m:
            continue
        phrase = m.group(1).strip(" .,;:\"'")
        phrase = re.sub(r"\s+", " ", phrase)
        if len(phrase) >= 3:
            return phrase
    return None


def author_phrase_tokens(phrase: str) -> list[str]:
    stop = _query_stopwords() | {"noor", "dr", "prof", "professor"}
    raw = re.findall(r"[a-z0-9]+", (phrase or "").lower())
    return [t for t in raw if len(t) >= 3 and t not in stop]


def parse_table_columns_from_query(query: str) -> list[str]:
    """Parse explicit column headers from numbered lists in the user query."""
    numbered = re.findall(r"\(\d+\)\s*([^,;\n]+)", query or "")
    cols = [c.strip().rstrip(".") for c in numbered if c.strip()]
    if cols:
        return cols
    return [
        "Title",
        "Year",
        "Venue",
        "Core topic / problem",
        "Main approach / contribution",
        "Key assumptions or scope",
    ]


def answer_has_table_truncation(answer: str) -> bool:
    return bool(TABLE_TRUNCATION_RE.search(answer or ""))


def filter_chunks_to_titles(
    chunks: list[dict[str, Any]],
    allowed_titles: list[str],
) -> list[dict[str, Any]]:
    if not allowed_titles:
        return chunks
    allowed = {t.strip() for t in allowed_titles}
    return [
        c for c in chunks
        if (c.get("metadata") or {}).get("title", "").strip() in allowed
    ]


def query_refers_to_missing_library_paper(query: str, papers_metadata: dict) -> bool:
    """
    True when the query names a specific author/surname that is not in the library.
    Used to return a clearer refusal than generic 'irrelevant context'.
    """
    if resolve_matching_paper_titles(query, papers_metadata):
        return False
    tokens = _significant_query_tokens(query)
    return any(len(t) >= 5 for t in tokens)


def resolve_matching_paper_titles(query: str, papers_metadata: dict) -> list[str]:
    """
    Map a natural-language question to paper title(s) in the local library inventory.
    Matches author surnames and distinctive title words against ChromaDB metadata,
    falling back to matching against filenames and metadata inside the ingestion manifest.
    
    Prioritizes author field matches over title matches for author-scoped queries.
    """
    if not papers_metadata:
        return []

    tokens = _significant_query_tokens(query)
    if not tokens:
        return []

    query_lower = query.lower()
    is_author_scoped = query_expects_named_author(query)

    matched: list[str] = []
    author_matches: list[str] = []

    # Match on explicit author phrase tokens first (full name / surname).
    author_phrase = extract_author_search_phrase(query)
    phrase_tokens = author_phrase_tokens(author_phrase) if author_phrase else []
    if phrase_tokens:
        for title, meta in papers_metadata.items():
            authors = (meta.get("authors") or "").lower()
            if all(t in authors for t in phrase_tokens):
                if title not in author_matches:
                    author_matches.append(title)
            elif len(phrase_tokens) >= 2 and phrase_tokens[-1] in authors:
                # Surname-only match when full phrase is not stored verbatim.
                if title not in author_matches:
                    author_matches.append(title)

    # 1. Try to match using the Ingestion Manifest (very robust for physical uploads / filenames)
    try:
        manifest_path = settings.BASE_DIR / "output" / "ingestion_manifest.json"
        if manifest_path.exists():
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for filename, meta in manifest.items():
                m_title = meta.get("title", "")
                m_authors = meta.get("authors", "")
                m_year = meta.get("year", "")
                
                for token in tokens:
                    if len(token) < 4:
                        continue
                    # Prioritize author field matches
                    if token in m_authors.lower():
                        # Resolve the corresponding title in papers_metadata
                        for db_title in papers_metadata.keys():
                            if (db_title.lower().strip() == m_title.lower().strip() or
                                    db_title.lower().strip() in m_title.lower().strip() or
                                    m_title.lower().strip() in db_title.lower().strip()):
                                if db_title not in author_matches:
                                    author_matches.append(db_title)
                                break
                    # Also match against filename and title as fallback
                    elif token in filename.lower() or token in m_title.lower():
                        for db_title in papers_metadata.keys():
                            if (db_title.lower().strip() == m_title.lower().strip() or
                                    db_title.lower().strip() in m_title.lower().strip() or
                                    m_title.lower().strip() in db_title.lower().strip()):
                                if db_title not in matched:
                                    matched.append(db_title)
                                break
    except Exception:
        # Prevent any manifest reading issue from crashing RAG
        pass

    # 2. Match against active ChromaDB papers metadata directly
    for title, meta in papers_metadata.items():
        authors = (meta.get("authors") or "").lower()
        title_l = (title or "").lower()
        for token in tokens:
            if len(token) < 4:
                continue
            # Prioritize author field matches
            if token in authors:
                if title not in author_matches:
                    author_matches.append(title)
                break
            # Title matches as fallback
            elif token in title_l:
                if title not in matched:
                    matched.append(title)
                break

    if author_matches:
        if is_author_scoped:
            matched = author_matches
        else:
            matched = author_matches + [m for m in matched if m not in author_matches]
    elif is_author_scoped:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge chunk lists by vector id while preserving order."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        cid = chunk.get("id") or id(chunk)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(chunk)
    return out


def retrieve_relevant_chunks(
    vector_store,
    query: str,
    limit: int,
    filter_title: str | None = None,
    *,
    scope_titles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve context chunks for RAG with author/paper-aware isolation.

    Pipeline:
      1. Detect if the query names a specific author/paper already in the library.
      2. Author-scoped path: fetch chunks EXCLUSIVELY from those papers — no semantic
         mixing that could contaminate results with other authors' papers.
      3. Unscoped path: standard semantic search → distance filter → token filter.
    """
    stats = vector_store.get_collection_stats()
    papers_metadata = stats.get("papers_metadata", {})

    inventory_titles = list(scope_titles) if scope_titles else resolve_matching_paper_titles(
        query, papers_metadata
    )
    if filter_title:
        inventory_titles = [filter_title]

    strict = getattr(settings, "RAG_STRICT_MODE", True)
    locked_scope = bool(inventory_titles) and (
        bool(scope_titles)
        or query_expects_named_author(query)
        or bool(filter_title)
    )

    # ── Author / Paper-scoped retrieval ───────────────────────────────────────
    # When the query names a specific author or paper that is already in the library,
    # pull chunks EXCLUSIVELY from those matched papers.  This prevents topically
    # similar papers written by *different* authors from polluting the context and
    # causing the LLM to attribute content to the wrong author.
    # Note: no [:3] cap — we fetch ALL matched papers so every one of the author's
    # works is represented in the context for exhaustive listing/tabulation queries.
    if inventory_titles:
        author_chunks: list[dict[str, Any]] = []
        per_paper = max(limit, 10)
        for title in inventory_titles:
            paper_chunks = vector_store.get_chunks_for_paper(title, max_chunks=per_paper)
            author_chunks = _dedupe_chunks(author_chunks + paper_chunks)
        if author_chunks:
            # Return all fetched chunks; rag_service will pass an enlarged limit for
            # listing/tabulation queries so no paper is silently dropped.
            return author_chunks

    # ── Standard semantic search path ─────────────────────────────────────────
    # Request extra candidates so post-filters still leave enough context.
    search_limit = max(limit * 3, limit + 8)
    raw = vector_store.query_similar_chunks(
        query, limit=search_limit, filter_title=filter_title
    )

    chunks = filter_chunks_by_relevance(raw, max_distance=settings.RAG_MAX_DISTANCE)

    if settings.RAG_REQUIRE_QUERY_TERM_MATCH:
        chunks = _filter_chunks_by_query_term_presence(
            chunks,
            query,
            skip_if_empty=False,
        )

    if inventory_titles and (locked_scope or query_expects_named_author(query)):
        chunks = filter_chunks_to_titles(chunks, inventory_titles)
        if not chunks:
            for title in inventory_titles:
                chunks = _dedupe_chunks(
                    chunks + vector_store.get_chunks_for_paper(title, max_chunks=max(limit, 12))
                )

    # Strict mode: never return unscoped semantic noise when the query is entity-locked.
    if strict and locked_scope:
        return filter_chunks_to_titles(chunks, inventory_titles)[:limit]

    if not chunks and inventory_titles and not strict:
        chunks = raw[:limit]
        chunks = filter_chunks_to_titles(chunks, inventory_titles) or chunks

    if not chunks and not locked_scope:
        chunks = raw[:limit]

    return chunks[:limit]


def chunk_citation_label(chunk: dict[str, Any], index: int) -> str:
    """Human-readable label for the UI sources panel (APA-style, not 'Source N')."""
    meta = chunk.get("metadata") or {}
    authors = meta.get("authors", "Unknown Authors")
    year = meta.get("year", "N/A")
    pages = meta.get("pages", "N/A")
    page_str = ""
    if pages and pages != "N/A":
        page_str = f", p. {pages}" if isinstance(pages, (int, float)) else f", pp. {pages}"
    return f"({authors}, {year}{page_str})"
