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
    """
    if not papers_metadata:
        return []

    tokens = _significant_query_tokens(query)
    if not tokens:
        return []

    matched: list[str] = []

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
                    # Match query token against filename, manifest title, or manifest authors
                    if (token in filename.lower() or 
                            token in m_title.lower() or 
                            token in m_authors.lower()):
                        # Resolve the corresponding title in papers_metadata
                        for db_title in papers_metadata.keys():
                            if (db_title.lower().strip() == m_title.lower().strip() or
                                    db_title.lower().strip() in m_title.lower().strip() or
                                    m_title.lower().strip() in db_title.lower().strip()):
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
            if token in authors or token in title_l:
                if title not in matched:
                    matched.append(title)
                break

    # Deduplicate while preserving order
    seen = set()
    out = []
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
) -> list[dict[str, Any]]:
    """
    Retrieve context chunks for RAG with author/paper-aware fallbacks.

    Pipeline:
      1. Vector search (semantic similarity).
      2. If query names an author/paper in the library, pull chunks for that paper directly.
      3. Apply distance threshold (relaxed when a library paper was explicitly matched).
      4. Optional lexical token gate — skipped when that would drop all matched-paper chunks.
    """
    stats = vector_store.get_collection_stats()
    papers_metadata = stats.get("papers_metadata", {})

    inventory_titles = resolve_matching_paper_titles(query, papers_metadata)
    if filter_title:
        inventory_titles = [filter_title]

    # Request extra candidates so post-filters still leave enough context.
    search_limit = max(limit * 3, limit + 8)
    raw = vector_store.query_similar_chunks(
        query, limit=search_limit, filter_title=filter_title
    )

    # Direct fetch for papers the user named (e.g. "Khamsani") even if embeddings miss.
    for title in inventory_titles[:3]:
        paper_chunks = vector_store.get_chunks_for_paper(title, max_chunks=max(limit * 2, 12))
        raw = _dedupe_chunks(raw + paper_chunks)

    relaxed_distance = settings.RAG_MAX_DISTANCE
    if inventory_titles:
        relaxed_distance = min(1.25, settings.RAG_MAX_DISTANCE + 0.35)

    chunks = filter_chunks_by_relevance(raw, max_distance=relaxed_distance)

    if settings.RAG_REQUIRE_QUERY_TERM_MATCH:
        chunks = _filter_chunks_by_query_term_presence(
            chunks,
            query,
            skip_if_empty=bool(inventory_titles),
        )

    # Last resort: user named a paper in inventory but filters removed everything.
    if not chunks and inventory_titles:
        fallback: list[dict[str, Any]] = []
        for title in inventory_titles[:2]:
            fallback.extend(
                vector_store.get_chunks_for_paper(title, max_chunks=max(limit, 10))
            )
        chunks = _dedupe_chunks(fallback)[:limit]

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
