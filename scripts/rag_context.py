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


def filter_chunks_by_relevance(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Drop chunks whose cosine distance exceeds RAG_MAX_DISTANCE in settings.
    Lower distance = more similar in ChromaDB.
    """
    threshold = settings.RAG_MAX_DISTANCE
    return [c for c in chunks if float(c.get("distance", 0.0)) <= threshold]


def retrieve_relevant_chunks(
    vector_store,
    query: str,
    limit: int,
    filter_title: str | None = None,
) -> list[dict[str, Any]]:
    """Query ChromaDB and apply the configured relevance distance threshold."""
    raw = vector_store.query_similar_chunks(query, limit=limit, filter_title=filter_title)
    chunks = filter_chunks_by_relevance(raw)
    if settings.RAG_REQUIRE_QUERY_TERM_MATCH:
        chunks = _filter_chunks_by_query_term_presence(chunks, query)
    return chunks


def _filter_chunks_by_query_term_presence(chunks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """
    Keep only chunks containing at least one significant query token.
    This is a lexical safety gate on top of embeddings to reduce hallucinations.
    """
    tokens = _significant_query_tokens(query)
    if not tokens or not chunks:
        return chunks

    kept: list[dict[str, Any]] = []
    for chunk in chunks:
        haystack = f"{chunk.get('text', '')} {(chunk.get('metadata') or {}).get('title', '')}".lower()
        if any(t in haystack for t in tokens):
            kept.append(chunk)
    return kept


def _significant_query_tokens(query: str) -> list[str]:
    """Extract lowercased meaningful tokens from user query."""
    stop = {
        "what", "which", "when", "where", "does", "about", "from", "with",
        "that", "this", "have", "into", "your", "their", "paper", "papers",
        "author", "authors", "nik", "hassan", "rushdi", "say", "says",
    }
    raw = re.findall(r"[a-z0-9]+", (query or "").lower())
    tokens = [t for t in raw if len(t) >= 5 and t not in stop]
    # Fallback: if query is short, still keep medium tokens.
    if not tokens:
        tokens = [t for t in raw if len(t) >= 4 and t not in stop]
    return tokens


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
