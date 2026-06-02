"""
rag_strict.py — Deterministic scope resolution and answer verification for RAG.

Ensures answers only reference papers/authors in scope and refuses when entities
are not in the knowledge base.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from config import settings
from rag_context import (
    NOT_IN_LIBRARY_REFUSAL,
    IRRELEVANT_REFUSAL,
    TABLE_TRUNCATION_RE,
    extract_author_search_phrase,
    infer_library_author_phrase,
    query_expects_named_author,
    query_has_author_intent,
    query_has_paper_focus,
    resolve_matching_paper_titles,
    resolve_papers_for_author_phrase,
    resolve_author_from_library,
    fuzzy_match_paper_titles,
    build_catalog_indexes,
    detect_topic_profile,
    resolve_topic_scoped_papers,
    _significant_query_tokens,
    _topic_specific_tokens,
    _refine_topic_papers_by_query,
    find_papers_by_metadata_keywords,
    query_has_library_topic_cue,
)

_COMPARE_QUERY_RE = re.compile(r"\bcompare\b", re.I)

_YEAR_RANGE_RE = re.compile(
    r"\b(?:between|from)\s+(20\d{2})\s+(?:and|to|-)\s+(20\d{2})\b|"
    r"\b(20\d{2})\s*-\s*(20\d{2})\b",
    re.I,
)

_YEAR_SINGLE_RE = re.compile(
    r"\b(?:published\s+in|in)\s+(20\d{2})\b|\b(20\d{2})\s+papers?\b",
    re.I,
)

_KEYWORD_DISCOVERY_RE = re.compile(
    r"\b(?:do|does|are there|which)\s+(?:any\s+)?(?:of\s+)?(?:my\s+)?papers?\s+"
    r"(?:discuss|mention|cover|address|use|include|contain)\b|"
    r"\bwhich\s+papers?\s+(?:discuss|mention|use|cover)\b|"
    r"\bpapers?\s+(?:on|about|regarding)\s+\w",
    re.I,
)

COMPARE_NEEDS_PICKER_MSG = (
    "To compare two papers, select **Paper A** and **Paper B** in the Focus on Paper "
    "controls (or use the compare template), then ask your comparison question. "
    "You can also quote both exact titles in your message."
)

VERIFICATION_FAILED_REFUSAL = (
    "I cannot provide this answer because it references papers or authors that are "
    "not in the retrieved scope for your question. Please narrow your query using "
    "an author name, paper filter, or a topic that exists in your ingested library."
)

_TOPIC_NOT_FOUND_REFUSAL = (
    "I could not find any ingested papers in your knowledge base that discuss that "
    "topic. Please ingest more on-topic papers or rephrase your question."
)


@dataclass
class QueryScope:
    """Resolved corpus scope for a single RAG request."""

    scoped_titles: list[str] = field(default_factory=list)
    requires_entity: bool = False
    entity_kind: str = "none"  # author | paper | topic | filter | none
    author_phrase: str | None = None
    topic_tokens: list[str] = field(default_factory=list)

    @property
    def is_locked(self) -> bool:
        """True when retrieval and answers must stay within scoped_titles."""
        if not self.scoped_titles:
            return False
        return self.entity_kind in ("author", "paper", "filter", "topic")


def _load_manifest() -> dict:
    path = settings.BASE_DIR / "output" / "ingestion_manifest.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _author_tokens_from_string(authors: str) -> set[str]:
    tokens: set[str] = set()
    for part in re.split(r"[,;&]| and ", (authors or "").lower()):
        for word in re.findall(r"[a-z]{4,}", part):
            tokens.add(word)
    return tokens


def list_distinct_authors(papers_metadata: dict) -> list[dict[str, Any]]:
    """Unique author strings with paper counts (deterministic, no LLM)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for title, meta in sorted((papers_metadata or {}).items()):
        authors = (meta.get("authors") or "Unknown Authors").strip()
        key = authors.lower()
        if key in seen:
            continue
        seen.add(key)
        count = sum(
            1
            for _, m in papers_metadata.items()
            if (m.get("authors") or "").strip().lower() == key
        )
        out.append({"authors": authors, "paper_count": count})
    return sorted(out, key=lambda x: x["authors"].lower())


def extract_quoted_paper_titles(query: str) -> list[str]:
    return re.findall(r'"([^"]{8,200})"', query or "")


def _fuzzy_title_match(query: str, papers_metadata: dict) -> list[str]:
    """Backward-compatible alias for paper title resolution."""
    return fuzzy_match_paper_titles(query, papers_metadata)


def _papers_matching_topic_tokens(
    topic_tokens: list[str],
    papers_metadata: dict,
) -> list[str]:
    if not topic_tokens:
        return []
    manifest = _load_manifest()
    matched: list[str] = []

    for title, meta in papers_metadata.items():
        abstract = ""
        for _fn, m in manifest.items():
            mt = (m.get("title") or "").lower()
            if mt == title.lower() or mt in title.lower() or title.lower() in mt:
                abstract = (m.get("abstract") or "").lower()
                break
        hay = " ".join(
            [
                title.lower(),
                (meta.get("authors") or "").lower(),
                (meta.get("venue") or "").lower(),
                abstract,
            ]
        )
        hits = sum(1 for t in topic_tokens if t in hay)
        if hits >= max(1, len(topic_tokens) // 2):
            matched.append(title)
    return matched


def is_keyword_discovery_query(query: str) -> bool:
    """Questions answerable from title/metadata before LLM (any topic)."""
    return bool(_KEYWORD_DISCOVERY_RE.search(query or ""))


def answer_keyword_discovery_query(query: str, papers_metadata: dict) -> str | None:
    """List papers whose metadata matches question keywords — no profile list needed."""
    if not papers_metadata or not is_keyword_discovery_query(query):
        return None
    titles = find_papers_by_metadata_keywords(query, papers_metadata)
    if not titles:
        return _TOPIC_NOT_FOUND_REFUSAL
    lines = []
    for i, title in enumerate(titles, 1):
        m = papers_metadata[title]
        lines.append(
            f"{i}. {m.get('authors', 'Unknown')} ({m.get('year', 'N/A')}). {title}"
        )
    return _format_numbered_list(
        f"Papers in your library that match this topic ({len(lines)} paper(s)):",
        lines,
    )


def apply_scope_resilience(
    scope: QueryScope,
    query: str,
    papers_metadata: dict,
) -> QueryScope:
    """
    Avoid hard failure when scope resolution is empty but the library may still
    contain relevant papers (new topics, wording variants, no profile match).
    """
    if scope.scoped_titles:
        return scope

    if scope.entity_kind == "author" and scope.requires_entity:
        # Explicit author name with zero papers → keep refusal path in caller.
        return scope

    if scope.entity_kind == "filter":
        return scope

    # Topic profile/token path found nothing — try metadata keywords once.
    if scope.entity_kind == "topic" or query_has_library_topic_cue(query):
        rescued = find_papers_by_metadata_keywords(query, papers_metadata)
        if rescued:
            return QueryScope(
                scoped_titles=rescued,
                requires_entity=True,
                entity_kind="topic",
                topic_tokens=_topic_specific_tokens(query),
            )

    # Open semantic search over full library (never refuse for "unknown topic name").
    return QueryScope(
        scoped_titles=[],
        requires_entity=False,
        entity_kind="none",
        topic_tokens=scope.topic_tokens,
    )


def compare_query_needs_paper_pickers(query: str, papers_metadata: dict) -> bool:
    """True when user asked to compare but did not select or quote two papers."""
    if not _COMPARE_QUERY_RE.search(query or ""):
        return False
    quoted = extract_quoted_paper_titles(query)
    if len(quoted) >= 2:
        resolved = fuzzy_match_paper_titles(query, papers_metadata)
        return len(resolved) < 2
    return True


def resolve_query_scope(
    query: str,
    papers_metadata: dict,
    *,
    filter_title: str | None = None,
) -> QueryScope:
    """
    Decide which papers may be used to answer this query.
    When requires_entity is True and scoped_titles is empty → caller must refuse.
    """
    if filter_title:
        if filter_title in papers_metadata:
            return QueryScope(
                scoped_titles=[filter_title],
                requires_entity=True,
                entity_kind="filter",
            )
        return QueryScope(requires_entity=True, entity_kind="filter")

    if query_has_paper_focus(query):
        paper_titles = fuzzy_match_paper_titles(query, papers_metadata)
        if paper_titles:
            return QueryScope(
                scoped_titles=paper_titles[:1] if len(paper_titles) == 1 else paper_titles,
                requires_entity=True,
                entity_kind="paper",
            )

    author_phrase, author_titles = resolve_author_from_library(query, papers_metadata)
    explicit_author = query_expects_named_author(query) or bool(
        extract_author_search_phrase(query)
    )
    if author_titles:
        return QueryScope(
            scoped_titles=author_titles,
            requires_entity=True,
            entity_kind="author",
            author_phrase=author_phrase,
        )
    if explicit_author:
        return QueryScope(
            scoped_titles=[],
            requires_entity=True,
            entity_kind="author",
            author_phrase=author_phrase,
        )

    topic_tokens = _significant_query_tokens(query)
    q_lower = (query or "").lower()
    profile = detect_topic_profile(query)
    if profile:
        topic_papers = resolve_topic_scoped_papers(query, papers_metadata, profile)
        if topic_papers:
            return QueryScope(
                scoped_titles=topic_papers,
                requires_entity=True,
                entity_kind="topic",
                topic_tokens=topic_tokens,
            )
    if topic_tokens and query_has_library_topic_cue(query):
        specific = _topic_specific_tokens(query) or topic_tokens
        topic_papers = _papers_matching_topic_tokens(specific, papers_metadata)
        topic_papers = _refine_topic_papers_by_query(query, topic_papers, papers_metadata)
        if topic_papers:
            return QueryScope(
                scoped_titles=topic_papers,
                requires_entity=True,
                entity_kind="topic",
                topic_tokens=specific,
            )
        rescued = find_papers_by_metadata_keywords(query, papers_metadata)
        if rescued:
            return QueryScope(
                scoped_titles=rescued,
                requires_entity=True,
                entity_kind="topic",
                topic_tokens=specific,
            )

    paper_titles = fuzzy_match_paper_titles(query, papers_metadata)
    if len(paper_titles) == 1:
        return QueryScope(
            scoped_titles=paper_titles,
            requires_entity=False,
            entity_kind="paper",
        )

    return QueryScope(
        scoped_titles=[],
        requires_entity=False,
        entity_kind="none",
        topic_tokens=topic_tokens,
    )


def scope_refusal_message(scope: QueryScope) -> str:
    if scope.entity_kind == "topic":
        return _TOPIC_NOT_FOUND_REFUSAL
    if scope.entity_kind in {"author", "paper", "filter"}:
        return NOT_IN_LIBRARY_REFUSAL
    return NOT_IN_LIBRARY_REFUSAL


def inventory_for_scope(
    papers_metadata: dict,
    scope: QueryScope,
) -> dict:
    if scope.scoped_titles:
        return {t: papers_metadata[t] for t in scope.scoped_titles if t in papers_metadata}
    return papers_metadata


def allowed_evidence_from_chunks(
    chunks: list[dict[str, Any]],
    scoped_titles: list[str],
    papers_metadata: dict,
) -> tuple[set[str], set[str]]:
    titles: set[str] = set(scoped_titles)
    authors: set[str] = set()
    for t in scoped_titles:
        authors |= _author_tokens_from_string((papers_metadata.get(t) or {}).get("authors", ""))
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        title = (meta.get("title") or "").strip()
        if title:
            titles.add(title)
        authors |= _author_tokens_from_string(meta.get("authors") or "")
    return titles, authors


# Common title words that must not alone trigger an out-of-scope paper match.
_TITLE_WORD_STOP = frozenset({
    "about", "based", "using", "approach", "system", "systems", "study",
    "analysis", "review", "model", "models", "method", "methods", "framework",
    "detection", "security", "cybersecurity", "network", "networks", "internet",
    "things", "smart", "cities", "learning", "deep", "machine", "traffic",
    "monitoring", "enhanced", "novel", "hybrid", "secure", "routing", "cloud",
    "computing", "internet", "enabled", "research", "design", "application",
})


def verify_answer_against_scope(
    answer: str,
    *,
    scoped_titles: list[str],
    papers_metadata: dict,
    chunks: list[dict[str, Any]],
    strict: bool = True,
    scope_locked: bool = False,
) -> tuple[bool, str]:
    """
    Return (ok, reason). When not ok, caller should replace answer with refusal.
    """
    if not strict or not answer or not papers_metadata:
        return True, ""

    if TABLE_TRUNCATION_RE.search(answer):
        return False, "table_truncation"

    locked = scope_locked or bool(scoped_titles)

    if locked and scoped_titles:
        # Scoped answers may only cite papers in the resolved corpus (not semantic noise).
        allowed_titles = set(scoped_titles)
        allowed_author_tokens: set[str] = set()
        for t in scoped_titles:
            allowed_author_tokens |= _author_tokens_from_string(
                (papers_metadata.get(t) or {}).get("authors", "")
            )
        chunks = [
            c for c in chunks
            if (c.get("metadata") or {}).get("title", "").strip() in allowed_titles
        ]
    else:
        allowed_titles, allowed_author_tokens = allowed_evidence_from_chunks(
            chunks, scoped_titles, papers_metadata
        )
        if not allowed_titles and scoped_titles:
            allowed_titles = set(scoped_titles)

    answer_lower = answer.lower()

    for title in papers_metadata:
        if title in allowed_titles:
            continue
        tl = title.lower()
        # Require a long, distinctive title substring — not generic keyword overlap.
        min_len = 28 if locked else 36
        if len(tl) < min_len:
            continue
        if tl in answer_lower:
            return False, f"out_of_scope_title:{title[:60]}"

    # Co-authors on scoped papers are allowed; do not run surname-token checks on synthesis answers.
    return True, ""


def is_catalog_metadata_query(query: str) -> bool:
    q = (query or "").lower()
    if re.search(r"\bpapers?\s+by\s+", q) or re.search(r"\barticles?\s+by\s+", q):
        return True
    if re.search(r"\b(list|show|name|who|all)\b.{0,40}\b(authors?|writers?)\b", q):
        return True
    if re.search(r"\bhow many\b.{0,30}\b(papers?|articles?)\b", q):
        return True
    if re.search(r"\blist\b.{0,40}\b(all\s+)?(ingested\s+)?(papers?|articles?)\b", q):
        return True
    if re.search(r"\bwhat\b.{0,20}\b(papers?|articles?)\b.{0,20}\b(library|ingested|knowledge base)\b", q):
        return True
    if _YEAR_RANGE_RE.search(q) and re.search(r"\b(papers?|articles?|publications?)\b", q):
        return True
    if _YEAR_SINGLE_RE.search(q) and re.search(r"\b(papers?|articles?|publications?)\b", q):
        return True
    return False


def _extract_year_range(query: str) -> tuple[int, int] | None:
    """
    Return an inclusive (start_year, end_year) if the query names a year range.
    Supports:
      - between 2021 and 2023
      - from 2021 to 2023
      - 2021-2023
    """
    q = (query or "").strip()
    if not q:
        return None
    m = _YEAR_RANGE_RE.search(q)
    if not m:
        return None
    years = [y for y in m.groups() if y]
    if len(years) != 2:
        return None
    try:
        a, b = int(years[0]), int(years[1])
    except ValueError:
        return None
    start, end = (a, b) if a <= b else (b, a)
    if start < 1900 or end > 2100:
        return None
    return start, end


def _extract_single_year(query: str) -> int | None:
    """
    Return a single year if the query asks for papers published in that year.
    Supports:
      - published in 2025
      - papers in 2025
      - 2025 papers
    """
    q = (query or "").strip()
    if not q:
        return None
    m = _YEAR_SINGLE_RE.search(q)
    if not m:
        return None
    year = next((g for g in m.groups() if g), None)
    if not year:
        return None
    try:
        yi = int(year)
    except ValueError:
        return None
    if yi < 1900 or yi > 2100:
        return None
    return yi


def extract_papers_by_author_phrase(query: str) -> str | None:
    """e.g. 'List papers by Jhanjhi' → 'Jhanjhi'."""
    patterns = [
        r"\b(?:list|show|give)?\s*(?:me\s+)?(?:all\s+)?papers?\s+by\s+(.+?)(?:\s*[\.,;]|$)",
        r"\barticles?\s+by\s+(.+?)(?:\s*[\.,;]|$)",
        r"\bpapers?\s+by\s+(.+?)(?:\s*[\.,;]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, (query or "").strip(), re.I)
        if m:
            phrase = m.group(1).strip(" .,;:\"'")
            if len(phrase) >= 2:
                return phrase
    return None


def _format_numbered_list(header: str, lines: list[str]) -> str:
    """Use blank lines between entries so the UI renders readable lists."""
    body = "\n\n".join(lines)
    return f"{header}\n\n{body}"


def answer_catalog_metadata_query(query: str, papers_metadata: dict) -> str | None:
    """Deterministic answers for library inventory questions (no LLM)."""
    if not papers_metadata or not is_catalog_metadata_query(query):
        return None
    q = (query or "").lower()

    year_range = _extract_year_range(query)
    if year_range:
        start_year, end_year = year_range
        in_range: list[tuple[str, dict[str, Any]]] = []
        for title, meta in papers_metadata.items():
            y = meta.get("year")
            try:
                yi = int(y) if y is not None and str(y).strip().isdigit() else None
            except Exception:
                yi = None
            if yi is None:
                continue
            if start_year <= yi <= end_year:
                in_range.append((title, meta))

        # Sort by year then title for stable output.
        in_range.sort(key=lambda tm: (int(tm[1].get("year") or 0), tm[0].lower()))
        lines = []
        for i, (title, meta) in enumerate(in_range, 1):
            lines.append(
                f"{i}. {meta.get('authors', 'Unknown')} ({meta.get('year', 'N/A')}). {title}"
            )
        header = (
            f"Papers in your ingested library published between {start_year} and {end_year} "
            f"({len(lines)} paper(s)):"
        )
        if not lines:
            return (
                f"I could not find any ingested papers in your knowledge base published between "
                f"{start_year} and {end_year}."
            )
        return _format_numbered_list(header, lines)

    year_single = _extract_single_year(query)
    if year_single is not None:
        in_year: list[tuple[str, dict[str, Any]]] = []
        for title, meta in papers_metadata.items():
            y = meta.get("year")
            try:
                yi = int(y) if y is not None and str(y).strip().isdigit() else None
            except Exception:
                yi = None
            if yi == year_single:
                in_year.append((title, meta))

        in_year.sort(key=lambda tm: (tm[0].lower(), str(tm[1].get("authors") or "").lower()))
        lines = []
        for i, (title, meta) in enumerate(in_year, 1):
            lines.append(
                f"{i}. {meta.get('authors', 'Unknown')} ({meta.get('year', 'N/A')}). {title}"
            )
        header = (
            f"Papers in your ingested library published in {year_single} "
            f"({len(lines)} paper(s)):"
        )
        if not lines:
            return (
                f"I could not find any ingested papers in your knowledge base published in {year_single}."
            )
        return _format_numbered_list(header, lines)

    author_phrase = extract_papers_by_author_phrase(query)
    if author_phrase or re.search(r"\bpapers?\s+by\s+|\barticles?\s+by\s+", q):
        phrase = author_phrase or re.split(r"\bby\s+", query, maxsplit=1, flags=re.I)[-1].strip()
        _ph, scoped = resolve_author_from_library(f"papers by {phrase}", papers_metadata)
        if not scoped:
            scoped = resolve_matching_paper_titles(f"papers by {phrase}", papers_metadata)
        if not scoped:
            return NOT_IN_LIBRARY_REFUSAL
        inv = {t: papers_metadata[t] for t in scoped if t in papers_metadata}
        lines = []
        for i, title in enumerate(sorted(inv.keys()), 1):
            m = inv[title]
            lines.append(
                f"{i}. {m.get('authors', 'Unknown')} ({m.get('year', 'N/A')}). {title}"
            )
        return _format_numbered_list(
            f"Papers by {phrase} in your ingested library ({len(lines)} paper(s)):",
            lines,
        )

    if re.search(r"\bhow many\b.{0,30}\b(papers?|articles?)\b", q):
        return (
            f"Your knowledge base contains **{len(papers_metadata)}** ingested paper(s)."
        )
    if re.search(r"\b(list|show|name|who|all)\b.{0,40}\b(authors?|writers?)\b", q):
        authors = list_distinct_authors(papers_metadata)
        if not authors:
            return "No authors found in the ingested library."
        lines = [f"{i}. {a['authors']} ({a['paper_count']} paper(s))" for i, a in enumerate(authors, 1)]
        return _format_numbered_list("Authors in your ingested library:", lines)

    if re.search(r"\blist\b.{0,40}\b(all\s+)?(ingested\s+)?(papers?|articles?)\b", q) and not re.search(
        r"\bpapers?\s+by\s+|\barticles?\s+by\s+", q
    ):
        titles = sorted(papers_metadata.keys())
        lines = []
        for i, title in enumerate(titles, 1):
            m = papers_metadata[title]
            lines.append(
                f"{i}. {m.get('authors', 'Unknown')} ({m.get('year', 'N/A')}). {title}"
            )
        return _format_numbered_list("Papers in your ingested library:", lines)

    return None


def apply_verification_or_refuse(
    answer: str,
    *,
    scope: QueryScope,
    papers_metadata: dict,
    chunks: list[dict[str, Any]],
) -> tuple[str, bool]:
    """Returns (final_answer, passed)."""
    if not getattr(settings, "RAG_STRICT_MODE", True):
        return answer, True
    ok, reason = verify_answer_against_scope(
        answer,
        scoped_titles=scope.scoped_titles,
        papers_metadata=papers_metadata,
        chunks=chunks,
        scope_locked=scope.is_locked,
    )
    if ok:
        return answer, True
    return VERIFICATION_FAILED_REFUSAL, False
