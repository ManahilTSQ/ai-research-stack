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
    author_phrase_tokens,
    query_expects_named_author,
    resolve_matching_paper_titles,
    _significant_query_tokens,
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
        return bool(self.scoped_titles) and self.requires_entity


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


def build_catalog_indexes(papers_metadata: dict) -> dict[str, Any]:
    """Build author → titles and normalized lookup structures from library metadata."""
    author_to_titles: dict[str, list[str]] = {}
    title_lower_map: dict[str, str] = {}

    for title, meta in (papers_metadata or {}).items():
        title_lower_map[title.lower().strip()] = title
        for token in _author_tokens_from_string(meta.get("authors") or ""):
            author_to_titles.setdefault(token, [])
            if title not in author_to_titles[token]:
                author_to_titles[token].append(title)

    manifest = _load_manifest()
    for _fn, meta in manifest.items():
        m_title = (meta.get("title") or "").strip()
        m_authors = meta.get("authors") or ""
        if not m_title:
            continue
        for db_title in papers_metadata:
            if (
                db_title.lower().strip() == m_title.lower()
                or db_title.lower() in m_title.lower()
                or m_title.lower() in db_title.lower()
            ):
                for token in _author_tokens_from_string(m_authors):
                    author_to_titles.setdefault(token, [])
                    if db_title not in author_to_titles[token]:
                        author_to_titles[token].append(db_title)
                break

    return {
        "author_to_titles": author_to_titles,
        "title_lower_map": title_lower_map,
        "all_titles": list(papers_metadata.keys()),
    }


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
    """Match paper titles via quoted strings or distinctive title-token overlap."""
    quoted = extract_quoted_paper_titles(query)
    matches: list[str] = []
    for q in quoted:
        ql = q.lower().strip()
        for title in papers_metadata:
            tl = title.lower()
            if ql in tl or tl in ql:
                if title not in matches:
                    matches.append(title)

    if matches:
        return matches

    tokens = _significant_query_tokens(query)
    if not tokens:
        return []

    scored: list[tuple[int, str]] = []
    for title, meta in papers_metadata.items():
        hay = f"{title} {meta.get('authors', '')}".lower()
        score = sum(1 for t in tokens if t in hay)
        if score >= 2 or (len(tokens) == 1 and tokens[0] in title.lower()):
            scored.append((score, title))
    if not scored:
        return []
    scored.sort(reverse=True)
    best_score = scored[0][0]
    return [t for s, t in scored if s == best_score and s >= 2]


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

    author_phrase = extract_author_search_phrase(query)
    if author_phrase or query_expects_named_author(query):
        titles = resolve_matching_paper_titles(query, papers_metadata)
        return QueryScope(
            scoped_titles=titles,
            requires_entity=True,
            entity_kind="author",
            author_phrase=author_phrase,
        )

    paper_titles = _fuzzy_title_match(query, papers_metadata)
    if paper_titles:
        return QueryScope(
            scoped_titles=paper_titles,
            requires_entity=True,
            entity_kind="paper",
        )

    # Topic-style queries: require at least one library paper to mention the topic.
    topic_tokens = _significant_query_tokens(query)
    q_lower = (query or "").lower()
    topic_cues = (
        "about", "on ", " regarding ", " related to ", " topic ", " theme ",
        "discuss", "discusses", "cover", "covers", "concerning",
    )
    if topic_tokens and any(c in q_lower for c in topic_cues):
        topic_papers = _papers_matching_topic_tokens(topic_tokens, papers_metadata)
        return QueryScope(
            scoped_titles=topic_papers,
            requires_entity=True,
            entity_kind="topic",
            topic_tokens=topic_tokens,
        )

    # General library question — no hard lock, but verification still applies.
    matched = resolve_matching_paper_titles(query, papers_metadata)
    return QueryScope(
        scoped_titles=matched,
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


def verify_answer_against_scope(
    answer: str,
    *,
    scoped_titles: list[str],
    papers_metadata: dict,
    chunks: list[dict[str, Any]],
    strict: bool = True,
) -> tuple[bool, str]:
    """
    Return (ok, reason). When not ok, caller should replace answer with refusal.
    """
    if not strict or not answer or not papers_metadata:
        return True, ""

    if TABLE_TRUNCATION_RE.search(answer):
        return False, "table_truncation"

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
        if len(tl) < 12:
            continue
        # Distinctive title substring appeared but paper not in scope.
        if tl in answer_lower:
            return False, f"out_of_scope_title:{title[:60]}"
        words = [w for w in re.findall(r"[a-z]{5,}", tl) if w not in {"paper", "study", "analysis"}]
        if len(words) >= 2 and all(w in answer_lower for w in words[:3]):
            return False, f"out_of_scope_title_words:{title[:60]}"

    all_author_tokens: dict[str, set[str]] = {}
    for title, meta in papers_metadata.items():
        if title in allowed_titles:
            continue
        for token in _author_tokens_from_string(meta.get("authors") or ""):
            all_author_tokens.setdefault(token, set()).add(title)

    for token, titles in all_author_tokens.items():
        if len(token) < 5:
            continue
        if token in allowed_author_tokens:
            continue
        if token in answer_lower:
            return False, f"out_of_scope_author:{token}"

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
    return False


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

    author_phrase = extract_papers_by_author_phrase(query)
    if author_phrase or re.search(r"\bpapers?\s+by\s+|\barticles?\s+by\s+", q):
        phrase = author_phrase or re.split(r"\bby\s+", query, maxsplit=1, flags=re.I)[-1].strip()
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
    )
    if ok:
        return answer, True
    return VERIFICATION_FAILED_REFUSAL, False
