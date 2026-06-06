"""
rag_strict.py — Deterministic scope resolution and answer verification for RAG.

Ensures answers only reference papers/authors in scope and refuses when entities
are not in the knowledge base.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any

from config import settings

logger = logging.getLogger(__name__)
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
    extract_multi_author_phrases,
    resolve_coauthored_papers,
    author_field_contains_token,
)

_COMPARE_QUERY_RE = re.compile(r"\bcompare\b", re.I)

_YEAR_RANGE_RE = re.compile(
    r"\b(?:between|from)\s+(20\d{2})\s+(?:and|to|-)\s+(20\d{2})\b|"
    r"\b(20\d{2})\s*-\s*(20\d{2})\b",
    re.I,
)

_YEAR_SINGLE_RE = re.compile(
    r"\b(?:published\s+in|in|from|of|year|during)?\s*(20\d{2})\b|\b(20\d{2})\s+papers?\b",
    re.I,
)

_KEYWORD_DISCOVERY_RE = re.compile(
    r"\b(?:do|does|are there|which)\s+(?:any\s+)?(?:of\s+)?(?:my\s+)?papers?\s+"
    r"(?:discuss|mention|cover|address|use|include|contain)\b|"
    r"\bwhich\s+papers?\s+(?:discuss|mention|use|cover)\b|"
    # IMPORTANT: only match plural "papers on/about/regarding ...".
    # Singular phrasing like "a paper on X" is commonly a WRITING request
    # (e.g., "Draft an introduction for a paper on X") and must NOT be routed
    # into library keyword discovery.
    r"\bpapers\s+(?:on|about|regarding)\s+\w|"
    r"\b(?:most\s+)?relevant\s+to\b|"
    r"\brelated\s+to\b|"
    r"\bcitation-?worthy\b|"
    r"\bfor\s+(?:a\s+)?survey\s+on\b|"
    r"\bbibliograph(?:y|ies)\b|"
    # Listing queries that should be handled by metadata filtering, not LLM
    r"\blist\s+(?:only\s+)?(?:papers?|articles?|studies?)\b",
    re.I,
)

_CATALOG_BLOCKERS = [
    r"\banalyze\b", r"\banalysis\b", r"\banalyzing\b",
    r"\bidentify\b", r"\bexamine\b", r"\binvestigate\b",
    r"\bstep-?by-?step\b", r"\bpipeline\b",
    r"\bthreshold\b", r"\bparameter\b", r"\bmethodolog",
    r"\bapproach\s+used\b", r"\btechnique\s+used\b",
    r"\binformation\s+gain\b", r"\bfeature\s+selection\b",
    r"\bwhat\s+(?:thresholds?|parameters?|criteria|method|approach)\b",
    r"\bhow\s+(?:did|does|do)\b",
    r"\bexact\s+(?:step|pipeline|method|parameter)\b",
    r"\bcompar(?:e|ison|ative)\b",
    r"\bcontribution\b", r"\bfinding\b", r"\bframework\b",
    r"\bexplain\b", r"\bdescribe\b", r"\bdetail\b",
    r"\bsummarize\b", r"\bsynthesize\b", r"\bcompare\b",
    r"\bevaluate\b", r"\bassess\b", r"\bdetermine\b",
    r"\bwhich\s+.*\s+(?:use|used|uses|using)\b",
    r"\bwhat\s+(?:method|approach|technique|strategy)\b",
    r"\bhow\s+(?:to|do|does|did)\b",
]

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
        return self.entity_kind in ("author", "paper", "filter")


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
    """Extract 4+ character surname tokens, with Unicode normalization.

    Handles fi/fl ligatures (e.g. Aldughay\ufb01q → aldughayfiq) and
    diacritic marks (e.g. Gra\xf1a → grana) so APA citation verification
    does not false-positive on papers with non-ASCII author names.
    """
    import unicodedata
    tokens: set[str] = set()
    # NFKD decomposition: ligatures + diacritics → base ASCII chars
    normalized = unicodedata.normalize("NFKD", (authors or ""))
    normalized = normalized.replace("\ufb01", "fi").replace("\ufb02", "fl")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    for part in re.split(r"[,;&]| and ", normalized.lower()):
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
    q = (query or "").lower()
    # Exclude meta-questions about missing full text papers
    if "full text" in q or "pdf" in q:
        if "did not" in q or "not find" in q or "would have" in q or "wanted" in q:
            return False
    # Check catalog blockers (analytical/synthesis queries)
    if any(re.search(pat, q) for pat in _CATALOG_BLOCKERS):
        return False
    return bool(_KEYWORD_DISCOVERY_RE.search(q))


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
    
    # Detect corpus-level comparison queries (comparing methodologies/topics across all papers)
    # These should NOT require paper pickers
    q = (query or "").lower()
    corpus_comparison_patterns = [
        r"\bacross\s+(all\s+)?(papers?|articles?|library)",
        r"\ball\s+(papers?|articles?|library)",
        r"\bmethodolog",
        r"\bapproaches?\s+across",
        r"\btechnolog",
    ]
    if any(re.search(pat, q) for pat in corpus_comparison_patterns):
        return False
    
    quoted = extract_quoted_paper_titles(query)
    if len(quoted) >= 2:
        resolved = fuzzy_match_paper_titles(query, papers_metadata)
        return len(resolved) < 2
    
    # NEW: Extract unquoted paper names from comparison queries
    # Pattern: "compare the X paper and the Y paper" or "compare X and Y"
    paper_name_patterns = [
        r"(?:the\s+)?([A-Z][A-Za-z0-9\s\-]{3,50})\s+paper",
        r"(?:the\s+)?([A-Z][A-Za-z0-9\s\-]{3,50})\s+(?:model|approach|method|system)",
    ]
    
    extracted_names = []
    for pattern in paper_name_patterns:
        matches = re.findall(pattern, query)
        extracted_names.extend([m.strip() for m in matches if len(m.strip()) >= 3])
    
    # Also try to extract capitalized phrases that might be paper names
    # Pattern: "Compare SimCLR-GRU and BERT ensemble"
    capitalized_phrases = re.findall(r'\b([A-Z][a-zA-Z0-9\-]{2,}(?:\s+[A-Z][a-zA-Z0-9\-]{2,})*)\b', query)
    extracted_names.extend(capitalized_phrases)
    
    # Remove duplicates and very short matches
    extracted_names = list(set([n for n in extracted_names if len(n) >= 3]))
    
    if len(extracted_names) >= 2:
        # Try to resolve these names against the library
        resolved_count = 0
        for name in extracted_names:
            matched = fuzzy_match_paper_titles(name, papers_metadata)
            if matched:
                resolved_count += 1
        
        # If we can resolve at least 2 papers, don't need pickers
        if resolved_count >= 2:
            logger.info(f"Comparison query resolved {resolved_count} papers from extracted names: {extracted_names}")
            return False
    
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

    # Check paper titles first to prevent false-positive author/topic matching
    # (e.g. queries matching "contribution of Xception" being treated as author queries)
    paper_titles = fuzzy_match_paper_titles(query, papers_metadata)
    if paper_titles:
        if len(paper_titles) == 1 or query_has_paper_focus(query):
            return QueryScope(
                scoped_titles=paper_titles,
                requires_entity=True,
                entity_kind="paper",
            )

    # ── Multi-author co-authorship: "papers by X and Y" → intersection ──────
    # Must run BEFORE single-author resolution so "Stiawan and Budiarto" is
    # not accidentally matched as just "Budiarto" (the second surname).
    multi_authors = extract_multi_author_phrases(query)
    if multi_authors:
        coauthored = resolve_coauthored_papers(multi_authors, papers_metadata)
        if coauthored:
            return QueryScope(
                scoped_titles=coauthored,
                requires_entity=True,
                entity_kind="author",
                author_phrase=" & ".join(multi_authors),
            )
        # Both authors exist individually but share no papers.
        return QueryScope(
            scoped_titles=[],
            requires_entity=True,
            entity_kind="author",
            author_phrase=" & ".join(multi_authors),
        )

    # ── Single-author co-authorship: "papers co-authored by X" ───────────────
    # Handle queries like "Which papers were co-authored by M. Humayun?"
    from rag_context import extract_single_author_coauthor_query
    single_coauthor = extract_single_author_coauthor_query(query)
    if single_coauthor:
        author_phrase, author_titles = resolve_author_from_library(query, papers_metadata)
        if author_titles:
            return QueryScope(
                scoped_titles=author_titles,
                requires_entity=True,
                entity_kind="author",
                author_phrase=author_phrase or single_coauthor,
            )
        # Author not found in library
        return QueryScope(
            scoped_titles=[],
            requires_entity=True,
            entity_kind="author",
            author_phrase=single_coauthor,
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

    # ── Comparison/synthesis query detection ──
    is_compare = bool(re.search(r"\b(?:compare|comparison|contrast|difference|differences|similarities|versus|vs)\b", q_lower))
    if is_compare:
        # Extract topic keywords (excluding comparison words and query stopwords)
        compare_stopwords = {"compare", "comparison", "contrast", "difference", "differences", 
                             "similarities", "versus", "vs", "all", "papers", "paper", "articles", "article",
                             "in", "my", "library", "ingested", "studies", "study"}
        tokens = [t for t in topic_tokens if t not in compare_stopwords]
        if tokens:
            matched_papers = find_papers_by_metadata_keywords(query, papers_metadata)
            if matched_papers:
                return QueryScope(
                    scoped_titles=matched_papers,
                    requires_entity=True,
                    entity_kind="topic",
                    topic_tokens=tokens,
                )

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
    
    # ── Direct listing query with topic keywords (e.g., "List SDN papers") ──
    # This handles cases where query_has_library_topic_cue might not trigger
    # but the query clearly has topic keywords for filtering
    if re.search(r'\b(?:list|show|table)\s+(?:\w+\s+){1,}(?:papers?|articles?|studies)\b', q_lower):
        # Extract topic keywords by removing listing words
        listing_words = {"list", "show", "table", "papers", "paper", "articles", "article", "studies", "study", "all"}
        topic_keywords = [t for t in topic_tokens if t not in listing_words]
        if topic_keywords:
            # Try to match papers by these topic keywords
            matched_papers = find_papers_by_metadata_keywords(query, papers_metadata)
            if matched_papers:
                return QueryScope(
                    scoped_titles=matched_papers,
                    requires_entity=True,
                    entity_kind="topic",
                    topic_tokens=topic_keywords,
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

_APA_CITATION_RE = re.compile(
    r"\(([^()]{1,80}?),\s*(20\d{2})(?:\s*[,)]|\s*p\.|\s*pp\.)",
    re.I,
)


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

    # Only enforce citation author-token scope when the query is locked to a specific
    # author/paper/topic/filter. For general library-wide questions, retrieved chunks
    # are a sample and may not include every in-library author token.
    if locked:
        # Disallow citations whose author token is not present in allowed evidence.
        # This blocks invented "Wang et al., 2022" style citations.
        for m in _APA_CITATION_RE.finditer(answer):
            author_blob = (m.group(1) or "").strip()
            year = m.group(2)
            # Extract candidate surname tokens (ignore "et al.", "&", initials).
            parts = re.split(r"&|and", author_blob)
            for part in parts:
                token = re.sub(r"[^A-Za-z\u00C0-\u017F\-’']", " ", part).strip().lower()
                if not token:
                    continue
                # Prefer last word as surname (e.g., "Noor Zaman Jhanjhi" -> "jhanjhi").
                words = [w for w in re.split(r"\s+", token) if w and w not in {"et", "al"}]
                if not words:
                    continue
                surname = words[-1]
                # Short surnames / initials are ignored.
                if len(surname) < 4:
                    continue
                if surname not in allowed_author_tokens:
                    # Grounded-citation fallback: if the surname appears in the text of
                    # the retrieved chunks, the LLM is citing a researcher mentioned
                    # *within* the scoped papers (e.g. a referenced dataset author,
                    # a standard algorithm like Quinlan/J48, or a cited dataset creator).
                    # This is accurate and expected — it is NOT a hallucination.
                    chunk_text_blob = " ".join(
                        (c.get("text") or "").lower() for c in chunks
                    )
                    if surname in chunk_text_blob:
                        # Citation is grounded in retrieved evidence — allow it.
                        continue
                    # Also allow very common algorithm/method author names that
                    # frequently appear in academic IDS / network security papers.
                    _COMMON_METHOD_AUTHORS = frozenset({
                        "quinlan", "breiman", "shannon", "vapnik", "friedman",
                        "lecun", "cortes", "Cover", "lashkari", "sharafaldin",
                        "habibi", "tavallaee",
                    })
                    if surname in _COMMON_METHOD_AUTHORS:
                        continue
                    return False, f"out_of_scope_citation:{surname}:{year}"

    # Co-authors on scoped papers are allowed; author-token checks above only target citations.
    return True, ""


def is_catalog_metadata_query(query: str) -> bool:
    q = (query or "").lower()

    # ── Meta-questions about missing papers/gaps ─────────────────────────────
    # These should be handled by the missing papers logic, not catalog metadata
    if "full text" in q or "pdf" in q:
        if "did not" in q or "not find" in q or "would have" in q or "wanted" in q:
            return False

    # ── Topic-search queries should NOT be routed to catalog metadata ─────────
    # "Find papers about ransomware" / "Show papers related to SDN" are topic searches.
    # Even if they mention "authors" or "metadata" as display fields, the primary
    # intent is topic search — not catalog listing. Return False so keyword discovery
    # or semantic search handles them.
    if re.search(
        r"\b(?:find|search|get|retrieve)\s+papers?\s+(?:about|on|related\s+to|concerning|mentioning)\b",
        q, re.I
    ):
        return False

    # ── Specific paper author query route-to-RAG ──────────────────────────────
    # Route queries asking about the authors / writers of a specific paper/topic to RAG
    if re.search(r"\b(?:who\s+(?:wrote|authored)|authors?\s+of)\s+(?:the|this|a|an)?\s*(?:papers?|articles?|publications?|studies|study)\b", q):
        return False

    # ── Multi-field metadata query with a specific paper title ────────────────
    # e.g. "What are the authors, venue, and year of the paper 'X'?"
    # Detected by: quoted title OR 'paper'/'article' mention + 2+ metadata fields.
    _META_FIELDS = {"author", "authors", "venue", "year", "doi", "journal", "published"}
    _meta_hits = sum(1 for f in _META_FIELDS if f in q)
    if _meta_hits >= 2 and re.search(
        r"[\"\'].*?[\"\']|\bpaper\b|\barticle\b|\bpublication\b", q
    ):
        return True

    # ── Individual paper metadata queries ─────────────────────────────────────
    if any(pat in q for pat in ["who wrote", "who authored", "who are the authors", "author of", "authors of", "who is the author",
                                "what year", "when was", "publication year", "published in what year",
                                "which journal", "which venue", "where was", "published in", "journal of the", "published by which",
                                "doi of", "what is the doi"]):
        if re.search(r"\b(papers?|articles?|publications?|studies|study)\b", q):
            return True

    # ── Content-analysis early exit ───────────────────────────────────────────
    # Even if the query contains "papers by X", if it also asks for analytical
    # content (pipelines, thresholds, methodology, etc.) it must NOT be handled
    # by the deterministic catalog path — route it to LLM content synthesis.
    if any(re.search(pat, q) for pat in _CATALOG_BLOCKERS):
        return False

    # Author query patterns - match any query asking about papers by a specific author
    author_patterns = [
        r"\bpapers?\s+by\s+",
        r"\barticles?\s+by\s+",
        r"\bauthored\s+by\s+",
        r"\bwritten\s+by\s+",
        r"\bwho\s+wrote\s+",
        r"\bwhich\s+paper\s+was\s+written\s+by\s+",
        r"\bwhich\s+paper\s+by\s+",
        r"\bwhich\s+article\s+by\s+",
        r"\bfind\s+papers?\s+by\s+",
        r"\bfind\s+articles?\s+by\s+",
        r"\bshow\s+papers?\s+by\s+",
        r"\bshow\s+articles?\s+by\s+",
        r"\blist\s+papers?\s+by\s+",
        r"\blist\s+articles?\s+by\s+",
        r"\bpapers?\s+from\s+",
        r"\barticles?\s+from\s+",
    ]
    if any(re.search(pat, q) for pat in author_patterns):
        return True
    # Match "list all authors" / "show all authors" but NOT:
    # - "list articles with Jhanjhi as author or co-author" (role context)
    # - "show full metadata including authors" (authors is a display field, not subject)
    # - "show papers about X ... authors and year" (topic search mentioning author field)
    if (re.search(r"\b(list|show|name|who|all)\b.{0,40}\b(all\s+)?(authors?|writers?)\b", q)
            and not re.search(r"\bas\s+(an?\s+)?(?:author|co-?author)\b", q)
            and not re.search(r"\b(?:including|metadata|complete|full)\b", q)
            and "and year" not in q and "and venue" not in q):
        return True
    if re.search(r"\bhow many\b.{0,30}\b(papers?|articles?)\b", q):
        return True
    # Check for topic-filtered listing queries (e.g., "List malware detection papers")
    # These should NOT be treated as general "list all papers" queries
    topic_filtered_list = re.search(r"\blist\s+(?:only\s+)?(?:\w+\s+){1,}(?:papers?|articles?|studies)\b", q)
    if topic_filtered_list:
        return False  # Let topic filtering handle this
    
    # General listing queries without topic filtering
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


def answer_individual_paper_metadata_query(query: str, papers_metadata: dict) -> str | None:
    q = (query or "").lower().strip()
    
    # 1. Identify target field (or "all" for multi-field requests)
    target_field = None
    _META_FIELD_COUNT = sum(1 for f in ("author", "authors", "venue", "year", "doi") if f in q)
    if (any(pat in q for pat in ["who wrote", "who authored", "who are the authors", "who is the author", "list the authors of", "show the authors of", "name the authors of", "what are the authors of", "what is the authors of"])
        or (("author" in q or "authors" in q) and any(pat in q for pat in ["what is the name of the", "what are the names of the", "names of the author", "name of the author"]))):
        target_field = "authors"
    elif any(pat in q for pat in ["what year", "when was", "publication year", "published in what year"]):
        target_field = "year"
    elif any(pat in q for pat in ["which journal", "which venue", "where was", "published in", "journal of the", "published by which", "which journal or conference"]):
        target_field = "venue"
    elif "doi" in q:
        target_field = "doi"

    # Multi-field request: "What are the authors, venue, and year of X?"
    # When 2+ metadata fields are requested together, return a combined answer.
    if not target_field and _META_FIELD_COUNT >= 2:
        target_field = "all"

    if not target_field:
        return None
        
    # 2. Match papers
    matched_papers = fuzzy_match_paper_titles(query, papers_metadata)
    if not matched_papers:
        # Extract title keywords with stricter matching
        words = [w.strip(".,:;?()[]\"'") for w in q.split()]
        stop_words = {
            "who", "wrote", "authored", "are", "the", "authors", "author", "of", "paper", "papers",
            "article", "articles", "publication", "publications", "on", "in", "what", "year", "when",
            "was", "published", "which", "journal", "venue", "where", "by", "doi", "the", "a", "an",
            "detection", "using", "based", "system", "approach", "model", "analysis", "for", "methods",
            "techniques", "algorithms", "studies", "study", "list", "show", "give", "name", "names"
        }
        keywords = [w for w in words if len(w) >= 3 and w not in stop_words]
        if keywords:
            scored = []
            for title in papers_metadata:
                title_lower = title.lower()
                # Require ALL keywords to be present for a match (stricter)
                hits = sum(1 for kw in keywords if kw in title_lower)
                # Only consider matches with high keyword coverage (>= 70% of keywords)
                if hits >= len(keywords) * 0.7:
                    scored.append((hits, title))
            if scored:
                scored.sort(reverse=True)
                best_hits = scored[0][0]
                # Only return papers with the best score, and require at least 2 keyword matches
                if best_hits >= 2:
                    matched_papers = [t for h, t in scored if h == best_hits]
                
    # Allow up to 5 matches for DOI queries to increase recall
    max_matches = 5 if target_field == "doi" else 3
    if not matched_papers or len(matched_papers) > max_matches:
        return None
        
    # 3. Format answer
    answers = []
    for title in matched_papers:
        meta = papers_metadata[title]

        if target_field == "all":
            # Return all available metadata fields in one structured answer
            authors_val = meta.get("authors", "Unknown Authors")
            year_val = meta.get("year", "N/A")
            venue_val = meta.get("venue") or "N/A"
            doi_val = meta.get("doi") or "N/A"
            parts = [
                f"**Title:** {title}",
                f"**Authors:** {authors_val}",
                f"**Year:** {year_val}",
                f"**Venue:** {venue_val}",
            ]
            if doi_val and doi_val != "N/A":
                parts.append(f"**DOI:** https://doi.org/{doi_val}")
            answers.append("\n".join(parts))
        else:
            val = meta.get(target_field)
            if not val or val == "N/A":
                continue
            if target_field == "authors":
                answers.append(f"The author(s) of the paper \"{title}\" are: {val}.")
            elif target_field == "year":
                answers.append(f"The paper \"{title}\" was published in {val}.")
            elif target_field == "venue":
                answers.append(f"The paper \"{title}\" was published in {val}.")
            elif target_field == "doi":
                answers.append(f"The DOI of the paper \"{title}\" is {val}.")

    if not answers:
        return None

    references_list = []
    for title in matched_papers:
        meta = papers_metadata[title]
        ref = f"{meta.get('authors', 'Unknown')} ({meta.get('year', 'N/A')}). {title}."
        v = meta.get("venue")
        if v and v != "N/A":
            ref += f" {v}."
        d = meta.get("doi")
        if d and d != "N/A":
            ref += f" https://doi.org/{d}"
        references_list.append(ref)

    ref_block = "\n\nReferences:\n\n" + "\n\n".join(references_list)
    return "\n\n".join(answers) + ref_block


def answer_catalog_metadata_query(query: str, papers_metadata: dict) -> str | None:
    """Deterministic answers for library inventory questions (no LLM)."""
    if not papers_metadata or not is_catalog_metadata_query(query):
        return None
    q = (query or "").lower()

    # ── Deterministic direct-match for exact keyword lookups in titles ──
    # Handle queries like "Which papers contain 'Deep Learning' in the title?"
    # Also handle variations without quotes: "Which papers contain Deep Learning in the title"
    title_keyword_match = re.search(r'\b(?:which|what|list)\s+(?:papers?|articles?)\s+(?:contain|have|with)\s+(?:[\'"]?(.+?)[\'"]?\s+)?(?:in\s+)?(?:the\s+)?title\b', q)
    if title_keyword_match:
        keyword = title_keyword_match.group(1).strip().lower()
        # Remove any trailing punctuation or stop words
        keyword = re.sub(r'[\'".,;:?!\s]+$', '', keyword)
        matched = []
        for title, meta in papers_metadata.items():
            if keyword in title.lower():
                matched.append((title, meta))
        if matched:
            lines = []
            for title, meta in matched:
                lines.append(f"{meta.get('authors', 'Unknown')} ({meta.get('year', 'N/A')}). {title}")
            header = f"Papers in your ingested library containing '{keyword}' in the title ({len(lines)} paper(s)):"
            return _format_numbered_list(header, lines)
        else:
            return f"No papers in your ingested library contain '{keyword}' in the title."

    # Deterministic metadata answers for specific paper questions
    individual_ans = answer_individual_paper_metadata_query(query, papers_metadata)
    if individual_ans:
        return individual_ans

    # ── Author existence check — refuse queries about authors not in library ──
    if query_expects_named_author(query):
        # Skip checking if the query is actually targeting a paper title in our library.
        has_paper_focus = False
        if fuzzy_match_paper_titles(query, papers_metadata):
            has_paper_focus = True
            
        author_phrase = extract_author_search_phrase(query)
        if not author_phrase:
            # Fallback
            author_phrase, _ = resolve_author_from_library(query, papers_metadata)
            
        if author_phrase and author_phrase in papers_metadata:
            has_paper_focus = True
            
        if not has_paper_focus and author_phrase:
            author_exists = verify_author_exists_in_library(author_phrase, papers_metadata)
            if not author_exists:
                # Double-check
                _, resolved_papers = resolve_author_from_library(query, papers_metadata)
                if not resolved_papers:
                    return f"No papers authored by {author_phrase} were found in the ingested library."

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

        # ── Multi-author AND intersection ─────────────────────────────────────
        # "papers by Stiawan and Budiarto" must return papers co-authored by
        # BOTH, not the union of each author's individual paper set.
        multi = extract_multi_author_phrases(query)
        if multi:
            scoped = resolve_coauthored_papers(multi, papers_metadata)
            if scoped:
                inv = {t: papers_metadata[t] for t in scoped if t in papers_metadata}
                lines = []
                for i, title in enumerate(sorted(inv.keys()), 1):
                    m = inv[title]
                    lines.append(
                        f"{i}. {m.get('authors', 'Unknown')} ({m.get('year', 'N/A')}). {title}"
                    )
                label = " & ".join(multi)
                return _format_numbered_list(
                    f"Papers co-authored by {label} in your ingested library ({len(lines)} paper(s)):",
                    lines,
                )
            else:
                return (
                    f"I could not find any papers in your knowledge base that are "
                    f"co-authored by both {' and '.join(multi)}. "
                    "Please check the author spellings or ingest more papers."
                )

        # ── Single-author listing ─────────────────────────────────────────────
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
        matched_venue = None

        # ── Skip venue matching for topic/relation queries ────────────────────
        # "List papers related to SDN-based security" should NOT accidentally match
        # "Journal of Information Security" via the word "security".
        # Only apply venue matching when the user explicitly names a venue/journal.
        _is_topic_query = bool(re.search(
            r"\b(?:related|about|on|concerning|regarding|focus|covering|mentioning)\b", q, re.I
        ))
        if not _is_topic_query:
            venues = set()
            for meta in papers_metadata.values():
                v = meta.get("venue")
                if v and v != "N/A":
                    venues.add(v)

            for v in venues:
                norm_v = v.lower().replace("&amp;", "&").strip()
                # Full venue name must appear literally in query
                if norm_v in q:
                    matched_venue = v
                    break
                # Keyword match: require a venue-specific word (not a generic topic word)
                _VENUE_GENERIC = {
                    "journal", "conference", "transactions", "letters",
                    "proceedings", "reports", "report", "review", "research",
                    # Common words that should NEVER alone trigger a venue match:
                    "security", "network", "computing", "intelligence", "science",
                    "engineering", "technology", "systems", "applications",
                    "information", "international", "digital",
                }
                words = [w.strip(".,:;()[]") for w in norm_v.split()]
                specific_words = [w for w in words if len(w) >= 5 and w not in _VENUE_GENERIC]
                # Require at least 2 specific venue words to match — prevents
                # 'security' alone matching 'Journal of Information Security'
                if len(specific_words) >= 2 and sum(1 for w in specific_words if w in q) >= 2:
                    matched_venue = v
                    break

        titles = sorted(papers_metadata.keys())
        if matched_venue:
            filtered_titles = []
            for t in titles:
                v = papers_metadata[t].get("venue", "")
                if v and (matched_venue.lower() in v.lower() or v.lower() in matched_venue.lower()):
                    filtered_titles.append(t)
            lines = []
            for i, title in enumerate(filtered_titles, 1):
                m = papers_metadata[title]
                lines.append(
                    f"{i}. {m.get('authors', 'Unknown')} ({m.get('year', 'N/A')}). {title}"
                )
            header = f"Papers in your ingested library published in {matched_venue} ({len(lines)} paper(s)):"
            if not lines:
                return f"I could not find any ingested papers in your library published in {matched_venue}."
            return _format_numbered_list(header, lines)

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
    """
    Returns (final_answer, passed).

    Scope-checks the answer body against the resolved paper scope.
    Reference integrity (hallucination blocking) is handled upstream
    by _build_safe_references() which enforces a strict library-catalog
    guard before any title is rendered.

    NOTE: strip_unverified_citations is intentionally NOT called here.
    By the time this function executes, _bind_citations_and_verify has
    already resolved all (doc_X) placeholders into (Author et al., Year)
    APA inline citations. Calling strip_unverified_citations at this point
    causes it to mis-match its own valid_pairs format and delete correct
    citations, which was the primary cause of citations disappearing.
    """
    if not answer or not answer.strip():
        return answer, True

    # Isolate References section from verification to prevent mutilation
    answer_body = answer
    refs_part = ""
    if "References:" in answer:
        parts = answer.split("References:", 1)
        answer_body = parts[0].strip()
        refs_part = "References:\n" + parts[1].strip()

    # Re-append references
    final_answer = answer_body
    if refs_part:
        final_answer = f"{answer_body}\n\n{refs_part}"

    # ── Bypass verification for broad/global queries ──────────────────────
    if scope.entity_kind == "none" and not scope.scoped_titles:
        return final_answer, True

    # ── For locked scopes (author/paper/topic/filter), enforce strict verification ──
    # Check scope ONLY on the answer body to prevent references list from causing false refusals
    if scope.is_locked:
        ok, reason = verify_answer_against_scope(
            answer_body,
            scoped_titles=scope.scoped_titles,
            papers_metadata=papers_metadata,
            chunks=chunks,
            scope_locked=True,
        )
        if ok:
            return final_answer, True
        logger.warning(f"Citation verification failed: {reason}")
        return VERIFICATION_FAILED_REFUSAL, False

    # For unlocked but non-empty scopes, apply lighter verification
    if scope.scoped_titles:
        ok, reason = verify_answer_against_scope(
            answer_body,
            scoped_titles=scope.scoped_titles,
            papers_metadata=papers_metadata,
            chunks=chunks,
            scope_locked=False,
        )
        if ok:
            return final_answer, True
        logger.warning(f"Citation verification failed (unlocked scope): {reason}")
        return final_answer, True

    return final_answer, True


def verify_author_exists_in_library(author_name: str, papers_metadata: dict) -> bool:
    """
    Check if a named author exists in the library using whole-word matching
    on normalized author fields. This prevents false positive substring matches
    (e.g., 'Ada' matching 'Adam', or 'E' matching 'Elon').
    """
    if not author_name or not papers_metadata:
        return False
        
    # Extract name tokens (ignore initials / short noise)
    tokens = [t for t in re.findall(r"[a-zA-Z\u00C0-\u017F]+", author_name.lower()) if len(t) >= 3]
    if not tokens:
        tokens = [t for t in re.findall(r"[a-zA-Z\u00C0-\u017F]+", author_name.lower()) if len(t) >= 2]
    if not tokens:
        return False
        
    # Check if any paper contains the author tokens as whole words
    for paper_meta in papers_metadata.values():
        authors_field = paper_meta.get("authors", "")
        # Check each token in the authors field
        for token in tokens:
            if author_field_contains_token(authors_field, token):
                return True
                
    return False


def is_broad_author_query(query: str, author_phrase: str) -> bool:
    """
    True if the query is a broad synthesis or summary request on an author
    without any specified topic keywords.
    """
    from rag_context import _significant_query_tokens, author_phrase_tokens
    
    sig_tokens = _significant_query_tokens(query)
    author_tokens = set(author_phrase_tokens(author_phrase))
    
    # Filter out author tokens from query significant tokens
    topic_tokens = [t for t in sig_tokens if t not in author_tokens]
    
    generic_qa_words = {
        "say", "says", "said", "write", "wrote", "written", "author", "authored",
        "work", "research", "paper", "papers", "article", "articles", "contribution",
        "contributions", "view", "views", "opinion", "opinions", "thought", "thoughts",
        "perspective", "perspectives", "find", "finding", "findings", "publish",
        "published", "discuss", "discusses", "discussed", "about", "content"
    }
    
    topic_tokens = [t for t in topic_tokens if t not in generic_qa_words]
    return len(topic_tokens) == 0
