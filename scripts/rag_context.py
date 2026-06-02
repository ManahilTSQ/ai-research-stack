"""
rag_context.py — Shared helpers for building RAG context strings and retrieval.

Used by server.py and rag_service.py so template mode and standard chat share
the same chunk formatting, relevance filtering, and two-paper compare logic.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from config import settings


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
    "list", "table", "tabulate", "all paper",
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
    # Comparative / analytical table keywords
    "comparative", "comparison", "algorithm", "technique", "accuracy",
    "dataset", "performance", "precision", "recall", "benchmark",
    "ml method", "approach used", "method used", "evaluation metric",
    "intrusion detection method", "classification method",
    # Research analysis / synthesis keywords — prevent misclassification as
    # simple inventory listings when the query also contains 'papers by'.
    "pipeline", "step-by-step", "threshold", "parameter", "parameters",
    "ranking", "criteria", "identify", "analyze", "analyse",
    "feature selection", "information gain", "exact step",
    "how did", "how do", "how does", "what did", "what approach",
    "what method", "what technique", "what strategy", "what threshold",
    "used for", "applied to", "drop", "weight", "rank",
)

_AUTHOR_PHRASE_PATTERNS = [
    re.compile(r"corpus of\s+(.+?)(?:'s|\u2019s)\s+(?:articles?|papers?|works?)", re.I),
    re.compile(r"corpus of\s+(.+?)\s+(?:articles?|papers?|works?)\s+on\b", re.I),
    re.compile(r"corpus of\s+(.+?)\s+on\b", re.I),
    re.compile(r"papers?\s+by\s+(.+?)(?:\s+on\b|[\.,;]|$)", re.I),
    re.compile(r"articles?\s+by\s+(.+?)(?:\s+on\b|[\.,;]|$)", re.I),
    # Author-scoped phrasing: "papers with D. Stiawan", "extract from the papers with <name>"
    re.compile(r"papers?\s+with\s+(.+?)(?:\s+on\b|[\.,;]|$)", re.I),
    re.compile(r"extract\s+from\s+the\s+papers?\s+with\s+(.+?)(?:\s+on\b|[\.,;]|$)", re.I),
    re.compile(r"(.+?)(?:'s|\u2019s)\s+(?:articles?|papers?|works?)\s+on\b", re.I),
    # Synthesis / stance questions: "thoughts of Jhanjhi", "contributions by X"
    re.compile(
        r"(?:thoughts?|views?|opinions?|ideas?|perspective|work|research|"
        r"contributions?|findings?)\s+(?:of|by)\s+(.+?)(?:\s*[\.,;\?]|$)",
        re.I,
    ),
    re.compile(
        r"(?:what\s+are|describe|summarize|explain)\s+(?:the\s+)?(?:main\s+)?"
        r"(?:thoughts?|views?|contributions?|work)\s+(?:of|by)\s+(.+?)(?:\s*[\.,;\?]|$)",
        re.I,
    ),
    # Possessive author queries: "Jie Li's work", "Noor Zaman Jhanjhi’s research"
    re.compile(r"(.+?)(?:'s|\u2019s)\s+work(?:\s+on\b|[\.,;]|$)", re.I),
    re.compile(r"(.+?)(?:'s|\u2019s)\s+research(?:\s+on\b|[\.,;]|$)", re.I),
    re.compile(r"according to\s+(.+?)(?:\s*[\.,;\?]|$)", re.I),
    re.compile(r"(?:who is|tell me about)\s+(.+?)(?:\s*[\.,;\?]|$)", re.I),
    re.compile(r"about\s+(?:the\s+)?(?:research(?:er)?|work of)\s+(.+?)(?:\s*[\.,;\?]|$)", re.I),
    re.compile(r"what does\s+(.+?)\s+research\b", re.I),
    re.compile(
        r"(?:main\s+)?contributions?\s+of\s+(.+?)(?:\s*[\.,;\?]|$)",
        re.I,
    ),
    re.compile(r"describe\s+(?:the\s+)?research\s+of\s+(.+?)(?:\s*[\.,;\?]|$)", re.I),
]

_AUTHOR_INTENT_RE = re.compile(
    r"\b(?:contributions?|thoughts?|views?|opinions?|research(?:er)?|"
    r"authored?|wrote|written|findings?|according to|"
    r"main work|what does|what did|what are|who is|tell me about)\b",
    re.I,
)

# Tokens too common for "what do my papers say about X" title matching.
_GENERIC_TOPIC_TOKENS = frozenset({
    "about", "papers", "paper", "research", "study", "studies", "using", "based",
    "approach", "approaches", "method", "methods", "model", "models", "system",
    "systems", "analysis", "review", "reviews", "survey", "surveys", "detection",
    "classification", "learning", "deep", "machine", "framework", "application",
    "applications", "novel", "enhanced", "hybrid", "secure", "security",
    "optimization", "optimize", "optimizing", "optimal",
    "internet", "things", "smart", "cities", "city", "network", "networks",
})

MAX_SCOPED_CONTEXT_CHUNKS = 48

_PRONOUN_AUTHOR_RE = re.compile(
    r"\b(?:his|her|their)\s+(?:main\s+)?(?:contributions?|thoughts?|views?|work|research|papers?)\b",
    re.I,
)

_PAPER_FOCUS_RE = re.compile(
    r"\b(?:summarize|summary of|paper titled|the paper|this paper|"
    r"according to the paper|in the paper)\b",
    re.I,
)

# Substrings that must never match inside another name word (e.g. hassan in Riskhan).
_FORBIDDEN_AUTHOR_SUBSTRINGS = frozenset({"hassan", "hasan"})

# Surnames that appear with many different given names — require fuller name in query.
_DISAMBIGUATE_SURNAMES = frozenset({
    "khan", "kumar", "singh", "ahmed", "ali", "smith", "lee", "wang", "zhang",
})


def normalize_for_match(text: str) -> str:
    """Lowercase + Unicode normalize (fi ligatures, accents) for author/title matching."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKD", text)
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl")
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Remove punctuation so "D. Stiawan" == "D Stiawan" in matching.
    s = re.sub(r"[^0-9A-Za-z\s]+", " ", s)
    return re.sub(r"\s+", " ", s).lower().strip()

# Topic profiles: require domain phrases in paper metadata/abstract, not generic "learning".
_TOPIC_PROFILES: list[dict[str, Any]] = [
    {
        "id": "medical_imaging",
        "query_patterns": [
            r"medical\s+imaging",
            r"medical\s+image",
            r"radiology",
            r"histopathology",
            r"dermoscopy",
            r"retinopathy",
            r"mri\s+brain",
            r"brain\s+tumor",
            r"chest\s+x-?ray",
            r"melanoma",
            r"breast\s+cancer\s+imag",
            r"colon\s+cancer\s+(?:tissue|image)",
            r"colon\s+cancer",
            r"pressure\s+ulcer",
            r"retinoblastoma",
            r"phishing\s+detect",
        ],
        "paper_markers": [
            "medical imaging", "medical image", "radiology", "histopathology",
            "dermoscopy", "retinopathy", "melanoma", "mammogram", "x-ray", "xray",
            "mri", "brain tumor", "tumor detection", "skin lesion", "skin cancer",
            "breast cancer", "lung carcinoma", "colon cancer", "pressure ulcer",
            "retinoblastoma", "diabetic retinopathy", "dermoscopy", "biomedical image",
            "plant leaf disease",
        ],
        "chunk_markers": [
            "medical", "imaging", "radiolog", "histopath", "dermoscop", "retinopath",
            "melanoma", "mammogram", "x-ray", "xray", " mri", "tumor", "lesion",
            "dermatolog", "oncolog", "diagnos", "biopsy", "histolog",
        ],
    },
    {
        "id": "smart_city_cyber",
        "query_patterns": [
            r"smart\s+cit",
            r"iot\s+cyber",
            r"cybersecurity.*smart",
        ],
        "paper_markers": [
            "smart cit", "internet of things", " iot", "cybersecurity", "cyber security",
            "5g-enabled", "v2x", "uav", "blockchain", "federated learning",
        ],
        "chunk_markers": [
            "smart cit", " iot", "internet of things", "cyber", "intrusion",
            "malware", "phishing", "blockchain", "federated",
        ],
    },
]

_AUTHOR_SCOPED_PATTERNS = [
    r"corpus of\s+",
    r"papers?\s+by\s+",
    r"articles?\s+by\s+",
    r"works?\s+by\s+",
    r"'s\s+(papers?|articles?|works?|work|research)",
    r"\u2019s\s+(papers?|articles?|works?)",
    r"\u2019s\s+(work|research)",
    r"articles?\s+with\s+.+?\s+as\s+(author|co-author)",
    r"list\s+(articles?|papers?)\s+with\s+",
    r"papers?\s+with\s+",
    r"papers?\s+authored?\s+by\s+",
    r".+?\s+authored?\s+(papers?|articles?)",
    r"as\s+(author|co-author)\s+or\s+co-author",
    r"(?:thoughts?|views?|opinions?|contributions?|perspective)\s+(?:of|by)\s+",
    r"main\s+contributions?\s+(?:of|by)\s+",
    r"what\s+are\s+(?:the\s+)?(?:his|her|their)\s+main\s+contributions",
]

TABLE_TRUNCATION_RE = re.compile(
    r"remaining\s+papers?|\.\.\.\s*\(|\.\.\.\s*\||\|\s*\.\.\.\s*\|",
    re.IGNORECASE,
)

_BOTH_INTENT_RE = re.compile(
    r"\bboth\b|\blist\b.*\b(?:summari[sz]e|explain|describe|what\s+does)\b|"
    r"\b(?:summari[sz]e|explain|describe)\b.*\band\b.*\blist\b|"
    r"\bplus\s+(?:a\s+)?(?:summary|synthesis)\b",
    re.I,
)

_CONTENT_INTENT_RE = re.compile(
    r"\bwhat\s+(?:does|did|is|are)\b|\bwhat\s+(?:he|she|they)\s+says?\b|"
    r"\bsays?\s+about\b|\bmain\s+contributions?\b|\bsummarize\b|"
    r"\bdescribe\s+(?:the\s+)?research\b|\bfindings?\b|\bexplain\b|"
    # Analytical research queries that happen to contain 'papers by'
    r"\banalyze\b|\banalyse\b|\bidentif\w+\b|\binvestigat\w+\b|"
    r"\bexamin\w+\b|\bpipeline\b|\bthreshold\b|\bstep-?by-?step\b|"
    r"\branking\s+criteri\b|\bfeature\s+selection\b|\binformation\s+gain\b",
    re.I,
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
        "thoughts", "views", "opinions", "ideas", "main",
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


def _topic_specific_tokens(query: str) -> list[str]:
    """Distinctive topic tokens (excludes generic words like 'detection', 'learning')."""
    return [t for t in _significant_query_tokens(query) if t not in _GENERIC_TOPIC_TOKENS]


def query_has_library_topic_cue(query: str) -> bool:
    """True for 'what do my papers say about X' style questions."""
    q_lower = (query or "").lower()
    cues = (
        "about", "on ", " regarding ", " related to ", " topic ", " theme ",
        "discuss", "discusses", "cover", "covers", "concerning", " say about",
    )
    return any(c in q_lower for c in cues)


# Tokens so generic that a single hit should NOT qualify a paper as on-topic.
_WEAK_TOPIC_TOKENS = frozenset({
    "data", "based", "using", "approach", "network", "deep",
    "learning", "model", "system", "detection", "classification",
    "performance", "result", "proposed", "method", "algorithm",
    "wireless", "communication", "protocol", "node", "cloud",
    "mobile", "user", "service", "layer",
})


def find_papers_by_metadata_keywords(
    query: str,
    papers_metadata: dict,
    *,
    min_token_hits: int | None = None,
) -> list[str]:
    """
    Match papers by distinctive words in title / authors / manifest abstract.
    Works for any ingested topic — no hardcoded profile required.

    Strictness rules:
    - When min_token_hits is not given we require at least 2 hits when the
      query has 2+ distinctive tokens, so that papers matching only a single
      generic word (e.g. "deep" or "network") are excluded.
    - Single-token matches are allowed only when the token is long and specific
      (>=7 characters, not in _WEAK_TOPIC_TOKENS).
    """
    if not papers_metadata:
        return []
    tokens = _topic_specific_tokens(query) or _significant_query_tokens(query)
    # Short tech tokens (6g, v2x, ai) from the raw query.
    for raw in re.findall(r"[a-z0-9]{2,}", (query or "").lower()):
        if raw not in tokens and raw not in _GENERIC_TOPIC_TOKENS:
            tokens.append(raw)
    if not tokens:
        return []
    need = min_token_hits
    if need is None:
        if len(tokens) >= 3:
            need = 2
        elif len(tokens) == 2:
            need = 2
        else:
            # Single-token query: only match if the token is specific enough.
            tok = tokens[0]
            if len(tok) < 7 or tok in _WEAK_TOPIC_TOKENS:
                need = 2  # Force no single-generic-token matches.
            else:
                need = 1
    matched: list[str] = []
    for title, meta in papers_metadata.items():
        hay = _paper_haystack(title, meta)
        hits = sum(1 for t in tokens if t in hay)
        if hits >= need:
            matched.append(title)
    return sorted(matched)


def _refine_topic_papers_by_query(
    query: str,
    paper_titles: list[str],
    papers_metadata: dict,
) -> list[str]:
    """
    Narrow broad profile matches using distinctive words from the question.
    Requires at least 2 token hits when possible so single-word incidental
    matches (e.g. a medical paper that mentions 'security' once) are dropped.
    """
    tokens = _topic_specific_tokens(query)
    if not tokens or not paper_titles:
        return paper_titles
    # Always require at least 2 hits when we have 2+ tokens — never accept
    # a paper that only matches 1 generic word from the query.
    need = 2 if len(tokens) >= 2 else 1
    refined: list[str] = []
    for title in paper_titles:
        meta = papers_metadata.get(title) or {}
        hay = _paper_haystack(title, meta)
        hits = sum(1 for t in tokens if t in hay)
        if hits >= need:
            refined.append(title)
    # If refinement drops everything keep the original list so we don't lose
    # a legitimate narrow topic (e.g. single-paper query).
    return refined if refined else paper_titles


def _chunk_query_overlap_score(chunk: dict[str, Any], query: str) -> int:
    """Higher = chunk text/metadata overlaps more with distinctive query terms."""
    tokens = _topic_specific_tokens(query) or _significant_query_tokens(query)
    if not tokens:
        return 0
    hay = _chunk_search_haystack(chunk)
    return sum(1 for t in tokens if t in hay)


def rank_and_cap_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    *,
    limit: int,
    max_total: int = MAX_SCOPED_CONTEXT_CHUNKS,
) -> list[dict[str, Any]]:
    """Prefer chunks that mention the question's topic; cap total context size."""
    if not chunks:
        return []
    cap = min(max_total, max(limit, 12))
    ranked = sorted(
        chunks,
        key=lambda c: (-_chunk_query_overlap_score(c, query), float(c.get("distance", 0.0))),
    )
    return ranked[:cap]


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
    has_listing_verb = any(kw in q for kw in LISTING_QUERY_KEYWORDS)
    if not has_listing_verb:
        return False
    # Inventory listing should be command-like ("list/show/table/..."), not a
    # semantic "what does X say about Y" question.
    semantic_qa_patterns = [
        r"\bwhat\s+(?:does|did|is|are)\b",
        r"\bwhat\s+(?:he|she|they)\s+says?\b",
        r"\bsays?\s+about\b",
        r"\bmain\s+contributions?\b",
        r"\bdescribe\s+(?:the\s+)?research\b",
        r"\bsummarize\b",
    ]
    if any(re.search(pat, q) for pat in semantic_qa_patterns):
        return False
    return True


def classify_query_mode(query: str) -> str:
    """
    Classify user intent for routing:
      - listing: metadata inventory/table/list only
      - content: synthesis/extraction from paper text
      - both: user explicitly wants list + summary
      - ambiguous: mixed signals, needs clarification
    """
    q = (query or "").strip()
    if not q:
        return "content"

    has_listing = is_listing_query(q)
    has_content = bool(_CONTENT_INTENT_RE.search(q)) or bool(
        any(sig in q.lower() for sig in CONTENT_EXTRACTION_SIGNALS)
    )
    wants_both = bool(_BOTH_INTENT_RE.search(q))
    topical_but_underspecified = bool(
        re.search(r"\b(?:about|on|regarding|related to)\b", q, re.I)
    ) and has_listing and query_has_author_intent(q) and not is_content_extraction_query(q)

    if wants_both:
        return "both"
    if topical_but_underspecified:
        return "ambiguous"
    if has_listing and has_content:
        # When the user's intent is clearly content-extraction (comparative tables,
        # per-paper methodology tables, etc.) don't ask for clarification — go straight
        # to the extraction path so the "Do you want (1) or (2)?" prompt is never shown.
        if is_content_extraction_query(q):
            return "content"
        return "ambiguous"
    if has_listing:
        return "listing"
    return "content"


def is_content_extraction_query(query: str) -> bool:
    q = (query or "").lower()
    # "extract ... what X says about Y" and similar semantic requests should be
    # handled as content extraction, not metadata inventory listing.
    if "extract" in q and re.search(r"\bwhat\b.*\bsays?\s+about\b", q):
        return True
    if re.search(r"\b(?:contributions?|findings?|approach|methodology|framework)\b", q):
        return True
    # Comparative / ML-method table requests need content extraction, not metadata.
    # e.g. "Generate a comparative table of ML methods used for intrusion detection"
    if is_listing_query(query) and re.search(
        r"\bcompar(?:e|ative|ison)\b|\bml\s*method|\balgorithm\b|\btechnique\b"
        r"|\baccuracy\b|\bdataset\b|\bperformance\b|\bevaluation\b",
        q, re.I,
    ):
        return True
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


def query_has_author_intent(query: str) -> bool:
    """True when the user is asking about a person's research (not a generic topic)."""
    q = query or ""
    return (
        bool(_AUTHOR_INTENT_RE.search(q))
        or bool(_PRONOUN_AUTHOR_RE.search(q))
        or query_expects_named_author(q)
    )


def query_has_paper_focus(query: str) -> bool:
    """True when the user is asking about a specific paper by title."""
    return bool(_PAPER_FOCUS_RE.search(query or "")) or bool(
        re.search(r'"[^"]{8,200}"', query or "")
    )


def _clean_author_phrase(phrase: str) -> str | None:
    phrase = re.sub(r"\s+", " ", (phrase or "").strip(" .,;:\"'?"))
    # Drop trailing pronouns / filler from "contributions, what are his"
    phrase = re.sub(
        r"\s*,\s*what\s+are\s+(?:his|her|their).*$",
        "",
        phrase,
        flags=re.I,
    ).strip()
    if len(phrase) < 2:
        return None
    return phrase


def extract_author_search_phrase(query: str) -> str | None:
    """Pull a human author phrase from queries like 'corpus of Noor Zaman Jhanjhi's articles'."""
    q = (query or "").strip()
    if not q:
        return None
    for pattern in _AUTHOR_PHRASE_PATTERNS:
        m = pattern.search(q)
        if not m:
            continue
        phrase = _clean_author_phrase(m.group(1))
        if phrase:
            return phrase
    return None


def author_phrase_tokens(phrase: str) -> list[str]:
    stop = _query_stopwords() | {"noor", "dr", "prof", "professor"}
    raw = re.findall(r"[a-z0-9]+", (phrase or "").lower())
    return [t for t in raw if len(t) >= 3 and t not in stop]


# ── Multi-author co-authorship (AND intersection) ────────────────────────────

# Patterns for "papers by X and Y", "co-authored by X and Y", etc.
_MULTI_AUTHOR_PATTERNS = [
    # "papers by Stiawan and Budiarto"
    re.compile(
        r"(?:papers?|articles?|works?)\s+(?:by|co-?authored?\s+by)\s+"
        r"([A-Za-z][A-Za-z\s\.\-]{1,40}?)\s+and\s+([A-Za-z][A-Za-z\s\.\-]{1,40}?)"
        r"(?:\s+(?:in|on|about|from|at|to)\b|[.,;?]|$)",
        re.I,
    ),
    # "co-authored papers by X and Y"
    re.compile(
        r"co-?authored?\s+(?:papers?|articles?|works?)?\s*by\s+"
        r"([A-Za-z][A-Za-z\s\.\-]{1,40}?)\s+and\s+([A-Za-z][A-Za-z\s\.\-]{1,40}?)"
        r"(?:\s+(?:in|on|about|from|at|to)\b|[.,;?]|$)",
        re.I,
    ),
    # "analyze/list/compare ... by X and Y"
    re.compile(
        r"(?:analyze|compare|list|show|find|get)\s+(?:all\s+)?(?:co-?authored?\s+)?"
        r"papers?\s+by\s+"
        r"([A-Za-z][A-Za-z\s\.\-]{1,40}?)\s+and\s+([A-Za-z][A-Za-z\s\.\-]{1,40}?)"
        r"(?:\s+(?:in|on|about|from|at|to)\b|[.,;?]|$)",
        re.I,
    ),
]


def extract_multi_author_phrases(query: str) -> list[str] | None:
    """
    Return [author1, author2] when the query explicitly asks for papers
    co-authored by two named people.  Returns None for single-author queries.

    Examples that match:
      - "papers by Stiawan and Budiarto"
      - "co-authored by Stiawan and Budiarto"
      - "Analyze co-authored papers by Stiawan and Budiarto"
    """
    q = (query or "").strip()
    for pat in _MULTI_AUTHOR_PATTERNS:
        m = pat.search(q)
        if m:
            a1 = m.group(1).strip().strip("\"',;.")
            a2 = m.group(2).strip().strip("\"',;.")
            if a1 and a2 and len(a1) >= 3 and len(a2) >= 3:
                return [a1, a2]
    return None


def resolve_coauthored_papers(
    author_phrases: list[str],
    papers_metadata: dict,
) -> list[str]:
    """
    Return papers that are co-authored by ALL listed author phrases (AND/intersection).

    For a query like "papers by Stiawan and Budiarto" this returns only the
    subset of papers that have BOTH Stiawan AND Budiarto in their authors field —
    not the union of all Stiawan papers plus all Budiarto papers.
    """
    if not author_phrases or not papers_metadata:
        return []

    sets: list[set[str]] = []
    for phrase in author_phrases:
        # Try full-phrase match first (most precise).
        titles = set(resolve_papers_for_author_phrase(phrase, papers_metadata))
        if not titles:
            # Surname-only fallback for abbreviated author names.
            tokens = author_phrase_tokens(phrase)
            if tokens:
                surname = tokens[-1]
                titles = {
                    t
                    for t, m in papers_metadata.items()
                    if author_field_contains_token(m.get("authors") or "", surname)
                }
        sets.append(titles)

    if not sets:
        return []

    # Strict AND: keep only papers that appear in every author's paper set.
    intersection: set[str] = sets[0]
    for s in sets[1:]:
        intersection = intersection & s

    return sorted(intersection)


def _load_ingestion_manifest() -> dict:
    path = settings.BASE_DIR / "output" / "ingestion_manifest.json"
    if not path.exists():
        return {}
    try:
        import json
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _split_author_segments(authors: str) -> list[str]:
    """Split an authors field into individual author name segments (from ingest metadata)."""
    if not authors or authors.strip().lower() in {"unknown authors", "unknown"}:
        return []
    parts = re.split(r",\s*|\s+et al\.?|\s*;\s*|\s+and\s+", authors, flags=re.I)
    out: list[str] = []
    for part in parts:
        part = re.sub(r"\s+", " ", part.strip())
        if len(part) >= 2:
            out.append(part)
    return out


def build_author_segment_catalog(papers_metadata: dict) -> dict[str, dict[str, Any]]:
    """
    Every distinct author string on ingested papers → list of paper titles.
    Rebuilt on each request from live ChromaDB metadata (includes newly ingested papers).
    """
    catalog: dict[str, dict[str, Any]] = {}
    for title, meta in (papers_metadata or {}).items():
        for segment in _split_author_segments(meta.get("authors") or ""):
            key = normalize_for_match(segment)
            if not key:
                continue
            if key not in catalog:
                catalog[key] = {"display": segment, "titles": []}
            if title not in catalog[key]["titles"]:
                catalog[key]["titles"].append(title)
    return catalog


def _papers_with_surname(surname: str, papers_metadata: dict) -> list[str]:
    """All papers where any author segment ends with this surname (any spelling variant)."""
    sn = normalize_for_match(surname)
    if len(sn) < 3:
        return []
    matched: set[str] = set()
    for title, meta in papers_metadata.items():
        for segment in _split_author_segments(meta.get("authors") or ""):
            words = _author_part_words(normalize_for_match(segment))
            if words and words[-1] == sn:
                matched.add(title)
    return sorted(matched)


def _edit_distance_leq1(a: str, b: str) -> bool:
    """
    Fast check for Levenshtein edit distance <= 1 (insert/delete/substitute).
    Used to tolerate small author surname typos (e.g., sitawan -> stiawan).
    """
    a = normalize_for_match(a)
    b = normalize_for_match(b)
    if not a or not b:
        return False
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # Same length: allow one substitution.
    if la == lb:
        mismatches = [i for i in range(la) if a[i] != b[i]]
        if len(mismatches) <= 1:
            return True
        # Allow one adjacent transposition (Damerau-style) for common typos:
        # e.g., "sitawan" vs "stiawan" (swap i and t).
        if len(mismatches) == 2:
            i, j = mismatches
            if j == i + 1:
                aa = list(a)
                aa[i], aa[j] = aa[j], aa[i]
                return "".join(aa) == b
        return False
    # Length differs by 1: allow one insertion/deletion.
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    return True


def _library_surnames(papers_metadata: dict) -> set[str]:
    """All distinct normalized surnames from author segments in the library."""
    out: set[str] = set()
    for _title, meta in (papers_metadata or {}).items():
        for segment in _split_author_segments(meta.get("authors") or ""):
            words = _author_part_words(normalize_for_match(segment))
            if words:
                out.add(words[-1])
    return out


def _given_name_keys_for_surname(surname: str, catalog: dict) -> set[str]:
    """Distinct given-name identities for a surname (to detect Khan vs Khan)."""
    sn = normalize_for_match(surname)
    keys: set[str] = set()
    for key, entry in catalog.items():
        words = _author_part_words(key)
        if not words or words[-1] != sn:
            continue
        if len(words) == 1:
            keys.add(sn)
        else:
            keys.add(" ".join(words[:-1]))
    return keys


def _token_is_ambiguous_surname(token: str, indexes: dict[str, Any], total_papers: int) -> bool:
    """True when this surname appears on many papers — require a fuller name in the query."""
    if len(token) < 4:
        return True
    count = len(indexes.get("author_to_titles", {}).get(token, []))
    if count == 0:
        return True
    return count > max(2, int(total_papers * 0.12))


def _query_mentions_word(query_lower: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", query_lower))


def resolve_author_from_library(
    query: str,
    papers_metadata: dict,
) -> tuple[str | None, list[str]]:
    """
    Match the question to author names exactly as stored on ingested papers.
    Works for any author in the library; updates automatically when new PDFs are ingested.
    """
    if not papers_metadata:
        return None, []

    catalog = build_author_segment_catalog(papers_metadata)
    indexes = build_catalog_indexes(papers_metadata)
    q_norm = normalize_for_match(query)

    explicit = extract_author_search_phrase(query)
    if explicit:
        titles = resolve_papers_for_author_phrase(explicit, papers_metadata, indexes=indexes)
        if titles:
            return explicit, titles
        ek = normalize_for_match(explicit)
        if ek in catalog:
            return catalog[ek]["display"], list(catalog[ek]["titles"])

    if not query_has_author_intent(query) and not explicit:
        return None, []

    # Longest normalized author segment contained in the query (Unicode-safe).
    best_key: str | None = None
    best_len = 0
    matched_titles: set[str] = set()

    for key, entry in catalog.items():
        if len(key) < 4:
            continue
        if key in q_norm and len(key) > best_len:
            best_key = key
            best_len = len(key)
            matched_titles.update(entry["titles"])
        else:
            words = _author_part_words(key)
            if len(words) >= 2 and all(
                re.search(rf"\b{re.escape(w)}\b", q_norm) for w in words
            ):
                if len(key) > best_len:
                    best_key = key
                    best_len = len(key)
                    matched_titles.update(entry["titles"])

    if matched_titles and best_key:
        return catalog[best_key]["display"], sorted(matched_titles)

    # Surname-only: union all papers for that surname (Jhanjhi, Aldughayfiq, etc.).
    surnames = _library_surnames(papers_metadata)
    for token in sorted(_significant_query_tokens(query), key=len, reverse=True):
        if len(token) < 4 or token in _FORBIDDEN_AUTHOR_SUBSTRINGS:
            continue
        if not re.search(rf"\b{re.escape(token)}\b", q_norm):
            continue
        papers = _papers_with_surname(token, papers_metadata)
        if not papers:
            # One-edit fuzzy surname rescue (common typos in user queries).
            close = [sn for sn in surnames if _edit_distance_leq1(token, sn)]
            if len(close) == 1:
                papers = _papers_with_surname(close[0], papers_metadata)
        if not papers:
            continue
        given_keys = _given_name_keys_for_surname(token, catalog)
        if len(given_keys) > 1 and token in _DISAMBIGUATE_SURNAMES:
            # e.g. "Khan" alone with many different Khans — need a fuller name.
            for key, entry in catalog.items():
                if key in q_norm:
                    return entry["display"], list(entry["titles"])
            return None, []
        return token.title(), papers

    return None, []


def build_catalog_indexes(papers_metadata: dict) -> dict[str, Any]:
    """Author token → paper titles and title lookup maps (deterministic, no LLM)."""
    author_to_titles: dict[str, list[str]] = {}
    title_lower_map: dict[str, str] = {}

    def _add_author_tokens(authors: str, title: str) -> None:
        for token in re.findall(r"[a-z]{3,}", (authors or "").lower()):
            author_to_titles.setdefault(token, [])
            if title not in author_to_titles[token]:
                author_to_titles[token].append(title)

    for title, meta in (papers_metadata or {}).items():
        title_lower_map[title.lower().strip()] = title
        _add_author_tokens(meta.get("authors") or "", title)

    for _fn, meta in _load_ingestion_manifest().items():
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
                _add_author_tokens(m_authors, db_title)
                break

    return {
        "author_to_titles": author_to_titles,
        "title_lower_map": title_lower_map,
        "all_titles": list(papers_metadata.keys()),
    }


def _author_part_words(part: str) -> list[str]:
    return re.findall(r"[a-z]+", normalize_for_match(part))


def author_field_contains_token(authors: str, token: str) -> bool:
    """
    True if token is a whole name word in an author segment (not a substring).
    Prevents false matches like hassan inside Riskhan.
    """
    token = (token or "").lower()
    if len(token) < 3:
        return False
    for part in re.split(r"[,;&]| and ", (authors or "").lower()):
        words = _author_part_words(part)
        if not words:
            continue
        if token in words:
            return True
        if words[-1] == token:
            return True
    return False


def author_field_matches_phrase(authors: str, phrase_tokens: list[str]) -> bool:
    """All phrase tokens must appear as whole words in the same author segment."""
    if not phrase_tokens:
        return False
    for part in re.split(r"[,;&]| and ", (authors or "").lower()):
        words = _author_part_words(part)
        if words and all(t in words for t in phrase_tokens):
            return True
    return False


def resolve_papers_for_author_phrase(
    phrase: str,
    papers_metadata: dict,
    *,
    indexes: dict[str, Any] | None = None,
) -> list[str]:
    """Library papers whose author field matches the given name (strict; no substring surnames)."""
    if not phrase or not papers_metadata:
        return []
    phrase_tokens = author_phrase_tokens(phrase)
    if not phrase_tokens:
        return []

    matched: set[str] = set()
    surname = phrase_tokens[-1]
    use_surname_only = len(phrase_tokens) == 1

    for title, meta in papers_metadata.items():
        authors = meta.get("authors") or ""
        if len(phrase_tokens) >= 2:
            if author_field_matches_phrase(authors, phrase_tokens):
                matched.add(title)
            else:
                # Surname-only fallback: catches abbreviated forms like
                # "N.Z. Jhanjhi" or "Jhanjhi, NZ" where given-name initials
                # don't spell out "zaman" but the surname is unambiguous.
                if author_field_contains_token(authors, surname):
                    matched.add(title)
            continue
        if author_field_contains_token(authors, surname):
            matched.add(title)

    if use_surname_only:
        catalog = build_author_segment_catalog(papers_metadata)
        entries = [
            e for k, e in catalog.items()
            if _author_part_words(k) and _author_part_words(k)[-1] == surname
        ]
        if len(entries) == 1:
            matched.update(entries[0]["titles"])
        elif len(entries) > 1:
            return []
        else:
            indexes = indexes or build_catalog_indexes(papers_metadata)
            if not _token_is_ambiguous_surname(surname, indexes, len(papers_metadata)):
                for title in indexes["author_to_titles"].get(surname, []):
                    matched.add(title)

    return sorted(matched)


def _expand_author_name_from_query(query: str, token: str) -> str:
    """Recover a fuller name from the query when only a surname token was matched."""
    pattern = rf"((?:[A-Z][A-Za-z]+|[A-Z]\.)(?:\s+(?:[A-Z][A-Za-z]+|[A-Z]\.)){{0,4}}\s+{re.escape(token)})"
    m = re.search(pattern, query, re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return token


def infer_library_author_phrase(query: str, papers_metadata: dict) -> str | None:
    """Detect an author from live library metadata (any ingested author, not a fixed list)."""
    phrase, titles = resolve_author_from_library(query, papers_metadata)
    if titles:
        return phrase
    return phrase if phrase else None


def detect_topic_profile(query: str) -> dict[str, Any] | None:
    """Return a domain topic profile when the query asks about a specialized research area."""
    q = (query or "").lower()
    for profile in _TOPIC_PROFILES:
        for pat in profile["query_patterns"]:
            if re.search(pat, q, re.I):
                return profile
    return None


def _paper_haystack(title: str, meta: dict) -> str:
    abstract = ""
    for _fn, m in _load_ingestion_manifest().items():
        mt = (m.get("title") or "").lower()
        if mt == title.lower() or mt in title.lower() or title.lower() in mt:
            abstract = (m.get("abstract") or "").lower()
            break
    return " ".join(
        [
            title.lower(),
            (meta.get("authors") or "").lower(),
            (meta.get("venue") or "").lower(),
            abstract,
        ]
    )


def resolve_topic_scoped_papers(
    query: str,
    papers_metadata: dict,
    profile: dict[str, Any],
) -> list[str]:
    """Papers whose title/metadata clearly belong to the topic domain (not generic 'learning')."""
    if not papers_metadata or not profile:
        return []
    markers = [m.lower() for m in profile.get("paper_markers", [])]
    matched: list[str] = []
    for title, meta in papers_metadata.items():
        hay = _paper_haystack(title, meta)
        if any(m in hay for m in markers):
            matched.append(title)
    matched = sorted(matched)
    return _refine_topic_papers_by_query(query, matched, papers_metadata)


def filter_chunks_for_topic_profile(
    chunks: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Drop chunks that are not about the topic domain (reduces phishing/traffic noise)."""
    markers = [m.lower() for m in profile.get("chunk_markers", [])]
    if not markers:
        return chunks
    kept: list[dict[str, Any]] = []
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        hay = " ".join(
            [
                chunk.get("text") or "",
                meta.get("title") or "",
                meta.get("authors") or "",
            ]
        ).lower()
        hits = sum(1 for m in markers if m in hay)
        if hits >= 2 or any(m in (meta.get("title") or "").lower() for m in markers):
            kept.append(chunk)
    return kept if kept else chunks


def fuzzy_match_paper_titles(query: str, papers_metadata: dict) -> list[str]:
    """Match paper titles via quoted strings or distinctive title-token overlap."""
    quoted = re.findall(r'"([^"]{8,200})"', query or "")
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
    Author matches always win over title-token matches to prevent wrong-paper answers.
    """
    if not papers_metadata:
        return []

    # Paper-title focus before author heuristics ("summarize" must not block quoted titles).
    if query_has_paper_focus(query):
        paper_matches = fuzzy_match_paper_titles(query, papers_metadata)
        if paper_matches:
            return paper_matches

    if query_expects_named_author(query) or (
        query_has_author_intent(query) and not query_has_paper_focus(query)
    ):
        _phrase, author_matches = resolve_author_from_library(query, papers_metadata)
        if author_matches:
            return author_matches
        if query_expects_named_author(query):
            return []

    indexes = build_catalog_indexes(papers_metadata)
    author_phrase = extract_author_search_phrase(query)
    is_author_scoped = query_expects_named_author(query) or bool(author_phrase)

    if author_phrase:
        author_matches = resolve_papers_for_author_phrase(
            author_phrase, papers_metadata, indexes=indexes
        )
        if author_matches:
            return author_matches
        if is_author_scoped:
            return []

    tokens = _significant_query_tokens(query)
    if not tokens:
        return []

    matched: list[str] = []
    author_matches: list[str] = []

    for title, meta in papers_metadata.items():
        authors = meta.get("authors") or ""
        title_l = (title or "").lower()
        for token in tokens:
            if len(token) < 4:
                continue
            if author_field_contains_token(authors, token):
                if title not in author_matches:
                    author_matches.append(title)
                break
            if not is_author_scoped and token in title_l:
                if title not in matched:
                    matched.append(title)
                break

    if len(tokens) == 1 and not _token_is_ambiguous_surname(
        tokens[0], indexes, len(papers_metadata)
    ):
        for title in indexes["author_to_titles"].get(tokens[0], []):
            if title not in author_matches:
                author_matches.append(title)

    if author_matches:
        if is_author_scoped or query_has_author_intent(query):
            return sorted(set(author_matches))
        matched = author_matches + [m for m in matched if m not in author_matches]
    elif is_author_scoped:
        return []

    return sorted(set(matched))


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
    # Lock retrieval whenever we resolved specific paper(s) — prevents cross-paper contamination.
    locked_scope = bool(scope_titles) or bool(inventory_titles) or bool(filter_title)

    # ── Author / Paper-scoped retrieval ───────────────────────────────────────
    # When the query names a specific author or paper that is already in the library,
    # pull chunks EXCLUSIVELY from those matched papers.  This prevents topically
    # similar papers written by *different* authors from polluting the context and
    # causing the LLM to attribute content to the wrong author.
    # Note: no [:3] cap — we fetch ALL matched papers so every one of the author's
    # works is represented in the context for exhaustive listing/tabulation queries.
    if inventory_titles:
        author_chunks: list[dict[str, Any]] = []
        n_papers = max(1, len(inventory_titles))
        per_paper = max(3, min(12, (limit * 2) // n_papers))
        for title in inventory_titles:
            paper_chunks = vector_store.get_chunks_for_paper(title, max_chunks=per_paper)
            author_chunks = _dedupe_chunks(author_chunks + paper_chunks)
        if author_chunks:
            return rank_and_cap_chunks(author_chunks, query, limit=limit)

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

    profile = detect_topic_profile(query)
    if profile and chunks:
        chunks = filter_chunks_for_topic_profile(chunks, profile)

    return rank_and_cap_chunks(chunks, query, limit=limit)


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
