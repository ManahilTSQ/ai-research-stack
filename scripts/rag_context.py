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
    r"authored?|wrote|written|findings?|summarize|summary|according to|"
    r"main work|what does|what did|what are|who is|tell me about)\b",
    re.I,
)

_PRONOUN_AUTHOR_RE = re.compile(
    r"\b(?:his|her|their)\s+(?:main\s+)?(?:contributions?|thoughts?|views?|work|research|papers?)\b",
    re.I,
)

_PAPER_FOCUS_RE = re.compile(
    r"\b(?:summarize|summary of|paper titled|the paper|this paper|"
    r"according to the paper|in the paper)\b",
    re.I,
)

# Tokens that appear as author surnames in many unrelated papers — skip for auto-detect.
_COMMON_AUTHOR_SURNAME_BLOCKLIST = frozenset({
    "kumar", "singh", "ahmed", "ali", "khan", "sharma", "patel", "smith",
    "wang", "chen", "zhang", "li", "kim", "lee", "roy", "das", "islam",
    "hassan", "hasan", "hossain", "rahman", "khan", "brohi", "humayun",
})

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
    r"'s\s+(papers?|articles?|works?)",
    r"\u2019s\s+(papers?|articles?|works?)",
    r"articles?\s+with\s+.+?\s+as\s+(author|co-author)",
    r"list\s+(articles?|papers?)\s+with\s+",
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
    return re.findall(r"[a-z]+", (part or "").lower())


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
            continue
        if author_field_contains_token(authors, surname):
            matched.add(title)

    # Index lookup only for single-token surnames (never union all "khan" papers for full names).
    if use_surname_only and surname not in _COMMON_AUTHOR_SURNAME_BLOCKLIST:
        indexes = indexes or build_catalog_indexes(papers_metadata)
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
    """
    Detect an author name present in the library from the query text alone.
    Used when the user did not say 'papers by X' but clearly asks about a researcher.
    """
    explicit = extract_author_search_phrase(query)
    if explicit:
        return explicit
    if not query_has_author_intent(query):
        return None

    indexes = build_catalog_indexes(papers_metadata)
    total = max(len(papers_metadata), 1)
    best_token: str | None = None
    best_count = 0

    for token in _significant_query_tokens(query):
        if len(token) < 4 or token in _COMMON_AUTHOR_SURNAME_BLOCKLIST:
            continue
        papers = indexes["author_to_titles"].get(token, [])
        if not papers:
            continue
        # Skip tokens that match too many papers (likely a topic word, not a surname).
        if len(papers) > max(5, int(total * 0.35)) and len(token) < 7:
            continue
        if len(papers) > best_count:
            best_count = len(papers)
            best_token = token

    if not best_token:
        return None
    return _expand_author_name_from_query(query, best_token)


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
    return sorted(matched)


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

    indexes = build_catalog_indexes(papers_metadata)
    author_phrase = extract_author_search_phrase(query) or infer_library_author_phrase(
        query, papers_metadata
    )
    is_author_scoped = query_expects_named_author(query) or bool(author_phrase)

    if author_phrase:
        author_matches = resolve_papers_for_author_phrase(
            author_phrase, papers_metadata, indexes=indexes
        )
        if author_matches:
            return author_matches
        if is_author_scoped or query_has_author_intent(query):
            return []

    # Paper-title focus (quoted title, summarize, etc.)
    if query_has_paper_focus(query):
        paper_matches = fuzzy_match_paper_titles(query, papers_metadata)
        if paper_matches:
            return paper_matches

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

    if len(tokens) == 1 and tokens[0] not in _COMMON_AUTHOR_SURNAME_BLOCKLIST:
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

    profile = detect_topic_profile(query)
    if profile and chunks:
        chunks = filter_chunks_for_topic_profile(chunks, profile)

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
