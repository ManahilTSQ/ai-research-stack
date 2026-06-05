"""
rag_context.py — Shared helpers for building RAG context strings and retrieval.

Used by server.py and rag_service.py so template mode and standard chat share
the same chunk formatting, relevance filtering, and two-paper compare logic.
"""

from __future__ import annotations

import re
import unicodedata
import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# Cross-encoder reranker for improved retrieval quality
_reranker = None

def _get_reranker():
    """Lazy-load the cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("BAAI/bge-reranker-base")
            logger.info("Cross-encoder reranker loaded successfully: BAAI/bge-reranker-base")
        except ImportError as e:
            logger.error(f"sentence-transformers not installed: {e}")
            logger.error("Reranking DISABLED. Install with: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder reranker: {e}")
            logger.error("Reranking DISABLED due to error")
    else:
        logger.debug("Using cached cross-encoder reranker")
    return _reranker

def rerank_chunks(question: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-rank chunks using cross-encoder for better relevance.
    
    Args:
        question: The user's query
        chunks: List of chunk dictionaries with 'text' field
        top_n: Number of top chunks to return
        
    Returns:
        Re-ranked list of chunks (top_n or fewer)
    """
    reranker = _get_reranker()
    if not reranker:
        logger.warning("Cross-encoder reranker not available, returning original chunks without reranking")
        return chunks[:top_n]
    
    if not chunks:
        return chunks
    
    try:
        logger.info(f"Cross-encoder reranking {len(chunks)} chunks for query: '{question[:60]}...'")
        # Extract chunk texts
        chunk_texts = [chunk.get("text", "") for chunk in chunks]
        
        # Create query-chunk pairs
        pairs = [[question, text] for text in chunk_texts]
        
        # Get relevance scores
        scores = reranker.predict(pairs)
        
        # Sort by score (descending)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        
        # Log top scores for debugging
        top_scores = [f"{s:.3f}" for s, _ in ranked[:5]]
        logger.info(f"Cross-encoder top scores: {top_scores}")
        
        # Return top_n chunks
        result = [chunk for _, chunk in ranked[:top_n]]
        logger.info(f"Cross-encoder reranking complete: returned {len(result)} chunks")
        return result
    except Exception as e:
        logger.error(f"Cross-encoder reranking failed: {e}, returning original chunks")
        return chunks[:top_n]


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

# Pattern to detect meta-questions about missing papers/gaps in knowledge base
# Includes contextual questions about "above questions" or "this chat"
_MISSING_PAPERS_QUERY_RE = re.compile(
    r"\b(?:what|which)\s+(?:papers?|articles?)\s+(?:would\s+(?:you\s+)?(?:have\s+)?(?:wanted|needed|liked)\s+to\s+(?:have|find)|"
    r"are\s+(?:missing|absent|not\s+(?:in|found))|"
    r"you\s+(?:did\s+)?not\s+(?:find|have)|"
    r"would\s+have\s+(?:helped|been\s+(?:useful|helpful|better)))\b|"
    r"\b(?:full\s+text|pdf)\s+(?:papers?|articles?)\s+(?:you\s+(?:did\s+)?not\s+(?:find|have)|would\s+(?:you\s+)?(?:have\s+)?(?:wanted|needed))\b",
    re.I
)

TABLE_TRUNCATION_REFUSAL = (
    "The table could not be completed for every paper in scope. "
    "Please try again with a paper filter, a smaller author corpus, or ask for "
    "title/year/venue only (metadata table)."
)

_STANDARD_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
})

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
    # Support 'written by' and 'authored by' phrases
    re.compile(r"(?:papers?|articles?|works?|publications?)?\s*(?:written|authored)\s+by\s+(.+?)(?:\s+on\b|[\.,;]|$)", re.I),
    re.compile(r"(?:who\s+wrote|who\s+authored)\s+(?:the\s+)?(?:paper|article|work)?\s*(?:by\s+)?(.+?)(?:\s*[\.,;\?]|$)", re.I),
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
    {
        "id": "coffee_landscape",
        "query_patterns": [
            r"coffee\s+cultural\s+landscape",
            r"colombian\s+coffee",
            r"coffee\s+landscape",
            r"coffee\s+communit",
            r"agroecology\s+coffee",
            r"\bcoffee\s+(?:related\s+)?papers?\b",
        ],
        "paper_markers": [
            "coffee cultural landscape", "colombian coffee", "coffee landscape",
            "coffee communities", "agroecology coffee", "coffee region",
            "sustainable tourism coffee", "coffee biodiversity",
        ],
        "chunk_markers": [
            "coffee", "cultural landscape", "colombian", "agroecology", "biodiversity",
            "tourism", "accessibility", "landscape patterns", "coffee farmers",
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
    r"(?:written|authored)\s+by\s+",
    r"who\s+(?:wrote|authored)\s+",
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


def chunks_to_context_string(
    chunks: list[dict[str, Any]],
    *,
    header: str = "Context Chunks",
    use_structured_shaping: bool = True
) -> str:
    """
    Join formatted chunk blocks; empty list yields a clear message.

    Args:
        chunks: List of retrieved chunks.
        header: Header for the context section.
        use_structured_shaping: If True, use context shaper for paper/section grouping.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "No relevant text passage chunks found for this query."

    # Use structured context shaping if enabled
    if use_structured_shaping:
        try:
            from context_shaper import ContextShaper
            shaper = ContextShaper()
            shaped_context = shaper.shape_context(chunks, query="")
            return f"{header}:\n{shaped_context}"
        except ImportError:
            logger.warning("Context shaper not available, using legacy formatting")
        except Exception as e:
            logger.warning(f"Context shaping failed: {e}, using legacy formatting")

    # Legacy formatting
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
        "review", "reviews", "empirical", "evidence", "conclusions",
        # Query verbs and action words to prevent keyword over-filtering
        "discussed", "discussing", "proposed", "proposing", "compared", "comparing",
        "described", "describing", "analyzed", "analyzing", "presented", "presenting",
        "investigated", "investigating", "used", "using", "found", "find", "seen",
        "report", "reports", "reporting", "summarized", "explained", "mention",
        "mentioned", "address", "addressed"
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

# Hierarchical topic mappings for broader term matching
# When a query uses a broad term, also search for its subtypes
_HIERARCHICAL_TOPIC_MAP = {
    "cancer": [
        "skin cancer", "melanoma", "brain tumor", "brain tumour", "glioma",
        "lung carcinoma", "lung cancer", "breast cancer", "breast tumor",
        "colon cancer", "colorectal cancer", "retinoblastoma", "prostate cancer",
        "leukemia", "lymphoma", "pancreatic cancer", "liver cancer",
        "ovarian cancer", "cervical cancer", "thyroid cancer", "kidney cancer",
        "bladder cancer", "esophageal cancer", "gastric cancer", "stomach cancer"
    ],
    "tumor": [
        "brain tumor", "brain tumour", "glioma", "meningioma", "pituitary tumor",
        "breast tumor", "lung tumor", "colon tumor", "liver tumor", "kidney tumor"
    ],
    "malware": [
        "ransomware", "trojan", "virus", "worm", "spyware", "adware",
        "botnet", "rootkit", "backdoor", "keylogger"
    ],
    "explainability": [
        "interpretable", "interpretability", "transparency", "shap", "lime",
        "attention", "saliency", "visualization", "visualisation"
    ],
    "medical": [
        "clinical", "diagnosis", "patient", "treatment", "therapy",
        "healthcare", "hospital", "medicine", "physician"
    ]
}


def find_papers_by_metadata_keywords(
    query: str,
    papers_metadata: dict,
    *,
    min_token_hits: int | None = None,
    use_expansion: bool = True,
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
    - For multi-token queries with specific terms (e.g., "SDN security"), require
      ALL significant tokens to be present to prevent over-retrieval.
    
    Query expansion: When use_expansion=True, expands query terms with synonyms
    to improve retrieval breadth for multi-term queries.
    """
    if not papers_metadata:
        return []
    
    # Base tokens from query
    tokens = _topic_specific_tokens(query) or _significant_query_tokens(query)
    stop = _query_stopwords().union(_STANDARD_STOPWORDS)
    # Short tech tokens (6g, v2x, ai) from the raw query.
    for raw in re.findall(r"[a-z0-9]{2,}", (query or "").lower()):
        if raw not in tokens and raw not in _GENERIC_TOPIC_TOKENS and raw not in stop:
            tokens.append(raw)
    
    # Identify significant tokens (non-generic) for strict matching
    significant_tokens = [t for t in tokens if t not in _GENERIC_TOPIC_TOKENS and len(t) >= 3]
    
    # Hierarchical topic expansion for broader terms
    # If query contains a broad term like "cancer", also search for its subtypes
    for broad_term, subtypes in _HIERARCHICAL_TOPIC_MAP.items():
        if broad_term in [t.lower() for t in tokens]:
            # Add all subtypes to the token list
            for subtype in subtypes:
                subtype_tokens = subtype.lower().split()
                for st in subtype_tokens:
                    if st not in [t.lower() for t in tokens]:
                        tokens.append(st)
            logger.debug(f"Hierarchical expansion: '{broad_term}' expanded to include {len(subtypes)} subtypes")
    
    # Query expansion for better retrieval breadth
    if use_expansion and len(tokens) >= 1:
        try:
            from query_expansion import QueryExpansion
            expander = QueryExpansion()
            # Expand key terms with synonyms
            expanded_terms = expander.expand_with_key_terms(tokens, max_terms=15)
            # Add expanded terms that aren't already in tokens
            for term in expanded_terms:
                if term.lower() not in [t.lower() for t in tokens]:
                    tokens.append(term.lower())
        except ImportError:
            pass  # Query expansion not available, continue with original tokens
    
    if not tokens:
        return []
    
    # Determine matching requirements
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
        
        # Strict mode: for multi-token queries with 2+ significant tokens,
        # require ALL significant tokens to be present
        if len(significant_tokens) >= 2 and need >= 2:
            significant_hits = sum(1 for t in significant_tokens if t in hay)
            if significant_hits < len(significant_tokens):
                continue  # Skip papers that don't have all significant tokens
        
        if hits >= need:
            matched.append(title)
    
    logger.info(f"Metadata keyword search: query='{query[:40]}...', tokens={len(tokens)}, significant={len(significant_tokens)}, matched={len(matched)} papers")
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
    # Allow contextual questions about chat history to pass through to RAG handler
    if "above questions" in query.lower() or "this chat" in query.lower():
        return False
    
    # If the query contains topic-specific keywords (e.g., "ransomware papers", "phishing detection papers"),
    # this is NOT a simple inventory listing - it requires topic filtering
    q_lower = (query or "").lower()
    
    # Check for explicit topic filtering patterns
    # These patterns indicate the user wants to filter by topic, not list all papers
    topic_filter_patterns = [
        r"\b(?:list|show|table)\s+(?:all\s+)?(?:\w+\s+){0,3}(?:papers?|articles?|studies)\b",
    ]
    
    # Extract potential topic keywords from the query
    # Remove common words to find the actual topic
    words = re.findall(r'\b[a-z]{3,}\b', q_lower)
    stop_words = {"list", "show", "table", "all", "papers", "paper", "articles", "article", "studies", "study", "for", "the", "and", "or", "in", "on", "about"}
    potential_topics = [w for w in words if w not in stop_words]
    
    # If we have potential topic keywords, this requires topic filtering via RAG
    if potential_topics:
        return False
    
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
    q = query or ""
    
    # Check for explicit paper focus patterns
    if _PAPER_FOCUS_RE.search(q):
        return True
    
    # Check for quoted titles
    if re.search(r'"[^"]{8,200}"', q):
        return True
    
    # Check for "Who wrote [paper title]" pattern
    # Extract the phrase after "Who wrote" and check if it matches a paper title
    who_wrote_match = re.search(r'who\s+(?:wrote|authored)\s+(.+?)(?:\s*[?\.]|$)', q, re.I)
    if who_wrote_match:
        potential_title = who_wrote_match.group(1).strip()
        # If the phrase is long enough (>= 4 words), it's likely a paper title not an author name
        if len(potential_title.split()) >= 4:
            return True
    
    return False


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
    if len(token) < 2:  # Lowered from 3 to 2 to support short surnames like Li, Ng
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

    # Extract acronyms from query (e.g., "SH-IDS", "YOLO")
    acronyms = re.findall(r'\b[A-Z]{2,}(?:-[A-Z]{2,})*\b', query)
    if acronyms:
        # Try to match papers by acronym in title
        for acronym in acronyms:
            acronym_lower = acronym.lower()
            for title in papers_metadata:
                title_lower = title.lower()
                # Check if acronym appears as a standalone word in title
                if re.search(rf'\b{re.escape(acronym_lower)}\b', title_lower):
                    if title not in matches:
                        matches.append(title)
        if matches:
            return matches

    # Enhanced unquoted title matching for "Tell me about X" queries
    # Check if the query looks like a paper title (long, specific phrases)
    q_lower = (query or "").lower().strip()
    # Remove common query prefixes
    q_clean = re.sub(r'^(tell me about|what is|describe|explain|summarize|summary of|about|the paper titled|paper titled|what dataset was used in|what deep learning architecture was used in|what performance metrics were reported in|what does jhanjhi think about)\s+', '', q_lower, flags=re.I)
    q_clean = q_clean.strip()
    
    # If the cleaned query is long enough (>= 3 words) and specific, try direct title matching
    words = q_clean.split()
    if len(words) >= 3:
        for title in papers_metadata:
            title_lower = title.lower()
            # Check for substantial overlap (at least 50% of words for 3-5 words, 60% for longer)
            title_words = set(title_lower.split())
            query_words = set(words)
            overlap = len(title_words & query_words)
            min_overlap = 2 if len(query_words) == 3 else min(3, len(query_words) * 0.5)
            if overlap >= min_overlap:
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
    min_required_score = 2 if len(tokens) >= 2 else 1
    return [t for s, t in scored if s == best_score and s >= min_required_score]


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
    # Allow contextual questions about chat history to pass through
    if "above questions" in query.lower() or "this chat" in query.lower():
        return False
    if resolve_matching_paper_titles(query, papers_metadata):
        return False
    tokens = _significant_query_tokens(query)
    return any(len(t) >= 5 for t in tokens)


def is_missing_papers_meta_query(query: str) -> bool:
    """
    Detect meta-questions about identifying gaps in the knowledge base.
    Examples: "what papers would you have wanted to have", "which papers are missing"
    """
    return bool(_MISSING_PAPERS_QUERY_RE.search(query or ""))


def resolve_matching_paper_titles(query: str, papers_metadata: dict) -> list[str]:
    """
    Map a natural-language question to paper title(s) in the local library inventory.
    Author matches always win over title-token matches to prevent wrong-paper answers.
    
    Enhanced to better handle topic-based queries like "Colombian Coffee Cultural Landscape"
    by requiring multiple token matches for topic queries.
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

    # For topic-based queries (not author-scoped), require multiple token matches
    # to prevent false positives like "Deep Learning" matching everything about learning
    is_topic_query = not is_author_scoped and not query_has_author_intent(query)
    min_token_matches = 2 if is_topic_query and len(tokens) >= 2 else 1

    for title, meta in papers_metadata.items():
        authors = meta.get("authors") or ""
        title_l = (title or "").lower()
        token_hits = 0
        
        for token in tokens:
            if len(token) < 4:
                continue
            if author_field_contains_token(authors, token):
                if title not in author_matches:
                    author_matches.append(title)
                break
            if not is_author_scoped and token in title_l:
                token_hits += 1
        
        # Only add to matched if we have enough token hits
        if token_hits >= min_token_matches and title not in matched:
            matched.append(title)

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


def is_bibliography_chunk(text: str) -> bool:
    """
    Detect if a chunk's text is primarily bibliography / references content.
    Uses robust heuristics to scan for lists of citations or year patterns.
    """
    if not text:
        return False
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return False
        
    # Heuristics:
    # 1. High density of years in parentheses (e.g. (2018), (2020))
    year_patterns = len(re.findall(r'\b(?:19|20)\d{2}\b', text))
    
    # 2. Count lines starting with citation markers like [1], [2], or author names
    brackets_citations = len(re.findall(r'^\[\d+\]', text, re.M))
    numbered_citations = len(re.findall(r'^\d+\.\s+[A-Z]', text, re.M))
    
    # 3. High density of common bibliography words
    bib_keywords = ("doi:", "proceedings", "journal of", "vol.", "no.", "pp.", "et al.", "press", "university", "editor")
    keyword_hits = sum(1 for kw in bib_keywords if kw in text.lower())
    
    # Calculate density metrics
    num_words = len(text.split())
    if num_words < 10:
        return False
        
    # If the text has many bracketed citations at line starts
    if brackets_citations >= 3 or numbered_citations >= 3:
        return True
        
    # If a high percentage of words or sentences are citation-heavy
    # e.g., if there is 1 year per 15 words and some keywords
    if year_patterns >= 3 and keyword_hits >= 2 and (year_patterns / num_words) > 0.03:
        return True
        
    # If it has a huge number of "et al." and "pp." or "doi"
    if text.lower().count("et al.") >= 4 or text.lower().count("doi:") >= 3:
        return True
        
    return False


def _ensure_rerank_scores(chunks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    for c in chunks:
        if "rerank_score" not in c:
            # Fallback calculation using distance: similarity score in [0.0, 1.0]
            distance = float(c.get("distance", 1.0))
            similarity = max(0.0, 1.0 - (distance / 2.0))
            overlap_count = _chunk_query_overlap_score(c, query)
            lexical_bonus = min(0.3, overlap_count * 0.1)
            c["rerank_score"] = min(1.0, similarity + lexical_bonus)
    return chunks


def retrieve_relevant_chunks(
    vector_store,
    query: str,
    limit: int,
    filter_title: str | None = None,
    *,
    scope_titles: list[str] | None = None,
    use_reranking: bool = True,
    over_retrieve_multiplier: float = 3.0,
    use_domain_filtering: bool = True,
    use_query_routing: bool = True,
) -> list[dict[str, Any]]:
    """
    Retrieve context chunks for RAG with author/paper-aware isolation and optional reranking.

    Pipeline:
      1. Query understanding (optional): classify intent and determine pipeline routing
      2. Detect if the query names a specific author/paper already in the library.
      3. Author-scoped path: fetch chunks EXCLUSIVELY from those papers — no semantic
         mixing that could contaminate results with other authors' papers.
      4. Unscoped path: standard semantic search → distance filter → token filter.
      5. (Optional) Over-retrieve more chunks and apply reranking for better accuracy.
      6. (Optional) Apply domain filtering to prevent cross-topic contamination.

    Args:
        vector_store: ChromaDB vector store instance.
        query: User's research question.
        limit: Final number of chunks to return.
        filter_title: Optional paper title filter.
        scope_titles: Optional list of paper titles to scope retrieval.
        use_reranking: If True, apply hybrid reranking after retrieval.
        over_retrieve_multiplier: Multiplier for initial retrieval (e.g., 3.0 = retrieve 3x limit).
        use_domain_filtering: If True, detect query domain and filter by domain metadata.
        use_query_routing: If True, use query understanding to control pipeline routing.

    Returns:
        List of retrieved and optionally reranked chunks.
    """
    stats = vector_store.get_collection_stats()
    papers_metadata = stats.get("papers_metadata", {})

    # ── Query understanding for pipeline routing ─────────────────────────────
    routing_config = {}
    if use_query_routing and not filter_title and not scope_titles:
        try:
            from query_understanding import QueryUnderstanding
            query_understanding = QueryUnderstanding()
            analysis = query_understanding.understand_query(query)
            routing_config = query_understanding.get_pipeline_routing(analysis)
            
            # Apply routing configuration
            if routing_config.get("retrieval_limit_multiplier"):
                over_retrieve_multiplier *= routing_config["retrieval_limit_multiplier"]
            
            if routing_config.get("strict_metadata_filter"):
                # Enforce strict metadata filtering
                if "paper_title" in analysis.constraints:
                    filter_title = analysis.constraints["paper_title"]
        except ImportError:
            logger.warning("Query understanding not available, skipping pipeline routing")
        except Exception as e:
            logger.warning(f"Pipeline routing failed: {e}")

    inventory_titles = list(scope_titles) if scope_titles else resolve_matching_paper_titles(
        query, papers_metadata
    )
    if filter_title:
        inventory_titles = [filter_title]

    strict = getattr(settings, "RAG_STRICT_MODE", True)
    # Lock retrieval whenever we resolved specific paper(s) — prevents cross-paper contamination.
    locked_scope = bool(scope_titles) or bool(inventory_titles) or bool(filter_title)

    # ── Domain filtering for multi-topic separation ───────────────────────────
    filter_domain = None
    if use_domain_filtering and not locked_scope:
        try:
            from topic_classifier import TopicClassifier
            classifier = TopicClassifier()
            filter_domain = classifier.get_domain_filter(query)
            if filter_domain:
                logger.info(f"Query detected as domain-specific: {filter_domain}")
                
                # HARD CONSTRAINT: Filter inventory to only papers in this domain
                domain_filtered_titles = []
                for title, meta in papers_metadata.items():
                    paper_domain = meta.get("domain", "").lower()
                    if paper_domain == filter_domain.lower():
                        domain_filtered_titles.append(title)
                
                if domain_filtered_titles:
                    if inventory_titles:
                        # Intersect with existing inventory titles
                        inventory_titles = list(set(inventory_titles) & set(domain_filtered_titles))
                    else:
                        inventory_titles = domain_filtered_titles
                    logger.info(f"Domain filtering applied: {len(inventory_titles)} papers in domain '{filter_domain}'")
                    
                    # HARD CONSTRAINT: If no papers match domain, return empty
                    if not inventory_titles:
                        logger.warning(f"No papers match domain constraint '{filter_domain}' for query: {query}")
                        return []
                else:
                    logger.warning(f"No papers found in domain '{filter_domain}'")
        except ImportError:
            logger.warning("Topic classifier not available, skipping domain filtering")
        except Exception as e:
            logger.warning(f"Domain filtering failed: {e}")

    # ── Metadata-driven pre-filtering ───────────────────────────────────────
    metadata_constraints = {}
    try:
        from metadata_filter import MetadataFilter
        metadata_filter = MetadataFilter()
        if metadata_filter.should_apply_metadata_filtering(query) or routing_config.get("strict_metadata_filter"):
            filtered_titles = metadata_filter.filter_papers_by_metadata(
                papers_metadata, query
            )
            if filtered_titles and len(filtered_titles) < len(papers_metadata):
                # Apply metadata filtering as an additional scope
                if inventory_titles:
                    # Intersect with existing inventory titles
                    inventory_titles = list(set(inventory_titles) & set(filtered_titles))
                else:
                    inventory_titles = filtered_titles
                logger.info(f"Metadata filtering applied: {len(inventory_titles)} papers in scope")
                
                # HARD CONSTRAINT: If no papers match metadata constraints, return empty
                if not inventory_titles:
                    logger.warning(f"No papers match metadata constraints for query: {query}")
                    return []
    except ImportError:
        logger.warning("Metadata filter not available, skipping metadata filtering")
    except Exception as e:
        logger.warning(f"Metadata filtering failed: {e}")

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
        # Over-retrieve for author-scoped queries too
        per_paper = max(3, min(12, int((limit * over_retrieve_multiplier) // n_papers)))
        for title in inventory_titles:
            paper_chunks = vector_store.get_chunks_for_paper(title, max_chunks=per_paper)
            # Filter out bibliography chunks
            paper_chunks = [c for c in paper_chunks if not is_bibliography_chunk(c.get("text", ""))]
            author_chunks = _dedupe_chunks(author_chunks + paper_chunks)
        if author_chunks:
            # Apply reranking to author-scoped chunks if enabled
            if use_reranking and len(author_chunks) > limit:
                # Use cross-encoder reranking for better precision
                author_chunks = rerank_chunks(query, author_chunks, top_n=limit)
                logger.info(f"Cross-encoder reranking applied to author-scoped chunks: {len(author_chunks)} chunks")
            return _ensure_rerank_scores(rank_and_cap_chunks(author_chunks, query, limit=limit), query)

    # ── Standard semantic search path with over-retrieval ─────────────────────
    # Over-retrieve: request extra candidates so reranking has more to work with
    search_limit = max(int(limit * over_retrieve_multiplier), limit + 8)
    
    # CRITICAL FIX: Apply domain filter at vector search level, not post-filter
    # This prevents vector search from retrieving irrelevant domains
    effective_filter_domain = filter_domain
    if not effective_filter_domain and inventory_titles:
        # If we have inventory_titles from metadata filtering, try to infer domain
        # to prevent cross-domain leakage
        first_title = inventory_titles[0] if inventory_titles else None
        if first_title and first_title in papers_metadata:
            paper_domain = papers_metadata[first_title].get("domain", "")
            if paper_domain:
                effective_filter_domain = paper_domain
                logger.info(f"Inferred domain from inventory_titles: {effective_filter_domain}")
    
    # Generate expanded queries for search to improve recall
    search_queries = [query]
    
    # 1. Cleaned query (e.g. replace hyphens with spaces)
    cleaned_q = query.replace("-", " ")
    if cleaned_q.lower() != query.lower():
        search_queries.append(cleaned_q)
        
    # 2. Extract acronyms/words
    acronyms = re.findall(r"\b[A-Z0-9\-]{2,}\b", query)
    if acronyms:
        for ac in acronyms:
            if "-" in ac:
                search_queries.append(ac.replace("-", " "))
            search_queries.append(ac)
            
    # 3. Title-based expansion
    for t in inventory_titles[:2]:
        title_words = t.split()[:6]
        title_q = " ".join(title_words)
        if title_q not in search_queries:
            search_queries.append(title_q)

    # 4. Author-based expansion (combining author surname with key terms)
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower())
    stop_words = {"papers", "paper", "article", "articles", "using", "uses", "with", "from", "library", "ingested"}
    sig_words = [w for w in words if w not in stop_words and not w.isdigit()]
    
    for title in inventory_titles:
        meta = papers_metadata.get(title) or {}
        authors_field = meta.get("authors") or ""
        for part in re.split(r"[,;&]| and ", authors_field):
            name_words = [w.strip() for w in part.split() if w.strip() and len(w.strip()) >= 3]
            if name_words:
                surname = name_words[-1]
                if sig_words:
                    author_q = f"{surname} " + " ".join(sig_words)
                    if author_q not in search_queries:
                        search_queries.append(author_q)
                    cleaned_sig = [w.replace("-", " ") for w in sig_words]
                    author_q_clean = f"{surname} " + " ".join(cleaned_sig)
                    if author_q_clean not in search_queries:
                        search_queries.append(author_q_clean)

    # Deduplicate queries
    search_queries = list(dict.fromkeys(search_queries))
    logger.info(f"Retrieving using expanded queries: {search_queries}")

    # Query vector store for all queries and merge/deduplicate
    raw = []
    seen_ids = set()
    for sq in search_queries:
        results = vector_store.query_similar_chunks(
            sq, limit=search_limit, filter_title=filter_title, filter_domain=effective_filter_domain
        )
        for r in results:
            rid = r.get("id")
            if rid not in seen_ids:
                seen_ids.add(rid)
                raw.append(r)
                
    # Sort merged results by distance ascending (most similar first)
    raw.sort(key=lambda x: x.get("distance", 1.0))

    # Filter out bibliography chunks
    raw = [c for c in raw if not is_bibliography_chunk(c.get("text", ""))]

    # Determine adaptive distance threshold based on query type
    _AGG_PATTERNS = (
        "what do all", "all papers say", "across all", "summarize all",
        "compare all", "conclusion of all", "what do these papers",
        "all studies say", "combined conclusion", "combined summary",
        "overall conclusion", "aggregate", "synthesis of all",
    )
    is_aggregation = any(p in query.lower() for p in _AGG_PATTERNS)
    is_single_paper = bool(filter_title) or (bool(inventory_titles) and len(inventory_titles) == 1)

    if is_aggregation:
        adaptive_distance = getattr(settings, "RAG_MAX_DISTANCE_AGGREGATION", 0.58)
        logger.info(f"Aggregation query detected. Using strict threshold {adaptive_distance}")
    elif is_single_paper:
        adaptive_distance = getattr(settings, "RAG_MAX_DISTANCE_SINGLE", 0.78)
        logger.info(f"Single-paper query detected. Using relaxed threshold {adaptive_distance}")
    else:
        adaptive_distance = getattr(settings, "RAG_MAX_DISTANCE_DEFAULT", 0.70)
        logger.info(f"General query. Using default threshold {adaptive_distance}")

    chunks = filter_chunks_by_relevance(raw, max_distance=adaptive_distance)

    if settings.RAG_REQUIRE_QUERY_TERM_MATCH:
        filtered_term_chunks = _filter_chunks_by_query_term_presence(
            chunks,
            query,
            skip_if_empty=False,
        )
        if filtered_term_chunks:
            chunks = filtered_term_chunks
        else:
            logger.info("Strict query term matching returned zero results. Falling back to pure embedding search to maximize recall.")

    # CRITICAL: When inventory_titles is set (specific paper matched), ALWAYS filter to those papers
    # This prevents cross-paper hallucination where content from wrong papers is cited
    if inventory_titles:
        # First, filter existing chunks to only matched papers
        chunks = filter_chunks_to_titles(chunks, inventory_titles)
        
        # If no chunks after filtering, retrieve directly from matched papers
        if not chunks:
            for title in inventory_titles:
                chunks = _dedupe_chunks(
                    chunks + vector_store.get_chunks_for_paper(title, max_chunks=max(limit, 12))
                )
            # Re-filter after direct retrieval to ensure only matched papers
            chunks = filter_chunks_to_titles(chunks, inventory_titles)
        
        # ALWAYS apply final filter to ensure no cross-paper contamination
        chunks = filter_chunks_to_titles(chunks, inventory_titles)

    # Strict mode: never return unscoped semantic noise when the query is entity-locked.
    if strict and locked_scope:
        filtered = filter_chunks_to_titles(chunks, inventory_titles)
        if use_reranking and len(filtered) > limit:
            # Use cross-encoder reranking for better precision
            filtered = rerank_chunks(query, filtered, top_n=limit)
            logger.info(f"Cross-encoder reranking applied in strict mode: {len(filtered)} chunks")
        return _ensure_rerank_scores(filtered[:limit], query)

    if not chunks and inventory_titles and not strict:
        chunks = raw[:limit]
        chunks = filter_chunks_to_titles(chunks, inventory_titles) or chunks

    if not chunks and not locked_scope:
        chunks = raw[:limit]

    profile = detect_topic_profile(query)
    if profile and chunks:
        chunks = filter_chunks_for_topic_profile(chunks, profile)

    # Apply reranking if enabled and we have enough chunks
    if use_reranking and len(chunks) > limit:
        # Use cross-encoder reranking for better precision
        chunks = rerank_chunks(query, chunks, top_n=limit)
        logger.info(f"Cross-encoder reranking applied: returned {len(chunks)} chunks")

    return _ensure_rerank_scores(rank_and_cap_chunks(chunks, query, limit=limit), query)


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
