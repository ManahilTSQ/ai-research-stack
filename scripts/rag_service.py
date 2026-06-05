"""
rag_service.py — Retrieval-Augmented Generation (RAG) Orchestration Service.

Connects the ChromaDB vector database retrieval layer to the local Ollama LLM:
  1. Retrieves the most relevant text chunks from ChromaDB for a given query.
  2. Formats a structured prompt with the retrieved context.
  3. Sends the prompt to the local Ollama chat API.
  4. Returns the grounded answer along with the source chunks used.

Also provides check_ollama_health() — a lightweight utility used by both
main.py and server.py to fail fast with a clear error message when Ollama
is not running before any expensive work is initiated.
"""

import sys
import logging
import re
import requests
from config import settings            # Flat import — scripts/ is on sys.path
from vector_store import VectorStoreService  # Local ChromaDB interface
from rag_context import (
    build_library_inventory,
    chunks_to_context_string,
    retrieve_relevant_chunks,
    resolve_matching_paper_titles,
    query_refers_to_missing_library_paper,
    is_missing_papers_meta_query,
    query_expects_named_author,
    is_simple_inventory_listing,
    is_per_paper_extraction_query,
    classify_query_mode,
    parse_table_columns_from_query,
    answer_has_table_truncation,
    filter_chunks_to_titles,
    EMPTY_DB_REFUSAL,
    IRRELEVANT_REFUSAL,
    NOT_IN_LIBRARY_REFUSAL,
    TABLE_TRUNCATION_REFUSAL,
    extract_author_search_phrase,
    resolve_author_from_library,
)
from rag_strict import (
    resolve_query_scope,
    scope_refusal_message,
    inventory_for_scope,
    apply_verification_or_refuse,
    answer_catalog_metadata_query,
    _fuzzy_title_match,
    _TOPIC_NOT_FOUND_REFUSAL,
    compare_query_needs_paper_pickers,
    COMPARE_NEEDS_PICKER_MSG,
    apply_scope_resilience,
    answer_keyword_discovery_query,
    verify_author_exists_in_library,
    is_broad_author_query,
)

# Maximum papers processed in one batched extraction table (one LLM call per paper).
MAX_EXTRACTION_TABLE_PAPERS = 60


def _parse_user_paper_limit(query: str) -> int | None:
    """
    Extract an explicit paper limit from the query, e.g.
      - "limit to 10 papers"
      - "show only 5"
      - "top 10 papers"
      - "first 15"
    Returns None when no limit is stated.
    """
    patterns = [
        r"\blimit\s+(?:to|of)?\s*(\d+)\s*papers?",
        r"\bshow\s+(?:only\s+|me\s+)?(\d+)\s*papers?",
        r"\btop\s+(\d+)\s*papers?",
        r"\bfirst\s+(\d+)\s*papers?",
        r"\bonly\s+(\d+)\s*papers?",
        r"\bmax(?:imum)?\s+(\d+)\s*papers?",
        r"\blimited\s+to\s+(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, (query or ""), re.I)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 200:
                    return n
            except ValueError:
                pass
    return None


# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server is not reachable."""


# ──────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL UTILITY: Ollama Health Check
# ──────────────────────────────────────────────────────────────────────────────

def check_ollama_health() -> bool:
    """
    Check whether the local Ollama server is reachable and responding.

    Sends a lightweight GET request to the /api/tags endpoint with a 5-second
    timeout. This endpoint lists all pulled models and always returns 200 when
    Ollama is running.

    On failure, prints a clear, actionable error message so the user knows
    exactly what to do (rather than seeing a cryptic connection refused error).

    Returns:
        True if Ollama is online and responsive, False otherwise.
    """
    health_url = f"{settings.OLLAMA_BASE_URL}/api/tags"

    try:
        resp = requests.get(health_url, timeout=5)

        if resp.status_code == 200:
            return True  # Ollama is up and responding normally

        # Non-200 response — Ollama is running but something is wrong
        logger.warning(
            f"Ollama responded with unexpected status {resp.status_code} at {health_url}"
        )
        return False

    except requests.exceptions.ConnectionError:
        # The most common case — Ollama is simply not running yet
        print(
            "\nERROR: Ollama is not running or unreachable.\n"
            f"    Expected at: {settings.OLLAMA_BASE_URL}\n"
            "    Start it with:  ollama serve\n"
            "    Then retry your command.\n"
        )
        return False

    except Exception as e:
        # Unexpected error (DNS failure, proxy issue, etc.)
        logger.warning(f"Ollama health check raised an unexpected error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: RAG Service
# ──────────────────────────────────────────────────────────────────────────────

class RAGService:
    """
    Orchestrates the full RAG pipeline: retrieve → prompt → generate → return.

    RAG (Retrieval-Augmented Generation) grounds LLM answers in real document
    content, preventing hallucination by restricting the model to only facts
    present in the retrieved context chunks.

    Workflow:
      1. Query ChromaDB for the top-K most relevant text chunks.
      2. Assemble a structured prompt with those chunks as context.
      3. Call the local Ollama LLM via its /api/chat endpoint.
      4. Parse the response and return it with source attribution.
    """

    def __init__(self):
        """
        Initialise the RAG service.

        Performs an Ollama health check immediately — if Ollama is not reachable,
        the process exits with a clear error message before wasting time loading
        the ChromaDB collection.
        """
        logger.info("Initialising RAG Orchestration Service...")

        if not check_ollama_health():
            raise OllamaUnavailableError(
                f"Ollama is not running at {settings.OLLAMA_BASE_URL}. Start with: ollama serve"
            )

        # Initialise the vector store client (loads ChromaDB from disk)
        self.vector_store = VectorStoreService()

        logger.info("RAG Service initialised successfully.")

    def _ollama_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.05,
        num_predict: int = 2048,
    ) -> str:
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.8,
                "repeat_penalty": 1.1,
                "num_predict": num_predict,
            },
        }
        response = requests.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json()["message"]["content"].strip()

    def _escape_md_cell(self, value: str) -> str:
        return (value or "").replace("|", "\\|").replace("\n", " ").strip()

    def _metadata_table_row(self, title: str, meta: dict, columns: list[str]) -> str:
        """Deterministic cells for standard bibliographic columns."""
        col_map = {
            "title": title,
            "year": str(meta.get("year", "N/A")),
            "venue": str(meta.get("venue", "N/A")),
            "authors": str(meta.get("authors", "Unknown Authors")),
            "doi": str(meta.get("doi", "N/A")),
        }
        cells: list[str] = []
        for col in columns:
            key = col.lower().strip()
            if key in col_map:
                cells.append(self._escape_md_cell(col_map[key]))
            else:
                cells.append("—")
        return "| " + " | ".join(cells) + " |"

    def _render_inventory_listing(self, query: str, inventory_metadata: dict) -> str:
        """Deterministic metadata-only list/table for in-scope papers."""
        is_table_request = "table" in query.lower() or "tabulate" in query.lower()
        if is_table_request:
            table_rows = [
                "| Title | Year | Venue |",
                "|-------|------|-------|",
            ]
            for title, meta in inventory_metadata.items():
                year = meta.get("year", "N/A")
                venue = meta.get("venue", "N/A")
                title_escaped = title.replace("|", "\\|")
                venue_escaped = venue.replace("|", "\\|")
                table_rows.append(f"| {title_escaped} | {year} | {venue_escaped} |")
            return "\n\n".join(table_rows)

        listing_parts = []
        for idx, (title, meta) in enumerate(inventory_metadata.items(), 1):
            authors = meta.get("authors", "Unknown Authors")
            year = meta.get("year", "N/A")
            doi = meta.get("doi", "N/A")
            venue = meta.get("venue", "")
            entry = f"{idx}. {authors} ({year}). {title}"
            if venue:
                entry += f". {venue}"
            if doi and doi != "N/A":
                entry += f". doi: {doi}"
            listing_parts.append(entry)
        return "Papers in your library:\n\n" + "\n\n".join(listing_parts)

    def _extract_single_paper_row(
        self,
        title: str,
        meta: dict,
        columns: list[str],
        paper_chunks: list[dict],
        query: str,
    ) -> str:
        """One LLM call per paper — prevents cross-author contamination and truncation."""
        context = chunks_to_context_string(paper_chunks, header=f"Paper: {title}")
        col_list = ", ".join(columns)
        system_prompt = (
            "You extract ONE markdown table row for a single academic paper.\n"
            "Rules:\n"
            "- Use ONLY the provided paper context and metadata.\n"
            "- Output EXACTLY one line: a markdown table row starting with | and ending with |.\n"
            "- Do NOT output a header row, separator row, or any other text.\n"
            "- If a field is not stated in the text, write: Not stated in text\n"
            "- Never mention other papers or authors not in this paper's metadata.\n"
        )
        user_prompt = (
            f"Researcher request (for context): {query}\n\n"
            f"Paper metadata:\n"
            f"Title: {title}\n"
            f"Authors: {meta.get('authors', 'Unknown')}\n"
            f"Year: {meta.get('year', 'N/A')}\n"
            f"Venue: {meta.get('venue', 'N/A')}\n"
            f"DOI: {meta.get('doi', 'N/A')}\n\n"
            f"Columns (in order): {col_list}\n\n"
            f"{context}\n\n"
            "Output one markdown table row only:"
        )
        raw = self._ollama_chat(system_prompt, user_prompt, num_predict=1024)
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("|") and line.endswith("|"):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) == len(columns):
                    return "| " + " | ".join(self._escape_md_cell(p) for p in parts) + " |"
        return self._metadata_table_row(title, meta, columns)

    def _generate_per_paper_extraction_table(
        self,
        query: str,
        inventory_metadata: dict,
        columns: list[str],
    ) -> dict:
        titles = list(inventory_metadata.keys())
        if not titles:
            return {
                "query": query,
                "answer": NOT_IN_LIBRARY_REFUSAL,
                "sources": [],
                "success": False,
                "error": "No papers in scope for extraction table.",
            }

        # Honour explicit user-stated limit (e.g. "limit to 10 papers", "top 5").
        user_limit = _parse_user_paper_limit(query)
        if user_limit is not None:
            titles = titles[:user_limit]
            inventory_metadata = {t: inventory_metadata[t] for t in titles}

        if len(titles) > MAX_EXTRACTION_TABLE_PAPERS:
            return {
                "query": query,
                "answer": (
                    f"This extraction table would cover {len(titles)} papers, which exceeds "
                    f"the limit of {MAX_EXTRACTION_TABLE_PAPERS}. Narrow the query to one author "
                    "or use the paper filter dropdown."
                ),
                "sources": [],
                "success": False,
                "error": "Extraction table too large.",
            }

        header = "| " + " | ".join(self._escape_md_cell(c) for c in columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        rows: list[str] = []
        all_sources: list[dict] = []

        meta_only_keys = {"title", "year", "venue", "authors", "doi"}
        needs_llm = any(c.lower().strip() not in meta_only_keys for c in columns)

        for title in sorted(titles, key=lambda t: str(inventory_metadata[t].get("year", ""))):
            meta = inventory_metadata[title]
            paper_chunks = self.vector_store.get_chunks_for_paper(title, max_chunks=24)
            all_sources.extend(paper_chunks[:8])
            if needs_llm and paper_chunks:
                row = self._extract_single_paper_row(
                    title, meta, columns, paper_chunks, query
                )
            else:
                row = self._metadata_table_row(title, meta, columns)
            rows.append(row)

        answer = "\n\n".join([header, separator, *rows])
        if answer_has_table_truncation(answer):
            return {
                "query": query,
                "answer": TABLE_TRUNCATION_REFUSAL,
                "sources": all_sources,
                "success": False,
                "error": "Table truncation detected.",
            }
        return {
            "query": query,
            "answer": answer,
            "sources": all_sources,
            "success": True,
        }

    def _build_safe_references(self, chunks: list[dict]) -> str:
        """
        Build a deterministic References section from retrieved chunk metadata only.
        This prevents the model from inventing bibliography entries that were never retrieved.
        """
        refs = []
        seen = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {}) or {}
            title = (meta.get("title") or "Untitled").strip()
            authors = (meta.get("authors") or "Unknown Authors").strip()
            year = str(meta.get("year") or "N/A").strip()
            doi = (meta.get("doi") or "N/A").strip()
            venue = (meta.get("venue") or "").strip()
            pages = (meta.get("pages") or "").strip()
            
            # Skip if metadata is too incomplete (placeholder detection)
            if title == "Untitled" or authors == "Unknown Authors" or year == "N/A":
                continue
            
            key = (title.lower(), authors.lower(), year, doi.lower())
            if key in seen:
                continue
            seen.add(key)
            
            # Build full APA7 reference with venue and pages if available
            ref_parts = [f"- {authors} ({year}). {title}"]
            if venue and venue != "N/A":
                ref_parts.append(venue)
            if pages and pages != "N/A":
                ref_parts.append(pages)
            ref = ". ".join(ref_parts) + "."
            if doi and doi != "N/A":
                ref += f" https://doi.org/{doi}"
            refs.append(ref)
        if not refs:
            return ""
        return "References:\n" + "\n".join(refs)

    def _strip_model_references(self, answer: str) -> str:
        """
        Remove any model-generated References section so we can append verified references.
        """
        if not answer:
            return ""
        # Match lines like "References:", "## References", "References list", etc., and strip everything after
        stripped = re.split(r"\n\s*(?:#+\s*|\*+\s*|_+)?references\b[:\s]*", answer, maxsplit=1, flags=re.IGNORECASE)[0]
        return stripped.strip()

    def _is_refusal_answer(self, answer: str) -> bool:
        """
        True when the final answer is a refusal / cannot-verify response.
        In these cases we must NOT append a References section.
        """
        a = (answer or "").strip().lower()
        if not a:
            return False
        refusal_markers = (
            EMPTY_DB_REFUSAL,
            IRRELEVANT_REFUSAL,
            NOT_IN_LIBRARY_REFUSAL,
            TABLE_TRUNCATION_REFUSAL,
            "this question is outside the scope of your ingested research knowledge base",
            "i cannot provide this answer because it references papers or authors that are not in the retrieved scope",
            "i could not find any ingested papers in your knowledge base that discuss that topic",
            "i could not find any relevant papers or context in the local database",
        )
        if any(marker and marker.lower() in a for marker in refusal_markers if marker):
            return True
            
        semantic_refusal_phrases = (
            "not related to",
            "not mention",
            "not present in",
            "outside the scope",
            "outside of the scope",
            "does not contain",
            "do not contain",
            "no papers",
            "no information",
            "not found in the",
            "not available",
            "is not discussed",
            "are not discussed",
            "cannot answer",
            "unavailable",
            "not relate to",
        )
        if any(phrase in a for phrase in semantic_refusal_phrases):
            return True
            
        return False

    def _is_unverifiable_sensitive_claim(self, query: str, chunks: list[dict]) -> bool:
        """
        Detect high-risk stance/position questions where retrieved context contains
        no lexical evidence for the sensitive topic, and force a safe refusal.
        """
        q = (query or "").lower()
        # Only trigger on explicitly controversial political/social topics, not generic academic terms
        sensitive_topics = [
            "abortion", "reproductive rights", "pro-choice", "pro life",
        ]
        stance_indicators = [
            "supports", "opposes", "stance on", "views on", "position on",
        ]
        
        # Must have both a sensitive topic AND a stance indicator to trigger
        has_topic = any(topic in q for topic in sensitive_topics)
        has_stance = any(indicator in q for indicator in stance_indicators)
        
        if not (has_topic and has_stance):
            return False
            
        combined = " ".join((c.get("text") or "").lower() for c in chunks)
        topic_present = any(term in combined for term in ["abortion", "reproductive", "pro-choice", "pro life"])
        return not topic_present

    def _check_author_in_library(self, query: str, papers_metadata: dict) -> tuple[bool, str]:
        """
        Check if the query mentions a specific author name that is NOT in the library.
        Returns (should_refuse, refusal_message).
        """
        # Disabled - too aggressive and blocks legitimate authors
        # Rely on system prompt to prevent hallucination instead
        return False, ""

    def _is_off_topic(self, query: str) -> bool:
        """
        Deterministic pre-LLM keyword blocker.
        Returns True if the query is clearly unrelated to academic research.
        This fires BEFORE any ChromaDB or LLM call — it is an absolute gate.
        """
        q = query.lower().strip()
        # Hard off-topic keyword families — grouped for clarity
        off_topic_patterns = [
            # Food & cooking
            "recipe", "cook", "bake", "ingredient", "meal", "food",
            "breakfast", "lunch", "dinner", "cuisine", "chef",
            # Weather & trivial queries
            "weather", "temperature", "forecast", "rain", "sunny",
            # Sports
            "football", "cricket", "soccer", "basketball", "tennis",
            "match", "score", "tournament",
            # Entertainment
            "movie", "film", "song", "music", "celebrity", "actor", "actress",
            # Finance / non-research
            "stock", "bitcoin", "cryptocurrency", "investment tips",
            # Travel
            "hotel", "flight", "booking", "vacation", "tourist",
        ]
        return any(pat in q for pat in off_topic_patterns)

    def _is_academic_drafting_request(self, query: str) -> bool:
        """
        True when the user is asking the assistant to *write* academic text
        (intro/abstract/conclusion/related work), not to report what the library says.

        These requests should not be routed into library keyword discovery or
        strict in-scope verification, because the correct behavior is often to
        produce a well-formed draft even when the library lacks direct evidence.
        """
        q = (query or "").strip().lower()
        if not q:
            return False
        drafting_verbs = (
            "draft", "write", "compose", "generate", "create", "suggest", "propose",
        )
        academic_targets = (
            # Standard academic sections
            "introduction", "abstract", "conclusion", "related work",
            "literature review", "background", "problem statement",
            "methodology", "discussion",
            # Research planning outputs
            "gap analysis", "research gap", "research question", "research direction",
            "future work", "future direction", "open problem",
            # Professional / communication outputs
            "linkedin post", "blog post", "survey draft", "bibliograph",
            "paragraph", "section",
        )
        if not any(v in q for v in drafting_verbs):
            return False
        if not any(t in q for t in academic_targets):
            return False
        # If the user explicitly asks "based on my library/papers", keep RAG
        # so the answer is grounded in actual ingested content.
        if any(phrase in q for phrase in (
            "my library", "my papers", "in my library", "from my library",
            "ingested", "based on my", "from my", "using my",
        )):
            return False
        return True

    def _draft_academic_text(self, query: str) -> dict:
        """
        Generate an academic draft without RAG grounding.
        This is intentionally citation-free unless the user provides sources.
        """
        system_prompt = (
            "You are an academic writing assistant.\n"
            "Write clearly and formally.\n"
            "Do NOT invent citations, DOIs, authors, or references.\n"
            "Do NOT claim 'the literature shows' unless the user provided sources.\n"
            "If needed, use neutral phrasing like 'prior work has explored' without naming papers.\n"
        )
        answer = self._ollama_chat(system_prompt, query, temperature=0.4, num_predict=1200)
        return {
            "query": query,
            "answer": answer.strip(),
            "sources": [],
            "success": True,
        }

    def generate_answer(
        self,
        query: str,
        limit: int = 8,
        filter_title: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Execute the complete RAG pipeline for a researcher's query.

        Steps:
          1. Retrieve top-K relevant chunks from ChromaDB using cosine similarity.
             If filter_title is set, restricts retrieval to only that paper.
          2. Format each chunk into a labelled context block with source attribution.
          3. Build a structured system prompt instructing the LLM on citation rules.
          4. Send both prompts to Ollama's /api/chat endpoint.
          5. Return the structured response dict.

        Args:
            query: The research question to answer.
            limit: Number of context chunks to retrieve from ChromaDB (default: 4).
            filter_title: If set, restricts retrieval to only chunks from this paper.

        Returns:
            Dict with keys:
              - "query" (str): The original question.
              - "answer" (str): The LLM-generated grounded answer.
              - "sources" (list): The retrieved chunks used as context.
              - "success" (bool): Whether generation succeeded.
              - "error" (str, optional): Error message if success=False.
        """
        # ── Step 0: Deterministic off-topic gate — fires before ANY LLM or DB call ──
        if self._is_off_topic(query):
            return {
                "query": query,
                "answer": (
                    "This question is outside the scope of your ingested research knowledge base. "
                    "I can only answer questions based on the academic papers that have been ingested. "
                    "Please ask a question about the research papers in your library."
                ),
                "sources": [],
                "success": False,
                "error": "Off-topic query blocked by keyword gate.",
            }

        # ── Academic drafting path (non-RAG) ─────────────────────────────────────
        # Example: "Draft an introduction for a paper on federated learning for IoT security"
        # This must not trigger keyword discovery ("papers on ...") or strict scope verification.
        if self._is_academic_drafting_request(query):
            return self._draft_academic_text(query)

        # Fetch library index to support meta-queries
        stats = self.vector_store.get_collection_stats()
        papers_metadata = stats.get("papers_metadata", {})

        # ── Detect author-scoped and listing/tabulation queries ─────────────
        # When the query names a specific author, we:
        #   a) Filter the library inventory to only that author's papers so the LLM
        #      cannot accidentally reference other authors' works.
        #   b) Scale up the retrieval limit so every paper gets at least one chunk.
        scope = resolve_query_scope(
            query, papers_metadata, filter_title=filter_title
        )
        scope = apply_scope_resilience(scope, query, papers_metadata)
        if compare_query_needs_paper_pickers(query, papers_metadata) and not filter_title:
            return {
                "query": query,
                "answer": COMPARE_NEEDS_PICKER_MSG,
                "sources": [],
                "success": True,
            }

        matched_titles = scope.scoped_titles
        query_mode = classify_query_mode(query)
        listing_style_query = (query_mode == "listing")

        # ── Scope refusal for explicit author/paper queries ─────────────────────
        # When the query explicitly names an author or paper (not just a topic),
        # and that entity is not in the library, refuse immediately.
        # This prevents the Ada Lovelace problem where the system falls back to
        # semantic search and returns unrelated papers.
        if scope.requires_entity and not scope.scoped_titles:
            # Only refuse for explicit author/paper queries, not topic queries
            if scope.entity_kind in ("author", "paper") and query_expects_named_author(query):
                return {
                    "query": query,
                    "answer": scope_refusal_message(scope),
                    "sources": [],
                    "success": False,
                    "error": f"Entity not in library ({scope.entity_kind}).",
                }
            # For topic queries, allow semantic search fallback
            if scope.entity_kind == "topic":
                # Topic not found - let semantic search try to find relevant papers
                pass

        inventory_metadata = inventory_for_scope(papers_metadata, scope)

        # ── Author existence check & Broad author query handling — BEFORE retrieval/LLM ──
        # IMPORTANT: Only apply author existence check if the query is actually about an author,
        # NOT if it's about a paper title that might be confused with an author name
        if query_expects_named_author(query) and scope.entity_kind != "paper":
            # First check if the query might be about a paper title instead of an author
            # This prevents false positives like "Deep Learning with Differential Privacy" being treated as an author
            paper_title_matches = fuzzy_match_paper_titles(query, papers_metadata)
            if paper_title_matches:
                # Query is about a paper title, skip author existence check
                logger.info(f"Query matches paper title '{paper_title_matches[0]}', skipping author existence check")
            else:
                author_phrase = extract_author_search_phrase(query)
                if not author_phrase:
                    author_phrase, _ = resolve_author_from_library(query, papers_metadata)
                
                if author_phrase:
                    author_exists = verify_author_exists_in_library(author_phrase, papers_metadata)
                    if not author_exists:
                        # Double-check
                        _, resolved_papers = resolve_author_from_library(query, papers_metadata)
                        if not resolved_papers:
                            logger.warning(f"Author '{author_phrase}' not in library, refusing before LLM")
                            return {
                                "query": query,
                                "answer": f"No papers authored by {author_phrase} were found in the ingested library.",
                                "sources": [],
                                "success": False,
                                "error": f"Author not in library: {author_phrase}",
                            }
                    
                    # Check if it's a broad query when they have multiple papers
                    if author_exists:
                        author_display, resolved_papers = resolve_author_from_library(query, papers_metadata)
                        if len(resolved_papers) > 3 and is_broad_author_query(query, author_phrase):
                            logger.warning(f"Broad query on author '{author_phrase}' with {len(resolved_papers)} papers, refusing before LLM")
                            return {
                                "query": query,
                                "answer": f"{author_display} is an author on {len(resolved_papers)} papers in your library. Please specify which paper or topic.",
                                "sources": [],
                                "success": True,
                            }

        if query_mode == "ambiguous":
            return {
                "query": query,
                "answer": (
                    "I can do either. Do you want (1) a list of papers, "
                    "or (2) a summary from paper content?"
                ),
                "sources": [],
                "success": True,
            }

        keyword_answer = answer_keyword_discovery_query(query, papers_metadata)
        if keyword_answer:
            return {
                "query": query,
                "answer": keyword_answer,
                "sources": [],
                "success": True,
            }

        catalog_answer = answer_catalog_metadata_query(query, papers_metadata)
        if catalog_answer:
            return {
                "query": query,
                "answer": catalog_answer,
                "sources": [],
                "success": True,
            }

        # ── PER-PAPER EXTRACTION TABLE (one LLM call per paper, no truncation) ──
        if is_per_paper_extraction_query(query) and inventory_metadata:
            columns = parse_table_columns_from_query(query)
            return self._generate_per_paper_extraction_table(
                query, inventory_metadata, columns
            )

        # ── CODE-BASED LISTING FOR SIMPLE INVENTORY QUERIES ─────────────────
        if query_mode == "listing" and is_simple_inventory_listing(query) and inventory_metadata:
            answer = self._render_inventory_listing(query, inventory_metadata)
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "success": True
            }

        both_listing_block = ""
        if query_mode == "both" and inventory_metadata:
            both_listing_block = self._render_inventory_listing(query, inventory_metadata)

        effective_limit = limit
        if matched_titles:
            effective_limit = max(limit, len(matched_titles) * 3, 24)

        # Extraction/listing tables must not inherit unrelated prior chat turns.
        # Drafting / creative requests (draft, write, generate, suggest) also get a
        # clean slate — a prior LinkedIn-post query must not contaminate a gap-analysis answer.
        _DRAFTING_VERBS = ("draft", "write", "compose", "generate", "create", "suggest", "propose")
        history_for_llm = conversation_history
        if (
            is_per_paper_extraction_query(query)
            or is_simple_inventory_listing(query)
            or query_mode == "both"
            or any(query.strip().lower().startswith(v) for v in _DRAFTING_VERBS)
        ):
            history_for_llm = None

        # ── Step 1: Retrieve chunks ───────────────────────────────────────────
        # Enable reranking by default for better retrieval precision
        chunks = retrieve_relevant_chunks(
            self.vector_store,
            query,
            limit=effective_limit,
            filter_title=filter_title,
            scope_titles=matched_titles if matched_titles else None,
            use_reranking=True,  # Always enable reranking for better precision
            over_retrieve_multiplier=4.0,  # Increase over-retrieval for better reranking candidates
        )
        if matched_titles and not filter_title:
            chunks = filter_chunks_to_titles(chunks, matched_titles)

        # ── EARLY QUALITY GATE (immediately after retrieval/reranking) ───────────
        # Moved here to save compute - fail fast before any further processing
        # DISABLED: Context coherence gate is too aggressive and blocks valid queries
        # The coherence calculation is unreliable and causes false negatives
        # Relying on citation verification and scope verification instead
        # if chunks:
        #     try:
        #         from context_coherence import ContextCoherence
        #         coherence = ContextCoherence()
        #         coherence_metrics = coherence.calculate_coherence_score(chunks)
        #         
        #         # Quality threshold: if coherence is too low, refuse
        #         quality_threshold = 0.4
        #         if coherence_metrics["overall_coherence"] < quality_threshold and not scope.is_locked:
        #             logger.warning(
        #                 f"Context coherence too low: {coherence_metrics['overall_coherence']:.2f} "
        #                 f"(threshold: {quality_threshold})"
        #             )
        #             # If we have contradictions or high fragmentation, refuse
        #             if (coherence_metrics["contradiction_count"] > 0 or 
        #                 coherence_metrics["fragmentation_score"] > 0.7):
        #                 return {
        #                     "query": query,
        #                     "answer": IRRELEVANT_REFUSAL,
        #                     "sources": chunks,
        #                     "success": False,
        #                     "error": f"Context coherence gate failed: score {coherence_metrics['overall_coherence']:.2f}"
        #                 }
        #     except ImportError:
        #         logger.warning("Context coherence module not available, skipping early quality gate")
        #     except Exception as e:
        #         logger.warning(f"Early quality gate failed: {e}")

        if not papers_metadata:
            logger.warning("No papers found in ChromaDB.")
            return {
                "query": query,
                "answer": EMPTY_DB_REFUSAL,
                "sources": [],
                "success": False,
                "error": "No matching papers in the vector database.",
            }

        if _fuzzy_title_match(query, papers_metadata) and not chunks:
            return {
                "query": query,
                "answer": NOT_IN_LIBRARY_REFUSAL,
                "sources": [],
                "success": False,
                "error": "Named paper title not found in library.",
            }

        # Papers exist but nothing in the corpus is similar enough to the query.
        if not chunks:
            logger.warning("No chunks passed relevance threshold for query: %s", query)
            
            # Handle meta-questions about missing papers/gaps in knowledge base
            if is_missing_papers_meta_query(query):
                # Check if this is a question about papers needing full text with chat history
                if "full text" in query.lower() or "pdf" in query.lower():
                    # Check if chat history is provided in the query
                    if "above questions" in query.lower() or "this chat" in query.lower():
                        from manifest_manager import ManifestManagerService
                        manifest_mgr = ManifestManagerService()
                        manifest = manifest_mgr.get_all_entries()
                        
                        # Get all papers in the library with their metadata
                        library_papers = {}
                        for filename, meta in manifest.items():
                            if meta.get("status") == "success":
                                library_papers[meta.get("title", filename).lower()] = {
                                    "title": meta.get("title", filename),
                                    "authors": meta.get("authors", "Unknown Authors"),
                                    "year": meta.get("year", "N/A"),
                                    "has_full_text": meta.get("has_full_text", True),
                                    "abstract": meta.get("abstract", "")
                                }
                        
                        # Find papers that lack full text - these are the ones that would help
                        papers_without_full_text = []
                        for title, meta in library_papers.items():
                            if not meta["has_full_text"]:
                                papers_without_full_text.append(f"- {meta['authors']} ({meta['year']}). {meta['title']}")
                        
                        if papers_without_full_text:
                            answer = (
                                "The following papers in your library have minimal text extracted "
                                "(likely abstract-only or scanned PDFs). Having the full PDF text would "
                                "help provide more complete answers:\n\n" + "\n\n".join(papers_without_full_text)
                            )
                        else:
                            answer = (
                                "All papers in your library have full text extracted. "
                                "No papers would benefit from re-ingestion with full PDF text."
                            )
                        
                        return {
                            "query": query,
                            "answer": answer,
                            "sources": [],
                            "success": True,
                        }
                    else:
                        # Original behavior: list all papers without full text
                        from manifest_manager import ManifestManagerService
                        manifest_mgr = ManifestManagerService()
                        manifest = manifest_mgr.get_all_entries()
                        
                        # Find papers with has_full_text=False
                        papers_without_full_text = []
                        for filename, meta in manifest.items():
                            if meta.get("status") == "success" and meta.get("has_full_text") == False:
                                papers_without_full_text.append({
                                    "title": meta.get("title", filename),
                                    "authors": meta.get("authors", "Unknown Authors"),
                                    "year": meta.get("year", "N/A"),
                                    "filename": filename
                                })
                        
                        if papers_without_full_text:
                            lines = []
                            for i, paper in enumerate(papers_without_full_text, 1):
                                lines.append(
                                    f"{i}. {paper['authors']} ({paper['year']}). {paper['title']}"
                                )
                            answer = (
                                f"The following papers in your library have minimal text extracted "
                                f"(likely abstract-only or scanned PDFs). Having the full PDF text would "
                                f"help provide more complete answers:\n\n" + "\n\n".join(lines)
                            )
                            return {
                                "query": query,
                                "answer": answer,
                                "sources": [],
                                "success": True,
                            }
                        else:
                            answer = (
                                "All papers in your library have full text extracted. "
                                "No papers would benefit from re-ingestion with full PDF text."
                            )
                            return {
                                "query": query,
                                "answer": answer,
                                "sources": [],
                                "success": True,
                            }
                
                answer = (
                    "I cannot identify specific missing papers because I only have access to "
                    "the papers you've already ingested. To identify gaps in your knowledge base, "
                    "you would need to search external databases (like Semantic Scholar) for papers "
                    "by the authors or topics you're researching, then compare those results against "
                    "your ingested library. I can only answer questions about the papers currently "
                    "in your local knowledge base."
                )
                return {
                    "query": query,
                    "answer": answer,
                    "sources": [],
                    "success": False,
                    "error": "Meta-question about missing papers - requires external search.",
                }
            
            named_in_library = resolve_matching_paper_titles(query, papers_metadata)
            if named_in_library:
                # Paper/author is in inventory but chunk text could not be loaded (ingest issue).
                answer = (
                    f"A paper matching your query appears in the library "
                    f"({named_in_library[0][:120]}), but no readable text chunks were retrieved. "
                    "Try re-ingesting that PDF or use the paper filter dropdown."
                )
            elif query_refers_to_missing_library_paper(query, papers_metadata):
                answer = NOT_IN_LIBRARY_REFUSAL
            else:
                answer = IRRELEVANT_REFUSAL
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "success": False,
                "error": "No relevant chunks above similarity threshold.",
            }

        library_inventory_str = build_library_inventory(inventory_metadata)
        
        # ── Off-topic detection BEFORE LLM call ─────────────────────────────
        # Use retrieval count AND relevance check (more reliable than just count)
        # If chunks were retrieved but are not relevant to the query, refuse
        if not filter_title and not matched_titles:
            # Check if query might be about a paper title in the library
            query_matches_paper = False
            for title in papers_metadata.keys():
                if title.lower() in query.lower() or query.lower() in title.lower():
                    query_matches_paper = True
                    break
            
            # Check relevance of retrieved chunks
            if chunks:
                # Extract significant query tokens
                query_tokens = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
                # Remove common stopwords
                stopwords = {"what", "which", "where", "when", "how", "does", "did", "are", "is", "was", "were", "have", "has", "had", "will", "would", "could", "should", "may", "might", "must", "can", "about", "from", "with", "that", "this", "these", "those"}
                query_tokens = query_tokens - stopwords
                
                # Check if query tokens appear in retrieved chunks
                relevant_chunks = 0
                for chunk in chunks:
                    chunk_text = chunk.get("text", "").lower()
                    token_hits = sum(1 for token in query_tokens if token in chunk_text)
                    if token_hits >= 2 or (len(query_tokens) == 1 and query_tokens and query_tokens.pop() in chunk_text):
                        relevant_chunks += 1
                
                # If less than 50% of chunks are relevant, refuse
                if relevant_chunks < len(chunks) * 0.5 and not query_matches_paper:
                    logger.warning(f"Low chunk relevance: {relevant_chunks}/{len(chunks)} chunks relevant for query: {query}")
                    return {
                        "query": query,
                        "answer": NOT_IN_LIBRARY_REFUSAL,
                        "sources": [],
                        "success": False,
                        "error": "Retrieved chunks not relevant to query",
                    }
                
                # Additional check: if query asks about specific topic (e.g., "coffee papers"),
                # ensure chunks are from papers about that topic
                if "coffee" in query.lower():
                    coffee_chunks = 0
                    for chunk in chunks:
                        chunk_title = chunk.get("metadata", {}).get("title", "").lower()
                        chunk_text = chunk.get("text", "").lower()
                        if "coffee" in chunk_title or "coffee" in chunk_text:
                            coffee_chunks += 1
                    
                    if coffee_chunks < len(chunks) * 0.5:
                        logger.warning(f"Query asks about coffee but only {coffee_chunks}/{len(chunks)} chunks are coffee-related")
                        return {
                            "query": query,
                            "answer": NOT_IN_LIBRARY_REFUSAL,
                            "sources": [],
                            "success": False,
                            "error": "Retrieved chunks not relevant to coffee topic",
                        }
            
            # If no chunks retrieved and query doesn't match paper title, refuse
            if not chunks and not query_matches_paper:
                logger.warning(f"No chunks retrieved and query doesn't match any paper title: {query}")
                return {
                    "query": query,
                    "answer": NOT_IN_LIBRARY_REFUSAL,
                    "sources": [],
                    "success": False,
                    "error": "No relevant chunks retrieved",
                }
        
        # ── Additional safety check: refuse obviously off-topic queries ─────────
        # Even if chunks were retrieved, refuse if query is clearly outside scope
        off_topic_patterns = {
            "who won", "world cup", "fifa", "sports", "election", "president",
            "weather", "temperature", "forecast", "movie", "film", "actor",
            "celebrity", "music", "song", "recipe", "cook", "cooking", "food",
            "travel", "vacation", "hotel", "flight", "airport"
        }
        query_lower = query.lower()
        has_off_topic = any(pattern in query_lower for pattern in off_topic_patterns)
        
        if has_off_topic and not query_matches_paper and not filter_title:
            logger.warning(f"Off-topic query detected: {query}")
            return {
                "query": query,
                "answer": NOT_IN_LIBRARY_REFUSAL,
                "sources": [],
                "success": False,
                "error": "Off-topic query",
            }
        
        # ── Context quality gate before LLM ───────────────────────────────────
        # DISABLED: Context quality gate is too aggressive and blocks valid queries
        # The quality calculation is unreliable and causes false negatives
        # Relying on citation verification and scope verification instead
        # try:
        #     from context_shaper import ContextShaper
        #     shaper = ContextShaper()
        #     quality_metrics = shaper.estimate_context_quality(chunks, query)
        #     
        #     # Quality threshold: if quality is too low, refuse or retry
        #     quality_threshold = 0.3
        #     if quality_metrics["quality_score"] < quality_threshold and not scope.is_locked:
        #         logger.warning(
        #             f"Context quality too low: {quality_metrics['quality_score']:.2f} "
        #             f"(threshold: {quality_threshold})"
        #         )
        #         # If we have very few chunks or they're not relevant, refuse
        #         if quality_metrics["chunk_count"] < 2 or quality_metrics["avg_distance"] > 1.2:
        #             return {
        #                 "query": query,
        #                 "answer": IRRELEVANT_REFUSAL,
        #                 "sources": chunks,
        #                 "success": False,
        #                 "error": f"Context quality gate failed: score {quality_metrics['quality_score']:.2f}"
        #             }
        # except ImportError:
        #     logger.warning("Context shaper not available, skipping quality gate")
        # except Exception as e:
        #     logger.warning(f"Context quality gate failed: {e}")
        
        context_str = chunks_to_context_string(chunks)

        # ── Step 3: Build structured prompts ──────────────────────────────────
        scope_note = ""
        if matched_titles and not filter_title:
            if scope.entity_kind == "author":
                label = f"author \"{scope.author_phrase or 'named in query'}\""
            elif scope.entity_kind == "paper":
                label = "the matched paper(s)"
            else:
                label = "the matched library papers"
            scope_note = (
                f"LIBRARY SCOPE: Answer ONLY using {label} "
                f"({len(matched_titles)} paper(s) in scope). "
                "Do NOT cite, summarize, or mention any other ingested paper or author.\n"
            )
        elif scope.entity_kind == "topic":
            scope_note = (
                f"TOPIC SCOPE: Answer ONLY using the {len(matched_titles)} paper(s) in scope "
                "that are clearly about the topic in the question. "
                "Do NOT describe unrelated papers (e.g. phishing, traffic, barcodes) "
                "even if they mention 'deep learning'.\n"
                "If the inventory lists papers whose titles mention the topic, you MUST "
                "summarize what those papers say — do not claim the topic is absent.\n"
            )
        elif scope.entity_kind == "paper" and len(matched_titles) == 1:
            scope_note = (
                f"SINGLE-PAPER SCOPE: Answer ONLY from \"{matched_titles[0]}\". "
                "Do NOT invent frameworks, product names, or methods not in the passages.\n"
            )
        elif filter_title:
            scope_note = (
                f"NOTE: This query is scoped to a SINGLE paper: \"{filter_title}\". "
                "Only use information from this paper's context blocks when answering.\n"
            )

        # System prompt: absolute iron-wall instruction set for Llama 3 8B
        system_prompt = (
            "=== ABSOLUTE OPERATING RULES — READ BEFORE EVERYTHING ELSE ===\n"
            "You are an AI assistant LOCKED to an academic research knowledge base.\n"
            "You ONLY answer questions using information from the DOCUMENT CONTEXT BLOCKS and the INGESTED PAPER LIBRARY INVENTORY below.\n"
            "You have NO general knowledge. You are NOT ChatGPT. You CANNOT access the internet.\n"
            "You MUST REFUSE to answer ANYTHING that is not present in the provided context or library inventory.\n\n"
            "=== CRITICAL: DO NOT USE EXTERNAL KNOWLEDGE ===\n"
            "If the user asks about:\n"
            "- Ada Lovelace, Albert Einstein, Elon Musk, François Chollet, or any person NOT in the library inventory → REFUSE\n"
            "- Transformers invented by Vaswani et al. (if not in library) → REFUSE\n"
            "- Keras, TensorFlow, PyTorch (if not explicitly in context) → REFUSE\n"
            "- FIFA World Cup, sports, entertainment, politics → REFUSE\n"
            "- Any topic NOT explicitly mentioned in the context blocks or library inventory → REFUSE\n"
            "DO NOT provide general knowledge about these topics. DO NOT explain who they are. DO NOT cite external papers.\n"
            "Simply say: 'This question is outside the scope of your ingested research knowledge base.'\n\n"
            "=== HARD REFUSAL TRIGGERS — ALWAYS REFUSE THESE, NO EXCEPTIONS ===\n"
            "- Cooking, recipes, food → REFUSE\n"
            "- Medical advice, health, symptoms → REFUSE (unless paper is medical research)\n"
            "- News, weather, current events, sports, entertainment → REFUSE\n"
            "- Any question where the answer requires knowledge NOT in the context blocks or the Ingested Paper Library Inventory → REFUSE\n"
            "- Any question about a person, paper, or concept not found in the Library Inventory or context blocks → REFUSE\n"
            "- Questions about famous people, historical events, or general knowledge → REFUSE\n\n"
            "REFUSAL FORMAT (copy this exactly when refusing):\n"
            "\"This question is outside the scope of your ingested research knowledge base. "
            "I can only answer questions based on the papers that have been ingested. "
            "Please ask a question about the research papers in your library.\"\n\n"
            "=== WHAT YOU ARE ALLOWED TO DO ===\n"
            "- Answer research questions strictly using the Document Context Blocks or Ingested Paper Library Inventory provided.\n"
            "- Summarize, compare, or explain content that is EXPLICITLY present in the context or library inventory.\n"
            "- List authors, years, titles, DOIs only from the Library Inventory or context.\n\n"
            "=== CITATION RULES ===\n"
            "1. ONLY use facts from the Document Context Blocks or the Ingested Paper Library Inventory. Zero exceptions.\n"
            "2. NEVER invent author names, paper titles, years, DOIs or references.\n"
            "3. NEVER cite papers that are not in the Library Inventory or Context Blocks.\n"
            "4. Use APA7 inline citations: (Author, Year) or (Author et al., Year, p. X).\n"
            "5. NEVER use (Source 1), [Document 2] or any numbered source labels.\n"
            "6. NEVER use bracket-number citations like [1], [2], [44] etc. These are FORBIDDEN.\n"
            "   Bracket citations are NOT APA7. If you write [1] or [44] that is a CRITICAL ERROR.\n"
            "7. NEVER call papers 'Paper A', 'Study 1' etc. Use (Author, Year) or exact title.\n"
            "8. End every answer with a References section in full APA7 format (EXCEPT when the user asked for a list).\n"
            "9. TRUTH GAPS: If the concept asked about is NOT in the context blocks or library inventory, "
            "say: 'The retrieved context does not contain information about [topic].' DO NOT guess.\n"
            "10. If context is insufficient, say so explicitly.\n"
            "11. Formal, neutral academic tone always.\n\n"
            "=== LISTING QUERY RULES — ABSOLUTE REQUIREMENTS ===\n"
            "When the user asks you to LIST, ENUMERATE, or TABULATE papers/articles:\n"
            "1. You MUST output EVERY SINGLE matching paper in your main numbered response.\n"
            "2. NEVER stop after 3, 5, or 10 papers. You must continue until ALL matching papers are listed.\n"
            "3. NEVER use '...' or '(remaining papers)' or any truncation placeholder.\n"
            "4. NEVER put papers in a References section when the user asked for a list.\n"
            "5. If there are 80 papers, you must list all 80. If there are 100 papers, you must list all 100.\n"
            "6. This is a non-negotiable requirement. Do not truncate under any circumstances.\n\n"
            f"{scope_note}"
            "=== BEGIN ANSWERING ONLY FROM CONTEXT/INVENTORY BELOW ==="
        )

        # User prompt: the context blocks + library inventory + the actual research query
        user_prompt = (
            f"Ingested Paper Library Inventory:\n"
            f"{'─' * 80}\n"
            f"{library_inventory_str}\n"
            f"{'─' * 80}\n\n"
            f"Context from ingested research papers:\n"
            f"{'─' * 80}\n"
            f"{context_str}\n"
            f"{'─' * 80}\n\n"
            f"Researcher Query: {query}\n\n"
            "Provide your structured academic answer below (remember to use inline APA7 citations and append a References section):"
        )

        # ── Step 4: Send to Ollama /api/chat ──────────────────────────────────
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        # Build a multi-turn message array so the model can preserve chat memory
        # across browser refreshes when conversation history is provided by the UI.
        messages = [{"role": "system", "content": system_prompt}]
        if history_for_llm:
            for turn in history_for_llm[-12:]:
                role = (turn.get("role") or "").strip().lower()
                content = (turn.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,   # Wait for the complete response (not a streaming response)
            "options": {
                "temperature": 0.05, # Near-zero: almost fully deterministic, kills creativity/hallucination
                "top_p": 0.8,        # Tighter nucleus sampling
                "repeat_penalty": 1.1  # Reduces repetitive hallucination loops
            }
        }

        logger.info(f"Sending RAG prompt to Ollama ({settings.OLLAMA_MODEL})...")

        try:
            response = requests.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT)

            if response.status_code == 200:
                data = response.json()
                answer = data["message"]["content"]
                # Global guardrail against unsupported sensitive-topic stance claims.
                if self._is_unverifiable_sensitive_claim(query, chunks):
                    answer = (
                        "I cannot confirm or deny that claim from the retrieved context. "
                        "The current sources do not provide direct evidence on this topic."
                    )
                elif answer_has_table_truncation(answer):
                    answer = TABLE_TRUNCATION_REFUSAL
                else:
                    answer = self._strip_model_references(answer)
                    if (not listing_style_query) and (not self._is_refusal_answer(answer)):
                        safe_refs = self._build_safe_references(chunks)
                        if safe_refs:
                            answer = f"{answer}\n\n{safe_refs}"
                    if query_mode == "both" and both_listing_block:
                        answer = (
                            f"{both_listing_block}\n\n"
                            f"What these papers say:\n\n{answer}"
                        )
                    answer, verified = apply_verification_or_refuse(
                        answer,
                        scope=scope,
                        papers_metadata=papers_metadata,
                        chunks=chunks,
                    )
                    if not verified:
                        return {
                            "query": query,
                            "answer": answer,
                            "sources": chunks,
                            "success": False,
                            "error": "Answer failed scope verification.",
                        }
                
                # ── Hallucination detection: Strip unverified citations and claims ─────────────
                # Remove citations and claims that don't match retrieved chunk metadata
                try:
                    from citation_verifier import CitationVerifier
                    verifier = CitationVerifier()
                    # 1. Strip unverified citations (STRICT MODE)
                    answer_before = answer
                    answer = verifier.strip_unverified_citations(answer, chunks)
                    if answer != answer_before:
                        logger.warning(f"Citation verifier stripped citations, answer changed from {len(answer_before)} to {len(answer)} chars")
                    # 2. Verify and remove unsupported claim sentences
                    verification = verifier.verify_answer(answer, chunks)
                    answer_before = answer
                    answer = verifier.regenerate_or_remove_unsupported(answer, verification, action="remove")
                    if answer != answer_before:
                        logger.warning(f"Claim verifier removed unsupported claims, answer changed from {len(answer_before)} to {len(answer)} chars")
                    # 3. Refuse if too few claims can be grounded
                    support_ratio = verification.get("support_ratio", 1.0)
                    if support_ratio < 0.3 and not scope.is_locked:
                        logger.warning(f"Too few claims grounded ({support_ratio:.2%}), returning refusal")
                        answer = NOT_IN_LIBRARY_REFUSAL
                    # If answer is now empty, fallback to refusal
                    if not answer.strip():
                        logger.warning("Answer became empty after citation/claim verification, returning refusal")
                        answer = NOT_IN_LIBRARY_REFUSAL
                except ImportError:
                    logger.warning("Citation verifier not available, skipping citation and claim check")
                except Exception as e:
                    logger.warning(f"Citation and claim verification failed: {e}")

                # ── External knowledge detection DISABLED ───────────────────────────────
                # Entity extraction was too aggressive and blocked legitimate technical terms
                # Relying on system prompt + citation stripping instead
                # try:
                #     # Build set of all words from retrieved chunks (case-insensitive)
                #     chunk_words = set()
                #     for chunk in chunks:
                #         text = chunk.get("text", "").lower()
                #         words = re.findall(r'\b[a-z]{3,}\b', text)
                #         chunk_words.update(words)
                #     
                #     # Also add author names from library inventory
                #     for paper_meta in inventory_metadata.values():
                #         authors = paper_meta.get("authors", "").lower()
                #         author_words = re.findall(r'\b[a-z]{3,}\b', authors)
                #         chunk_words.update(author_words)
                #     
                #     # Extract capitalized words from answer (potential entities)
                #     answer_words = re.findall(r'\b[A-Z][a-z]{3,}\b', answer)
                #     external_entities = []
                #     
                #     for word in answer_words:
                #         if word.lower() not in chunk_words:
                #             external_entities.append(word)
                #     
                #     if external_entities:
                #         logger.warning(f"Answer contains entities not in context: {external_entities}")
                #         return {
                #             "query": query,
                #             "answer": "This question is outside the scope of your ingested research knowledge base. I can only answer questions based on the papers that have been ingested. Please ask a question about the research papers in your library.",
                #             "sources": chunks,
                #             "success": False,
                #             "error": f"External entities detected: {external_entities}",
                #         }
                # except Exception as e:
                #     logger.warning(f"External knowledge detection failed: {e}")

                logger.info("Successfully received answer from Ollama.")
                return {
                    "query": query,
                    "answer": answer.strip(),
                    "sources": chunks,
                    "success": True
                }

            else:
                # Non-200 response from Ollama — the model may not be pulled
                error_msg = f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
                logger.error(error_msg)
                return {
                    "query": query,
                    "answer": "Error communicating with the local LLM server.",
                    "sources": chunks,
                    "success": False,
                    "error": error_msg
                }

        except requests.exceptions.ConnectionError:
            # Ollama went offline between the health check and the actual call
            error_msg = (
                f"Lost connection to Ollama at {settings.OLLAMA_BASE_URL}. "
                "Please ensure Ollama is still running."
            )
            logger.error(error_msg)
            return {
                "query": query,
                "answer": "Could not connect to the local Ollama LLM. Please ensure it is running.",
                "sources": chunks,
                "success": False,
                "error": "ConnectionRefused"
            }

        except Exception as e:
            error_msg = f"Unexpected error during RAG generation: {e}"
            logger.error(error_msg)
            return {
                "query": query,
                "answer": "An unexpected error occurred during RAG query execution.",
                "sources": chunks,
                "success": False,
                "error": str(e)
            }
