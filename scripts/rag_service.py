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
    fuzzy_match_paper_titles,
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

    def _sanitize_metadata_field(self, field_name: str, value: str, papers_metadata: dict | None) -> str:
        if not value or value.strip().upper() == "N/A":
            return ""
        val = value.strip()
        
        if not papers_metadata:
            return val
            
        # 1. Clean up pages: should only contain digits, commas, hyphens, spaces, or "p"/"pp"
        if field_name == "pages":
            if not re.match(r"^[0-9\s,\-p\.]+$", val, re.IGNORECASE):
                # Try to search for page numbers like "1-12" or "pp. 6-7"
                m = re.search(r"\b(?:pp\.?|p\.?)\s*(\d+(?:\s*-\s*\d+)?)\b", val, re.I)
                if m:
                    return m.group(1)
                return ""
                
        # 2. Clean up venue/authors: should not contain titles of other papers
        if field_name in ("venue", "authors"):
            val_lower = val.lower()
            # If venue/authors is long and matches another paper's title or contains parts of it, clean it
            for other_title in papers_metadata.keys():
                if len(other_title) > 20 and other_title.lower() in val_lower:
                    idx = val_lower.find(other_title.lower())
                    val = val[:idx].strip()
                    val_lower = val.lower()
            # Remove trailing dots, commas, spaces
            val = val.rstrip(".,; ")
            # If a venue is extremely long (>120 chars), it is likely stitched/corrupted metadata
            if field_name == "venue" and len(val) > 120:
                return ""
                
        return val

    def _build_safe_references(self, chunks: list[dict], papers_metadata: dict | None = None) -> str:
        """
        Build a deterministic References section from retrieved chunk metadata.

        Guards applied (in order):
          1. Title must exist in the ingested library catalog (papers_metadata). Any title
             not found in the catalog is treated as a hallucination and dropped.
          2. Venue is capped at 80 characters (first clause only).
          3. Authors field must not contain other paper titles.
          4. Each unique *title* appears only once regardless of how many chunks came from it.
          5. Pages are omitted entirely — they are unreliably stored in chunk metadata and
             produce formatting artifacts like trailing "0.".
          6. If the assembled reference string exceeds 350 characters the entire entry is
             dropped — it is almost certainly corrupted / stitched metadata.
        """
        refs: list[str] = []
        seen_titles: set[str] = set()          # deduplicate by title alone

        # Build a fast lower-case lookup of the library catalog titles for Guard 1
        catalog_lower: dict[str, str] = {}
        if papers_metadata:
            for t in papers_metadata:
                catalog_lower[t.lower()] = t

        for chunk in chunks:
            meta = chunk.get("metadata", {}) or {}
            title  = (meta.get("title")   or "").strip()
            authors= (meta.get("authors") or "").strip()
            year   = str(meta.get("year") or "N/A").strip()
            doi    = (meta.get("doi")     or "N/A").strip()
            venue  = (meta.get("venue")   or "").strip()

            # ── Guard 1: skip obviously incomplete metadata ────────────────
            if not title or title == "Untitled":
                continue
            if not authors or authors in ("Unknown Authors", "N/A"):
                continue
            if year == "N/A":
                continue

            # ── Guard 2: LIBRARY CATALOG CHECK — block hallucinated papers ─
            # Every reference MUST correspond to a paper that was actually ingested.
            # If the title is not in papers_metadata, it is a hallucination and must
            # be silently dropped before it reaches the user.
            if catalog_lower:  # only enforce when catalog is available
                title_lower = title.lower()
                if title_lower not in catalog_lower:
                    # Allow partial match for very long titles that get truncated
                    partial_match = any(
                        len(ct) > 30 and (title_lower in ct or ct in title_lower)
                        for ct in catalog_lower
                    )
                    if not partial_match:
                        logger.warning(
                            f"Dropping hallucinated reference (not in library catalog): {title[:80]}"
                        )
                        continue

            # ── Guard 3: deduplicate by title ─────────────────────────────
            title_key = title.lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            # ── Guard 4: sanitize venue — 80 chars, no embedded titles ────
            venue_clean = ""
            if venue:
                # Take only the first clause (split on "." or ":") and cap length
                first_clause = re.split(r"[.:]\s+", venue)[0].strip()
                if len(first_clause) <= 80:
                    venue_clean = first_clause
                elif len(venue) <= 80:
                    venue_clean = venue
                # If papers_metadata provided, check for title contamination
                if papers_metadata and venue_clean:
                    v_lower = venue_clean.lower()
                    for other_title in papers_metadata:
                        if len(other_title) > 20 and other_title.lower() in v_lower:
                            venue_clean = ""
                            break

            # ── Guard 5: sanitize authors — no embedded titles ────────────
            authors_clean = authors
            if papers_metadata:
                a_lower = authors_clean.lower()
                for other_title in papers_metadata:
                    if len(other_title) > 20 and other_title.lower() in a_lower:
                        idx = a_lower.find(other_title.lower())
                        authors_clean = authors_clean[:idx].strip(", ")
                        a_lower = authors_clean.lower()

            # ── Assemble reference (pages deliberately omitted — unreliable) ──
            ref_parts = [f"- {authors_clean} ({year}). {title}"]
            if venue_clean:
                ref_parts.append(venue_clean)
            ref = ". ".join(ref_parts) + "."

            if doi and doi not in ("N/A", ""):
                clean_doi = doi.replace("https://doi.org/", "").replace("doi.org/", "").strip()
                ref += f" https://doi.org/{clean_doi}"

            # ── Guard 6: total length cap — corrupted if > 350 chars ──────
            if len(ref) > 350:
                logger.warning(f"Skipping corrupted reference (len={len(ref)}): {ref[:80]}...")
                continue

            refs.append(ref)

        if not refs:
            return ""
        return "References:\n" + "\n".join(refs)

    def _strip_body_reference_fragments(self, text: str, chunks: list[dict] | None = None) -> str:
        """
        Remove orphaned reference-list fragments the LLM injects into the answer body.

        Uses a UNIVERSAL title-word matching approach:
        1. Build a vocabulary of significant words from ALL known paper titles.
        2. For each line in the body (after the real References section is gone),
           compute what fraction of that line's words come from known paper titles.
        3. Lines that are predominantly title vocabulary (>= 50%) with no verb are
           bibliography fragments and are stripped.
        4. Lines that are bare DOI/URL are always stripped.

        This works for any domain (medical, coffee, AI safety, etc.) without
        needing domain-specific regex patterns.
        """
        if not text:
            return text

        # Build title-word vocabulary from retrieved chunks (if provided)
        title_words: set[str] = set()
        _STOP = {
            "the", "this", "that", "with", "and", "for", "from", "into",
            "which", "their", "also", "based", "using", "used", "study",
            "paper", "approach", "review", "survey", "deep", "learning",
            "machine", "analysis", "method", "model", "data", "image",
        }
        if chunks:
            for c in chunks:
                t = (c.get("metadata", {}).get("title") or "").lower()
                for w in re.findall(r"\b[a-z]{4,}\b", t):
                    if w not in _STOP:
                        title_words.add(w)

        lines = text.split("\n")
        kept: list[str] = []
        for line in lines:
            stripped_line = line.strip()

            # Always strip bare URL/DOI lines
            if re.match(r'^https?://\S+$', stripped_line):
                continue

            # Always strip lines that contain a DOI and nothing else meaningful
            if 'doi.org/' in stripped_line and len(stripped_line) < 120:
                word_count = len(re.findall(r'\b[a-z]{4,}\b', stripped_line.lower()))
                if word_count < 6:  # mostly DOI, not a real sentence
                    continue

            # Skip short lines and lines that look like normal sentences (have verbs)
            if len(stripped_line) < 20:
                kept.append(line)
                continue

            # Check if this line is predominantly title vocabulary
            if title_words:
                line_words = re.findall(r'\b[a-z]{4,}\b', stripped_line.lower())
                if line_words:
                    title_hits = sum(1 for w in line_words if w in title_words)
                    ratio = title_hits / len(line_words)
                    # Heuristic: if >=50% of words are from known titles AND
                    # the line has no sentence-ending period mid-line after a space gap,
                    # it's likely a bibliography fragment
                    has_year = bool(re.search(r'\(\d{4}\)', stripped_line))
                    has_doi  = 'doi' in stripped_line.lower() or 'https://' in stripped_line
                    if ratio >= 0.50 and (has_year or has_doi):
                        continue  # strip this line

            kept.append(line)

        return "\n".join(kept).strip()

    def _strip_model_references(self, answer: str, chunks: list[dict] | None = None) -> str:
        """
        Remove any model-generated References section and orphaned reference
        fragments so we can append a clean verified References section.
        """
        if not answer:
            return ""
        # Step 1: strip everything after a "References:" heading
        stripped = re.split(
            r"\n\s*(?:#+\s*|\*+\s*|_+)?references\b[:\s]*",
            answer, maxsplit=1, flags=re.IGNORECASE
        )[0]
        # Step 2: strip orphaned reference fragments using title-word matching
        stripped = self._strip_body_reference_fragments(stripped, chunks=chunks)
        return stripped.strip()

    def _is_refusal_answer(self, answer: str) -> bool:
        """
        True ONLY when the final answer is an explicit system-level refusal.
        Must NOT trigger on partial-evidence answers that happen to contain
        phrases like 'no papers discuss X' or 'does not contain details about Y'
        (those are valid grounded statements, not refusals).
        """
        a = (answer or "").strip().lower()
        if not a:
            return False

        # Hard refusal markers: exact constant strings generated by this system
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

        # Narrow semantic triggers: only fire when the ENTIRE answer is a refusal,
        # not when a single sentence inside a valid answer uses these words.
        # Require the phrase to appear near the START of the answer (first 200 chars).
        a_start = a[:200]
        strict_start_markers = (
            "i cannot provide",
            "i am unable to answer",
            "this is outside the scope",
            "outside the scope of your",
            "i cannot answer this",
            "no valid answer",
        )
        if any(phrase in a_start for phrase in strict_start_markers):
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

    def _has_keyword_match_in_chunks(self, query: str, chunks: list[dict]) -> bool:
        if not chunks:
            return False
        ignore_words = {"paper", "papers", "summarize", "summary", "list", "show", "describe", "all", "related"}
        words = re.findall(r"\b[a-z]{3,}\b", (query or "").lower())
        keywords = [w for w in words if w not in ignore_words]
        if not keywords:
            return False
            
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            meta = chunk.get("metadata", {}) or {}
            title = meta.get("title", "").lower()
            authors = meta.get("authors", "").lower()
            for kw in keywords:
                if kw in text or kw in title or kw in authors:
                    return True
        return False

    def _build_retrieved_set_summary(self, chunks: list[dict]) -> str:
        from collections import Counter
        summary_lines = []
        seen = set()
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {}) or {}
            title = meta.get("title", "Untitled")
            authors = meta.get("authors", "Unknown Authors")
            year = meta.get("year", "N/A")
            
            # Simple keyword extraction
            text = chunk.get("text", "")
            words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            stop_words = {"this", "that", "with", "from", "were", "they", "have", "been", "using", "based", "paper", "study", "results", "analysis", "their", "about", "there"}
            keywords = [w for w, c in Counter(words).most_common(5) if w not in stop_words]
            kw_str = ", ".join(keywords)
            
            key = (title.lower(), authors.lower(), year)
            if key not in seen:
                seen.add(key)
                # Map rerank score to retrieval confidence weighting
                rerank_score = chunk.get("rerank_score", 0.5)
                if rerank_score >= 0.60:
                    confidence = "High"
                elif rerank_score >= 0.45:
                    confidence = "Medium"
                else:
                    confidence = "Low/Weak"

                summary_lines.append(
                    f"- Document ID: doc_{idx}\n"
                    f"  Title: {title}\n"
                    f"  Authors: {authors} ({year})\n"
                    f"  Retrieval Confidence Weighting: [{confidence}] (Score: {rerank_score:.2f})\n"
                    f"  Keywords in chunk: {kw_str}"
                )
        return "\n".join(summary_lines)

    # ─────────────────────────────────────────────────────────────────────────────
    # ANSWER DECISION GATE
    # Single controller that replaces N independent threshold checks.
    # Returns one of three exclusive modes:
    #   'confident'  → ≥2 chunks with rerank_score ≥ 0.45
    #   'partial'    → at least 1 chunk with score 0.20–0.45 (OR ≥2 unscored chunks)
    #   'refuse'     → 0 chunks with score ≥ 0.20 (no usable evidence)
    # Modes NEVER overlap; transitions are deterministic.
    # ─────────────────────────────────────────────────────────────────────────────
    _CONFIDENT_THRESHOLD = 0.45
    _PARTIAL_THRESHOLD   = 0.20

    def _compute_answer_decision(
        self,
        chunks: list[dict],
        query: str,
    ) -> tuple[str, str]:
        """
        Determine answer mode from rerank scores.

        Returns
        -------
        (mode, partial_notice)
          mode            : 'confident' | 'partial' | 'refuse'
          partial_notice  : extra instruction to inject into the user prompt
                            when mode == 'partial'; empty string otherwise.
        """
        if not chunks:
            return "refuse", ""

        scored = sorted(
            [c for c in chunks if "rerank_score" in c],
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        if not scored:
            # Fall back on chunk count heuristic if no scores are present
            if len(chunks) >= 3:
                return "confident", ""
            return "partial", (
                "NOTE: Limited evidence retrieved. Describe only what the passages "
                "explicitly state. Do not extrapolate."
            )

        # Compute normalized query-level confidence metrics
        top_n = min(3, len(scored))
        avg_top_3 = sum(c["rerank_score"] for c in scored[:top_n]) / top_n
        max_score = scored[0]["rerank_score"]

        logger.info(f"Query confidence metrics: avg_top_3={avg_top_3:.3f}, max_score={max_score:.3f}")
        # Routing based on normalized query-level confidence
        if avg_top_3 >= 0.52 and max_score >= 0.58:
            return "confident", ""

        # Default to partial evidence mode if scores are lower but chunks are retrieved
        return "partial", (
            "PARTIAL EVIDENCE MODE: The retrieved documents are only partially or "
            "weakly related to this query. You MUST:\n"
            "1. State explicitly: 'The library contains limited or partially related "
            "evidence on this topic.'\n"
            "2. Summarize ONLY what the context passages explicitly say. Do not fill "
            "gaps with general knowledge.\n"
            "3. Never deny the existence of papers that appear in the Retrieved "
            "Document Set above.\n"
        )

    def _parse_constrained_claims(self, text: str) -> list[dict]:
        blocks = []
        # Pattern to match CLAIM:, SOURCE:, and QUOTE: blocks
        # CLAIM: <text>\nSOURCE: doc_<num>\nQUOTE: <text>
        pattern = r"CLAIM:\s*(.*?)\nSOURCE:\s*(doc_\d+)\nQUOTE:\s*(.*?)(?=\nCLAIM:|\Z)"
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for m in matches:
            claim = m.group(1).strip()
            source = m.group(2).strip().lower()
            quote = m.group(3).strip()
            blocks.append({
                "claim": claim,
                "source": source,
                "quote": quote
            })
        return blocks

    def _validate_claims(self, claims: list[dict], chunks: list[dict]) -> list[dict]:
        valid_claims = []
        for c in claims:
            source_id = c["source"]
            try:
                # doc_1 maps to chunks[0], doc_2 to chunks[1]...
                idx = int(source_id.split("_")[1]) - 1
                if idx < 0 or idx >= len(chunks):
                    logger.warning(f"Invalid source ID in claim: {source_id}")
                    continue
                chunk = chunks[idx]
                chunk_text = chunk.get("text", "")
                
                # Check quote match
                quote = c["quote"]
                # Normalize spaces and lower case for comparison
                norm_quote = re.sub(r"\s+", " ", quote.lower().strip())
                norm_chunk = re.sub(r"\s+", " ", chunk_text.lower().strip())
                
                # Check cleaned alphanumeric content to handle punctuation / quote differences
                clean_quote = re.sub(r"[^\w\s]", "", norm_quote)
                clean_chunk = re.sub(r"[^\w\s]", "", norm_chunk)
                
                if clean_quote in clean_chunk or norm_quote in norm_chunk:
                    valid_claims.append({
                        "claim": c["claim"],
                        "source": source_id,
                        "chunk_idx": idx,
                        "chunk": chunk
                    })
                else:
                    logger.warning(f"Verbatim quote validation failed for claim: {c['claim'][:50]}... Quote: {quote[:50]}...")
            except (IndexError, ValueError) as e:
                logger.warning(f"Failed to parse source ID {source_id}: {e}")
        return valid_claims

    def _synthesize_prose(self, valid_claims: list[dict], query: str, chunks: list[dict], papers_metadata: dict, listing_style_query: bool) -> str:
        # Group claims by paper
        paper_claims = {}
        for c in valid_claims:
            chunk = c["chunk"]
            meta = chunk.get("metadata", {}) or {}
            title = meta.get("title", "Untitled Paper")
            authors = meta.get("authors", "Unknown Authors")
            year = meta.get("year", "N/A")
            domain = meta.get("domain", "General Research")
            
            key = (title, authors, year, domain)
            if key not in paper_claims:
                paper_claims[key] = []
            paper_claims[key].append(c)
            
        if not paper_claims:
            return ""

        if listing_style_query:
            lines = []
            seen_papers = set()
            for key, claims in paper_claims.items():
                title, authors, year, domain = key
                paper_str = f"{authors} ({year}). {title}"
                if paper_str not in seen_papers:
                    seen_papers.add(paper_str)
                    lines.append(f"{len(lines)+1}. {paper_str}")
            return "\n".join(lines)

        # Check if it is an aggregation query
        _AGG_PATTERNS = (
            "what do all", "all papers say", "across all", "summarize all",
            "compare all", "conclusion of all", "what do these papers",
            "all studies say", "combined conclusion", "combined summary",
            "overall conclusion", "aggregate", "synthesis of all",
        )
        is_aggregation = any(p in query.lower() for p in _AGG_PATTERNS)

        sections = []
        
        if is_aggregation:
            # Group by domain
            domain_groups = {}
            for key, claims in paper_claims.items():
                title, authors, year, domain = key
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append((key, claims))
                
            for domain, paper_list in domain_groups.items():
                sections.append(f"### {domain.upper()} DOMAIN")
                for key, claims in paper_list:
                    title, authors, year, _ = key
                    # Get the confidence level of the first chunk from this paper
                    chunk = claims[0]["chunk"]
                    rerank_score = chunk.get("rerank_score", 0.5)
                    confidence = "High" if rerank_score >= 0.60 else ("Medium" if rerank_score >= 0.45 else "Low/Weak")
                    
                    sentences = []
                    for c in claims:
                        sent = c["claim"]
                        # Make sure sentence ends with the correct parenthetical citation
                        if not sent.endswith(f"({c['source']})"):
                            if sent.endswith("."):
                                sent = sent[:-1].strip()
                            sent = f"{sent} ({c['source']})."
                        sentences.append(sent)
                    
                    paper_prose = " ".join(sentences)
                    sections.append(
                        f"In the study \"{title}\" by {authors} ({year}) [Retrieval Confidence: {confidence}]:\n"
                        f"{paper_prose}"
                    )
                sections.append("")
        else:
            # Standard single-paper or multi-paper factual synthesis
            for key, claims in paper_claims.items():
                title, authors, year, domain = key
                chunk = claims[0]["chunk"]
                rerank_score = chunk.get("rerank_score", 0.5)
                confidence = "High" if rerank_score >= 0.60 else ("Medium" if rerank_score >= 0.45 else "Low/Weak")
                
                sentences = []
                for c in claims:
                    sent = c["claim"]
                    if not sent.endswith(f"({c['source']})"):
                        if sent.endswith("."):
                            sent = sent[:-1].strip()
                        sent = f"{sent} ({c['source']})."
                    sentences.append(sent)
                
                paper_prose = " ".join(sentences)
                sections.append(
                    f"According to \"{title}\" by {authors} ({year}) [Retrieval Confidence: {confidence}]:\n"
                    f"{paper_prose}"
                )
                
        return "\n\n".join(sections).strip()

    def _build_context_with_ids(self, chunks: list[dict]) -> str:
        blocks = []
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {}) or {}
            # Only expose clean, safe fields to the LLM.
            # pages/venue/doi are often corrupted (stitched across papers) and
            # cause the LLM to reproduce garbage in its answer body.
            title   = (meta.get("title")   or "Untitled").strip()
            authors = (meta.get("authors") or "Unknown Authors").strip()
            year    = str(meta.get("year") or "N/A").strip()
            section = (meta.get("section") or "").strip()
            text    = chunk.get("text", "")

            header = (
                f"=== DOCUMENT ID: doc_{idx} ===\n"
                f"Title: {title}\n"
                f"Authors: {authors} ({year})"
            )
            if section:
                header += f"\nSection: {section}"
            header += f"\nPassage:\n{text}\n"
            blocks.append(header)
        return "\n\n".join(blocks)

    # Specific model/architecture names that must appear in the cited chunk
    # if the sentence claims to be about that architecture.
    _ARCHITECTURE_NAMES: frozenset = frozenset({
        "rnn", "rnns", "lstm", "lstms", "gru", "grus",
        "autoencoder", "autoencoders",
        "transformer", "transformers",
        "xception", "resnet", "vgg", "inception", "densenet",
        "mobilenet", "efficientnet", "alexnet",
        "unet", "u-net",
        "bert", "gpt",
        "yolo",
    })

    def _verify_claim_chunk_support(self, sentence: str, doc_num: int, chunks: list[dict]) -> bool:
        """
        Span-level evidence check: verify that the chunk cited by doc_<doc_num> actually
        contains keyword evidence for the claim made in `sentence`.

        Rules (in order):
        1. Architecture-name check: if the sentence names a specific model/architecture
           (e.g. RNN, LSTM, GAN, Xception), that name MUST appear in the cited chunk.
           This prevents e.g. "RNNs used in security" being attributed to an agriculture paper.
        2. Token-overlap check:
           - Long sentences (5+ significant tokens): require >= 2 matching tokens.
           - Short sentences (< 5 tokens): require >= 1 matching token.
           - Title-word match from the cited paper also counts as a pass.
        """
        if doc_num < 1 or doc_num > len(chunks):
            return False
        chunk = chunks[doc_num - 1]
        chunk_text = (chunk.get("text") or "").lower()
        chunk_title = (chunk.get("metadata", {}).get("title") or "").lower()
        chunk_full = chunk_text + " " + chunk_title

        # ── Rule 1: Architecture-name gate ──────────────────────────────────
        # If the sentence explicitly mentions a specific architecture or model,
        # the cited chunk MUST contain that name.  This eliminates the pattern
        # where the LLM says "RNNs are used in security (doc_3)" when doc_3 is
        # an agriculture paper that never mentions RNNs.
        sent_lower = sentence.lower()
        for arch in self._ARCHITECTURE_NAMES:
            # Only fire when the architecture name appears as a whole word in the sentence
            if re.search(rf"\b{re.escape(arch)}\b", sent_lower):
                if arch not in chunk_full:
                    logger.debug(
                        f"Architecture gate: '{arch}' in sentence but absent from "
                        f"cited chunk '{chunk_title[:60]}' — rejecting"
                    )
                    return False

        # ── Rule 2: Token-overlap check ──────────────────────────────────────
        _STOP = {
            "the", "this", "that", "is", "are", "was", "were", "has", "have",
            "had", "with", "and", "for", "from", "into", "which", "their",
            "also", "based", "using", "used", "study", "paper", "research",
            "approach", "method", "result", "shows", "show", "shown",
        }
        # Extract tokens from the sentence (4+ chars, not stop-words)
        sent_tokens = [
            t for t in re.findall(r"\b[a-z]{4,}\b", sentence.lower())
            if t not in _STOP
        ]
        if not sent_tokens:
            return True  # can’t verify, give benefit of doubt

        # Count how many sentence tokens appear in chunk text
        hits = sum(1 for t in sent_tokens if t in chunk_text)

        # Also accept if chunk title appears directly in sentence
        title_words = [t for t in re.findall(r"\b[a-z]{4,}\b", chunk_title) if t not in _STOP]
        title_hit = any(t in sentence.lower() for t in title_words[:4])

        # Weaken token overlap check to allow semantic paraphrases:
        # Pass if at least 1 token matches (hits >= 1) OR the chunk's title matches,
        # OR the sentence contains too few checkable tokens to verify.
        return hits >= 1 or title_hit or len(sent_tokens) <= 2

    def _enforce_hard_grounding_rules(self, answer: str, chunks: list[dict]) -> str:
        # Split answer into claims/sentences
        from citation_verifier import CitationVerifier
        verifier = CitationVerifier()

        # Separate references if present
        answer_body = answer
        refs_part = ""
        if "References:" in answer:
            parts = answer.split("References:", 1)
            answer_body = parts[0].strip()
            refs_part = "References:\n" + parts[1].strip()

        sentences = verifier.split_into_claims(answer_body)
        kept_sentences = []

        for sent in sentences:
            # Always keep generic/structural sentences (headings, transitions)
            if verifier.is_generic_sentence(sent):
                kept_sentences.append(sent)
                continue

            # Find all doc_X IDs cited in this sentence
            doc_ids = re.findall(r"\bdoc_(\d+)\b", sent)

            if not doc_ids:
                # No citation at all — strip the sentence
                logger.warning(f"Removing uncited sentence: '{sent[:80]}'")
                continue

            # Span-level check: at least ONE cited chunk must support this sentence
            has_support = any(
                self._verify_claim_chunk_support(sent, int(d), chunks)
                for d in doc_ids
            )
            if has_support:
                kept_sentences.append(sent)
            else:
                logger.warning(
                    f"Removing sentence — cited chunks do not support claim: '{sent[:80]}'"
                )

        new_body = " ".join(kept_sentences)
        if refs_part:
            return f"{new_body}\n\n{refs_part}"
        return new_body

    def _strip_generic_sentences(self, answer: str, chunks: list[dict]) -> str:
        """
        Remove sentences from `answer` that share NO significant keywords with any
        retrieved chunk. 
        
        [RELAXED]: Now bypassed to allow full semantic paraphrasing and prevent 
        aggressive sentence stripping. Simply returns the answer as-is.
        """
        return answer

    def _bind_citations_and_verify(self, answer: str, chunks: list[dict]) -> tuple[str, bool]:
        # Let's map each doc_X to its clean APA citation
        def get_clean_citation(doc_num_str):
            try:
                idx = int(doc_num_str) - 1
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    meta = chunk.get("metadata", {}) or {}
                    authors = meta.get("authors", "Unknown Authors")
                    year = meta.get("year", "N/A")

                    # Clean first author surname
                    first_author = "Unknown"
                    for part in re.split(r"[,;&]| and ", authors):
                        words = [w for w in re.findall(r"[a-zA-Z\u00C0-\u017F]+", part)
                                 if w.lower() not in {"et", "al"}]
                        if words:
                            first_author = words[-1]
                            break

                    author_parts = [p for p in re.split(r"[,;&]| and ", authors) if p.strip()]
                    num_authors = len(author_parts)
                    if num_authors > 2:
                        citation = f"{first_author} et al., {year}"
                    elif num_authors == 2:
                        second_author = "Unknown"
                        words2 = [w for w in re.findall(r"[a-zA-Z\u00C0-\u017F]+", author_parts[1])
                                  if w.lower() not in {"et", "al"}]
                        if words2:
                            second_author = words2[-1]
                        citation = f"{first_author} & {second_author}, {year}"
                    else:
                        citation = f"{first_author}, {year}"
                    return citation
            except Exception:
                pass
            return None

        # Replace doc_X in text, tracking resolved citations for deduplication
        modified_answer = answer
        has_citations = False
        # Track per-sentence citation counts to prevent excessive repetition
        paper_citation_counts: dict[str, int] = {}

        pattern = r"\bdoc_(\d+)\b"

        def repl(match):
            nonlocal has_citations
            doc_num = match.group(1)
            cit = get_clean_citation(doc_num)
            if cit:
                # Deduplication: if this exact citation has appeared > 2 times, suppress
                count = paper_citation_counts.get(cit, 0)
                if count >= 3:
                    logger.debug(f"Suppressing duplicate citation: {cit} (count={count})")
                    return ""  # remove the duplicate
                paper_citation_counts[cit] = count + 1
                has_citations = True
                return cit
            return f"doc_{doc_num}"

        modified_answer = re.sub(pattern, repl, modified_answer)

        # Clean up empty parentheticals: "( )" or "(, )" or "()"
        modified_answer = re.sub(r"\(\s*,?\s*\)", "", modified_answer)
        # Clean up double parentheticals: "((X))"
        modified_answer = re.sub(r"\(\(([^()]+)\)\)", r"(\1)", modified_answer)
        # Clean up trailing commas before closing paren: "(X, )"
        modified_answer = re.sub(r",\s*\)", ")", modified_answer)

        return modified_answer, has_citations

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
        if scope.entity_kind == "author":
            author_phrase = scope.author_phrase
            if author_phrase:
                author_exists = verify_author_exists_in_library(author_phrase, papers_metadata)
                if not author_exists:
                    logger.warning(f"Author '{author_phrase}' not in library, refusing before LLM")
                    return {
                        "query": query,
                        "answer": f"No papers authored by {author_phrase} were found in the ingested library.",
                        "sources": [],
                        "success": False,
                        "error": f"Author not in library: {author_phrase}",
                    }
                
                # Check if it's a broad query when they have multiple papers
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
            # Cap to 20 chunks max. More chunks dilute relevance and cause citation drift.
            # Use 2 chunks per matched paper (enough coverage without flooding context).
            effective_limit = min(max(limit, len(matched_titles) * 2), 20)

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
            over_retrieve_multiplier=2.5,  # Reduced from 4.0 — prevents citation drift from irrelevant chunks
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
        
        # ── Check if query matches a paper title (used in multiple checks below) ─────
        query_matches_paper = False
        for title in papers_metadata.keys():
            if title.lower() in query.lower() or query.lower() in title.lower():
                query_matches_paper = True
                break
        
        # ── Off-topic detection BEFORE LLM call ─────────────────────────────
        # Use retrieval count AND relevance check (more reliable than just count)
        # If chunks were retrieved but are not relevant to the query, refuse
        if not filter_title and not matched_titles:
            # Check relevance of retrieved chunks
            if chunks:
                # Extract significant query tokens
                query_tokens = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
                # Remove common stopwords
                stopwords = {"what", "which", "where", "when", "how", "does", "did", "are", "is", "was", "were", "have", "has", "had", "will", "would", "could", "should", "may", "might", "must", "can", "about", "from", "with", "that", "this", "these", "those"}
                query_tokens = query_tokens - stopwords
                
                # Check if ANY query tokens appear in retrieved chunks
                # Use a very soft threshold: at least 1 chunk must match at least 1 token
                # The Answer Decision Gate (below) handles confidence; this check only
                # blocks truly off-topic retrieval where NOTHING in the query appears anywhere.
                relevant_chunks = 0
                for chunk in chunks:
                    chunk_text = chunk.get("text", "").lower()
                    chunk_title = chunk.get("metadata", {}).get("title", "").lower()
                    search_text = chunk_text + " " + chunk_title
                    token_hits = sum(1 for token in query_tokens if token in search_text)
                    if token_hits >= 1:  # lowered from 2 — any single meaningful token match counts
                        relevant_chunks += 1

                # Refuse only if <20% of chunks have ANY token match (truly off-topic retrieval)
                if relevant_chunks < max(1, len(chunks) * 0.2) and not query_matches_paper:
                    logger.warning(
                        f"Very low chunk relevance: {relevant_chunks}/{len(chunks)} for query: {query}"
                    )
                    return {
                        "query": query,
                        "answer": NOT_IN_LIBRARY_REFUSAL,
                        "sources": [],
                        "success": False,
                        "error": "Retrieved chunks not relevant to query",
                    }

                # Universal domain-term zero-match check.
                # If the query contains a high-specificity term (5+ chars, not a stopword)
                # and that term appears in ZERO retrieved chunks (text OR title),
                # then refuse — retrieval missed the domain entirely.
                # This is domain-agnostic: works for coffee, blockchain, UAV, etc.
                _COMMON = {
                    "paper", "study", "model", "method", "approach", "system",
                    "learning", "detection", "classification", "analysis", "review",
                    "using", "based", "these", "their", "which", "about",
                }
                specific_terms = [
                    t for t in query_tokens
                    if len(t) >= 5 and t not in _COMMON
                ]
                if specific_terms:
                    for term in specific_terms:
                        term_in_any_chunk = any(
                            term in c.get("text", "").lower()
                            or term in c.get("metadata", {}).get("title", "").lower()
                            for c in chunks
                        )
                        if not term_in_any_chunk:
                            logger.info(
                                f"Domain term '{term}' absent from all chunks — "
                                "but other terms match; continuing (not refusing)"
                            )
                    # Only refuse if ALL specific terms are absent from ALL chunks
                    all_terms_absent = all(
                        not any(
                            term in c.get("text", "").lower()
                            or term in c.get("metadata", {}).get("title", "").lower()
                            for c in chunks
                        )
                        for term in specific_terms
                    )
                    if all_terms_absent and not query_matches_paper:
                        logger.warning(
                            f"All specific query terms absent from chunks: {specific_terms}. Refusing."
                        )
                        return {
                            "query": query,
                            "answer": NOT_IN_LIBRARY_REFUSAL,
                            "sources": [],
                            "success": False,
                            "error": "Domain terms not present in retrieved chunks",
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
        
        # ── Evidence-based off-topic gate ─────────────────────────────────────
        # ONLY refuse on truly non-research structural patterns: sport scores,
        # entertainment gossip, personal advice, weather forecasts.
        # Do NOT refuse on domain/topic terms (cryptocurrency, autonomous driving,
        # blockchain) — if the library has no evidence, the Answer Decision Gate
        # (below) will handle it via the refuse mode.
        NON_RESEARCH_PATTERNS = (
            "who won", "world cup", "fifa", "match score", "election result",
            "weather forecast", "temperature today", "what's on tv",
            "movie review", "celebrity gossip", "song lyrics",
            "vacation plan", "hotel booking", "flight ticket",
        )
        query_lower = query.lower()
        is_non_research = any(p in query_lower for p in NON_RESEARCH_PATTERNS)

        if is_non_research and not query_matches_paper and not filter_title:
            logger.warning(f"Non-research structural query detected: {query}")
            return {
                "query": query,
                "answer": NOT_IN_LIBRARY_REFUSAL,
                "sources": [],
                "success": False,
                "error": "Non-research query",
            }
        
        # ── Context quality gate before LLM ───────────────────────────────────
        # DISABLED: Context quality gate is too aggressive and blocks valid queries
        # The quality calculation is unreliable and causes false negatives
        # Relying on citation verification and scope verification instead
        # try:
        #     from context_shaper import ContextShaper
        #     shaper = ContextShaper()

        # Build structured retrieved set summary and context blocks with IDs
        retrieved_summary_str = self._build_retrieved_set_summary(chunks)
        context_str = self._build_context_with_ids(chunks)

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

        # System prompt: iron-wall instruction set for grounded academic answers
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
            "=== OPERATING RULES FOR KEYWORDS & GROUNDING ===\n"
            "1. NEVER say 'none exist', 'no papers exist', or 'there are no papers on this topic' if any papers in the Ingested Paper Library Inventory or the Document Context Blocks match or mention the query keywords (e.g. 'coffee').\n"
            "2. WEAK RELEVANCE FALLBACK: If the retrieved papers are only partially or weakly related to the query, you MUST NOT deny their existence. Instead, clearly state: 'There are partially related studies, but they focus on [X, Y, Z] aspects of [topic]' and summarize what they do say.\n"
            "3. NO CITATION STITCHING/BLENDING: Each statement or claim must map directly to its specific source. Do NOT blend findings from multiple papers into one sentence. Keep claims from different papers in separate sentences, each with its own specific (doc_X) citation.\n\n"
            "=== CITATION RULES ===\n"
            "1. ONLY use facts from the Document Context Blocks or the Ingested Paper Library Inventory. Zero exceptions.\n"
            "2. NEVER invent author names, paper titles, years, DOIs or references.\n"
            "3. NEVER cite papers that are not in the Library Inventory or Context Blocks.\n"
            "4. You MUST cite using Document IDs in parentheticals at the end of each sentence or claim: (doc_1) or (doc_3, doc_4). Every non-generic sentence MUST contain at least one doc_X reference.\n"
            "5. NEVER use (Source 1), [Document 2] or bracket-number citations like [1], [2]. These are FORBIDDEN.\n"
            "6. End every answer with a References section in APA7 format (EXCEPT when the user asked for a list).\n"
            "7. TRUTH GAPS: If the concept asked about is NOT in the context, say: 'The retrieved context does not contain information about [topic].' DO NOT guess.\n"
            "8. ONE PAPER PER SENTENCE: In multi-paper answers, each sentence may cite at most ONE paper (one doc_X). Never blend two papers into one sentence.\n"
            "9. WRITE NOTHING YOU CANNOT TRACE: If you cannot attach a specific doc_X citation to a sentence, do not write that sentence.\n\n"
            "=== PAPER-SPECIFIC ANSWER RULES — MANDATORY ===\n"
            "When describing what a paper found, concluded, or contributed, you MUST include concrete specifics:\n"
            "- ALWAYS state the exact accuracy %, F1 score, dataset name, model name, or other metric if it appears in the passage. Example: 'achieves 98.3% accuracy on the ISIC dataset (doc_2)' NOT 'achieves high accuracy'.\n"
            "- ALWAYS name the specific method or architecture used. Example: 'uses a ResNet-50 backbone (doc_3)' NOT 'uses deep learning'.\n"
            "- If no specific numeric value appears in the retrieved passage, write: 'reports improved [metric] but specific figures are not in the retrieved passage (doc_X)'.\n"
            "- NEVER use generic phrases like 'reduces data requirements', 'improves performance', 'state of the art' without citing the specific number or paper-reported result.\n"
            "- NEVER describe online platforms, e-commerce, social media, or other topics that are NOT in the retrieved context passages.\n\n"
            "=== LISTING QUERY RULES — ABSOLUTE REQUIREMENTS ===\n"
            "When the user asks you to LIST, ENUMERATE, or TABULATE papers/articles:\n"
            "1. You MUST output EVERY SINGLE matching paper in your main numbered response.\n"
            "2. NEVER stop after 3, 5, or 10 papers. You must continue until ALL matching papers are listed.\n"
            "3. NEVER use '...' or '(remaining papers)' or any truncation placeholder.\n"
            "4. NEVER put papers in a References section when the user asked for a list.\n\n"
            "=== RETRIEVAL CONFIDENCE WEIGHTING RULES ===\n"
            "Each document in the Retrieved Document Set has a Retrieval Confidence Weighting: High, Medium, or Low/Weak.\n"
            "- High confidence [≥0.60]: Directly relevant. Use as primary evidence.\n"
            "- Medium confidence [0.45–0.60]: Moderately relevant. Use with appropriate hedging.\n"
            "- Low/Weak confidence [<0.45]: Weakly related. Mention only if directly asked; clearly flag as peripheral.\n"
            "Always weight and prioritize claims from High confidence documents.\n\n"
            f"{scope_note}"
            "=== BEGIN ANSWERING ONLY FROM CONTEXT/INVENTORY BELOW ==="
        )

        # ── Step 3b: Answer Decision Gate ────────────────────────────────────
        # Single deterministic controller: confident / partial / refuse.
        _answer_mode, _partial_notice = self._compute_answer_decision(chunks, query)
        logger.info(f"Answer Decision Gate: mode='{_answer_mode}' for query: {query[:80]}")

        # Hard refuse: scored below minimum threshold — return before calling LLM
        if _answer_mode == "refuse" and not listing_style_query:
            titles_found = list({
                c.get("metadata", {}).get("title", "")
                for c in chunks if c.get("metadata", {}).get("title")
            })
            if titles_found:
                titles_str = "; ".join(titles_found[:3])
                return {
                    "query": query,
                    "answer": (
                        f"The library contains papers related to this area ({titles_str}) "
                        "but the retrieved evidence scored below the minimum confidence "
                        "threshold to generate a reliable answer. "
                        "Try a more specific query or ask about a particular paper directly."
                    ),
                    "sources": chunks,
                    "success": False,
                    "error": "Below confidence threshold.",
                }
            else:
                return {
                    "query": query,
                    "answer": NOT_IN_LIBRARY_REFUSAL,
                    "sources": [],
                    "success": False,
                    "error": "No usable evidence retrieved.",
                }

        # ── Aggregation query detection ────────────────────────────────────────
        # Detect "what do all papers say / compare all / summarize all" queries.
        # Inject a strict aggregation instruction to prevent invented consensus.
        _AGG_PATTERNS = (
            "what do all", "all papers say", "across all", "summarize all",
            "compare all", "conclusion of all", "what do these papers",
            "all studies say", "combined conclusion", "combined summary",
            "overall conclusion", "aggregate", "synthesis of all",
        )
        _is_aggregation_query = any(p in query.lower() for p in _AGG_PATTERNS)
        _aggregation_notice = (
            "AGGREGATION MODE — STRICT STRUCTURE REQUIRED:\n"
            "You are synthesizing across papers from MULTIPLE DOMAINS. Follow this exact structure:\n\n"
            "STEP 1 — DOMAIN SEGMENTATION (mandatory first):\n"
            "  Group the retrieved documents by research domain (e.g., Medical Imaging, AI Safety, "
            "Agriculture, NLP). For each domain group, summarize ONLY what those papers say. "
            "Cite each claim with its specific doc_X. Do NOT mix domains in one paragraph.\n\n"
            "STEP 2 — SHARED THEMES (only if real overlap exists):\n"
            "  After per-domain summaries, list only themes explicitly mentioned by "
            "2 or more papers from DIFFERENT domains. Name the doc_IDs. "
            "If no cross-domain overlap exists, write: 'No significant cross-domain themes identified.'\n\n"
            "STEP 3 — DIVERGENCES AND GAPS:\n"
            "  State where papers disagree or cover different aspects.\n\n"
            "FORBIDDEN: Do NOT produce a single unified conclusion across all papers. "
            "Do NOT invent consensus. Do NOT blend claims from different domains into one sentence."
        ) if _is_aggregation_query else ""

        # User prompt: library inventory + doc-ID summary + full passages
        # Inject partial-notice from the gate when mode == 'partial'
        user_prompt = (
            f"Ingested Paper Library Inventory:\n"
            f"{'─' * 80}\n"
            f"{library_inventory_str}\n"
            f"{'─' * 80}\n\n"
            f"Retrieved Document Set (reference these doc_X IDs in every claim):\n"
            f"{'─' * 80}\n"
            f"{retrieved_summary_str}\n"
            f"{'─' * 80}\n\n"
            f"Full Context Passages from Retrieved Documents:\n"
            f"{'─' * 80}\n"
            f"{context_str}\n"
            f"{'─' * 80}\n\n"
            + (_partial_notice + "\n\n" if _partial_notice else "")
            + (_aggregation_notice + "\n\n" if _aggregation_notice else "")
            + f"Researcher Query: {query}\n\n"
            "Provide your structured academic answer below. "
            "You MUST cite every factual claim using doc_X IDs from the Retrieved Document Set above "
            "(e.g. (doc_1), (doc_3, doc_5)). Append a References section at the end."
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
                
                # Fallback logic: if the LLM refuses, but we have chunks containing query terms
                if self._is_refusal_answer(answer) and self._has_keyword_match_in_chunks(query, chunks):
                    logger.warning("LLM returned refusal, but chunks contain query keywords. Re-querying with fallback grounding prompt...")
                    fallback_system_prompt = (
                        f"{system_prompt}\n\n"
                        "=== CRITICAL INSTRUCTION FOR WEAK RELEVANCE ===\n"
                        "You previously claimed that no information or papers exist on this topic. This is WRONG.\n"
                        "There ARE matching papers in the provided library/context blocks.\n"
                        "You MUST NOT refuse to answer, and you MUST NOT say that no papers exist.\n"
                        "If the retrieved papers are only partially or weakly related, state that 'partially related papers exist, but they focus on [X, Y, Z] aspects' and summarize their findings.\n"
                        "Report whatever evidence is present in the context, even if it is a weak match. Do not deny existence.\n"
                    )
                    
                    messages_fallback = [{"role": "system", "content": fallback_system_prompt}]
                    if history_for_llm:
                        for turn in history_for_llm[-12:]:
                            role = (turn.get("role") or "").strip().lower()
                            content = (turn.get("content") or "").strip()
                            if role in {"user", "assistant"} and content:
                                messages_fallback.append({"role": role, "content": content})
                    messages_fallback.append({"role": "user", "content": user_prompt})
                    
                    payload_fallback = {
                        "model": settings.OLLAMA_MODEL,
                        "messages": messages_fallback,
                        "stream": False,
                        "options": {
                            "temperature": 0.05,
                            "top_p": 0.8,
                            "repeat_penalty": 1.1
                        }
                    }
                    
                    try:
                        fallback_response = requests.post(url, json=payload_fallback, timeout=settings.OLLAMA_TIMEOUT)
                        if fallback_response.status_code == 200:
                            fallback_data = fallback_response.json()
                            fallback_answer = fallback_data["message"]["content"]
                            if not self._is_refusal_answer(fallback_answer):
                                logger.info("Fallback prompt successfully generated grounded answer.")
                                answer = fallback_answer
                            else:
                                logger.warning("Fallback prompt also returned a refusal.")
                    except Exception as fe:
                        logger.warning(f"Fallback generation failed: {fe}")

                # Global guardrail against unsupported sensitive-topic stance claims.
                if self._is_unverifiable_sensitive_claim(query, chunks):
                    answer = (
                        "I cannot confirm or deny that claim from the retrieved context. "
                        "The current sources do not provide direct evidence on this topic."
                    )
                elif answer_has_table_truncation(answer):
                    answer = TABLE_TRUNCATION_REFUSAL
                else:
                    if not self._is_refusal_answer(answer):
                        # Optional claim-level enhancement: try to parse CLAIM/SOURCE/QUOTE blocks
                        # produced by some LLM outputs. If enough valid claims are found, synthesize
                        # prose from them (higher quality). If not — e.g. the 8B model wrote natural
                        # language instead of structured blocks — keep the raw LLM answer as-is.
                        parsed_blocks = self._parse_constrained_claims(answer)
                        valid_claims = self._validate_claims(parsed_blocks, chunks)
                        if valid_claims:
                            answer = self._synthesize_prose(valid_claims, query, chunks, papers_metadata, listing_style_query)
                            logger.info(f"Synthesized prose from {len(valid_claims)} valid claims.")
                        else:
                            logger.info("Claim parser found no structured blocks — keeping raw LLM answer.")

                    answer = self._strip_model_references(answer, chunks=chunks)

                    # FIX 2: Hard grounding — strip any sentence not backed by a doc_X reference.
                    # This must run BEFORE citation binding so uncited claims are removed first.
                    if not listing_style_query and not self._is_refusal_answer(answer):
                        try:
                            answer = self._enforce_hard_grounding_rules(answer, chunks)
                        except Exception as _eg:
                            logger.warning(f"Hard grounding enforcement failed: {_eg}")

                    # FIX 4: Citation binding — convert doc_X placeholders to APA inline.
                    # Must run after grounding enforcement, before reference section is appended.
                    if not listing_style_query and not self._is_refusal_answer(answer):
                        try:
                            answer, _had_citations = self._bind_citations_and_verify(answer, chunks)
                            if not _had_citations:
                                logger.warning("No doc_X citations found in LLM answer after grounding.")
                        except Exception as _cb:
                            logger.warning(f"Citation binding failed: {_cb}")

                    # FIX 4b: Generic-knowledge injection filter.
                    # Strips sentences that contain no doc_X anchor and look like general knowledge.
                    if not listing_style_query and not self._is_refusal_answer(answer):
                        try:
                            answer = self._strip_generic_sentences(answer, chunks)
                        except Exception as _gs:
                            logger.warning(f"Generic sentence filter failed: {_gs}")
                    if (not listing_style_query) and (not self._is_refusal_answer(answer)):
                        safe_refs = self._build_safe_references(chunks, papers_metadata)
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
