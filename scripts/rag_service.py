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
from query_router import QueryRouter, route_query  # Query routing and metadata filtering
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
    is_bibliography_chunk,
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
        num_predict: int = 4096,
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
                t = ((c.get("metadata") or {}).get("title") or "").lower()
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
        # Split on any References heading variant
        stripped = re.split(
            r'\n\s*(?:#+\s*|\*+\s*|_+)?[Rr]eferences\b[:\s]*',
            answer, maxsplit=1
        )[0]
        # Strip lines that are ONLY a bare URL or DOI — these are orphaned ref lines
        lines = stripped.split('\n')
        clean_lines = []
        for line in lines:
            stripped_line = line.strip()
            if re.match(r'^https?://\S+$', stripped_line):
                continue
            if re.match(r'^doi:\s*\S+$', stripped_line, re.I):
                continue
            # Strip bullet lines that look exactly like: "- Author (Year). Title"
            # but only if title matches a known paper (to avoid stripping real sentences)
            if re.match(r'^[-•]\s+\S.+\(\d{4}\)\.\s+\S', stripped_line) and chunks:
                known_titles = {(c.get("metadata") or {}).get("title", "").lower()
                               for c in (chunks or [])}
                # If 3+ words of this line appear in any known title, it's a ref fragment
                words = re.findall(r'\b[a-z]{4,}\b', stripped_line.lower())
                if any(sum(1 for w in words if w in t) >= 3 for t in known_titles):
                    continue
            clean_lines.append(line)
        return '\n'.join(clean_lines).strip()

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
        # Hard off-topic keyword families — only unambiguous non-academic terms
        off_topic_patterns = [
            # Food & cooking (full phrases only to avoid false positives)
            "recipe for", "how to cook", "bake a", "cooking instructions",
            "breakfast recipe", "lunch recipe", "dinner recipe",
            # Sports scores (not academic)
            "football score", "cricket score", "soccer score",
            "sports tournament", "nba score", "nfl score",
            # Entertainment (specific)
            "movie review", "film review", "song lyrics", "music video",
            "celebrity gossip",
            # Finance / non-research
            "bitcoin price", "stock price", "cryptocurrency price",
            "investment tips",
            # Travel bookings
            "hotel booking", "flight booking", "book a vacation",
        ]
        if any(pat in q for pat in off_topic_patterns):
            return True
        # Also block if query is very short AND matches no library paper title
        # (handled downstream, so return False here for borderline cases)
        return False

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
    #   'confident'  → ≥2 chunks with rerank_score ≥ 0.35
    #   'partial'    → at least 1 chunk with score 0.15–0.35 (OR ≥2 unscored chunks)
    #   'refuse'     → 0 chunks with score ≥ 0.15 (no usable evidence)
    # Modes NEVER overlap; transitions are deterministic.
    # ─────────────────────────────────────────────────────────────────────────────
    _CONFIDENT_THRESHOLD = 0.35
    _PARTIAL_THRESHOLD   = 0.08  # Lowered from 0.15 to prevent aggressive refusals

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

        # ── Fix 6: Multi-Paper Retrieval Floor for Comparison Queries ──
        _COMP_CUES = {
            "compare", "comparison", "contrast", "difference", "differences",
            "similarities", "versus", "vs"
        }
        q_lower = query.lower()
        is_comparison = any(cue in q_lower for cue in _COMP_CUES)
        if is_comparison:
            unique_papers = {(c.get("metadata") or {}).get("title", "") for c in chunks if (c.get("metadata") or {}).get("title")}
            if len(unique_papers) < 2:
                # Instead of hard refusal, allow the LLM to try to answer from what it has
                logger.warning(
                    f"Comparison query '{query}' retrieved only {len(unique_papers)} paper(s). "
                    "Proceeding with caution instead of hard refusal."
                )

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

        # ── Improved caution mode using retrieval scores (not keyword counting) ──────
        # More robust thresholds based on reranker scores:
        # - max_score < 0.08: refuse (chunks too irrelevant even for cautious answer)
        # - 0.08 <= max_score < 0.25: partial/caution mode
        # - max_score >= 0.25: confident or partial based on avg_top_3
        if max_score < self._PARTIAL_THRESHOLD:
            logger.warning(
                f"Refusing query '{query}': max_score={max_score:.3f} < {self._PARTIAL_THRESHOLD}. "
                "Chunks are too irrelevant for even a cautious answer."
            )
            return "refuse", ""
        
        if max_score < 0.25:
            logger.warning(
                f"Weak retrieval for query '{query}': max_score={max_score:.3f} < 0.25. "
                "Switching to caution/partial mode."
            )
            return "partial", (
                "⚠️ WEAK MATCH NOTICE: The retrieved documents scored below the normal "
                "confidence threshold. You MUST begin your answer with exactly this sentence:\n"
                "'I found weak matches in your library; this answer may be approximate.'\n"
                "Then summarize ONLY what the retrieved passages explicitly state. "
                "Do not extrapolate, invent details, or use general knowledge. "
                "If a passage is only tangentially related, say so explicitly."
            )


        # Routing based on normalized query-level confidence
        if avg_top_3 >= 0.40 and max_score >= 0.45:
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

    # Step 6a: Removed dead code - _parse_constrained_claims, _validate_claims, _synthesize_prose
    # These methods were not being called and added latency without benefit.
    # The 8B model doesn't reliably produce structured CLAIM/SOURCE/QUOTE blocks.

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

    def _deduplicate_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Remove chunks with >80% text overlap to prevent robotic stuttering.
        Uses Jaccard similarity to detect near-duplicate chunks.
        """
        if len(chunks) <= 1:
            return chunks

        def get_text_set(chunk: dict) -> set[str]:
            """Extract word set from chunk text for similarity comparison."""
            text = (chunk.get("text") or "").lower()
            # Tokenize into words (3+ chars to be meaningful)
            words = set(re.findall(r"\b[a-z]{3,}\b", text))
            return words

        def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
            """Calculate Jaccard similarity between two sets."""
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0

        deduplicated = []
        seen_texts = []

        for chunk in chunks:
            chunk_text_set = get_text_set(chunk)
            is_duplicate = False

            for seen_set in seen_texts:
                similarity = jaccard_similarity(chunk_text_set, seen_set)
                if similarity > 0.8:  # 80% threshold
                    is_duplicate = True
                    logger.debug(f"Dedup: removed chunk with {similarity:.2f} similarity")
                    break

            if not is_duplicate:
                deduplicated.append(chunk)
                seen_texts.append(chunk_text_set)

        logger.info(f"Deduplication: {len(chunks)} -> {len(deduplicated)} chunks")
        return deduplicated

    def _verify_claim_chunk_support(self, sentence: str, doc_num: int, chunks: list[dict]) -> bool:
        """
        Span-level evidence check: verify that the chunk cited by doc_<doc_num> actually
        contains keyword evidence for the claim made in `sentence`.

        Rules (in order):
        1. Architecture-name check: if the sentence names a specific model/architecture
           (e.g. RNN, LSTM, GAN, Xception), that name MUST appear in the cited chunk.
           This prevents e.g. "RNNs used in security" being attributed to an agriculture paper.
        2. Fuzzy title matching: if the sentence cites a paper by title words, those words
           must overlap with the cited chunk's title. This is more robust than hardcoded domain groups.
        3. Token-overlap check:
           - Adaptively requires hits based on sentence length (1 for <=2 tokens, 2 for <=5, 3 otherwise).
           - Title-word match from the cited paper also counts as a pass.
        """
        if doc_num < 1 or doc_num > len(chunks):
            return False
        chunk = chunks[doc_num - 1]
        chunk_text = (chunk.get("text") or "").lower()
        chunk_title = ((chunk.get("metadata") or {}).get("title") or "").lower()
        chunk_full = chunk_text + " " + chunk_title

        # ── Rule 1: Architecture-name gate ──────────────────────────────────
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

        # ── Rule 2: Fuzzy title matching ─────────────────────────────────────
        # Extract significant words from the chunk title (4+ chars, not common stopwords)
        _TITLE_STOP = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
            "using", "based", "approach", "method", "system", "model", "study", "research",
            "paper", "analysis", "detection", "classification"
        }
        title_significant = [t for t in re.findall(r"\b[a-z]{4,}\b", chunk_title) if t not in _TITLE_STOP]
        # If the sentence contains 2+ significant words from the chunk title, it's likely citing that paper
        title_word_hits = sum(1 for t in title_significant if t in sent_lower)
        if title_word_hits >= 2:
            # Strong title match - accept the claim
            return True

        # ── Rule 3: Token-overlap check ──────────────────────────────────────
        _STOP = {
            "the", "this", "that", "is", "are", "was", "were", "has", "have",
            "had", "with", "and", "for", "from", "into", "which", "their",
            "also", "based", "using", "used", "study", "paper", "research",
            "approach", "method", "result", "shows", "show", "shown",
            # Expanded stopwords to block generic matches on technical commonalities
            "detection", "classification", "accuracy", "performance", "proposed",
            "results", "system", "model", "data", "applications", "analysis",
            "evaluation", "methods", "approach"
        }
        # Extract tokens from the sentence (4+ chars, not stop-words)
        sent_tokens = [
            t for t in re.findall(r"\b[a-z]{4,}\b", sentence.lower())
            if t not in _STOP
        ]
        if not sent_tokens:
            return True  # can't verify, give benefit of doubt

        # Count how many sentence tokens appear in chunk text
        hits = sum(1 for t in sent_tokens if t in chunk_text)

        # Also accept if chunk title appears directly in sentence
        title_hit = any(t in sentence.lower() for t in title_significant[:4])

        # Adaptive thresholding — cap at 2 to avoid stripping valid factual sentences.
        # 1 hit  : very short claims (<=2 meaningful tokens)
        # 2 hits : all longer claims (>2 meaningful tokens)
        # Title-word match is always an independent pass.
        required_hits = 1 if len(sent_tokens) <= 2 else 2

        return hits >= required_hits or title_hit or len(sent_tokens) <= 1

    def _enforce_hard_grounding_rules(self, answer: str, chunks: list[dict]) -> str:
        # Split answer into claims/sentences
        from services import citation_verifier as verifier

        # Separate references if present
        answer_body = answer
        refs_part = ""
        if "References:" in answer:
            parts = answer.split("References:", 1)
            answer_body = parts[0].strip()
            refs_part = "References:\n" + parts[1].strip()

        sentences = verifier.split_into_claims(answer_body)
        kept_sentences = []

        GENERIC_FILLER_PATTERNS = [
            "future research",
            "further investigation",
            "further study",
            "further studies",
            "interdisciplinary approach",
            "deeper understanding",
            "complex phenomenon",
            "novel insights",
            "future directions",
            "research directions",
            "investigate the phenomenon",
            "another direction",
            "integrating insights",
            "addressing methodological",
            "firstly, there is a need",
            "another direction for",
            "integrating insights from",
            "addressing methodological limitations",
        ]

        # Pattern to detect bracketed placeholders like [specific phenomenon or concept]
        PLACEHOLDER_PATTERN = re.compile(r'\[([^\]]+)\]', re.IGNORECASE)

        for sent in sentences:
            # Check for generic academic filler patterns
            is_filler = any(p in sent.lower() for p in GENERIC_FILLER_PATTERNS)
            
            # Check for bracketed placeholders (template text)
            has_placeholder = bool(PLACEHOLDER_PATTERN.search(sent))
            if has_placeholder:
                logger.warning(f"Removing sentence with placeholder template: '{sent[:80]}'")
                continue

            # Always keep generic/structural sentences (headings, transitions)
            # EXCEPT when they contain academic filler pattern words (which need grounding)
            if verifier.is_generic_sentence(sent) and not is_filler:
                kept_sentences.append(sent)
                continue

            # Find all doc_X IDs cited in this sentence
            doc_ids = re.findall(r"\bdoc_(\d+)\b", sent)

            if not doc_ids:
                # Preserve synthesis/conclusion sentences (they summarize cited content)
                SYNTHESIS_STARTERS = [
                    "these ", "this ", "together, ", "overall, ", "in summary",
                    "in conclusion", "collectively", "taken together",
                    "highlight", "demonstrate", "suggest", "indicate",
                    "show that", "reveal", "confirm",
                ]
                is_synthesis = any(sent.lower().startswith(p) or p in sent.lower()[:40]
                                   for p in SYNTHESIS_STARTERS)
                
                # PRESERVE honest "not found" admissions — these are correct, not errors
                ADMISSION_PHRASES = [
                    "do not contain", "does not contain", "no information about",
                    "not found in", "cannot find", "not in the provided",
                    "no relevant", "not mentioned", "not discussed", "not addressed",
                    "no papers", "no chunks", "insufficient", "no evidence",
                    "provided documents do not", "context does not",
                    "ingested papers do not",
                ]
                is_admission = any(p in sent.lower() for p in ADMISSION_PHRASES)
                
                if is_synthesis or is_admission:
                    kept_sentences.append(sent)
                    continue  # Keep it — don't strip synthesis or honest admissions

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

        new_body = " ".join(kept_sentences).strip()

        # Safety net: if ALL sentences were stripped, check if this is an abstract-only library
        # (chunks are very short — typically <500 chars each). In that case, return the raw
        # answer since strict citation enforcement cannot work without full paper text.
        if not new_body:
            avg_chunk_len = sum(len(c.get("text","")) for c in chunks) / max(len(chunks),1)
            if avg_chunk_len < 600:
                logger.warning(
                    "_enforce_hard_grounding_rules stripped every sentence but chunks are "
                    f"abstract-only (avg {avg_chunk_len:.0f} chars). Returning raw answer."
                )
                # Return the original answer body without strict citation enforcement
                return answer_body
            logger.warning(
                "_enforce_hard_grounding_rules stripped every sentence — "
                "returning refusal message instead of unverified answer."
            )
            return "I could not find sufficient grounded information in the ingested papers to answer this question."

        if refs_part:
            return f"{new_body}\n\n{refs_part}"
        return new_body

    def _check_answer_faithfulness(self, answer: str, chunks: list[dict]) -> list[str]:
        """
        Check if the answer contains claims not supported by the chunks.
        
        Returns a list of issues found (empty if faithful).
        """
        from services import citation_verifier as verifier
        
        issues = []
        sentences = verifier.split_into_claims(answer)
        
        for sent in sentences:
            # Skip generic/structural sentences
            if verifier.is_generic_sentence(sent):
                continue
            
            # Check for chunk citations
            doc_ids = re.findall(r"\bdoc_(\d+)\b", sent)
            
            if not doc_ids:
                # Sentence has no citation - potential issue
                # Check if it looks like a factual claim
                factual_indicators = [
                    "found", "showed", "demonstrated", "reported", "concluded",
                    "achieved", "improved", "reduced", "increased", "used",
                    "proposed", "developed", "implemented", "evaluated"
                ]
                if any(indicator in sent.lower() for indicator in factual_indicators):
                    issues.append(f"Uncited factual claim: '{sent[:80]}...'")
        
        return issues

    def _log_retrieval_metrics(self, query: str, chunks: list[dict], mode: str, faithfulness_issues: list[str]) -> None:
        """
        Log detailed retrieval metrics for debugging.
        """
        logger.info("=" * 80)
        logger.info("RETRIEVAL METRICS DASHBOARD")
        logger.info("=" * 80)
        logger.info(f"Query: {query[:100]}")
        logger.info(f"Retrieved chunks: {len(chunks)}")
        logger.info(f"Answer mode: {mode}")
        logger.info(f"Faithfulness issues: {len(faithfulness_issues)}")
        
        if chunks:
            # Check for reference section pollution
            bibliography_count = 0
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                if is_bibliography_chunk(text):
                    bibliography_count += 1
            
            ref_percentage = (bibliography_count / len(chunks)) * 100 if chunks else 0
            logger.info(f"Bibliography chunks: {bibliography_count}/{len(chunks)} ({ref_percentage:.1f}%)")
            
            if ref_percentage > 15:
                logger.warning(f"High reference section pollution: {ref_percentage:.1f}% of chunks are from bibliography sections")
            
            # Log top 5 chunks with scores
            logger.info("Top 5 chunks:")
            for i, chunk in enumerate(chunks[:5], start=1):
                meta = chunk.get("metadata", {}) or {}
                title = meta.get("title", "Unknown")[:60]
                distance = chunk.get("distance", "N/A")
                rerank_score = chunk.get("rerank_score", "N/A")
                is_bib = "[BIB]" if is_bibliography_chunk(chunk.get("text", "")) else ""
                logger.info(f"  {i}. {title}... | distance={distance:.3f} | rerank={rerank_score} {is_bib}")
        
        if faithfulness_issues:
            logger.info("Faithfulness issues:")
            for issue in faithfulness_issues[:5]:
                logger.info(f"  - {issue}")
        
        logger.info("=" * 80)

    def _strip_generic_sentences(self, answer: str, chunks: list[dict]) -> str:
        """
        Remove sentences from `answer` that share NO significant keywords with any
        retrieved chunk.

        [RELAXED]: Now bypassed to allow full semantic paraphrasing and prevent
        aggressive sentence stripping. Simply returns the answer as-is.
        """
        return answer

    def _extract_key_sentences(self, chunks: list[dict], query: str) -> list[dict]:
        """
        Extract key sentences from chunks for extractive QA (Issue 3).
        Returns a list of dicts with 'sentence', 'doc_id', and 'source' info.
        """
        extracted = []
        query_tokens = set(re.findall(r"\b[a-z]{4,}\b", query.lower()))
        
        for idx, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            meta = chunk.get("metadata") or {}
            title = meta.get("title", "Untitled")
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 30:  # Skip very short sentences
                    continue
                
                sent_tokens = set(re.findall(r"\b[a-z]{4,}\b", sent.lower()))
                
                # Score sentence based on:
                # 1. Overlap with query tokens
                # 2. Sentence length (prefer medium-length sentences)
                # 3. Contains numbers/dates (often factual)
                query_overlap = len(query_tokens & sent_tokens)
                length_score = min(1.0, len(sent) / 200.0)  # Prefer ~200 char sentences
                has_facts = bool(re.search(r'\d{4}|\d+\.\d+|\d+%|\d+/\d+', sent))
                
                score = query_overlap * 2.0 + length_score + (1.0 if has_facts else 0)
                
                if score >= 1.0:  # Minimum threshold
                    extracted.append({
                        'sentence': sent,
                        'doc_id': f'doc_{idx}',
                        'title': title,
                        'score': score
                    })
        
        # Sort by score and take top sentences per chunk
        extracted.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to top 3 sentences per chunk to avoid overwhelming the LLM
        doc_counts = {}
        filtered = []
        for item in extracted:
            doc_id = item['doc_id']
            if doc_counts.get(doc_id, 0) < 3:
                filtered.append(item)
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        return filtered

    def _try_answer_from_metadata(self, query: str, papers_metadata: dict) -> dict | None:
        """
        Fix 2: Metadata query interceptor.
        Intercept queries about paper metadata or content mentions before RAG retrieval.
        Returns a dict with answer if query can be answered from metadata/chunks, None otherwise.
        """
        q = query.lower().strip()
        
        # Pattern: "which/what papers use/mention/discuss X"
        use_match = re.search(
            r'\b(?:which|what)\s+papers?\s+'
            r'(?:use|mention|discuss|implement|apply|contain|employ)\s+'
            r'(.+?)[\?\.]?$', q
        )
        if use_match:
            term = use_match.group(1).strip()
            matched = []
            for title, meta in papers_metadata.items():
                chunks = self.vector_store.get_chunks_for_paper(title)
                for chunk in chunks:
                    if term in chunk.get('text', '').lower():
                        matched.append(
                            f"{meta.get('authors','Unknown')} "
                            f"({meta.get('year','N/A')}). {title}"
                        )
                        break
            if matched:
                return {
                    "query": query,
                    "answer": (
                        f"Papers mentioning '{term}':\n\n" + 
                        "\n\n".join(matched)
                    ),
                    "sources": [], "success": True
                }
            return {
                "query": query,
                "answer": (
                    f"No papers in your library explicitly "
                    f"mention '{term}' in their text."
                ),
                "sources": [], "success": True
            }
        
        # Pattern: DOI/year/venue/authors lookup
        field_match = re.search(
            r'\b(?:what\s+is|what\'s|give me|show me|find)\s+'
            r'(?:the\s+)?'
            r'(doi|year|venue|journal|authors?|publisher)\s+'
            r'(?:of|for)\s+(.+?)[\?\.]?$', q
        )
        if field_match:
            field = field_match.group(1).strip()
            paper_query = field_match.group(2).strip()
            
            # Fuzzy match paper title
            best_match = None
            best_score = 0
            for title in papers_metadata.keys():
                title_lower = title.lower()
                if paper_query in title_lower or title_lower in paper_query:
                    score = len(paper_query) / max(len(title_lower), 1)
                    if score > best_score:
                        best_score = score
                        best_match = title
            
            if best_match and best_score > 0.3:
                meta = papers_metadata[best_match]
                field_map = {
                    "doi": meta.get("doi", "N/A"),
                    "year": meta.get("year", "N/A"),
                    "venue": meta.get("venue", "N/A"),
                    "journal": meta.get("venue", "N/A"),
                    "authors": meta.get("authors", "Unknown"),
                    "author": meta.get("authors", "Unknown"),
                    "publisher": meta.get("venue", "N/A"),
                }
                value = field_map.get(field, "N/A")
                return {
                    "query": query,
                    "answer": (
                        f"The {field} of \"{best_match}\" is: {value}"
                    ),
                    "sources": [], "success": True
                }
        
        return None  # Not a metadata query, continue to RAG

    def _query_is_out_of_scope(self, query: str, chunks: list[dict]) -> bool:
        """
        Fix 4: Irrelevant topic refusal.
        Returns True if the query's core topic is absent from ALL retrieved chunks.
        Uses 30% match threshold for key terms.
        """
        # Extract multi-word technical phrases (2+ word sequences)
        # These are more specific than single tokens
        words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
        stopwords = {
            "what", "which", "when", "where", "does", "about",
            "from", "with", "that", "this", "have", "into",
            "paper", "papers", "using", "latest", "modern",
            "these", "their", "used", "show", "tell", "give"
        }
        key_terms = [w for w in words if w not in stopwords]
        
        if len(key_terms) < 2:
            return False
        
        # Build full text of all chunks
        all_chunk_text = " ".join(
            c.get("text", "").lower() + " " + 
            (c.get("metadata") or {}).get("title", "").lower()
            for c in chunks
        )
        
        # Count how many key terms appear in ANY chunk
        matches = sum(1 for t in key_terms if t in all_chunk_text)
        match_ratio = matches / len(key_terms)
        
        # If fewer than 30% of key terms appear anywhere,
        # the topic is genuinely not in the library
        return match_ratio < 0.30

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
                # Deduplication: if this exact citation has appeared > 9 times, suppress
                count = paper_citation_counts.get(cit, 0)
                if count >= 10:  # Only suppress truly pathological repetition
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

        # ── Fix 2: Metadata query interceptor — fires before retrieval ─────────────
        # Fetch stats ONCE at the top - never fetch again inside this function
        stats = self.vector_store.get_collection_stats() or {}
        papers_metadata = stats.get("papers_metadata", {}) or {}
        
        if not papers_metadata:
            return {"query": query, "answer": EMPTY_DB_REFUSAL, 
                    "sources": [], "success": False}
        
        metadata_answer = self._try_answer_from_metadata(query, papers_metadata)
        if metadata_answer:
            return metadata_answer

        # ── Academic drafting path (non-RAG) ─────────────────────────────────────
        # Example: "Draft an introduction for a paper on federated learning for IoT security"
        # This must not trigger keyword discovery ("papers on ...") or strict scope verification.
        if self._is_academic_drafting_request(query):
            return self._draft_academic_text(query)

        # ── Detect author-scoped and listing/tabulation queries ─────────────
        # When the query names a specific author, we:
        #   a) Filter the library inventory to only that author's papers so the LLM
        #      cannot accidentally reference other authors' works.
        #   b) Scale up the retrieval limit so every paper gets at least one chunk.
        scope = resolve_query_scope(
            query, papers_metadata, filter_title=filter_title
        )
        scope = apply_scope_resilience(scope, query, papers_metadata)
        
        # ── Pre-retrieval paper existence check for specific paper queries ─────
        # Detect queries asking for a specific paper by title/name and refuse if not in library
        # This prevents hallucination of summaries for non-existent papers
        q_lower = (query or "").lower()
        paper_specific_patterns = [
            r"\b(?:summarize|summary of|describe|explain|the paper|this paper)\b.*\b(?:titled|called|named)\b",
            r"\b(?:summarize|describe|explain)\b\s+(?:the\s+)?paper\s+",
            r"\bthe\s+paper\s+",
            r"^\s*(?:summarize|describe|explain)\s+[A-Z]",  # Starts with action verb + capitalized title
        ]
        is_paper_specific = any(re.search(pat, q_lower) for pat in paper_specific_patterns)
        
        # Additional check: if query starts with "Summarize" or "Describe" followed by a capitalized phrase
        # that looks like a paper title (contains multiple words, some capitalized), treat as paper-specific
        if not is_paper_specific:
            action_pattern = r"^\s*(?:summarize|describe|explain)\s+([A-Z][A-Za-z0-9\s\-:]{10,100})"
            match = re.match(action_pattern, query)
            if match:
                potential_title = match.group(1).strip()
                # Check if this looks like a title (has multiple words, some capitals)
                if len(potential_title.split()) >= 3 and re.search(r'[A-Z]', potential_title):
                    is_paper_specific = True
                    logger.info(f"Detected paper-specific query by action+title pattern: '{query[:60]}...'")
        
        if is_paper_specific and not filter_title:
            # Extract potential paper title from query
            from rag_strict import fuzzy_match_paper_titles
            matched_papers = fuzzy_match_paper_titles(query, papers_metadata)
            if not matched_papers:
                # No paper found - refuse immediately
                logger.warning(f"Paper-specific query with no match in library: '{query[:80]}'")
                return {
                    "query": query,
                    "answer": (
                        "I could not find a paper with that title in your ingested library. "
                        "Please check the spelling or ingest the paper first. "
                        "I cannot summarize papers that are not in your knowledge base."
                    ),
                    "sources": [],
                    "success": False,
                    "error": "Paper not found in library.",
                }
            else:
                # Found the paper — set filter_title so retrieval is scoped to that paper only
                filter_title = matched_papers[0]
        
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
                    logger.warning(f"Broad query on author '{author_phrase}' with {len(resolved_papers)} papers, showing paper listing")
                    # Fix 6: Show their papers grouped by topic instead of refusing
                    author_papers_meta = {
                        t: papers_metadata[t]
                        for t in resolved_papers
                        if t in papers_metadata
                    }
                    listing = self._render_inventory_listing(query, author_papers_meta)
                    return {
                        "query": query,
                        "answer": (
                            f"{author_display} has {len(resolved_papers)} papers "
                            f"in your library:\n\n{listing}\n\n"
                            "Please ask about a specific paper or topic from the list above."
                        ),
                        "sources": [],
                        "success": True
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

        # Handle pure "list all papers" with no topic filter
        _SHOW_ALL_RE = re.compile(
            r'^(?:list|show|give\s+me|what\s+are|display)\s+(?:all\s+)?(?:the\s+)?papers?\s*'
            r'(?:in\s+(?:my\s+)?(?:the\s+)?library|in\s+chromadb|you\s+have)?\.?\s*$',
            re.I
        )
        if _SHOW_ALL_RE.match(query.strip()):
            listing = self._render_inventory_listing(query, papers_metadata)
            return {
                "query": query,
                "answer": f"Your library contains {len(papers_metadata)} papers:\n\n{listing}",
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
        # Fix 1: Topic-filtered listing - extract topic and filter inventory_metadata
        if query_mode == "listing" and is_simple_inventory_listing(query) and inventory_metadata:
            # Extract what topic they want - support both patterns:
            # - "List papers on SDN" (topic after papers)
            # - "List SDN papers" (topic before papers)
            topic_match = re.search(
                r'\blist\s+(?:only\s+)?papers?\s+(?:about|on|related to|regarding|covering|for|in)?\s*(.+?)[\?\.]?$',
                query, re.I
            )
            if not topic_match:
                # Try pattern where topic comes before "papers": "List SDN papers"
                topic_match = re.search(
                    r'\blist\s+(?:only\s+)?(.+)\s+papers?[\?\.]?$',
                    query, re.I
                )
            if not topic_match:
                # Try pattern with "articles" or "studies": "List SDN articles"
                topic_match = re.search(
                    r'\blist\s+(?:only\s+)?(.+)\s+(?:articles?|studies?)[\?\.]?$',
                    query, re.I
                )
            
            if topic_match:
                topic = topic_match.group(1).strip().lower()
                # Remove common trailing words
                topic = re.sub(r'\b(papers?|articles?|studies?)\s*$', '', topic).strip()
                
                # Special case: if the topic is just "all", don't filter
                if topic == "all":
                    topic_match = None
                else:
                    # Search chunk texts and titles for the topic
                    matched_titles = set()
                    for title, meta in inventory_metadata.items():
                        title_lower = title.lower()
                        if topic in title_lower:
                            matched_titles.add(title)
                            continue
                        # Search actual chunk text
                        chunks = self.vector_store.get_chunks_for_paper(title)
                        for chunk in chunks:
                            if topic in chunk.get('text', '').lower():
                                matched_titles.add(title)
                                break
                    
                    if matched_titles:
                        inventory_metadata = {
                            t: papers_metadata[t]
                            for t in matched_titles
                        }
                    else:
                        return {
                            "query": query,
                            "answer": f"No papers in your library explicitly cover '{topic}'.",
                            "sources": [],
                            "success": True
                        }

            # else: no topic filter, show all (for "list all papers")

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

        # Fix 4: Increase chunk retrieval for deep methodology/dataset/limitation questions
        # These questions require more context to find specific information deep in papers
        deep_content_patterns = [
            r'\b(?:what|which)\s+(?:dataset|data|methodology|method|approach|technique|algorithm|framework|architecture|performance metric|evaluation metric|limitation|weakness|drawback|challenge)\b',
            r'\b(?:how|describe|explain)\s+(?:the\s+)?(?:method|approach|technique|algorithm|framework|architecture)\b',
            r'\b(?:what\s+are\s+the\s+)?(?:limitations|weaknesses|drawbacks|challenges)\b',
        ]
        if any(re.search(pat, query, re.I) for pat in deep_content_patterns):
            effective_limit = max(effective_limit, 10)  # Increase to at least 10 chunks for deep questions

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
        # Detect aggregation queries — must NOT scope to a single paper
        _AGGREGATION_CUES = {
            "across", "library", "all papers", "in general", "most common",
            "overall", "throughout", "collectively", "combined"
        }
        q_lower = query.lower()
        is_aggregation_query = any(cue in q_lower for cue in _AGGREGATION_CUES)
        
        # Enable reranking by default for better retrieval precision
        chunks = retrieve_relevant_chunks(
            self.vector_store,
            query,
            limit=effective_limit,
            filter_title=filter_title,
            scope_titles=None if is_aggregation_query else (matched_titles if matched_titles and scope.entity_kind != "topic" else None),
            use_reranking=True,  # Always enable reranking for better precision
            over_retrieve_multiplier=2.5,  # Reduced from 4.0 — prevents citation drift from irrelevant chunks
        )
        if matched_titles and not filter_title and scope.entity_kind != "topic":
            chunks = filter_chunks_to_titles(chunks, matched_titles)

        # ── Cap chunks before deduplication to prevent CUDA OOM on reranker ──
        # Broad author queries can return 39+ chunks which overflows GPU memory.
        # Limit to 20 chunks — enough for a good answer without OOM.
        if len(chunks) > 20:
            logger.info(f"Capping chunks from {len(chunks)} to 20 to prevent CUDA OOM")
            chunks = chunks[:20]

        # ── Text deduplication to prevent robotic stuttering ─────────────────
        # Remove chunks with >80% text overlap to avoid redundant information
        chunks = self._deduplicate_chunks(chunks)

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
                    chunk_title = (chunk.get("metadata") or {}).get("title", "").lower()
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
                        "answer": IRRELEVANT_REFUSAL,
                        "sources": [],
                        "success": False,
                        "error": "Very low chunk relevance"
                    }

        # ── Step 2: Build context string with doc IDs ────────────────────────
        context_str = self._build_context_with_ids(chunks)
        retrieved_set_summary = self._build_retrieved_set_summary(chunks)

        # ── Step 3: Determine answer mode (confident / partial / refuse) ─────
        mode, partial_notice = self._compute_answer_decision(chunks, query)

        if mode == "refuse":
            logger.warning(f"Answer decision gate: REFUSE for query: {query[:80]}")
            return {
                "query": query,
                "answer": IRRELEVANT_REFUSAL,
                "sources": chunks,
                "success": False,
                "error": "Answer decision gate: refuse mode (no usable evidence).",
            }

        # ── Step 4: Build system prompt ──────────────────────────────────────
        system_prompt = (
            "You are a precise academic research assistant. "
            "Your ONLY knowledge source is the retrieved document passages below. "
            "You must not use any outside knowledge or general training data.\n\n"

            "CITATION RULES (mandatory — no exceptions):\n"
            "- Every factual sentence MUST end with a citation: (doc_N)\n"
            "  where N is the Document ID number from the context block.\n"
            "- Multiple sources for one sentence: (doc_1, doc_3)\n"
            "- NEVER cite a doc_N that does not appear in the Retrieved Document Set.\n"
            "- NEVER invent author names, paper titles, DOIs, venues, or years.\n\n"

            "WHEN THE CONTEXT IS INSUFFICIENT:\n"
            "- If the retrieved passages do not contain enough information to answer "
            "the question, state this explicitly WITHOUT a citation:\n"
            "  'The ingested papers do not contain sufficient information about [topic].'\n"
            "- Do NOT attempt to answer from general knowledge.\n"
            "- Do NOT apologize or add filler. Just state what is and is not present.\n\n"

            "SCOPE RULES (mandatory):\n"
            f"{library_inventory_str}\n\n"
            "- Only reference papers listed in the Library Inventory above.\n"
            "- If the query is about a specific author, ONLY discuss that author's papers.\n"
            "- Do NOT mention papers not listed in the Retrieved Document Set.\n\n"

            "FORMAT RULES:\n"
            "- Be concise. No generic academic filler sentences.\n"
            "- Do NOT generate a References section — one will be appended automatically.\n"
            "- Write in plain prose. Do not use markdown headers inside your answer.\n"
        )

        # ── Step 5: Build user prompt ────────────────────────────────────────
        history_block = ""
        if history_for_llm:
            history_lines = []
            for turn in history_for_llm[-6:]:  # Last 6 turns max
                role = turn.get("role", "user")
                content = (turn.get("content") or "")[:400]
                history_lines.append(f"{role.upper()}: {content}")
            if history_lines:
                history_block = "CONVERSATION HISTORY (for context only):\n" + "\n".join(history_lines) + "\n\n"

        user_prompt = (
            f"{history_block}"
            f"{context_str}\n\n"
            f"{partial_notice + chr(10) + chr(10) if partial_notice else ''}"
            f"Research Question: {query}\n\n"
            f"Answer (cite every factual sentence with doc_N):"
        )

        # ── Step 6: Call Ollama ──────────────────────────────────────────────
        try:
            raw_answer = self._ollama_chat(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return {
                "query": query,
                "answer": "The language model is currently unavailable. Please try again shortly.",
                "sources": chunks,
                "success": False,
                "error": str(e),
            }

        # ── Step 7: Enforce hard grounding rules (strip uncited sentences) ───
        grounded_answer = self._enforce_hard_grounding_rules(raw_answer, chunks)

        if self._is_refusal_answer(grounded_answer):
            return {
                "query": query,
                "answer": grounded_answer,
                "sources": chunks,
                "success": False,
                "error": "Grounding enforcement produced a refusal.",
            }

        # ── Step 8: Replace doc_N placeholders with APA citations ────────────
        final_answer, has_citations = self._bind_citations_and_verify(grounded_answer, chunks)

        # ── Step 9: Strip any model-generated references + append verified ones ─
        papers_meta_for_refs = papers_metadata if papers_metadata else None
        final_answer = self._strip_model_references(final_answer, chunks=chunks)
        verified_refs = self._build_safe_references(chunks, papers_metadata=papers_meta_for_refs)
        if verified_refs:
            final_answer = f"{final_answer}\n\n{verified_refs}"

        # ── Step 10: Post-verification ────────────────────────────────────────
        if not self._is_refusal_answer(final_answer):
            verified, _ = apply_verification_or_refuse(
                final_answer,
                scope=scope,
                papers_metadata=papers_metadata,
                chunks=chunks,
            )
            final_answer = verified

        # ── Step 11: Log faithfulness metrics ────────────────────────────────
        faithfulness_issues = self._check_answer_faithfulness(final_answer, chunks)
        self._log_retrieval_metrics(query, chunks, mode, faithfulness_issues)

        return {
            "query": query,
            "answer": final_answer,
            "sources": chunks,
            "success": True,
        }
