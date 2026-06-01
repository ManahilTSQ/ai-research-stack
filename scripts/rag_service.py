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
    EMPTY_DB_REFUSAL,
    IRRELEVANT_REFUSAL,
    NOT_IN_LIBRARY_REFUSAL,
)


# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


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
            "\n❌  Ollama is not running or unreachable.\n"
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

        # Early health check — exit immediately if Ollama is not available
        if not check_ollama_health():
            print("    RAG service cannot start without a working Ollama instance.\n")
            sys.exit(1)

        # Initialise the vector store client (loads ChromaDB from disk)
        self.vector_store = VectorStoreService()

        logger.info("RAG Service initialised successfully.")

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
            key = (title.lower(), authors.lower(), year, doi.lower())
            if key in seen:
                continue
            seen.add(key)
            doi_suffix = f" https://doi.org/{doi}" if doi and doi != "N/A" else ""
            refs.append(f"- {authors} ({year}). {title}.{doi_suffix}")
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

        # Fetch library index to support meta-queries
        stats = self.vector_store.get_collection_stats()
        papers_metadata = stats.get("papers_metadata", {})

        # ── Detect author-scoped and listing/tabulation queries ─────────────
        # When the query names a specific author, we:
        #   a) Filter the library inventory to only that author's papers so the LLM
        #      cannot accidentally reference other authors' works.
        #   b) Scale up the retrieval limit so every paper gets at least one chunk.
        matched_titles = resolve_matching_paper_titles(query, papers_metadata)
        _listing_kw = (
            "list", "table", "tabulate", "extract", "all paper",
            "each paper", "for each", "structured", "enumerate",
        )
        is_listing_query = any(kw in query.lower() for kw in _listing_kw)

        # Build an author-filtered inventory when an author is identified
        inventory_metadata = papers_metadata
        if matched_titles and not filter_title:
            filtered = {t: papers_metadata[t] for t in matched_titles if t in papers_metadata}
            if filtered:
                inventory_metadata = filtered

        # For listing queries over many papers, give each paper at least 3 chunks
        effective_limit = limit
        if is_listing_query and matched_titles:
            effective_limit = max(limit, len(matched_titles) * 3, 24)

        # ── Step 1: Retrieve chunks ───────────────────────────────────────────
        chunks = retrieve_relevant_chunks(
            self.vector_store, query, limit=effective_limit, filter_title=filter_title
        )

        if not papers_metadata:
            logger.warning("No papers found in ChromaDB.")
            return {
                "query": query,
                "answer": EMPTY_DB_REFUSAL,
                "sources": [],
                "success": False,
                "error": "No matching papers in the vector database.",
            }

        # Papers exist but nothing in the corpus is similar enough to the query.
        if not chunks:
            logger.warning("No chunks passed relevance threshold for query: %s", query)
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
        context_str = chunks_to_context_string(chunks)

        # ── Step 3: Build structured prompts ──────────────────────────────────
        # Note if the query is scoped to a specific paper
        scope_note = ""
        if filter_title:
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
            "=== HARD REFUSAL TRIGGERS — ALWAYS REFUSE THESE, NO EXCEPTIONS ===\n"
            "- Cooking, recipes, food → REFUSE\n"
            "- Medical advice, health, symptoms → REFUSE (unless paper is medical research)\n"
            "- News, weather, current events → REFUSE\n"
            "- Any question where the answer requires knowledge NOT in the context blocks or the Ingested Paper Library Inventory → REFUSE\n"
            "- Any question about a person, paper, or concept not found in the Library Inventory or context blocks → REFUSE\n\n"
            "REFUSAL FORMAT (copy this exactly when refusing):\n"
            "\"This question is outside the scope of your ingested research knowledge base. "
            "I can only answer questions based on the papers that have been ingested. "
            "Please ask a question about the research papers in your library.\"\n\n"
            "=== WHAT YOU ARE ALLOWED TO DO ===\n"
            "- Answer research questions strictly using the Document Context Blocks or Ingested Paper Library Inventory provided.\n"
            "- Summarize, compare, or explain content that is EXPLICITLY present in the context or library inventory.\n"
            "- List authors, years, titles, DOIs only from the Library Inventory or context.\n"
            "- If a query asks to list, count, or tabulate papers/articles in the library, use the Ingested Paper Library Inventory to answer exhaustively.\n"
            "- You MUST list ALL relevant papers and articles. Do NOT truncate lists or tables with '... (remaining papers)' or similar placeholders. You must output all rows exhaustively without omitting any records.\n\n"
            "=== CITATION RULES ===\n"
            "1. ONLY use facts from the Document Context Blocks or the Ingested Paper Library Inventory. Zero exceptions.\n"
            "2. NEVER invent author names, paper titles, years, DOIs or references.\n"
            "3. Use APA7 inline citations: (Author, Year) or (Author et al., Year, p. X).\n"
            "4. NEVER use (Source 1), [Document 2] or any numbered source labels.\n"
            "5. NEVER call papers 'Paper A', 'Study 1' etc. Use (Author, Year) or exact title.\n"
            "6. End every answer with a References section in full APA7 format.\n"
            "7. TRUTH GAPS: If the concept asked about is NOT in the context blocks or library inventory, "
            "say: 'The retrieved context does not contain information about [topic].' DO NOT guess.\n"
            "8. If context is insufficient, say so explicitly.\n"
            "9. Formal, neutral academic tone always.\n\n"
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
        if conversation_history:
            for turn in conversation_history[-12:]:
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
                else:
                    # Keep model narrative, but replace free-form references with deterministic ones.
                    answer = self._strip_model_references(answer)
                    safe_refs = self._build_safe_references(chunks)
                    if safe_refs:
                        answer = f"{answer}\n\n{safe_refs}"
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
