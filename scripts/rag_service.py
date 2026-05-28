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
        stripped = re.split(r"\n\s*references\s*:\s*\n", answer, maxsplit=1, flags=re.IGNORECASE)[0]
        return stripped.strip()

    def _is_unverifiable_sensitive_claim(self, query: str, chunks: list[dict]) -> bool:
        """
        Detect high-risk stance/position questions where retrieved context contains
        no lexical evidence for the sensitive topic, and force a safe refusal.
        """
        q = (query or "").lower()
        sensitive_terms = [
            "abortion", "reproductive rights", "pro-choice", "pro life",
            "supports", "opposes", "stance", "position", "views on",
        ]
        if not any(term in q for term in sensitive_terms):
            return False
        combined = " ".join((c.get("text") or "").lower() for c in chunks)
        topic_present = any(term in combined for term in ["abortion", "reproductive", "pro-choice", "pro life"])
        return not topic_present

    def generate_answer(
        self,
        query: str,
        limit: int = 4,
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
        # Fetch library index to support meta-queries
        stats = self.vector_store.get_collection_stats()
        papers_metadata = stats.get("papers_metadata", {})

        # ── Step 1: Retrieve chunks and drop low-similarity (off-topic) matches ──
        chunks = retrieve_relevant_chunks(
            self.vector_store, query, limit=limit, filter_title=filter_title
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

        library_inventory_str = build_library_inventory(papers_metadata)
        context_str = chunks_to_context_string(chunks)

        # ── Step 3: Build structured prompts ──────────────────────────────────
        # Note if the query is scoped to a specific paper
        scope_note = ""
        if filter_title:
            scope_note = (
                f"NOTE: This query is scoped to a SINGLE paper: \"{filter_title}\". "
                "Only use information from this paper's context blocks when answering.\n"
            )

        # System prompt: sets the LLM's role and strict citation rules
        system_prompt = (
            "You are a professional, self-hosted academic AI research assistant.\n"
            "Your task is to answer the researcher's query based strictly on the provided Document context blocks and the Ingested Paper Library Inventory.\n\n"
            f"{scope_note}"
            "STRICT CITATION AND WRITING RULES — you MUST follow ALL of these:\n"
            "1. ONLY use facts from the provided Document context blocks or Library Inventory. No pre-trained knowledge or invented details.\n"
            "2. INLINE CITATIONS: Always use APA7 parenthetical format: (Author, Year) or (Author & Author, Year) or (Author et al., Year). \n"
            "   If citing a specific passage, add the page: (Author, Year, p. X).\n"
            "3. NEVER use bracketed source numbers like '(Source 1)', '[Source 2]', 'Document Source 1', 'Document 1' etc. in the text. These are internal labels only.\n"
            "   ALWAYS convert internal source labels to proper (Author, Year) citations using the Authors and Year in each Document block.\n"
            "4. NEVER refer to papers as 'Paper A', 'Paper B', 'Study 1', 'Study 2', or any similar generic label. \n"
            "   Always identify papers by: their EXACT title in quotes, or using (Author, Year) notation.\n"
            "5. REFERENCES SECTION: End your response with a 'References' section listing all cited papers in full APA7 bibliography format:\n"
            "   Author, A. A., & Author, B. B. (Year). Title of article. Journal Name, volume(issue), pages. https://doi.org/xxxxx\n"
            "6. If asked about an author or paper not in the context or Inventory, respond EXACTLY: \n"
            "   'I could not find any relevant papers or context in the local database to answer your question. Please ingest papers first.'\n"
            "7. If context lacks sufficient detail, state that clearly, then summarise what the context does say.\n"
            "8. Maintain a formal, neutral, and academic tone throughout.\n"
            "9. Do NOT infer personal stances (politics, religion, abortion, legal or moral views) unless directly stated in retrieved context.\n"
            "10. Never fabricate citations, references, or source details."
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
                "temperature": 0.2,  # Low temperature = more deterministic, less creative
                "top_p": 0.9         # Nucleus sampling threshold
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
