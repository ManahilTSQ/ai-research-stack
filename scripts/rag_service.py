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
import requests
from config import settings            # Flat import — scripts/ is on sys.path
from vector_store import VectorStoreService  # Local ChromaDB interface


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

    def generate_answer(self, query: str, limit: int = 4) -> dict:
        """
        Execute the complete RAG pipeline for a researcher's query.

        Steps:
          1. Retrieve top-K relevant chunks from ChromaDB using cosine similarity.
          2. Format each chunk into a labelled context block with source attribution.
          3. Build a structured system prompt instructing the LLM on citation rules.
          4. Send both prompts to Ollama's /api/chat endpoint.
          5. Return the structured response dict.

        Args:
            query: The research question to answer.
            limit: Number of context chunks to retrieve from ChromaDB (default: 4).

        Returns:
            Dict with keys:
              - "query" (str): The original question.
              - "answer" (str): The LLM-generated grounded answer.
              - "sources" (list): The retrieved chunks used as context.
              - "success" (bool): Whether generation succeeded.
              - "error" (str, optional): Error message if success=False.
        """
        # ── Step 1: Retrieve relevant context chunks from ChromaDB ─────────────
        chunks = self.vector_store.query_similar_chunks(query, limit=limit)

        if not chunks:
            # No relevant context found — return a polite refusal rather than hallucinating
            logger.warning("No relevant chunks found in ChromaDB for this query.")
            return {
                "query": query,
                "answer": (
                    "I could not find any relevant papers or context in the local database "
                    "to answer your question. Please ingest papers first."
                ),
                "sources": [],
                "success": False,
                "error": "No matching context in the vector database."
            }

        # ── Step 2: Format context blocks for the prompt ───────────────────────
        # Each chunk is labelled [Source N] so the LLM can reference it by number.
        context_blocks = []
        for idx, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            title = meta.get("title", "Untitled Paper")
            pages = meta.get("pages", "N/A")
            text = chunk["text"]

            # Format: [Source 1] "Paper Title" (Pages: 3,4)\nContent: <text>
            block = (
                f'[Source {idx + 1}] "{title}" (Pages: {pages})\n'
                f"Content: {text}\n"
            )
            context_blocks.append(block)

        # Join all blocks into one context string for the prompt
        context_str = "\n".join(context_blocks)

        # ── Step 3: Build structured prompts ──────────────────────────────────
        # System prompt: sets the LLM's role and strict citation rules
        system_prompt = (
            "You are a professional, self-hosted academic AI research assistant. "
            "Your task is to answer the researcher's query based strictly on the provided context blocks from ingested documents.\n\n"
            "Rules:\n"
            "1. Use ONLY facts stated in the provided context blocks. Do NOT use your pre-trained knowledge or make up any details.\n"
            "2. For every claim you make, cite the source number (e.g. [Source 1]) and page numbers.\n"
            "3. If a query asks about a named author (e.g. Hassan, Smith) or a specific paper title that does NOT appear anywhere in the context blocks, respond with EXACTLY: 'I could not find any relevant papers or context in the local database to answer your question. Please ingest papers first.'\n"
            "4. If the context blocks are on the right general topic but lack sufficient detail for the specific question, honestly state that the available ingested papers do not contain enough detail on that aspect, then summarise what the context DOES say on the topic.\n"
            "5. Maintain a formal, neutral, and academic tone throughout."
        )

        # User prompt: the context blocks + the actual research question
        user_prompt = (
            f"Context from ingested research papers:\n"
            f"{'─' * 80}\n"
            f"{context_str}\n"
            f"{'─' * 80}\n\n"
            f"Researcher Query: {query}\n\n"
            "Provide your structured academic answer below "
            "(remember to inline cite sources and page numbers):"
        )

        # ── Step 4: Send to Ollama /api/chat ──────────────────────────────────
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
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
