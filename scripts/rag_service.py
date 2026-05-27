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

    def generate_answer(self, query: str, limit: int = 4, filter_title: str | None = None) -> dict:
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

        # ── Step 1: Retrieve relevant context chunks from ChromaDB ─────────────
        chunks = self.vector_store.query_similar_chunks(query, limit=limit, filter_title=filter_title)

        if not chunks and not papers_metadata:
            # No papers at all in database — return a polite refusal
            logger.warning("No relevant chunks and no papers found in ChromaDB.")
            return {
                "query": query,
                "answer": (
                    "I could not find any relevant papers or context in the local database "
                    "to answer your question. Please ingest papers first."
                ),
                "sources": [],
                "success": False,
                "error": "No matching papers in the vector database."
            }

        # Format Library Inventory string
        library_inventory_blocks = []
        for i, (title, meta) in enumerate(papers_metadata.items()):
            library_inventory_blocks.append(
                f"- Paper {i+1}: \"{title}\" | Authors: {meta.get('authors', 'Unknown Authors')} | Year: {meta.get('year', 'N/A')} | DOI: {meta.get('doi', 'N/A')}"
            )
        library_inventory_str = "\n".join(library_inventory_blocks) if library_inventory_blocks else "No papers in database library."

        # ── Step 2: Format context blocks for the prompt ───────────────────────
        context_blocks = []
        for idx, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            title = meta.get("title", "Untitled Paper")
            authors = meta.get("authors", "Unknown Authors")
            year = meta.get("year", "N/A")
            doi = meta.get("doi", "N/A")
            pages = meta.get("pages", "N/A")
            text = chunk["text"]

            # Detailed structured metadata format for LLM reference
            block = (
                f'Document [Source {idx + 1}]:\n'
                f'Title: "{title}"\n'
                f'Authors: {authors}\n'
                f'Year: {year}\n'
                f'DOI: {doi}\n'
                f'Pages: {pages}\n'
                f'Content: {text}\n'
            )
            context_blocks.append(block)

        # Join all blocks into one context string for the prompt
        context_str = "\n".join(context_blocks) if context_blocks else "No relevant text passage chunks found for this query."

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
            "Rules:\n"
            "1. Use ONLY facts stated in the provided Document context blocks or the Ingested Paper Library Inventory. Do NOT use your pre-trained knowledge or make up any details.\n"
            "2. Cite your sources inline using APA7 style, for example: (Hassan, 2020) or (Smith & Jones, 2018, p. 12). If citing specific pages, use the Pages metadata from the Document block (e.g. p. 45).\n"
            "3. Do NOT use bracketed source numbers like '[Source 1]' or 'Source 1' in your inline citations. Convert them to proper (Author, Year) citations using the Authors and Year metadata provided in each Document block.\n"
            "4. At the end of your response, you MUST compile a 'References' section containing all the papers you cited in your answer. Format each reference in proper APA7 bibliography style, using the Authors, Year, Title, and DOI/URL if available in the metadata.\n"
            "5. If a query asks about a named author or paper that does not appear in the context blocks or Library Inventory, respond with EXACTLY: 'I could not find any relevant papers or context in the local database to answer your question. Please ingest papers first.'\n"
            "6. If the context blocks are on the right general topic but lack sufficient detail for the specific question, honestly state that the available ingested papers do not contain enough detail on that aspect, then summarise what the context DOES say on the topic.\n"
            "7. Maintain a formal, neutral, and academic tone throughout."
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
