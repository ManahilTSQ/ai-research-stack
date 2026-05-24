"""
vector_store.py — ChromaDB Vector Database Interface.

Manages all interactions with the local ChromaDB persistent vector database:
  - Initialises the ChromaDB client and ONNX embedding function on startup.
  - Upserts text chunks (with metadata) for ingested papers.
  - Performs cosine-similarity vector search for RAG retrieval.
  - Returns collection statistics (chunk counts, paper list).

ChromaDB runs entirely locally — no cloud or internet connection required.
Embeddings are generated using a bundled ONNX MiniLM model (also local/offline).
"""

import re
import logging
import chromadb
from chromadb.utils import embedding_functions
from config import settings   # Flat import — scripts/ is on sys.path


# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Service to manage the ChromaDB persistent local vector database.

    Responsibilities:
      1. Initialise a persistent ChromaDB client at the vectordb/ path.
      2. Set up a local ONNX-based embedding function (no GPU, no internet).
      3. Get-or-create the "research_papers" collection with cosine distance.
      4. Provide upsert (add/update) and query (similarity search) operations.
      5. Return collection health statistics.

    The collection uses cosine similarity distance (hnsw:space = "cosine"),
    which is the standard metric for comparing text embedding vectors.
    """

    def __init__(self):
        """
        Initialise the ChromaDB client and embedding function.

        Raises:
            Exception: If ChromaDB cannot initialise at the configured path.
        """
        logger.info(f"Initialising ChromaDB persistent client at: {settings.VECTOR_DB_DIR}")

        try:
            # ── ChromaDB Persistent Client ────────────────────────────────────
            # PersistentClient saves the entire database to disk at the given path.
            # The database survives process restarts automatically.
            self.client = chromadb.PersistentClient(path=str(settings.VECTOR_DB_DIR))

            # ── Local ONNX Embedding Function ─────────────────────────────────
            # ONNXMiniLM_L6_V2 is a bundled sentence-transformer model that runs
            # using onnxruntime — no GPU, no internet, no separate download needed.
            # It produces 384-dimensional dense vector embeddings.
            self.embedding_function = embedding_functions.ONNXMiniLM_L6_V2()

            # ── Collection Setup ──────────────────────────────────────────────
            # get_or_create_collection is idempotent — safe to call on every startup.
            # hnsw:space = "cosine" sets cosine similarity as the distance metric.
            # Cosine similarity is recommended for normalised text embeddings because
            # it measures angular distance (topic similarity) rather than magnitude.
            self.collection = self.client.get_or_create_collection(
                name="research_papers",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(
                "ChromaDB client and ONNX embedding function initialised successfully. "
                f"Collection contains {self.collection.count()} chunks."
            )

        except Exception as e:
            logger.error(f"Failed to initialise ChromaDB: {e}")
            raise  # Critical failure — let the caller (server.py / main.py) handle it

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Slug Generation
    # ──────────────────────────────────────────────────────────────────────────

    def _slugify(self, text: str) -> str:
        """
        Convert an arbitrary string into a safe slug for use as a ChromaDB vector ID.

        ChromaDB vector IDs must not contain special characters.
        This replaces anything that isn't alphanumeric, underscore, or hyphen
        with an underscore, then lowercases the result.

        Args:
            text: Input string (e.g. a DOI or paper title).

        Returns:
            A filesystem-and-DB-safe lowercase slug string.
        """
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", text).lower()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Add or Update Paper Chunks
    # ──────────────────────────────────────────────────────────────────────────

    def add_paper_chunks(
        self,
        paper_title: str,
        doi: str | None,
        chunks: list[dict]
    ) -> bool:
        """
        Upsert (insert or update) all text chunks for a single paper into ChromaDB.

        "Upsert" means: if a vector with the same ID already exists it is updated;
        if it does not exist it is inserted. This makes the operation idempotent —
        re-ingesting the same paper will update rather than duplicate its chunks.

        Vector ID format: <paper_slug>_chunk_<index>
          e.g. "10_48550_arxiv_1706_03762_chunk_0"

        ChromaDB metadata values must be primitive types (str, int, float, bool).
        Lists are serialised to comma-separated strings where required.

        Args:
            paper_title: Human-readable title of the paper.
            doi: DOI identifier (used to generate stable IDs). May be None.
            chunks: List of chunk dicts as produced by PDFProcessorService.chunk_text().

        Returns:
            True on successful upsert, False if no chunks were provided or on error.
        """
        if not chunks:
            logger.warning(f"No chunks provided for paper: '{paper_title}' — skipping.")
            return False

        # Generate a stable, unique identifier for this paper
        # Prefer DOI (globally unique) over title (may have duplicates or typos)
        paper_id = self._slugify(doi) if doi else self._slugify(paper_title[:40])
        logger.info(f"Upserting {len(chunks)} chunks for '{paper_title}' (ID: {paper_id})")

        # Prepare parallel lists for the ChromaDB batch upsert call
        ids = []           # Unique string ID for each vector
        documents = []     # The text content to embed
        metadatas = []     # Searchable metadata attached to each vector

        for chunk in chunks:
            chunk_idx = chunk["chunk_index"]
            chunk_text = chunk["text"]
            chunk_meta = chunk["metadata"]

            # Unique vector ID: paper slug + chunk index
            vector_id = f"{paper_id}_chunk_{chunk_idx}"

            # Metadata is stored alongside the vector for display in search results.
            # All values must be primitive types — convert list to comma-separated string.
            metadata = {
                "title": paper_title,
                "doi": doi or "N/A",
                # Convert the list of page numbers to a comma-separated string
                "pages": ",".join(map(str, chunk_meta.get("pages", []))),
                "char_start": int(chunk_meta.get("char_start", 0)),
                "char_end": int(chunk_meta.get("char_end", 0)),
                "length": int(chunk_meta.get("length", len(chunk_text)))
            }

            ids.append(vector_id)
            documents.append(chunk_text)
            metadatas.append(metadata)

        try:
            # Perform a single atomic batch upsert for all chunks of this paper.
            # ChromaDB will embed all documents in parallel using the ONNX model.
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(
                f"Successfully upserted {len(chunks)} chunks for '{paper_title}' into ChromaDB."
            )
            return True

        except Exception as e:
            logger.error(f"ChromaDB upsert failed for '{paper_title}': {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Vector Similarity Search
    # ──────────────────────────────────────────────────────────────────────────

    def query_similar_chunks(self, query: str, limit: int = 4) -> list[dict]:
        """
        Perform a cosine-similarity vector search across all ingested paper chunks.

        The query string is embedded using the same ONNX model used during ingestion,
        and the top-K nearest vectors (by cosine distance) are returned.

        Lower distance = higher similarity (ChromaDB uses distance, not score).

        Args:
            query: The researcher's natural language question or search phrase.
            limit: Maximum number of similar chunks to return (default: 4).

        Returns:
            List of result dicts, each containing:
              - "id": The vector ID string
              - "text": The chunk text content
              - "metadata": dict with title, doi, pages, char offsets, length
              - "distance": Float cosine distance (lower = more similar, range: 0–2)
        """
        logger.info(f"Querying ChromaDB: '{query}' (top {limit} chunks)")

        try:
            # query() accepts a list of query texts (we always pass exactly one)
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )

            formatted_results = []

            # ChromaDB returns results as nested lists (one list per query text)
            # We always send exactly one query, so we index [0] throughout.
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                ids = results["ids"][0]
                metadatas = results["metadatas"][0]
                # distances may be absent for older ChromaDB versions — default to 0.0
                distances = (
                    results["distances"][0]
                    if "distances" in results
                    else [0.0] * len(docs)
                )

                for i in range(len(docs)):
                    formatted_results.append({
                        "id": ids[i],
                        "text": docs[i],
                        "metadata": metadatas[i],
                        "distance": float(distances[i])
                    })

            logger.info(f"Retrieved {len(formatted_results)} matching chunks.")
            return formatted_results

        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Collection Statistics
    # ──────────────────────────────────────────────────────────────────────────

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current ChromaDB collection.

        Fetches all metadata entries to identify distinct paper titles.
        Used by the health check endpoint and the web UI's stats badges.

        Returns:
            Dict with:
              - "total_chunks" (int): Total number of vectors stored.
              - "total_papers" (int): Number of unique paper titles.
              - "papers_list" (list[str]): Sorted list of unique paper titles.
        """
        try:
            total_chunks = self.collection.count()
            unique_papers = set()

            if total_chunks > 0:
                # Fetch only metadata (not document text) to keep the response small
                data = self.collection.get(include=["metadatas"])
                for meta in data.get("metadatas", []):
                    if meta and "title" in meta:
                        unique_papers.add(meta["title"])

            return {
                "total_chunks": total_chunks,
                "total_papers": len(unique_papers),
                "papers_list": sorted(list(unique_papers))
            }

        except Exception as e:
            logger.error(f"Failed to fetch ChromaDB collection stats: {e}")
            # Return safe defaults so the health check endpoint doesn't crash
            return {
                "total_chunks": 0,
                "total_papers": 0,
                "papers_list": []
            }
