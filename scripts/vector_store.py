"""
vector_store.py — ChromaDB Vector Database Interface.

Manages all interactions with the local ChromaDB persistent vector database:
  - Initialises the ChromaDB client and sentence-transformer embedding function on startup.
  - Upserts text chunks (with metadata) for ingested papers.
  - Performs cosine-similarity vector search for RAG retrieval.
  - Returns collection statistics (chunk counts, paper list).

ChromaDB runs entirely locally — no cloud or internet connection required.
Embeddings are generated using BAAI/bge-base-en-v1.5 (requires re-ingestion after model change).
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
      2. Set up a sentence-transformer embedding function using BAAI/bge-base-en-v1.5.
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

            # ── Sentence-Transformer Embedding Function ─────────────────────────────
            # BAAI/bge-base-en-v1.5 is a high-performance sentence-transformer model
            # that produces 768-dimensional dense vector embeddings.
            # This model requires re-ingestion of all papers after switching from ONNXMiniLM_L6_V2.
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-base-en-v1.5",
                device="cpu"
            )

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

            # Step 2b: Run startup database metadata normalization migration only once
            # Use a flag file to track if normalization has been done
            normalization_flag = Path(settings.VECTOR_DB_DIR) / ".metadata_normalized"
            if not normalization_flag.exists():
                self._normalize_stored_metadata()
                normalization_flag.touch()
                logger.info("Metadata normalization completed and flag file created.")
            else:
                logger.info("Metadata normalization already completed (flag file exists).")

            logger.info(
                "ChromaDB client and sentence-transformer embedding function initialised successfully. "
                f"Collection contains {self.collection.count()} chunks."
            )

        except Exception as e:
            logger.error(f"Failed to initialise ChromaDB: {e}")
            raise  # Critical failure — let the caller (server.py / main.py) handle it

    def _normalize_stored_metadata(self) -> None:
        """
        One-time startup check to clean up newlines and consecutive spaces
        in stored chunk titles and metadata inside ChromaDB.
        """
        try:
            total_chunks = self.collection.count()
            if total_chunks == 0:
                return
                
            data = self.collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
            ids = data.get("ids", [])
            
            updates_ids = []
            updates_metadatas = []
            
            for idx, meta in enumerate(metadatas):
                if not meta:
                    continue
                changed = False
                
                # Normalize title
                if "title" in meta:
                    orig_title = meta["title"]
                    clean_title = re.sub(r"\s+", " ", orig_title.strip())
                    if clean_title != orig_title:
                        meta["title"] = clean_title
                        changed = True
                        
                # Normalize authors
                if "authors" in meta:
                    orig_authors = meta["authors"]
                    clean_authors = re.sub(r"\s+", " ", orig_authors.strip())
                    if clean_authors != orig_authors:
                        meta["authors"] = clean_authors
                        changed = True
                        
                # Normalize venue
                if "venue" in meta:
                    orig_venue = meta["venue"]
                    clean_venue = re.sub(r"\s+", " ", orig_venue.strip())
                    if clean_venue != orig_venue:
                        meta["venue"] = clean_venue
                        changed = True
                        
                if changed:
                    updates_ids.append(ids[idx])
                    updates_metadatas.append(meta)
                    
            if updates_ids:
                logger.info(f"Normalizing stored metadata for {len(updates_ids)} chunks in ChromaDB...")
                # Update in batches of 500 to avoid limits
                for i in range(0, len(updates_ids), 500):
                    batch_ids = updates_ids[i:i+500]
                    batch_metas = updates_metadatas[i:i+500]
                    self.collection.update(ids=batch_ids, metadatas=batch_metas)
                logger.info("ChromaDB metadata normalization complete.")
        except Exception as e:
            logger.warning(f"Metadata normalization migration failed: {e}")

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
        chunks: list[dict],
        authors: str | None = None,
        year: int | None = None,
        venue: str | None = None,
        domain: str | None = None
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
            authors: Formatted string of authors. May be None.
            year: Publication year. May be None.
            venue: Publication venue (journal/conference name). May be None.
            domain: Research domain (e.g., "computer_science", "medical_imaging"). May be None.

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
                "authors": authors or "Unknown Authors",
                "year": str(year) if year else "N/A",
                "venue": venue or "N/A",
                "domain": domain or "unknown",
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

    def delete_paper(self, title: str, doi: str | None = None) -> bool:
        """
        Delete all chunks associated with a specific paper in ChromaDB.
        Uses title or doi as criteria.
        """
        try:
            # Delete by DOI if available (more precise), otherwise by title
            if doi and doi != "N/A":
                self.collection.delete(where={"doi": doi})
                logger.info(f"Deleted paper chunks from ChromaDB for DOI: '{doi}'")
            else:
                self.collection.delete(where={"title": title})
                logger.info(f"Deleted paper chunks from ChromaDB for Title: '{title}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete paper chunks from ChromaDB: {e}")
            return False

    def update_paper_metadata(self, title: str, authors: str, year: int | str, doi: str | None = None, venue: str | None = None, new_title: str | None = None) -> bool:
        """
        Update metadata (authors, year, doi, venue, title) in-place for all chunks of a paper.
        """
        try:
            # Retrieve all chunks belonging to this paper title
            data = self.collection.get(where={"title": title}, include=["metadatas"])
            if not data or not data.get("ids"):
                logger.warning(f"No chunks found in ChromaDB to update metadata for paper title: '{title}'")
                return False

            ids = data["ids"]
            metadatas = data["metadatas"]
            
            # Update metadata fields using dictionary merge to avoid overwriting existing fields with None
            new_fields = {}
            if authors is not None:
                new_fields["authors"] = authors
            if year is not None:
                new_fields["year"] = str(year)
            if venue and venue != "N/A":
                new_fields["venue"] = venue
            if doi and doi != "N/A":
                new_fields["doi"] = doi
            if new_title:
                new_fields["title"] = new_title

            for meta in metadatas:
                meta.update(new_fields)

            # Batch update in ChromaDB
            self.collection.update(ids=ids, metadatas=metadatas)
            logger.info(f"Successfully updated metadata in ChromaDB for '{title}' ({len(ids)} chunks).")
            return True
        except Exception as e:
            logger.error(f"Failed to update paper metadata in ChromaDB for '{title}': {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Vector Similarity Search
    # ──────────────────────────────────────────────────────────────────────────

    def query_similar_chunks(
        self,
        query: str,
        limit: int = 4,
        filter_title: str | None = None,
        filter_domain: str | None = None
    ) -> list[dict]:
        """
        Perform a cosine-similarity vector search across all ingested paper chunks.

        The query string is embedded using the same ONNX model used during ingestion,
        and the top-K nearest vectors (by cosine distance) are returned.

        Lower distance = higher similarity (ChromaDB uses distance, not score).

        Args:
            query: The researcher's natural language question or search phrase.
            limit: Maximum number of similar chunks to return (default: 4).
            filter_title: If set, restricts search to chunks from this paper title only.
            filter_domain: If set, restricts search to chunks from this research domain only.

        Returns:
            List of result dicts, each containing:
              - "id": The vector ID string
              - "text": The chunk text content
              - "metadata": dict with title, doi, pages, char offsets, length, domain
              - "distance": Float cosine distance (lower = more similar, range: 0–2)
        """
        logger.info(
            f"Querying ChromaDB: '{query}' (top {limit} chunks, "
            f"filter_title={filter_title!r}, filter_domain={filter_domain!r})"
        )

        try:
            # Build optional where clause to restrict to a specific paper or domain
            where_conditions = {}
            if filter_title:
                where_conditions["title"] = filter_title
            if filter_domain:
                where_conditions["domain"] = filter_domain

            where_clause = where_conditions if where_conditions else None

            # query() accepts a list of query texts (we always pass exactly one)
            query_kwargs = {
                "query_texts": [query],
                "n_results": limit
            }
            if where_clause:
                query_kwargs["where"] = where_clause

            results = self.collection.query(**query_kwargs)

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

    def get_chunks_for_paper(self, paper_title: str, max_chunks: int = 30) -> list[dict]:
        """
        Fetch stored chunks for one paper by exact metadata title match.
        Used when the user names an author/paper in the library inventory but vector
        search + lexical filters would otherwise return nothing.
        """
        if not paper_title:
            return []
        try:
            # Try exact match first
            data = self.collection.get(
                where={"title": paper_title},
                include=["documents", "metadatas"],
                limit=max_chunks,
            )
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            
            # If exact match returns no results, try case-insensitive match via title_lower field
            if not docs:
                data = self.collection.get(
                    where={"title_lower": paper_title.lower().strip()},
                    include=["documents", "metadatas"],
                    limit=max_chunks,
                )
                ids = data.get("ids") or []
                docs = data.get("documents") or []
                metas = data.get("metadatas") or []
            
            results = []
            for i, doc_text in enumerate(docs):
                if not doc_text:
                    continue
                results.append({
                    "id": ids[i] if i < len(ids) else f"paper_{i}",
                    "text": doc_text,
                    "metadata": metas[i] if i < len(metas) else {},
                    # Direct inventory fetch — treat as highly relevant to the named paper.
                    "distance": 0.0,
                })
            logger.info(
                "Loaded %s chunk(s) from inventory for paper '%s'.",
                len(results),
                paper_title[:80],
            )
            return results
        except Exception as e:
            logger.error("Failed to load chunks for paper '%s': %s", paper_title, e)
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Collection Statistics
    # ──────────────────────────────────────────────────────────────────────────

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current ChromaDB collection.

        Fetches all metadata entries to identify distinct paper titles.
        Used by the health check endpoint and the web UI's stats badges.

        Step 2a: Added 30-second cache to avoid loading ALL metadata on every query.

        Returns:
            Dict with:
              - "total_chunks" (int): Total number of vectors stored.
              - "total_papers" (int): Number of unique paper titles.
              - "papers_list" (list[str]): Sorted list of unique paper titles.
              - "papers_metadata" (dict): Mapping from paper title (str) to dict containing 'authors', 'year', and 'doi'.
        """
        import time
        now = time.time()
        if hasattr(self, '_stats_cache') and (now - self._stats_cache_time) < 30:
            return self._stats_cache

        try:
            total_chunks = self.collection.count()
            unique_papers = set()
            papers_metadata = {}

            if total_chunks > 0:
                # Fetch only metadata (not document text) to keep the response small
                data = self.collection.get(include=["metadatas"])
                for meta in data.get("metadatas", []):
                    if meta and "title" in meta:
                        title = meta["title"]
                        unique_papers.add(title)
                        if title not in papers_metadata:
                            papers_metadata[title] = {
                                "authors": meta.get("authors", "Unknown Authors"),
                                "year": meta.get("year", "N/A"),
                                "doi": meta.get("doi", "N/A"),
                                "venue": meta.get("venue", "N/A")
                            }

            result = {
                "total_chunks": total_chunks,
                "total_papers": len(unique_papers),
                "papers_list": sorted(list(unique_papers)),
                "papers_metadata": papers_metadata
            }
            self._stats_cache = result
            self._stats_cache_time = now
            return result

        except Exception as e:
            logger.error(f"Failed to fetch ChromaDB collection stats: {e}")
            # Return safe defaults so the health check endpoint doesn't crash
            return {
                "total_chunks": 0,
                "total_papers": 0,
                "papers_list": [],
                "papers_metadata": {}
            }
