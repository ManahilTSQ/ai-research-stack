"""
context_builder.py — Context Builder Service for RAG.

Cleans, compresses, and structures retrieved chunks before sending to the LLM.
Performs deduplication, removes irrelevant sentences, and attaches metadata.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds clean, structured context from retrieved chunks.
    """

    def __init__(self):
        """Initialize the context builder."""
        logger.info("Context builder initialized")

    def deduplicate_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Remove duplicate chunks based on text similarity.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Deduplicated list of chunks.
        """
        if not chunks:
            return []
        
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', text.lower())
            
            if normalized and normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_chunks.append(chunk)
        
        if len(unique_chunks) < len(chunks):
            logger.info(f"Deduplicated: {len(chunks)} → {len(unique_chunks)} chunks")
        
        return unique_chunks

    def remove_irrelevant_sentences(
        self,
        chunk_text: str,
        query_terms: list[str]
    ) -> str:
        """
        Remove sentences from a chunk that don't contain query-relevant terms.

        Args:
            chunk_text: The chunk text to filter.
            query_terms: Key terms from the query.

        Returns:
            Filtered chunk text.
        """
        if not query_terms:
            return chunk_text
        
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
        
        # Convert query terms to lowercase for matching
        query_terms_lower = [t.lower() for t in query_terms]
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains at least one query term
            if any(term in sentence_lower for term in query_terms_lower):
                relevant_sentences.append(sentence)
        
        # If no sentences match, keep the original (better to have context than none)
        if not relevant_sentences:
            return chunk_text
        
        return " ".join(relevant_sentences)

    def compress_chunk(self, chunk: dict[str, Any], max_length: int = 800) -> dict[str, Any]:
        """
        Compress a chunk to fit within max_length characters.

        Args:
            chunk: The chunk to compress.
            max_length: Maximum character length.

        Returns:
            Compressed chunk.
        """
        text = chunk.get("text", "")
        
        if len(text) <= max_length:
            return chunk
        
        # Truncate at sentence boundary
        compressed = text[:max_length]
        last_sentence_end = compressed.rfind('.')
        
        if last_sentence_end > max_length * 0.7:  # If we can keep at least 70%
            compressed = compressed[:last_sentence_end + 1]
        else:
            # Fall back to hard truncate with ellipsis
            compressed = compressed[:max_length - 3] + "..."
        
        compressed_chunk = chunk.copy()
        compressed_chunk["text"] = compressed
        compressed_chunk["metadata"]["length"] = len(compressed)
        
        return compressed_chunk

    def attach_metadata_to_context(
        self,
        chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Ensure all chunks have complete metadata attached.

        Args:
            chunks: List of chunks.

        Returns:
            Chunks with complete metadata.
        """
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            
            # Ensure required fields exist
            if "title" not in meta:
                meta["title"] = "Unknown Paper"
            if "authors" not in meta:
                meta["authors"] = "Unknown Authors"
            if "year" not in meta:
                meta["year"] = "N/A"
            if "venue" not in meta:
                meta["venue"] = "N/A"
            if "domain" not in meta:
                meta["domain"] = "unknown"
            if "section" not in meta:
                meta["section"] = "Unknown Section"
            if "pages" not in meta:
                meta["pages"] = "N/A"
            
            chunk["metadata"] = meta
        
        return chunks

    def build_context_blocks(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        query_terms: list[str] | None = None,
        max_chunk_length: int = 800,
        remove_irrelevant: bool = True
    ) -> list[dict[str, Any]]:
        """
        Build clean context blocks from retrieved chunks.

        Pipeline:
          1. Deduplicate chunks
          2. Attach complete metadata
          3. Remove irrelevant sentences (optional)
          4. Compress to max length
          5. Return clean blocks

        Args:
            chunks: Retrieved chunks.
            query: Original query (for relevance checking).
            query_terms: Key terms for relevance filtering.
            max_chunk_length: Maximum length per chunk.
            remove_irrelevant: Whether to remove irrelevant sentences.

        Returns:
            List of clean context blocks.
        """
        if not chunks:
            return []
        
        # Step 1: Deduplicate
        chunks = self.deduplicate_chunks(chunks)
        
        # Step 2: Attach metadata
        chunks = self.attach_metadata_to_context(chunks)
        
        # Step 3: Remove irrelevant sentences
        if remove_irrelevant and query_terms:
            for chunk in chunks:
                chunk["text"] = self.remove_irrelevant_sentences(
                    chunk["text"],
                    query_terms
                )
        
        # Step 4: Compress chunks
        compressed_chunks = []
        for chunk in chunks:
            compressed = self.compress_chunk(chunk, max_chunk_length)
            if compressed.get("text", "").strip():
                compressed_chunks.append(compressed)
        
        logger.info(
            f"Built {len(compressed_chunks)} context blocks from {len(chunks)} chunks"
        )
        
        return compressed_chunks

    def format_context_block(self, chunk: dict[str, Any], index: int) -> str:
        """
        Format a single chunk as a context block with metadata.

        Args:
            chunk: The chunk to format.
            index: Chunk index for labeling.

        Returns:
            Formatted context block string.
        """
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Unknown Paper")
        authors = meta.get("authors", "Unknown Authors")
        year = meta.get("year", "N/A")
        section = meta.get("section", "Unknown Section")
        pages = meta.get("pages", "N/A")
        text = chunk.get("text", "")
        
        block = (
            f"--- Context Block {index + 1} ---\n"
            f"Paper: {title}\n"
            f"Authors: {authors} ({year})\n"
            f"Section: {section}\n"
            f"Pages: {pages}\n"
            f"Content: {text}\n"
        )
        
        return block

    def build_context_string(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        header: str = "Retrieved Context"
    ) -> str:
        """
        Build a complete context string from chunks.

        Args:
            chunks: Retrieved chunks.
            query: Original query.
            header: Section header.

        Returns:
            Complete context string ready for LLM prompt.
        """
        if not chunks:
            return f"{header}:\nNo relevant context found."
        
        # Build clean context blocks
        query_terms = re.findall(r'\b[a-z]{3,}\b', query.lower())
        clean_chunks = self.build_context_blocks(
            chunks,
            query,
            query_terms=query_terms
        )
        
        # Format blocks
        blocks = [
            self.format_context_block(chunk, i)
            for i, chunk in enumerate(clean_chunks)
        ]
        
        context_string = f"{header}:\n" + "\n\n".join(blocks)
        
        logger.info(
            f"Built context string: {len(clean_chunks)} blocks, "
            f"{len(context_string)} characters"
        )
        
        return context_string

    def estimate_context_size(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Estimate the size of the context after building.

        Args:
            chunks: Retrieved chunks.

        Returns:
            Dict with size estimates.
        """
        total_chars = sum(len(c.get("text", "")) for c in chunks)
        total_chunks = len(chunks)
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = total_chars // 4
        
        return {
            "total_characters": total_chars,
            "total_chunks": total_chunks,
            "estimated_tokens": estimated_tokens,
            "fits_in_context": estimated_tokens < 8000  # Common context window limit
        }
