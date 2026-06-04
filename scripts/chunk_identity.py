"""
chunk_identity.py — Chunk Identity System for True Provenance Tracking.

Assigns every chunk a unique identity (chunk_id, paper_id, section_id)
for unambiguous citation and verification. This removes all ambiguity
in provenance tracking.
"""

import uuid
import logging
from typing import Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkIdentity:
    """Unique identity for a chunk."""
    chunk_id: str
    paper_id: str
    section_id: str
    global_id: str  # Combined unique identifier


class ChunkIdentityManager:
    """
    Manages chunk identities for provenance tracking.
    """

    def __init__(self):
        """Initialize the chunk identity manager."""
        logger.info("Chunk identity manager initialized")

    def assign_chunk_identity(
        self,
        chunk: dict[str, Any],
        paper_title: str,
        section: str
    ) -> dict[str, Any]:
        """
        Assign unique identity to a chunk.

        Args:
            chunk: The chunk to identify.
            paper_title: Paper title for paper_id.
            section: Section name for section_id.

        Returns:
            Chunk with identity metadata added.
        """
        # Generate paper_id from title (stable, reproducible)
        paper_id = self._generate_paper_id(paper_title)
        
        # Generate section_id from section name
        section_id = self._generate_section_id(section)
        
        # Generate chunk_id (unique per chunk)
        chunk_id = self._generate_chunk_id()
        
        # Generate global_id (combined)
        global_id = f"{paper_id}_{section_id}_{chunk_id}"
        
        # Add identity to chunk metadata
        chunk_with_identity = chunk.copy()
        if "metadata" not in chunk_with_identity:
            chunk_with_identity["metadata"] = {}
        
        chunk_with_identity["metadata"]["chunk_id"] = chunk_id
        chunk_with_identity["metadata"]["paper_id"] = paper_id
        chunk_with_identity["metadata"]["section_id"] = section_id
        chunk_with_identity["metadata"]["global_id"] = global_id
        
        return chunk_with_identity

    def _generate_paper_id(self, paper_title: str) -> str:
        """
        Generate stable paper_id from title.

        Uses a hash of the title for reproducibility.
        """
        import hashlib
        # Use first 8 characters of MD5 hash
        hash_obj = hashlib.md5(paper_title.lower().encode())
        return f"paper_{hash_obj.hexdigest()[:8]}"

    def _generate_section_id(self, section: str) -> str:
        """
        Generate section_id from section name.
        """
        # Normalize section name
        normalized = section.lower().replace(" ", "_").replace("-", "_")
        # Remove non-alphanumeric
        normalized = "".join(c for c in normalized if c.isalnum() or c == "_")
        return f"section_{normalized[:20]}"  # Limit to 20 chars

    def _generate_chunk_id(self) -> str:
        """
        Generate unique chunk_id.
        """
        return f"chunk_{uuid.uuid4().hex[:8]}"

    def assign_identities_to_chunks(
        self,
        chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Assign identities to a list of chunks.

        Args:
            chunks: List of chunks.

        Returns:
            Chunks with identity metadata added.
        """
        identified_chunks = []
        
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            paper_title = meta.get("title", "Unknown Paper")
            section = meta.get("section", "Unknown Section")
            
            chunk_with_id = self.assign_chunk_identity(chunk, paper_title, section)
            identified_chunks.append(chunk_with_id)
        
        logger.info(f"Assigned identities to {len(identified_chunks)} chunks")
        return identified_chunks

    def extract_chunk_id_from_citation(self, citation: str) -> Optional[str]:
        """
        Extract chunk_id from a citation string.

        Args:
            citation: Citation string (e.g., "[chunk_abc123]" or "(Smith, 2020) [chunk_abc123]").

        Returns:
            chunk_id or None if not found.
        """
        # Pattern: [chunk_xxxxxx] or chunk_xxxxxx
        match = re.search(r'\[?(chunk_[a-f0-9]{8})\]?', citation)
        return match.group(1) if match else None

    def format_citation_with_chunk_id(
        self,
        chunk: dict[str, Any],
        traditional_citation: str
    ) -> str:
        """
        Format a citation that includes chunk_id.

        Args:
            chunk: The chunk being cited.
            traditional_citation: Traditional APA citation (e.g., "(Smith, 2020)").

        Returns:
            Citation with chunk_id appended.
        """
        chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
        if chunk_id:
            return f"{traditional_citation} [{chunk_id}]"
        return traditional_citation

    def verify_chunk_id_citation(
        self,
        citation: str,
        available_chunks: list[dict[str, Any]]
    ) -> bool:
        """
        Verify that a chunk_id in a citation is valid.

        Args:
            citation: Citation string.
            available_chunks: Chunks that were provided as context.

        Returns:
            True if chunk_id is valid, False otherwise.
        """
        chunk_id = self.extract_chunk_id_from_citation(citation)
        if not chunk_id:
            # No chunk_id in citation, can't verify
            return True
        
        # Check if chunk_id exists in available chunks
        available_ids = set()
        for chunk in available_chunks:
            cid = chunk.get("metadata", {}).get("chunk_id", "")
            if cid:
                available_ids.add(cid)
        
        return chunk_id in available_ids

    def get_chunk_by_id(
        self,
        chunk_id: str,
        chunks: list[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """
        Retrieve a chunk by its chunk_id.

        Args:
            chunk_id: The chunk_id to look up.
            chunks: List of chunks to search.

        Returns:
            Chunk dict or None if not found.
        """
        for chunk in chunks:
            cid = chunk.get("metadata", {}).get("chunk_id", "")
            if cid == chunk_id:
                return chunk
        return None

    def build_chunk_id_index(self, chunks: list[dict[str, Any]]) -> dict[str, dict]:
        """
        Build an index mapping chunk_id to chunk for fast lookup.

        Args:
            chunks: List of chunks.

        Returns:
            Dict mapping chunk_id to chunk.
        """
        index = {}
        for chunk in chunks:
            chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
            if chunk_id:
                index[chunk_id] = chunk
        return index

    def add_chunk_ids_to_answer(
        self,
        answer: str,
        chunks: list[dict[str, Any]]
    ) -> str:
        """
        Add chunk_ids to an answer's citations.

        This is a best-effort approach to add chunk_ids to traditional citations.

        Args:
            answer: The LLM-generated answer.
            chunks: Chunks that were used as context.

        Returns:
            Answer with chunk_ids added to citations.
        """
        # Build chunk_id index
        chunk_index = self.build_chunk_id_index(chunks)
        
        # Find traditional citations and add chunk_ids
        # This is heuristic - assumes the LLM cited the right chunks
        lines = answer.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Check if line has a citation
            if '(' in line and ')' in line:
                # Extract citation
                citation_match = re.search(r'\([^)]+\)', line)
                if citation_match:
                    traditional_citation = citation_match.group(0)
                    
                    # Try to find which chunk this citation refers to
                    # This is imperfect - we use the first matching chunk
                    matching_chunk = self._find_chunk_for_citation(
                        traditional_citation,
                        chunks
                    )
                    
                    if matching_chunk:
                        chunk_id = matching_chunk.get("metadata", {}).get("chunk_id", "")
                        if chunk_id:
                            # Add chunk_id to citation
                            enhanced_citation = f"{traditional_citation} [{chunk_id}]"
                            line = line.replace(traditional_citation, enhanced_citation)
            
            enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)

    def _find_chunk_for_citation(
        self,
        citation: str,
        chunks: list[dict[str, Any]]
    ) -> Optional[dict]:
        """
        Find the chunk that a citation likely refers to.

        This is heuristic - matches author/year from citation to chunk metadata.

        Args:
            citation: Traditional citation (e.g., "(Smith, 2020)").
            chunks: Available chunks.

        Returns:
            Matching chunk or None.
        """
        # Extract author and year from citation
        match = re.search(r'([A-Z][a-zA-Z]+)(?:\s+et\s+al\.)?,\s*(\d{4})', citation)
        if not match:
            return None
        
        author = match.group(1).lower()
        year = match.group(2)
        
        # Find matching chunk
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            authors = meta.get("authors", "").lower()
            chunk_year = str(meta.get("year", ""))
            
            if author in authors and year in chunk_year:
                return chunk
        
        return None
