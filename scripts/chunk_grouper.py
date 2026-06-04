"""
chunk_grouper.py — Chunk Grouping Service for RAG.

Groups retrieved chunks by document and section to help the LLM
understand relationships instead of seeing random disconnected text.
"""

import logging
from typing import Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ChunkGrouper:
    """
    Groups chunks by document and section for structured reasoning.
    """

    def __init__(self):
        """Initialize the chunk grouper."""
        logger.info("Chunk grouper initialized")

    def group_by_document(self, chunks: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """
        Group chunks by document (paper title).

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Dict mapping paper titles to their chunks.
        """
        grouped = defaultdict(list)
        
        for chunk in chunks:
            title = chunk.get("metadata", {}).get("title", "Unknown Paper")
            grouped[title].append(chunk)
        
        logger.debug(f"Grouped {len(chunks)} chunks into {len(grouped)} documents")
        return dict(grouped)

    def group_by_section(self, chunks: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """
        Group chunks by section within documents.

        Args:
            chunks: List of retrieved chunks (should be from a single document).

        Returns:
            Dict mapping section names to their chunks.
        """
        grouped = defaultdict(list)
        
        for chunk in chunks:
            section = chunk.get("metadata", {}).get("section", "Unknown Section")
            grouped[section].append(chunk)
        
        return dict(grouped)

    def group_by_document_and_section(self, chunks: list[dict[str, Any]]) -> dict[str, dict[str, list[dict]]]:
        """
        Group chunks by document, then by section within each document.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Nested dict: {document_title: {section_name: [chunks]}}
        """
        doc_groups = self.group_by_document(chunks)
        
        nested_groups = {}
        for doc_title, doc_chunks in doc_groups.items():
            section_groups = self.group_by_section(doc_chunks)
            nested_groups[doc_title] = section_groups
        
        logger.debug(
            f"Grouped {len(chunks)} chunks into {len(nested_groups)} documents "
            f"with nested sections"
        )
        return nested_groups

    def sort_chunks_within_groups(
        self,
        chunks: list[dict[str, Any]],
        sort_by: str = "distance"
    ) -> list[dict[str, Any]]:
        """
        Sort chunks within each group.

        Args:
            chunks: List of chunks.
            sort_by: Sort criterion ("distance", "page", "section").

        Returns:
            Sorted list of chunks.
        """
        if sort_by == "distance":
            # Sort by semantic distance (lower is better)
            return sorted(chunks, key=lambda c: c.get("distance", 1.0))
        elif sort_by == "page":
            # Sort by page number
            return sorted(chunks, key=lambda c: int(c.get("metadata", {}).get("pages", "999").split(",")[0] if c.get("metadata", {}).get("pages") else 999))
        elif sort_by == "section":
            # Sort by section (custom order)
            section_order = {
                "Abstract": 0, "Introduction": 1, "Background": 2,
                "Methodology": 3, "Methods": 3, "Method": 3,
                "Results": 4, "Discussion": 5, "Conclusion": 6,
                "Unknown": 99
            }
            return sorted(
                chunks,
                key=lambda c: section_order.get(
                    c.get("metadata", {}).get("section", "Unknown"),
                    99
                )
            )
        else:
            return chunks

    def limit_chunks_per_group(
        self,
        grouped_chunks: dict[str, list[dict]],
        max_per_group: int = 8
    ) -> dict[str, list[dict]]:
        """
        Limit the number of chunks per group.

        Args:
            grouped_chunks: Dict of grouped chunks.
            max_per_group: Maximum chunks to keep per group.

        Returns:
            Dict with limited chunks per group.
        """
        limited = {}
        for group_name, chunks in grouped_chunks.items():
            # Sort by distance and keep top N
            sorted_chunks = self.sort_chunks_within_groups(chunks, sort_by="distance")
            limited[group_name] = sorted_chunks[:max_per_group]
        
        return limited

    def merge_nearby_chunks(
        self,
        chunks: list[dict[str, Any]],
        max_distance: float = 0.3
    ) -> list[dict[str, Any]]:
        """
        Merge chunks that are semantically similar and from the same document/section.

        Args:
            chunks: List of chunks.
            max_distance: Maximum distance to consider chunks similar.

        Returns:
            List of merged chunks.
        """
        if not chunks:
            return []
        
        # Group by document and section first
        nested_groups = self.group_by_document_and_section(chunks)
        
        merged = []
        for doc_title, section_groups in nested_groups.items():
            for section_name, section_chunks in section_groups.items():
                # Sort by distance
                section_chunks = self.sort_chunks_within_groups(section_chunks, sort_by="distance")
                
                # Merge similar chunks
                if len(section_chunks) > 1:
                    current_group = [section_chunks[0]]
                    for chunk in section_chunks[1:]:
                        # Check if distance to last chunk in group is small
                        last_chunk = current_group[-1]
                        distance_diff = abs(chunk.get("distance", 1.0) - last_chunk.get("distance", 1.0))
                        
                        if distance_diff < max_distance:
                            current_group.append(chunk)
                        else:
                            # Merge current group and start new one
                            merged_chunk = self._merge_chunk_group(current_group)
                            if merged_chunk:
                                merged.append(merged_chunk)
                            current_group = [chunk]
                    
                    # Don't forget the last group
                    if current_group:
                        merged_chunk = self._merge_chunk_group(current_group)
                        if merged_chunk:
                            merged.append(merged_chunk)
                else:
                    merged.extend(section_chunks)
        
        logger.debug(f"Merged chunks: {len(chunks)} → {len(merged)}")
        return merged

    def _merge_chunk_group(self, chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Merge a group of similar chunks into one.

        Args:
            chunks: List of chunks to merge.

        Returns:
            Merged chunk dict or None if chunks is empty.
        """
        if not chunks:
            return None
        
        # Use the first chunk as base
        base = chunks[0].copy()
        
        # Combine text
        combined_text = " ".join(c.get("text", "") for c in chunks)
        base["text"] = combined_text
        
        # Update metadata
        all_pages = set()
        for chunk in chunks:
            pages = chunk.get("metadata", {}).get("pages", "")
            if pages:
                for page in pages.split(","):
                    all_pages.add(page.strip())
        
        base["metadata"]["pages"] = ",".join(sorted(all_pages))
        base["metadata"]["length"] = len(combined_text)
        
        # Use the minimum distance (most relevant)
        base["distance"] = min(c.get("distance", 1.0) for c in chunks)
        
        return base

    def get_group_statistics(self, grouped_chunks: dict[str, list[dict]]) -> dict[str, Any]:
        """
        Get statistics about grouped chunks.

        Args:
            grouped_chunks: Dict of grouped chunks.

        Returns:
            Dict with statistics.
        """
        stats = {
            "total_groups": len(grouped_chunks),
            "total_chunks": sum(len(chunks) for chunks in grouped_chunks.values()),
            "avg_chunks_per_group": 0,
            "max_chunks_in_group": 0,
            "min_chunks_in_group": 0,
        }
        
        if grouped_chunks:
            chunk_counts = [len(chunks) for chunks in grouped_chunks.values()]
            stats["avg_chunks_per_group"] = sum(chunk_counts) / len(chunk_counts)
            stats["max_chunks_in_group"] = max(chunk_counts)
            stats["min_chunks_in_group"] = min(chunk_counts)
        
        return stats
