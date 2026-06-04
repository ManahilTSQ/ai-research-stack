"""
context_shaper.py — Context Shaping Service for RAG.

Organizes retrieved chunks into a structured format before sending to the LLM.
Instead of a flat text wall, chunks are grouped by paper, then by section,
which helps the model understand relationships and build structured reasoning.
"""

import logging
from typing import Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContextShaper:
    """
    Shapes retrieved chunks into a structured context format.
    """

    def __init__(self):
        """Initialize the context shaper."""
        logger.info("Context shaper initialized")

    def group_chunks_by_paper(self, chunks: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """
        Group chunks by paper title.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Dict mapping paper titles to their chunks.
        """
        grouped = defaultdict(list)
        
        for chunk in chunks:
            title = chunk.get("metadata", {}).get("title", "Unknown Paper")
            grouped[title].append(chunk)
        
        logger.debug(f"Grouped {len(chunks)} chunks into {len(grouped)} papers")
        return dict(grouped)

    def group_chunks_by_section(self, chunks: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """
        Group chunks by section within each paper.

        Args:
            chunks: List of retrieved chunks (should be from a single paper).

        Returns:
            Dict mapping section names to their chunks.
        """
        grouped = defaultdict(list)
        
        for chunk in chunks:
            section = chunk.get("metadata", {}).get("section", "Unknown Section")
            grouped[section].append(chunk)
        
        return dict(grouped)

    def shape_context(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        max_chunks_per_paper: int = 8
    ) -> str:
        """
        Shape chunks into a structured context string with semantic blocks.

        Organization:
          1. Group by paper
          2. Within each paper, group by section
          3. Add clear headers and metadata
          4. Limit chunks per paper to avoid dominance

        Args:
            chunks: List of retrieved chunks.
            query: Original query (for logging).
            max_chunks_per_paper: Maximum chunks to include per paper.

        Returns:
            Structured context string ready for LLM prompt.
        """
        if not chunks:
            return "No relevant context found."

        # Group by paper
        papers = self.group_chunks_by_paper(chunks)
        
        # Sort papers by number of chunks (most relevant first)
        sorted_papers = sorted(
            papers.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        context_parts = []
        
        for paper_title, paper_chunks in sorted_papers:
            # Limit chunks per paper
            paper_chunks = paper_chunks[:max_chunks_per_paper]
            
            # Get paper metadata
            meta = paper_chunks[0].get("metadata", {})
            authors = meta.get("authors", "Unknown Authors")
            year = meta.get("year", "N/A")
            chunk_id = meta.get("chunk_id", "N/A")
            
            # Group by section within this paper
            sections = self.group_chunks_by_section(paper_chunks)
            
            # Sort sections by relevance (more chunks = more relevant)
            sorted_sections = sorted(
                sections.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            
            # Build paper section
            paper_header = f"## Paper: {paper_title}\n**{authors} ({year})**\n"
            paper_content = []
            
            for section_name, section_chunks in sorted_sections:
                section_content = []
                for chunk in section_chunks:
                    text = chunk.get("text", "").strip()
                    pages = chunk.get("metadata", {}).get("pages", "N/A")
                    chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
                    
                    # Format with chunk_id for provenance tracking
                    if pages and pages != "N/A":
                        if chunk_id:
                            section_content.append(f"[pp. {pages}] [{chunk_id}] {text}")
                        else:
                            section_content.append(f"[pp. {pages}] {text}")
                    else:
                        if chunk_id:
                            section_content.append(f"[{chunk_id}] {text}")
                        else:
                            section_content.append(text)
                
                if section_content:
                    paper_content.append(f"\n### {section_name}\n" + "\n".join(section_content))
            
            if paper_content:
                context_parts.append(paper_header + "\n".join(paper_content))
        
        shaped_context = "\n\n---\n\n".join(context_parts)
        
        logger.info(
            f"Shaped context: {len(chunks)} chunks from {len(papers)} papers, "
            f"{len(shaped_context)} characters"
        )
        
        return shaped_context

    def shape_context_semantic_blocks(
        self,
        chunks: list[dict[str, Any]],
        query: str
    ) -> dict[str, Any]:
        """
        Shape chunks into semantic blocks (not just text).

        Returns structured data: Paper → Section → Claim → Evidence → Chunk ID

        Args:
            chunks: List of retrieved chunks.
            query: Original query.

        Returns:
            Dict with structured semantic blocks.
        """
        if not chunks:
            return {"papers": {}}

        # Group by paper
        papers = self.group_chunks_by_paper(chunks)
        
        semantic_blocks = {"papers": {}}
        
        for paper_title, paper_chunks in papers.items():
            # Get paper metadata
            meta = paper_chunks[0].get("metadata", {})
            paper_id = meta.get("paper_id", "")
            
            # Group by section
            sections = self.group_chunks_by_section(paper_chunks)
            
            paper_block = {
                "paper_id": paper_id,
                "title": paper_title,
                "authors": meta.get("authors", ""),
                "year": meta.get("year", ""),
                "sections": {}
            }
            
            for section_name, section_chunks in sections.items():
                section_block = {
                    "section_id": meta.get("section_id", ""),
                    "chunks": []
                }
                
                for chunk in section_chunks:
                    chunk_meta = chunk.get("metadata", {})
                    chunk_block = {
                        "chunk_id": chunk_meta.get("chunk_id", ""),
                        "text": chunk.get("text", ""),
                        "pages": chunk_meta.get("pages", ""),
                        "context_role": chunk.get("context_role", "supporting_evidence"),
                        "distance": chunk.get("distance", 0.0)
                    }
                    section_block["chunks"].append(chunk_block)
                
                paper_block["sections"][section_name] = section_block
            
            semantic_blocks["papers"][paper_title] = paper_block
        
        logger.info(
            f"Shaped semantic blocks: {len(chunks)} chunks from {len(papers)} papers"
        )
        
        return semantic_blocks

    def shape_context_compact(
        self,
        chunks: list[dict[str, Any]],
        max_total_chunks: int = 12
    ) -> str:
        """
        Compact context shaping for shorter contexts.

        Simpler format: paper-level grouping only, no section headers.
        """
        if not chunks:
            return "No relevant context found."

        # Limit total chunks
        chunks = chunks[:max_total_chunks]
        
        # Group by paper
        papers = self.group_chunks_by_paper(chunks)
        
        context_parts = []
        
        for paper_title, paper_chunks in papers.items():
            meta = paper_chunks[0].get("metadata", {})
            authors = meta.get("authors", "Unknown Authors")
            year = meta.get("year", "N/A")
            
            paper_text = []
            for chunk in paper_chunks:
                text = chunk.get("text", "").strip()
                pages = chunk.get("metadata", {}).get("pages", "N/A")
                if pages and pages != "N/A":
                    paper_text.append(f"[{pages}] {text}")
                else:
                    paper_text.append(text)
            
            if paper_text:
                header = f"**{authors} ({year})** - {paper_title}"
                context_parts.append(f"{header}\n" + " ".join(paper_text))
        
        return "\n\n".join(context_parts)

    def deduplicate_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Remove duplicate chunks based on text similarity.

        Simple exact match deduplication for now.
        """
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_chunks.append(chunk)
        
        if len(unique_chunks) < len(chunks):
            logger.info(f"Deduplicated: {len(chunks)} → {len(unique_chunks)} chunks")
        
        return unique_chunks

    def estimate_context_quality(
        self,
        chunks: list[dict[str, Any]],
        query: str
    ) -> dict[str, Any]:
        """
        Estimate the quality of retrieved context before LLM call.

        Returns quality metrics:
            - chunk_count: Number of chunks
            - paper_diversity: Number of unique papers
            - section_diversity: Number of unique sections
            - avg_distance: Average semantic distance
            - quality_score: Overall quality score (0-1)
        """
        if not chunks:
            return {
                "chunk_count": 0,
                "paper_diversity": 0,
                "section_diversity": 0,
                "avg_distance": 1.0,
                "quality_score": 0.0
            }
        
        # Count unique papers
        papers = self.group_chunks_by_paper(chunks)
        paper_diversity = len(papers)
        
        # Count unique sections
        sections = set()
        for chunk in chunks:
            section = chunk.get("metadata", {}).get("section", "Unknown")
            sections.add(section)
        section_diversity = len(sections)
        
        # Average distance
        distances = [chunk.get("distance", 1.0) for chunk in chunks]
        avg_distance = sum(distances) / len(distances)
        
        # Quality score (higher is better)
        # Factors: more papers (diversity), more sections (coverage), lower distance (relevance)
        diversity_score = min(paper_diversity / 3.0, 1.0)  # Cap at 3 papers
        relevance_score = max(0.0, 1.0 - (avg_distance / 2.0))  # Distance 0-2
        section_score = min(section_diversity / 2.0, 1.0)  # Cap at 2 sections
        
        quality_score = 0.4 * relevance_score + 0.3 * diversity_score + 0.3 * section_score
        
        return {
            "chunk_count": len(chunks),
            "paper_diversity": paper_diversity,
            "section_diversity": section_diversity,
            "avg_distance": avg_distance,
            "quality_score": quality_score
        }
