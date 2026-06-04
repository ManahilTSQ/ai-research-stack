"""
context_coherence.py — Context Coherence Scoring for RAG.

Measures cross-chunk coherence to detect:
- Redundancy (duplicate information across chunks)
- Contradiction (conflicting claims)
- Fragmentation (disconnected information)

This addresses the biggest failure point: chunks good individually but inconsistent as a group.
"""

import re
import logging
from typing import Any
from collections import Counter

logger = logging.getLogger(__name__)


class ContextCoherence:
    """
    Scores coherence of retrieved chunk groups.
    """

    def __init__(self):
        """Initialize the context coherence scorer."""
        logger.info("Context coherence scorer initialized")

    def _extract_significant_words(self, text: str) -> set[str]:
        """
        Extract significant words from text for comparison.

        Args:
            text: Input text.

        Returns:
            Set of significant words (3+ characters, not stop words).
        """
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "has", "have", "been", "this", "that",
            "with", "they", "from", "what", "when", "which", "will", "more", "about",
            "would", "there", "their", "than", "then", "them", "some", "such", "into",
            "its", "who", "also", "get", "may", "other", "these", "only", "new",
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return {w for w in words if w not in stop_words}

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score between 0 and 1.
        """
        words1 = self._extract_significant_words(text1)
        words2 = self._extract_significant_words(text2)
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0

    def calculate_redundancy_ratio(self, chunks: list[dict[str, Any]]) -> float:
        """
        Calculate redundancy ratio across chunks.

        High redundancy means chunks contain duplicate information.

        Args:
            chunks: List of chunks.

        Returns:
            Redundancy ratio between 0 and 1 (higher = more redundant).
        """
        if len(chunks) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                text1 = chunks[i].get("text", "")
                text2 = chunks[j].get("text", "")
                sim = self._calculate_text_similarity(text1, text2)
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        # Average similarity as redundancy measure
        avg_similarity = sum(similarities) / len(similarities)
        
        logger.debug(f"Redundancy ratio: {avg_similarity:.3f}")
        return avg_similarity

    def detect_contradictions(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Detect potential contradictions between chunks using light heuristics.

        Args:
            chunks: List of chunks.

        Returns:
            List of potential contradiction pairs.
        """
        contradictions = []
        
        if len(chunks) < 2:
            return contradictions
        
        # Contradiction indicators
        contradiction_patterns = [
            (r"\b(?:however|but|although|nevertheless|conversely)\b", "contrast"),
            (r"\b(?:not|no|never|neither)\s+(?:the|a|an)\b", "negation"),
            (r"\b(?:different|opposite|contrary)\b", "opposition"),
        ]
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                text1 = chunks[i].get("text", "").lower()
                text2 = chunks[j].get("text", "").lower()
                
                # Check for contradiction indicators
                has_contradiction = False
                for pattern, indicator in contradiction_patterns:
                    if re.search(pattern, text1) and re.search(pattern, text2):
                        has_contradiction = True
                        break
                
                # Check for numerical contradictions (e.g., "accuracy 95%" vs "accuracy 80%")
                numbers1 = re.findall(r'\d+(?:\.\d+)?%', text1)
                numbers2 = re.findall(r'\d+(?:\.\d+)?%', text2)
                
                if numbers1 and numbers2:
                    # If both have percentages and they differ significantly
                    val1 = float(numbers1[0].rstrip('%'))
                    val2 = float(numbers2[0].rstrip('%'))
                    if abs(val1 - val2) > 10:  # More than 10% difference
                        has_contradiction = True
                
                if has_contradiction:
                    contradictions.append({
                        "chunk_i": i,
                        "chunk_j": j,
                        "text_i": text1[:100],
                        "text_j": text2[:100]
                    })
        
        logger.debug(f"Detected {len(contradictions)} potential contradictions")
        return contradictions

    def calculate_fragmentation_score(self, chunks: list[dict[str, Any]]) -> float:
        """
        Calculate fragmentation score.

        High fragmentation means chunks are disconnected and don't form a coherent narrative.

        Args:
            chunks: List of chunks.

        Returns:
            Fragmentation score between 0 and 1 (higher = more fragmented).
        """
        if len(chunks) < 2:
            return 0.0
        
        # Check if chunks are from different papers
        papers = set()
        for chunk in chunks:
            title = chunk.get("metadata", {}).get("title", "")
            papers.add(title)
        
        paper_diversity = len(papers) / len(chunks)
        
        # Check if chunks are from different sections
        sections = set()
        for chunk in chunks:
            section = chunk.get("metadata", {}).get("section", "")
            sections.add(section)
        
        section_diversity = len(sections) / len(chunks)
        
        # Fragmentation = high diversity across papers and sections
        fragmentation = 0.5 * paper_diversity + 0.5 * section_diversity
        
        logger.debug(f"Fragmentation score: {fragmentation:.3f}")
        return fragmentation

    def calculate_coherence_score(
        self,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Calculate overall coherence score for a group of chunks.

        Args:
            chunks: List of chunks.

        Returns:
            Dict with:
                - redundancy_ratio: float (0-1, lower is better)
                - contradiction_count: int
                - fragmentation_score: float (0-1, lower is better)
                - overall_coherence: float (0-1, higher is better)
        """
        if not chunks:
            return {
                "redundancy_ratio": 0.0,
                "contradiction_count": 0,
                "fragmentation_score": 0.0,
                "overall_coherence": 0.0
            }
        
        redundancy = self.calculate_redundancy_ratio(chunks)
        contradictions = self.detect_contradictions(chunks)
        fragmentation = self.calculate_fragmentation_score(chunks)
        
        # Overall coherence: high when redundancy is moderate, no contradictions, low fragmentation
        # Ideal: moderate redundancy (0.2-0.4) for reinforcement, no contradictions, low fragmentation
        redundancy_score = 1.0 - abs(redundancy - 0.3)  # Penalize deviation from 0.3
        contradiction_penalty = min(len(contradictions) * 0.2, 1.0)
        fragmentation_penalty = fragmentation
        
        overall_coherence = max(0.0, redundancy_score - contradiction_penalty - fragmentation_penalty)
        
        result = {
            "redundancy_ratio": redundancy,
            "contradiction_count": len(contradictions),
            "fragmentation_score": fragmentation,
            "overall_coherence": overall_coherence,
            "contradictions": contradictions
        }
        
        logger.info(
            f"Coherence score: {overall_coherence:.3f} "
            f"(redundancy: {redundancy:.3f}, contradictions: {len(contradictions)}, "
            f"fragmentation: {fragmentation:.3f})"
        )
        
        return result

    def should_filter_for_coherence(
        self,
        chunks: list[dict[str, Any]],
        min_coherence: float = 0.4
    ) -> tuple[bool, dict[str, Any]]:
        """
        Determine if chunks should be filtered based on coherence.

        Args:
            chunks: List of chunks.
            min_coherence: Minimum acceptable coherence score.

        Returns:
            Tuple of (should_filter, coherence_metrics).
        """
        metrics = self.calculate_coherence_score(chunks)
        should_filter = metrics["overall_coherence"] < min_coherence
        
        return should_filter, metrics

    def improve_coherence(
        self,
        chunks: list[dict[str, Any]],
        target_count: int = 8
    ) -> list[dict[str, Any]]:
        """
        Improve coherence by selecting a more coherent subset of chunks.

        Args:
            chunks: List of chunks.
            target_count: Target number of chunks to return.

        Returns:
            More coherent subset of chunks.
        """
        if len(chunks) <= target_count:
            return chunks
        
        # Greedy selection: start with most relevant, then add chunks that improve coherence
        selected = []
        remaining = chunks.copy()
        
        # Sort by distance (most relevant first)
        remaining.sort(key=lambda c: c.get("distance", 1.0))
        
        # Add first chunk
        selected.append(remaining.pop(0))
        
        # Greedily add chunks that improve coherence
        while len(selected) < target_count and remaining:
            best_chunk = None
            best_coherence = 0.0
            
            for chunk in remaining:
                test_set = selected + [chunk]
                coherence = self.calculate_coherence_score(test_set)["overall_coherence"]
                
                if coherence > best_coherence:
                    best_coherence = coherence
                    best_chunk = chunk
            
            if best_chunk:
                selected.append(best_chunk)
                remaining.remove(best_chunk)
            else:
                # No improvement, just add the next most relevant
                selected.append(remaining.pop(0))
        
        logger.info(f"Improved coherence: {len(chunks)} → {len(selected)} chunks")
        return selected
