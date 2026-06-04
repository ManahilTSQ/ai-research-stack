"""
reranker.py — Hybrid Reranking Service for RAG.

Implements a multi-signal reranking approach that combines:
  - Semantic similarity (from vector search distance)
  - Lexical overlap with query terms
  - Section relevance (if section metadata available)
  - Paper diversity (avoid too many chunks from same paper)

This provides immediate accuracy improvement without requiring
external APIs or new model downloads.
"""

import re
import logging
from typing import Any
from collections import Counter

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Hybrid reranker that reorders retrieved chunks based on multi-signal relevance.
    
    The reranker takes a pool of retrieved chunks (typically 15-30) and scores them
    on multiple dimensions, then returns the top-k most relevant chunks.
    """

    def __init__(self):
        """Initialize the reranker service."""
        logger.info("Reranker service initialized")

    def _extract_query_tokens(self, query: str) -> set[str]:
        """
        Extract meaningful tokens from the query for lexical matching.
        Removes stop words and short tokens.
        """
        stop_words = {
            "what", "which", "when", "where", "does", "about", "from", "with",
            "that", "this", "have", "into", "your", "their", "paper", "papers",
            "author", "authors", "say", "says", "line", "summarize", "summary",
            "brief", "explain", "describe", "tell", "give", "please", "would",
            "thoughts", "views", "opinions", "ideas", "main", "the", "a", "an",
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "shall", "can", "need", "dare", "ought",
        }
        
        tokens = re.findall(r"\b[a-z0-9]{3,}\b", query.lower())
        return {t for t in tokens if t not in stop_words}

    def _lexical_overlap_score(self, chunk: dict[str, Any], query_tokens: set[str]) -> float:
        """
        Score chunk based on lexical overlap with query tokens.
        
        Returns a score between 0 and 1 based on how many query tokens
        appear in the chunk text and metadata.
        """
        if not query_tokens:
            return 0.0
        
        # Build haystack from chunk text and metadata
        meta = chunk.get("metadata", {})
        haystack = " ".join([
            chunk.get("text", ""),
            meta.get("title", ""),
            meta.get("authors", ""),
            meta.get("section", ""),
        ]).lower()
        
        # Count matches
        matches = sum(1 for token in query_tokens if token in haystack)
        
        # Normalize by number of query tokens
        return matches / len(query_tokens) if query_tokens else 0.0

    def _section_relevance_score(self, chunk: dict[str, Any], query: str) -> float:
        """
        Score chunk based on section relevance to query type.
        
        Different query types prefer different sections:
        - Methodology queries → Methods/Methodology sections
        - Result queries → Results sections
        - Overview queries → Introduction/Abstract sections
        """
        section = chunk.get("metadata", {}).get("section", "Unknown").lower()
        query_lower = query.lower()
        
        # Query type detection
        if any(word in query_lower for word in ["method", "approach", "technique", "algorithm", "how"]):
            # Prefer methodology sections
            if section in ["methodology", "methods", "method", "experimental setup"]:
                return 1.0
            elif section in ["introduction", "background"]:
                return 0.5
            else:
                return 0.3
        
        elif any(word in query_lower for word in ["result", "finding", "outcome", "performance", "accuracy"]):
            # Prefer results sections
            if section in ["results", "discussion"]:
                return 1.0
            elif section in ["methodology", "methods"]:
                return 0.6
            else:
                return 0.3
        
        elif any(word in query_lower for word in ["overview", "summary", "introduction", "what is", "describe"]):
            # Prefer introduction/abstract
            if section in ["introduction", "abstract", "background"]:
                return 1.0
            elif section in ["conclusion", "conclusions"]:
                return 0.7
            else:
                return 0.5
        
        # Default: no strong section preference
        return 0.5

    def _classify_context_role(self, chunk: dict[str, Any], query: str) -> str:
        """
        Classify the role of a chunk in the context.

        Returns one of: definition, method, result, supporting_evidence, background.
        """
        section = chunk.get("metadata", {}).get("section", "Unknown").lower()
        text = chunk.get("text", "").lower()
        query_lower = query.lower()
        
        # Definition role
        if section in ["abstract", "introduction", "background"]:
            if any(word in text for word in ["define", "definition", "refers to", "means", "is defined as"]):
                return "definition"
            return "background"
        
        # Method role
        if section in ["methodology", "methods", "method", "experimental setup"]:
            return "method"
        
        # Result role
        if section in ["results", "discussion"]:
            if any(word in text for word in ["result", "finding", "achieve", "obtain", "performance", "accuracy"]):
                return "result"
            return "supporting_evidence"
        
        # Supporting evidence (default)
        return "supporting_evidence"

    def _semantic_score(self, chunk: dict[str, Any]) -> float:
        """
        Convert cosine distance to a similarity score.
        
        Lower distance = higher similarity. ChromaDB returns distance in range [0, 2].
        We convert this to a score in range [0, 1] where 1 = most similar.
        """
        distance = chunk.get("distance", 1.0)
        # Convert distance to similarity: distance 0 → score 1, distance 2 → score 0
        similarity = max(0.0, 1.0 - (distance / 2.0))
        return similarity

    def _diversity_penalty(self, chunk: dict[str, Any], paper_counts: Counter) -> float:
        """
        Apply penalty if too many chunks come from the same paper.
        
        This ensures diversity across papers in the final result set.
        """
        title = chunk.get("metadata", {}).get("title", "")
        if not title:
            return 0.0
        
        count = paper_counts.get(title, 0)
        
        # Penalty increases with each additional chunk from the same paper
        # 1st chunk: 0 penalty, 2nd: 0.1, 3rd: 0.2, etc.
        penalty = max(0.0, (count - 1) * 0.1)
        return penalty

    def rerank(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        top_k: int = 8,
        weights: dict[str, float] | None = None
    ) -> list[dict[str, Any]]:
        """
        Rerank chunks using multi-signal scoring.
        
        Args:
            chunks: List of retrieved chunks from vector search.
            query: The original user query.
            top_k: Number of top chunks to return.
            weights: Optional weights for different signals. Defaults to:
                - semantic: 0.4
                - lexical: 0.3
                - section: 0.2
                - diversity: 0.1
        
        Returns:
            List of top-k reranked chunks.
        """
        if not chunks:
            return []
        
        if len(chunks) <= top_k:
            logger.info(f"Only {len(chunks)} chunks retrieved, no reranking needed")
            return chunks
        
        # Default weights
        if weights is None:
            weights = {
                "semantic": 0.4,
                "lexical": 0.3,
                "section": 0.2,
                "diversity": 0.1
            }
        
        query_tokens = self._extract_query_tokens(query)
        
        # Count papers for diversity penalty
        paper_counts = Counter()
        scored_chunks = []
        
        # Score each chunk
        for chunk in chunks:
            title = chunk.get("metadata", {}).get("title", "")
            paper_counts[title] += 1
            
            semantic = self._semantic_score(chunk)
            lexical = self._lexical_overlap_score(chunk, query_tokens)
            section = self._section_relevance_score(chunk, query)
            diversity_penalty = self._diversity_penalty(chunk, paper_counts)
            
            # Combined score
            total_score = (
                weights["semantic"] * semantic +
                weights["lexical"] * lexical +
                weights["section"] * section -
                weights["diversity"] * diversity_penalty
            )
            
            # Classify context role
            context_role = self._classify_context_role(chunk, query)
            
            scored_chunks.append({
                "chunk": chunk,
                "score": total_score,
                "context_role": context_role,
                "signals": {
                    "semantic": semantic,
                    "lexical": lexical,
                    "section": section,
                    "diversity_penalty": diversity_penalty
                }
            })
        
        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k chunks
        top_chunks = [item["chunk"] for item in scored_chunks[:top_k]]
        
        logger.info(
            f"Reranked {len(chunks)} chunks → {len(top_chunks)} top chunks. "
            f"Top score: {scored_chunks[0]['score']:.3f}, "
            f"Bottom score: {scored_chunks[top_k-1]['score']:.3f if top_k > 0 else 0:.3f}"
        )
        
        return top_chunks

    def rerank_with_scores(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        top_k: int = 8,
        weights: dict[str, float] | None = None
    ) -> list[dict[str, Any]]:
        """
        Rerank chunks and return them with scoring metadata for debugging.
        
        This is useful for understanding why certain chunks were ranked higher.
        """
        if not chunks:
            return []
        
        if weights is None:
            weights = {
                "semantic": 0.4,
                "lexical": 0.3,
                "section": 0.2,
                "diversity": 0.1
            }
        
        query_tokens = self._extract_query_tokens(query)
        paper_counts = Counter()
        scored_chunks = []
        
        for chunk in chunks:
            title = chunk.get("metadata", {}).get("title", "")
            paper_counts[title] += 1
            
            semantic = self._semantic_score(chunk)
            lexical = self._lexical_overlap_score(chunk, query_tokens)
            section = self._section_relevance_score(chunk, query)
            diversity_penalty = self._diversity_penalty(chunk, paper_counts)
            
            total_score = (
                weights["semantic"] * semantic +
                weights["lexical"] * lexical +
                weights["section"] * section -
                weights["diversity"] * diversity_penalty
            )
            
            # Add score metadata to chunk
            chunk_with_scores = chunk.copy()
            chunk_with_scores["rerank_score"] = total_score
            chunk_with_scores["rerank_signals"] = {
                "semantic": semantic,
                "lexical": lexical,
                "section": section,
                "diversity_penalty": diversity_penalty
            }
            
            scored_chunks.append(chunk_with_scores)
        
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_chunks[:top_k]
