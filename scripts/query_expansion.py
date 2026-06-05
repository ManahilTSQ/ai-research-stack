"""
query_expansion.py — Controlled Query Expansion Service for RAG.

Generates synonyms and academic variations of query terms to improve
retrieval recall without destroying precision.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class QueryExpansion:
    """
    Expands queries with controlled synonyms and academic variations.
    """

    # Academic term synonyms and variations (generic only, no domain-specific terms)
    ACADEMIC_SYNONYMS = {
        "algorithm": [
            "method", "approach", "technique", "procedure", "strategy",
            "computational method", "optimization method"
        ],
        "performance": [
            "accuracy", "efficiency", "effectiveness", "results", "outcome",
            "metric", "benchmark", "evaluation"
        ],
        "dataset": [
            "data", "corpus", "database", "collection", "samples",
            "training data", "test data"
        ],
        "model": [
            "architecture", "network", "system", "framework", "approach",
            "predictive model", "classification model"
        ],
        "feature": [
            "attribute", "characteristic", "property", "variable",
            "input variable", "predictor"
        ],
        "experiment": [
            "study", "trial", "test", "evaluation", "assessment",
            "empirical study", "experimental evaluation"
        ],
        "conclusion": [
            "finding", "result", "outcome", "observation", "insight",
            "key finding", "main result"
        ],
        "methodology": [
            "method", "approach", "technique", "procedure", "design",
            "experimental design", "research method"
        ],
        "framework": [
            "architecture", "system", "model", "structure", "paradigm",
            "conceptual framework", "theoretical framework"
        ],
        "implementation": [
            "deployment", "realization", "execution", "application",
            "system implementation", "practical implementation"
        ],
    }

    # Academic prefixes/suffixes for variations
    ACADEMIC_PREFIXES = ["semi-", "quasi-", "pseudo-", "multi-", "hyper-", "meta-"]
    ACADEMIC_SUFFIXES = [
        "-based", "-driven", "-aware", "-oriented", "-centric",
        "-level", "-scale", "-wise", "-time"
    ]

    def __init__(self):
        """Initialize the query expansion service."""
        logger.info("Query expansion service initialized")

    def get_synonyms(self, term: str) -> list[str]:
        """
        Get synonyms for a given term.

        Args:
            term: The term to find synonyms for.

        Returns:
            List of synonyms (may be empty if no synonyms found).
        """
        term_lower = term.lower()
        
        # Direct match
        if term_lower in self.ACADEMIC_SYNONYMS:
            return self.ACADEMIC_SYNONYMS[term_lower]
        
        # Partial match (e.g., "machine" matches "machine learning")
        for key, synonyms in self.ACADEMIC_SYNONYMS.items():
            if term_lower in key or key in term_lower:
                # Return synonyms that contain the term or are related
                related = [s for s in synonyms if term_lower in s.lower()]
                if related:
                    return related
        
        return []

    def generate_variations(self, term: str) -> list[str]:
        """
        Generate academic variations of a term.

        Args:
            term: The term to generate variations for.

        Returns:
            List of term variations.
        """
        variations = []
        
        # Add academic prefixes
        for prefix in self.ACADEMIC_PREFIXES:
            variations.append(prefix + term)
        
        # Add academic suffixes
        for suffix in self.ACADEMIC_SUFFIXES:
            variations.append(term + suffix)
        
        return variations

    def expand_query(
        self,
        query: str,
        max_expansions: int = 5,
        include_synonyms: bool = True,
        include_variations: bool = False
    ) -> list[str]:
        """
        Expand a query with controlled synonyms and variations.

        Args:
            query: Original query.
            max_expansions: Maximum number of expanded queries to generate.
            include_synonyms: Whether to include synonym-based expansions.
            include_variations: Whether to include morphological variations.

        Returns:
            List of expanded queries (original + expansions).
        """
        # Extract key terms
        words = re.findall(r'\b[a-z]{3,}\b', query.lower())
        
        if not words:
            return [query]
        
        expanded_queries = [query]
        
        # Generate synonym-based expansions
        if include_synonyms:
            for word in words:
                synonyms = self.get_synonyms(word)
                for synonym in synonyms[:2]:  # Limit to 2 synonyms per term
                    expanded = query.lower().replace(word, synonym)
                    if expanded not in expanded_queries:
                        expanded_queries.append(expanded)
                        if len(expanded_queries) >= max_expansions + 1:
                            return expanded_queries
        
        # Generate variation-based expansions
        if include_variations:
            for word in words:
                variations = self.generate_variations(word)
                for variation in variations[:1]:  # Limit to 1 variation per term
                    expanded = query.lower().replace(word, variation)
                    if expanded not in expanded_queries:
                        expanded_queries.append(expanded)
                        if len(expanded_queries) >= max_expansions + 1:
                            return expanded_queries
        
        logger.info(f"Expanded query into {len(expanded_queries)} variations")
        return expanded_queries[:max_expansions + 1]

    def expand_with_key_terms(
        self,
        key_terms: list[str],
        max_terms: int = 10
    ) -> list[str]:
        """
        Expand a list of key terms with synonyms.

        Args:
            key_terms: Original key terms.
            max_terms: Maximum total terms to return.

        Returns:
            Expanded list of terms.
        """
        expanded_terms = list(key_terms)
        
        for term in key_terms:
            synonyms = self.get_synonyms(term)
            for synonym in synonyms[:2]:
                if synonym not in expanded_terms:
                    expanded_terms.append(synonym)
                    if len(expanded_terms) >= max_terms:
                        return expanded_terms
        
        return expanded_terms

    def should_expand_query(self, query: str) -> bool:
        """
        Determine if a query should be expanded.

        Returns False for very specific queries (e.g., with quotes, specific names).
        """
        # Don't expand if query has quoted text (specific paper title)
        if '"' in query:
            return False
        
        # Don't expand if query is very short (< 3 words)
        if len(query.split()) < 3:
            return False
        
        # Don't expand if query has specific author names (capitalized words)
        # This is a heuristic - may have false positives
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        if len(capitalized_words) >= 2:
            return False
        
        return True
