"""
metadata_filter.py — Metadata-Driven Pre-Filtering Service.

Provides strong metadata-based filtering before vector search to prevent
irrelevant retrieval. This is a control layer that narrows scope using
paper metadata like title, author, year, domain, or tags.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    Filters paper metadata based on query constraints before vector search.
    """

    def __init__(self):
        """Initialize the metadata filter."""
        logger.info("Metadata filter initialized")

    def extract_year_constraint(self, query: str) -> int | None:
        """
        Extract year constraint from query.

        Returns the year if specified, otherwise None.
        Examples:
            "papers from 2020" → 2020
            "recent papers after 2019" → None (range not supported yet)
            "2021 papers" → 2021
        """
        # Pattern: "papers from/in YEAR" or "YEAR papers"
        patterns = [
            r"\b(?:papers?|articles?|studies?)\s+(?:from|in|of)\s*(\d{4})\b",
            r"\b(\d{4})\s*(?:papers?|articles?|studies?)\b",
            r"\bpublished\s+in\s+(\d{4})\b",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    year = int(match.group(1))
                    if 1900 <= year <= 2100:  # Reasonable year range
                        logger.debug(f"Extracted year constraint: {year}")
                        return year
                except ValueError:
                    pass
        
        return None

    def extract_venue_constraint(self, query: str) -> str | None:
        """
        Extract venue constraint from query.

        Returns the venue name if specified, otherwise None.
        Examples:
            "papers from CVPR" → "CVPR"
            "NeurIPS papers" → "NeurIPS"
        """
        # Common academic venues
        venue_keywords = [
            "CVPR", "ICCV", "NeurIPS", "ICML", "AAAI", "IJCAI", "KDD", "SIGIR",
            "WWW", "ACL", "EMNLP", "NAACL", "ICLR", "ICRA", "ECCV",
            "Nature", "Science", "Cell", "PNAS", "IEEE", "ACM",
        ]
        
        query_upper = query.upper()
        for venue in venue_keywords:
            if venue.upper() in query_upper:
                logger.debug(f"Extracted venue constraint: {venue}")
                return venue
        
        return None

    def filter_papers_by_metadata(
        self,
        papers_metadata: dict[str, dict],
        query: str,
        constraints: dict[str, Any] | None = None
    ) -> list[str]:
        """
        Filter papers based on metadata constraints extracted from query.

        Args:
            papers_metadata: Dict mapping paper titles to metadata dicts.
            query: User's query string.
            constraints: Optional explicit constraints dict.

        Returns:
            List of paper titles that match the constraints.
        """
        if not papers_metadata:
            return []
        
        # Extract constraints from query if not provided
        if constraints is None:
            constraints = {}
            year_constraint = self.extract_year_constraint(query)
            if year_constraint:
                constraints["year"] = year_constraint
            
            venue_constraint = self.extract_venue_constraint(query)
            if venue_constraint:
                constraints["venue"] = venue_constraint
        
        if not constraints:
            # No constraints, return all papers
            return list(papers_metadata.keys())
        
        # Apply filters
        filtered_titles = []
        
        for title, meta in papers_metadata.items():
            match = True
            
            # Year filter
            if "year" in constraints:
                paper_year = str(meta.get("year", ""))
                if paper_year != str(constraints["year"]):
                    match = False
            
            # Venue filter (case-insensitive partial match)
            if match and "venue" in constraints:
                paper_venue = (meta.get("venue", "")).upper()
                constraint_venue = constraints["venue"].upper()
                if constraint_venue not in paper_venue:
                    match = False
            
            # Domain filter
            if match and "domain" in constraints:
                paper_domain = meta.get("domain", "unknown")
                if paper_domain != constraints["domain"]:
                    match = False
            
            if match:
                filtered_titles.append(title)
        
        logger.debug(
            f"Filtered {len(papers_metadata)} papers → {len(filtered_titles)} "
            f"using constraints: {constraints}"
        )
        
        return filtered_titles

    def build_metadata_where_clause(
        self,
        constraints: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Build ChromaDB where clause from metadata constraints.

        Args:
            constraints: Dict of metadata constraints.

        Returns:
            ChromaDB where clause dict or None if no constraints.
        """
        if not constraints:
            return None
        
        where_clause = {}
        
        if "year" in constraints:
            where_clause["year"] = str(constraints["year"])
        
        if "venue" in constraints:
            # For venue, we need to use contains since ChromaDB doesn't support
            # partial string matching in where clauses natively
            # This is a limitation - we'll filter after retrieval
            pass
        
        if "domain" in constraints:
            where_clause["domain"] = constraints["domain"]
        
        if "title" in constraints:
            where_clause["title"] = constraints["title"]
        
        return where_clause if where_clause else None

    def should_apply_metadata_filtering(self, query: str) -> bool:
        """
        Determine if metadata filtering should be applied based on query.

        Returns True if the query contains explicit metadata constraints.
        """
        # Check for year mentions
        if re.search(r"\b\d{4}\b", query):
            return True
        
        # Check for venue mentions
        venue_keywords = ["CVPR", "ICCV", "NeurIPS", "ICML", "AAAI", "Nature", "Science"]
        if any(venue.lower() in query.lower() for venue in venue_keywords):
            return True
        
        # Check for explicit constraint phrases
        constraint_phrases = [
            "published in", "from", "venue", "conference", "journal",
            "year", "domain", "field"
        ]
        if any(phrase in query.lower() for phrase in constraint_phrases):
            return True
        
        return False
