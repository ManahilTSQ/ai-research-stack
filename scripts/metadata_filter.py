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
            "only 2024 papers" → 2024
            "papers in 2024" → 2024
        """
        # Pattern: "papers from/in YEAR" or "YEAR papers" - enhanced patterns
        patterns = [
            r"\b(?:papers?|articles?|studies?)\s+(?:from|in|of|published\s+in)\s*(\d{4})\b",
            r"\b(?:only|just|strictly)?\s*(\d{4})\s*(?:papers?|articles?|studies?)\b",
            r"\bpublished\s+in\s+(\d{4})\b",
            r"\b(?:show|list|display)?\s*(\d{4})\s*(?:papers?|articles?|studies?)\b",
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

    def extract_year_constraints(self, query: str) -> dict[str, Any] | None:
        """
        Extract year constraints (including operators) from query.
        Returns a dict like:
            {"year": 2020, "op": "before"} or {"year": 2020, "op": "after"}
            or {"year_range": (2020, 2025), "op": "range"}
            or {"year": 2020, "op": "eq"}
        """
        q = (query or "").lower()
        
        # 1. Check for year range: "between 2020 and 2025" or "2020-2025"
        range_match = re.search(
            r"\b(?:between|from)\s+(19\d{2}|20\d{2})\s+(?:and|to|-)\s+(19\d{2}|20\d{2})\b|"
            r"\b(19\d{2}|20\d{2})\s*-\s*(19\d{2}|20\d{2})\b",
            q
        )
        if range_match:
            years = [int(y) for y in range_match.groups() if y]
            if len(years) == 2:
                start, end = min(years), max(years)
                return {"year_range": (start, end), "op": "range"}
        
        # 2. Check for before/after/since constraints:
        # "before 2020", "prior to 2020", "earlier than 2020", "up to 2020", "< 2020"
        before_match = re.search(
            r"\b(?:before|prior\s+to|earlier\s+than|up\s+to|before|published\s+before)\s+(\d{4})\b|(?:\b<\s*(\d{4})\b)",
            q
        )
        if before_match:
            year_str = before_match.group(1) or before_match.group(2)
            return {"year": int(year_str), "op": "before"}
            
        # "after 2020", "since 2020", "later than 2020", "published\s+after", "> 2020"
        after_match = re.search(
            r"\b(?:after|since|later\s+than|published\s+after|post)\s+(\d{4})\b|(?:\b>\s*(\d{4})\b)",
            q
        )
        if after_match:
            year_str = after_match.group(1) or after_match.group(2)
            return {"year": int(year_str), "op": "after"}
            
        # 3. Check for exact year match:
        year_val = self.extract_year_constraint(query)
        if year_val:
            return {"year": year_val, "op": "eq"}
            
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

        This module ONLY does FILTERING (removes papers that don't match constraints).
        It does NOT reorder or annotate - that must happen elsewhere.

        Args:
            papers_metadata: Dict mapping paper titles to metadata dicts.
            query: User's query string.
            constraints: Optional explicit constraints dict.

        Returns:
            List of paper titles that match the constraints (preserves input order).
        """
        if not papers_metadata:
            return []
        
        # Extract constraints from query if not provided
        if constraints is None:
            constraints = {}
            year_info = self.extract_year_constraints(query)
            if year_info:
                constraints["year_info"] = year_info
            
            venue_constraint = self.extract_venue_constraint(query)
            if venue_constraint:
                constraints["venue"] = venue_constraint
        
        if not constraints:
            # No constraints, return all papers (preserves input order)
            return list(papers_metadata.keys())
        
        # Apply filters (preserves input order - no sorting)
        filtered_titles = []
        
        for title, meta in papers_metadata.items():
            match = True
            
            # Year info filter
            if "year_info" in constraints:
                paper_year_str = str(meta.get("year", ""))
                try:
                    paper_year = int(paper_year_str) if paper_year_str.isdigit() else None
                except ValueError:
                    paper_year = None
                
                if paper_year is None:
                    match = False
                else:
                    info = constraints["year_info"]
                    op = info["op"]
                    if op == "eq" and paper_year != info["year"]:
                        match = False
                    elif op == "before" and paper_year >= info["year"]:
                        match = False
                    elif op == "after" and paper_year <= info["year"]:
                        match = False
                    elif op == "range":
                        start, end = info["year_range"]
                        if not (start <= paper_year <= end):
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
        
        # Check for explicit constraint phrases (only specific phrases, not generic words)
        constraint_phrases = [
            "published in", "venue", "conference", "journal"
        ]
        if any(phrase in query.lower() for phrase in constraint_phrases):
            return True
        
        return False
