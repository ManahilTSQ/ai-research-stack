"""
query_router.py — Query Router and Structured Metadata Filter.

Implements a router pattern to separate global vs local queries, and parses
structured metadata constraints (year, DOI, author) for native ChromaDB filtering.

Global queries bypass vector search and use direct metadata lookups:
- "How many papers..."
- "List all papers..."
- "Show all DOIs..."
- "Which is the newest paper..."
- "Papers published before 2022..."

Local queries proceed to semantic/hybrid search:
- "What is the main contribution of X..."
- "Summarize Y..."
- "Explain the methodology in..."
"""

import re
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Routes user queries to appropriate retrieval strategies.
    
    Separates global library inventory queries from local content queries,
    and extracts structured metadata filters for native ChromaDB filtering.
    """
    
    # Patterns for global inventory queries
    GLOBAL_QUERY_PATTERNS = [
        r"\bhow\s+many\s+papers?\b",
        r"\blist\s+(?:all\s+)?papers?\b",
        r"\bshow\s+(?:all\s+)?papers?\b",
        r"\bshow\s+all\s+dois?\b",
        r"\bwhich\s+is\s+(?:the\s+)?(?:newest|latest|oldest)\s+paper\b",
        r"\bwhat\s+(?:is\s+the\s+)?(?:newest|latest|oldest)\s+paper\b",
        r"\bcount\s+(?:of\s+)?papers?\b",
        r"\btotal\s+(?:number\s+of\s+)?papers?\b",
        r"\ball\s+papers?\s+by\s+",
        r"\bpapers?\s+published\s+(?:before|after|in|during)\s+\d{4}",
        r"\bpapers?\s+from\s+\d{4}",
        r"\bdois?\s+(?:of\s+)?(?:all\s+)?papers?\b",
        r"\bshow\s+(?:all\s+)?papers?\s+with\s+doi\b",
        r"\bshow\s+doi\s+information\b",
        r"\benumerate\s+papers?\b",
    ]
    
    # Patterns for local content queries
    LOCAL_QUERY_PATTERNS = [
        r"\bwhat\s+is\s+(?:the\s+)?main\s+(?:contribution|finding|result|idea)\b",
        r"\bsummarize\s+(?:the\s+)?paper\b",
        r"\bexplain\s+(?:the\s+)?(?:methodology|method|approach)\b",
        r"\bdescribe\s+(?:the\s+)?(?:framework|system|model)\b",
        r"\bhow\s+does\s+(?:it|this|the\s+paper)\b",
        r"\bwhat\s+(?:are\s+)?(?:the\s+)?(?:limitations|challenges|problems)\b",
        r"\bwhat\s+(?:are\s+)?(?:the\s+)?(?:assumptions|hypotheses)\b",
    ]
    
    def __init__(self):
        """Initialize the query router."""
        logger.info("Query router initialized")
    
    def classify_query(self, query: str) -> str:
        """
        Classify a query as 'global' or 'local'.
        
        Args:
            query: The user's query string.
            
        Returns:
            'global' for inventory/metadata queries, 'local' for content queries.
        """
        q_lower = query.lower()
        
        # Check global patterns first
        for pattern in self.GLOBAL_QUERY_PATTERNS:
            if re.search(pattern, q_lower):
                logger.debug(f"Query classified as GLOBAL: '{query[:60]}...'")
                return "global"
        
        # Check local patterns
        for pattern in self.LOCAL_QUERY_PATTERNS:
            if re.search(pattern, q_lower):
                logger.debug(f"Query classified as LOCAL: '{query[:60]}...'")
                return "local"
        
        # Default to local if no pattern matches
        logger.debug(f"Query classified as LOCAL (default): '{query[:60]}...'")
        return "local"
    
    def parse_metadata_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract structured metadata filters from a query.
        
        Parses year constraints, DOI constraints, and author constraints
        for native ChromaDB where-clause filtering.
        
        Args:
            query: The user's query string.
            
        Returns:
            Dict with filter keys: 'year', 'doi', 'authors', 'venue'.
            Values are ChromaDB-compatible filter expressions.
        """
        filters = {}
        q_lower = query.lower()
        
        # Parse year constraints
        # "before 2022", "after 2020", "in 2021", "from 2020 to 2022"
        year_before = re.search(r'\b(?:before|prior to|earlier than)\s+(\d{4})\b', q_lower)
        if year_before:
            year = int(year_before.group(1))
            filters['year'] = {"$lt": str(year)}
            logger.info(f"Parsed year filter: before {year}")
        
        year_after = re.search(r'\b(?:after|since|later than)\s+(\d{4})\b', q_lower)
        if year_after:
            year = int(year_after.group(1))
            filters['year'] = {"$gte": str(year)}
            logger.info(f"Parsed year filter: after {year}")
        
        year_in = re.search(r'\b(?:in|from|of)\s+(\d{4})\b', q_lower)
        if year_in and 'year' not in filters:
            year = int(year_in.group(1))
            filters['year'] = str(year)
            logger.info(f"Parsed year filter: in {year}")
        
        year_range = re.search(r'\b(?:from|between)\s+(\d{4})\s+(?:to|and|-)\s+(\d{4})\b', q_lower)
        if year_range:
            start_year = int(year_range.group(1))
            end_year = int(year_range.group(2))
            filters['year'] = {"$gte": str(start_year), "$lte": str(end_year)}
            logger.info(f"Parsed year filter: {start_year} to {end_year}")
        
        # Parse DOI constraints
        doi_match = re.search(r'\bdoi[:\s]+([^\s,]+)\b', q_lower)
        if doi_match:
            doi = doi_match.group(1).strip()
            filters['doi'] = doi
            logger.info(f"Parsed DOI filter: {doi}")
        
        # Parse author constraints
        # "by John Smith", "authored by Jane Doe"
        author_match = re.search(r'\b(?:by|authored by|written by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        if author_match:
            author = author_match.group(1).strip()
            # Use $contains for partial author name matching
            filters['authors'] = {"$contains": author}
            logger.info(f"Parsed author filter: {author}")
        
        # Parse venue constraints
        venue_match = re.search(r'\b(?:in|at|published in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        if venue_match:
            venue = venue_match.group(1).strip()
            filters['venue'] = {"$contains": venue}
            logger.info(f"Parsed venue filter: {venue}")
        
        # Special handling for "newest" and "oldest" queries
        # These need special sorting, not just filtering
        if re.search(r'\b(?:newest|latest)\b', q_lower):
            filters['_sort_by_year'] = 'desc'
            logger.info("Detected newest paper query - will sort by year descending")
        elif re.search(r'\boldest\b', q_lower):
            filters['_sort_by_year'] = 'asc'
            logger.info("Detected oldest paper query - will sort by year ascending")
        
        return filters
    
    def should_use_metadata_filter(self, query: str) -> bool:
        """
        Determine if a query should use metadata filtering instead of pure semantic search.
        
        Args:
            query: The user's query string.
            
        Returns:
            True if metadata filters should be applied, False otherwise.
        """
        filters = self.parse_metadata_filters(query)
        return len(filters) > 0


def route_query(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to route a query and extract filters.
    
    Args:
        query: The user's query string.
        
    Returns:
        Tuple of (query_type, filters) where query_type is 'global' or 'local',
        and filters is a dict of ChromaDB-compatible filter expressions.
    """
    router = QueryRouter()
    query_type = router.classify_query(query)
    filters = router.parse_metadata_filters(query)
    return query_type, filters
