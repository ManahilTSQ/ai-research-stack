"""
metadata_service.py — Centralized metadata fetching with multi-source cascade.

Implements the priority cascade for paper metadata:
  1. Crossref API (primary - authoritative publisher metadata)
  2. OpenAlex API (fallback - 250M+ works, global coverage)
  3. Semantic Scholar (enrichment - abstracts, citations)

This ensures clean, authoritative metadata while maximizing coverage.
"""

import logging
import re
import requests
import time
from urllib.parse import quote
from typing import Dict, Optional, Any
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# ── Cross-process rate-limit state file for Crossref ─────────────────────────────
_CROSSREF_THROTTLE_FILE = settings.BASE_DIR / "output" / ".last_crossref_call"


def _read_last_crossref_call_time() -> float:
    """Read the Unix timestamp of the last Crossref API call from disk."""
    try:
        if _CROSSREF_THROTTLE_FILE.exists():
            return float(_CROSSREF_THROTTLE_FILE.read_text().strip())
    except Exception:
        pass
    return 0.0


def _write_last_crossref_call_time(ts: float) -> None:
    """Persist the Unix timestamp of the most recent Crossref API call to disk."""
    try:
        _CROSSREF_THROTTLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CROSSREF_THROTTLE_FILE.write_text(str(ts))
    except Exception:
        pass


class TokenBucket:
    """
    Token bucket rate limiter for Crossref API.
    
    Crossref allows 50 requests/second for polite users with proper User-Agent.
    This token bucket allows bursts up to the capacity while maintaining the average rate.
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens per second (e.g., 50.0 for 50 req/s)
            capacity: Maximum burst capacity (e.g., 10 for burst of 10 requests)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume (default: 1)
            
        Returns:
            True if tokens were consumed, False if not enough tokens available
        """
        now = time.time()
        # Add tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def wait_for_token(self, tokens: int = 1) -> None:
        """
        Wait until enough tokens are available.
        
        Args:
            tokens: Number of tokens needed (default: 1)
        """
        while not self.consume(tokens):
            # Calculate how long to wait
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens < tokens:
                # Wait for enough tokens to accumulate
                wait_time = (tokens - self.tokens) / self.rate
                time.sleep(wait_time)


class MetadataService:
    """
    Centralized service for fetching paper metadata from multiple sources.
    
    Uses a cascading priority structure:
    - Crossref: Primary source for clean publisher metadata
    - OpenAlex: Fallback for papers not in Crossref
    - Semantic Scholar: Enrichment for abstracts and citation data
    """

    def __init__(self):
        """Initialize the metadata service."""
        self.crossref_base = "https://api.crossref.org/works"
        self.openalex_base = "https://api.openalex.org/works"
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1"
        
        # Headers for API requests
        self.headers = {
            "User-Agent": f"AIResearchStack/1.0 (mailto:{settings.UNPAYWALL_EMAIL})"
        }
        
        # Rate limiting state
        self.last_crossref_call = _read_last_crossref_call_time()
        # Token bucket for Crossref: 50 requests/second with burst capacity of 10
        self.crossref_token_bucket = TokenBucket(rate=50.0, capacity=10)
        
        # Semantic Scholar API key if available
        if settings.SEMANTIC_SCHOLAR_API_KEY:
            self.semantic_headers = self.headers.copy()
            self.semantic_headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
            logger.info("Semantic Scholar API Key loaded for metadata service")
        else:
            self.semantic_headers = self.headers
            logger.info("No Semantic Scholar API Key - using public rate limits")

    def get_paper_metadata(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Fetch paper metadata using the cascade: Crossref → OpenAlex → Semantic Scholar.
        
        Args:
            identifier: DOI (with or without https://doi.org/), arXiv ID, or CorpusID
            
        Returns:
            Dict with keys: title, authors, year, venue, doi, abstract, externalIds
            Returns None if all sources fail
        """
        logger.info(f"Fetching metadata for identifier: {identifier}")
        
        # Normalize identifier
        doi = self._extract_doi(identifier)
        arxiv_id = self._extract_arxiv_id(identifier)
        
        # Step 1: Try Crossref (primary source)
        if doi:
            logger.info(f"Step 1: Querying Crossref for DOI: {doi}")
            crossref_data = self._fetch_crossref_metadata(doi)
            if crossref_data:
                logger.info(f"✓ Crossref found: '{crossref_data['title']}'")
                # Enrich with Semantic Scholar for abstract/citations
                enriched = self._enrich_with_semantic_scholar(crossref_data, identifier)
                return enriched
            else:
                logger.info(f"✗ Crossref not found for DOI: {doi}")
        
        # Step 2: Try OpenAlex (fallback)
        if doi:
            logger.info(f"Step 2: Querying OpenAlex for DOI: {doi}")
            openalex_data = self._fetch_openalex_metadata(doi)
            if openalex_data:
                logger.info(f"✓ OpenAlex found: '{openalex_data['title']}'")
                # Enrich with Semantic Scholar for abstract/citations
                enriched = self._enrich_with_semantic_scholar(openalex_data, identifier)
                return enriched
            else:
                logger.info(f"✗ OpenAlex not found for DOI: {doi}")
        
        # Step 3: Try OpenAlex by title search (if identifier looks like a title)
        if not doi and not arxiv_id:
            logger.info(f"Step 3: Querying OpenAlex by title search: '{identifier}'")
            openalex_data = self._fetch_openalex_metadata(identifier)
            if openalex_data:
                logger.info(f"✓ OpenAlex found by title: '{openalex_data['title']}'")
                return openalex_data
            else:
                logger.info(f"✗ OpenAlex not found by title: '{identifier}'")
        
        # Step 4: Try Semantic Scholar directly (final fallback)
        logger.info(f"Step 4: Querying Semantic Scholar for: {identifier}")
        s2_data = self._fetch_semantic_scholar_metadata(identifier)
        if s2_data:
            logger.info(f"✓ Semantic Scholar found: '{s2_data['title']}'")
            return s2_data
        else:
            logger.info(f"✗ Semantic Scholar not found for: {identifier}")
        
        logger.error(f"All metadata sources failed for identifier: {identifier}")
        return None

    def _extract_doi(self, identifier: str) -> Optional[str]:
        """Extract bare DOI from identifier string."""
        if not identifier:
            return None
        
        # Remove URL prefix
        doi = identifier.replace("https://doi.org/", "").replace("doi.org/", "").strip()
        
        # Check if it looks like a DOI (starts with 10.)
        if re.match(r"^10\.\d{4,9}/", doi):
            return doi
        
        return None

    def _extract_arxiv_id(self, identifier: str) -> Optional[str]:
        """
        Safely extracts an arXiv ID without misidentifying portions of a DOI string.
        
        Rejects if it looks like a standard publisher DOI and only matches if the identifier
        explicitly mentions "arxiv" or matches standard standalone patterns.
        """
        if not identifier:
            return None
        
        # Reject if it looks like a standard publisher DOI
        if "10." in identifier and "/" in identifier:
            # It's a DOI, check if it explicitly mentions arXiv in the path
            if "arxiv" not in identifier.lower():
                return None
        
        # Strict regex pattern for old and modern arXiv IDs
        # Matches: arXiv:YYMM.NNNNN, arxiv/YYMMNNN, or standalone modern patterns
        arxiv_pattern = re.compile(r'(?:arxiv[:/])?(\d{4}\.\d{4,5})(?:v\d+)?', re.IGNORECASE)
        match = arxiv_pattern.search(identifier)
        
        if match:
            return match.group(1)
        return None

    def _throttle_request(self, use_token_bucket: bool = False) -> None:
        """
        Enforce rate limiting for API requests.
        
        Args:
            use_token_bucket: If True, use token bucket for Crossref (50 req/s, burst 10).
                            If False, use simple 1.0s throttle for OpenAlex.
        """
        if use_token_bucket:
            # Use token bucket for Crossref (allows bursts up to 10, maintains 50 req/s average)
            self.crossref_token_bucket.wait_for_token(tokens=1)
            # Log when approaching rate limit (less than 3 tokens remaining)
            if self.crossref_token_bucket.tokens < 3:
                logger.info(f"Crossref rate limit approaching: {self.crossref_token_bucket.tokens:.1f} tokens remaining")
        else:
            # Simple throttle for OpenAlex (1.0s minimum between requests)
            now = time.time()
            elapsed = now - self.last_crossref_call
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
        
        # Record and persist the current call time
        self.last_crossref_call = time.time()
        _write_last_crossref_call_time(self.last_crossref_call)

    def _fetch_crossref_metadata(self, doi: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata from Crossref API with rate limiting and exponential backoff."""
        for attempt in range(5):
            # Apply throttle before request (use token bucket for Crossref)
            self._throttle_request(use_token_bucket=True)
            
            try:
                url = f"{self.crossref_base}/{quote(doi, safe='/')}"
                resp = requests.get(url, headers=self.headers, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    message = data.get("message", {})
                    
                    # Title
                    titles = message.get("title", [])
                    title = titles[0].strip() if titles else "Untitled Paper"
                    
                    # Authors
                    crossref_authors = message.get("author", [])
                    authors = []
                    for a in crossref_authors:
                        given = a.get("given", "").strip()
                        family = a.get("family", "").strip()
                        name = f"{given} {family}".strip()
                        if name:
                            authors.append({"name": name})
                    
                    # Year
                    year = "N/A"
                    for date_source in [message.get("published-print"), message.get("published-online"), message.get("created")]:
                        if date_source:
                            date_parts = date_source.get("date-parts", [])
                            if date_parts and date_parts[0]:
                                year = str(date_parts[0][0])
                                break
                    
                    # Venue
                    container = message.get("container-title", [])
                    venue = container[0].strip() if container else "N/A"
                    
                    # External IDs
                    external_ids = {
                        "DOI": doi,
                        "CorpusID": message.get("corpus_id")
                    }
                    
                    return {
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "venue": venue,
                        "doi": doi,
                        "abstract": None,  # Crossref doesn't provide abstracts
                        "externalIds": external_ids,
                        "citationCount": 0,
                        "source": "crossref"
                    }
                
                elif resp.status_code == 404:
                    logger.info(f"Crossref: DOI '{doi}' not found")
                    break
                
                elif resp.status_code == 429:
                    # Rate limited - wait with exponential backoff
                    wait_time = 3.0 * (2 ** attempt)
                    logger.warning(f"Crossref rate limit (429) hit. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    # Reset timer after wait
                    self.last_crossref_call = time.time()
                    _write_last_crossref_call_time(self.last_crossref_call)
                
                else:
                    logger.warning(f"Crossref returned HTTP {resp.status_code} for DOI '{doi}'")
                    break
            
            except Exception as e:
                logger.error(f"Crossref lookup failed for '{doi}': {e}")
                if attempt < 4:
                    time.sleep(3.0)
        
        return None

    def _fetch_openalex_metadata(self, doi_or_title: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata from OpenAlex API with rate limiting and exponential backoff."""
        for attempt in range(5):
            # Apply throttle before request (use simple throttle for OpenAlex)
            self._throttle_request(use_token_bucket=False)
            
            try:
                doi_or_title = doi_or_title.strip()
                
                # Determine if it's a DOI or title query
                is_doi = bool(re.match(r"^(10\.\d{4,9}/[-._;()/:A-Z0-9]+)$", doi_or_title, re.IGNORECASE) or "doi.org" in doi_or_title)
                
                if is_doi:
                    bare_doi = doi_or_title.replace("https://doi.org/", "").strip()
                    url = f"{self.openalex_base}/doi:{bare_doi}"
                    params = {}
                    logger.info(f"OpenAlex: Querying by DOI: {bare_doi}")
                else:
                    url = self.openalex_base
                    params = {"search": doi_or_title, "per_page": 1}
                    logger.info(f"OpenAlex: Querying by title: '{doi_or_title}'")
                
                resp = requests.get(url, headers=self.headers, params=params, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    # If searching, data is a list under "results"
                    if not is_doi:
                        results = data.get("results", [])
                        if not results:
                            logger.info("OpenAlex title search returned no results")
                            return None
                        work = results[0]
                    else:
                        work = data
                    
                    # Title
                    title = work.get("title") or "Untitled Paper"
                    
                    # Authors
                    authorships = work.get("authorships", [])
                    authors = []
                    for auth in authorships:
                        author_meta = auth.get("author", {})
                        name = author_meta.get("display_name", "").strip()
                        if name:
                            authors.append({"name": name})
                    
                    # Year
                    year = "N/A"
                    pub_year = work.get("publication_year")
                    if pub_year:
                        year = str(pub_year)
                    
                    # Venue
                    primary_location = work.get("primary_location", {})
                    source = primary_location.get("source", {})
                    venue = source.get("display_name") or "N/A"
                    
                    # DOI
                    doi = work.get("doi") or "N/A"
                    
                    # External IDs
                    external_ids = {
                        "DOI": doi if doi != "N/A" else None,
                        "CorpusID": work.get("id", "").replace("https://openalex.org/", "")
                    }
                    
                    return {
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "venue": venue,
                        "doi": doi,
                        "abstract": None,  # OpenAlex doesn't provide abstracts
                        "externalIds": external_ids,
                        "citationCount": work.get("cited_by_count", 0),
                        "source": "openalex"
                    }
                
                elif resp.status_code == 404:
                    logger.info(f"OpenAlex: '{doi_or_title}' not found")
                    break
                
                elif resp.status_code == 429:
                    # Rate limited - wait with exponential backoff
                    wait_time = 3.0 * (2 ** attempt)
                    logger.warning(f"OpenAlex rate limit (429) hit. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    # Reset timer after wait
                    self.last_crossref_call = time.time()
                    _write_last_crossref_call_time(self.last_crossref_call)
                
                else:
                    logger.warning(f"OpenAlex returned HTTP {resp.status_code}")
                    break
            
            except Exception as e:
                logger.error(f"OpenAlex lookup failed: {e}")
                if attempt < 4:
                    time.sleep(3.0)
        
        return None

    def _fetch_semantic_scholar_metadata(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata from Semantic Scholar API."""
        try:
            # Use the existing paper_discovery service for S2 lookup
            from paper_discovery import PaperDiscoveryService
            discover_service = PaperDiscoveryService()
            
            data = discover_service.get_paper_details(identifier)
            if data:
                # Add source field
                data["source"] = "semantic_scholar"
                return data
        
        except Exception as e:
            logger.error(f"Semantic Scholar lookup failed: {e}")
        
        return None

    def _enrich_with_semantic_scholar(self, base_metadata: Dict[str, Any], identifier: str) -> Dict[str, Any]:
        """
        Enrich base metadata with Semantic Scholar data (abstract, citations).
        
        Args:
            base_metadata: Metadata from Crossref or OpenAlex
            identifier: Original identifier for S2 lookup
            
        Returns:
            Enriched metadata dict
        """
        try:
            from paper_discovery import PaperDiscoveryService
            discover_service = PaperDiscoveryService()
            
            s2_data = discover_service.get_paper_details(identifier)
            if s2_data:
                # Merge abstract and citation data
                if s2_data.get("abstract"):
                    base_metadata["abstract"] = s2_data["abstract"]
                if s2_data.get("citationCount"):
                    base_metadata["citationCount"] = s2_data["citationCount"]
                if s2_data.get("externalIds"):
                    # Merge external IDs
                    for key, value in s2_data["externalIds"].items():
                        if value and key not in base_metadata.get("externalIds", {}):
                            base_metadata.setdefault("externalIds", {})[key] = value
                
                logger.info("Enriched with Semantic Scholar abstract and citation data")
        
        except Exception as e:
            logger.warning(f"Failed to enrich with Semantic Scholar: {e}")
        
        return base_metadata


# Module-level singleton
metadata_service = MetadataService()
