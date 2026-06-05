"""
paper_discovery.py — Academic Paper Search, PDF Resolution, and Download Service.

Handles all external API interactions:
  - Semantic Scholar Graph API v1  : paper search, details, citations
  - Unpaywall API                  : open-access PDF URL resolution
  - CrossRef public API            : DOI → title resolution fallback
  - arXiv direct                   : free PDF download for arXiv papers
  - HTTP PDF download              : streams PDFs to the papers/ directory

Rate limiting: Enforces a 1.2 second minimum gap between Semantic Scholar
requests. The last-call timestamp is persisted to disk (output/.last_api_call)
so the throttle is respected across multiple CLI invocations in the same session.
"""

import time
import logging
import requests
from urllib.parse import quote
from pathlib import Path
from tqdm import tqdm          # Progress bar for PDF downloads
from config import settings   # Flat import — scripts/ is on sys.path


# ── Logger setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Cross-process rate-limit state file ──────────────────────────────────────
# Stored in output/.last_api_call — a plain text file containing a Unix timestamp.
# Reading and writing this file allows multiple CLI processes to share rate-limit
# state without spinning up a dedicated rate-limit server.
_THROTTLE_FILE = settings.BASE_DIR / "output" / ".last_api_call"


def _read_last_call_time() -> float:
    """
    Read the Unix timestamp of the last Semantic Scholar API call from disk.

    Returns:
        Float timestamp, or 0.0 if the file does not exist or cannot be read.
        0.0 effectively means "no previous call" → no throttle delay needed.
    """
    try:
        if _THROTTLE_FILE.exists():
            return float(_THROTTLE_FILE.read_text().strip())
    except Exception:
        pass  # Silently fall back to 0.0 if the file is corrupt or unreadable
    return 0.0


def _write_last_call_time(ts: float) -> None:
    """
    Persist the Unix timestamp of the most recent API call to disk.

    Args:
        ts: Unix timestamp (float) as returned by time.time().
    """
    try:
        _THROTTLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _THROTTLE_FILE.write_text(str(ts))
    except Exception:
        pass  # Non-critical — worst case the throttle is not enforced cross-process


class PaperDiscoveryService:
    """
    Service for discovering academic papers and resolving open-access PDFs.

    Attributes:
        SEMANTIC_SCHOLAR_BASE: Base URL for the Semantic Scholar Graph API v1.
        UNPAYWALL_BASE: Base URL for the Unpaywall REST API v2.
    """

    SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
    UNPAYWALL_BASE = "https://api.unpaywall.org/v2"

    def __init__(self):
        """
        Initialise the service and load the API key + last call timestamp.
        """
        # HTTP headers sent with every Semantic Scholar request
        self.headers = {}

        # Load the last call timestamp from disk for cross-process throttling
        self.last_request_time = _read_last_call_time()

        # Inject the API key if one is configured
        if settings.SEMANTIC_SCHOLAR_API_KEY:
            self.headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
            logger.info("Semantic Scholar API Key loaded (elevated rate limits active).")
        else:
            logger.info("No Semantic Scholar API Key configured (using public rate limits).")

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Throttled HTTP GET with exponential backoff
    # ──────────────────────────────────────────────────────────────────────────

    def _get_request(
        self,
        url: str,
        params: dict = None,
        retries: int = 5,
        backoff: float = 3.0
    ) -> dict | None:
        """
        Make a throttled GET request with automatic retry on rate-limit errors.

        Throttle logic:
          - Enforces at least 1.2 seconds between any two S2 requests.
          - If a 429 (Too Many Requests) is returned, waits backoff * 2^attempt
            seconds before retrying (exponential backoff).
          - The throttle timestamp is persisted to disk after each request so
            back-to-back CLI runs also respect the rate limit.

        Args:
            url: Full URL to request.
            params: Optional query parameters dict.
            retries: Maximum number of retry attempts (default: 5).
            backoff: Base back-off time in seconds (default: 3.0).

        Returns:
            Parsed JSON dict on success, or None if all attempts fail.
        """
        for attempt in range(retries):
            # ── Proactive throttle ─────────────────────────────────────────────
            # Calculate how long since the last request and sleep if needed.
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < 1.2:
                time.sleep(1.2 - elapsed)  # Sleep for the remaining fraction of 1.2s

            # Record and persist the current call time before making the request
            self.last_request_time = time.time()
            _write_last_call_time(self.last_request_time)

            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=15  # 15 second connection + read timeout
                )

                if response.status_code == 200:
                    return response.json()  # Success — return parsed JSON

                elif response.status_code in (401, 403):
                    # API Key authentication failure — fall back dynamically to public API by clearing the key
                    logger.warning(
                        f"HTTP {response.status_code} - Configured Semantic Scholar API key is invalid or unauthorized. "
                        "Falling back to public rate-limited endpoint (no key)."
                    )
                    if "x-api-key" in self.headers:
                        del self.headers["x-api-key"]
                    
                    # Immediately retry this single request without the key header
                    response = requests.get(
                        url,
                        headers=self.headers,
                        params=params,
                        timeout=15
                    )
                    if response.status_code == 200:
                        return response.json()
                    
                    logger.error(f"Fallback HTTP {response.status_code} for {url}: {response.text[:200]}")
                    break

                elif response.status_code == 429:
                    # Rate limited — wait with exponential backoff then retry
                    wait_time = backoff * (2 ** attempt)
                    logger.warning(f"Rate limit (429) hit. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    # Reset the timer so the next attempt starts fresh after the wait
                    self.last_request_time = time.time()
                    _write_last_call_time(self.last_request_time)

                else:
                    # Non-retryable HTTP error (404, 500, etc.)
                    logger.error(f"HTTP {response.status_code} for {url}: {response.text[:200]}")
                    break  # Do not retry non-429 errors

            except Exception as e:
                logger.error(f"Request exception for {url}: {e}")
                time.sleep(backoff)  # Wait before next retry after a connection error

        return None  # All attempts exhausted

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Paper Search
    # ──────────────────────────────────────────────────────────────────────────

    def search_papers(self, query: str, limit: int = 5, offset: int = 0) -> list[dict]:
        """
        Search for academic papers on Semantic Scholar.

        Args:
            query: Free-text search query (keywords, title, author name, etc.).
            limit: Maximum number of results to return (default: 5, max: 100).
            offset: Offset of the first result to return (default: 0).

        Returns:
            List of paper metadata dicts, each including title, authors, year,
            venue, externalIds (DOI, ArXiv), abstract, and citation counts.
            Returns empty list on failure.
        """
        url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            # Request only the fields we actually use — keeps response small
            "fields": "title,authors,venue,year,externalIds,abstract,citationCount,referenceCount"
        }
        logger.info(f"Searching Semantic Scholar: '{query}' (limit={limit}, offset={offset})")
        data = self._get_request(url, params=params)

        # S2 returns results nested under a "data" key
        if data and "data" in data:
            return data["data"]
        return []

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Paper Detail Lookup (5-strategy cascade)
    # ──────────────────────────────────────────────────────────────────────────

    def get_paper_details(self, paper_id: str) -> dict | None:
        """
        Fetch full metadata for a single paper using a 5-strategy lookup cascade.

        Strategies tried in order (first success wins):
          1. DOI:<doi>  prefix in URL path
          2. Bare DOI directly in URL path
          3. Full-text search, verified by exact DOI match
          4. Keyword search using the DOI suffix
          5. CrossRef → paper title → S2 title search (for Nature/Elsevier papers)

        Args:
            paper_id: Can be:
              - A canonical 40-char hex S2 paper ID
              - A DOI (with or without "DOI:" prefix)
              - An arXiv ID (with "ArXiv:" prefix)
              - A CorpusID (e.g. "CorpusID:13756489")

        Returns:
            Paper metadata dict on success, or None if all strategies fail.
        """
        params = {
            "fields": "title,authors,venue,year,externalIds,abstract,citationCount,referenceCount"
        }

        # ── Fast path: canonical 40-character hexadecimal S2 paper ID ─────────
        # S2 paper IDs are always exactly 40 lowercase hex characters.
        is_canonical = (
            len(paper_id) == 40 and
            all(c in "0123456789abcdefABCDEF" for c in paper_id)
        )
        if is_canonical:
            url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}"
            logger.info(f"Canonical S2 ID lookup: {url}")
            return self._get_request(url, params=params)

        # ── Pre-strategy: Extract and try direct arXiv lookup if identifier contains arXiv ID ──
        # e.g., 10.48550/arXiv.1706.03762, arXiv:1706.03762, or bare 1706.03762
        import re
        arxiv_match = re.search(r"(?:arXiv[:\.])?(\d{4}\.\d{4,5}(?:v\d+)?)", paper_id, re.IGNORECASE)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/ArXiv:{arxiv_id}"
            logger.info(f"Auto-detected arXiv ID '{arxiv_id}' from identifier. Attempting direct arXiv lookup: {url}")
            res = self._get_request(url, params=params)
            if res:
                return res

        # ── Normalise identifier to a bare DOI ────────────────────────────────
        if paper_id.upper().startswith("DOI:"):
            bare_doi = paper_id[4:]  # Strip "DOI:" prefix
        elif paper_id.upper().startswith("ARXIV:"):
            # arXiv IDs have a direct lookup endpoint in S2
            url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}"
            logger.info(f"arXiv direct lookup: {url}")
            return self._get_request(url, params=params)
        else:
            bare_doi = paper_id  # Treat as bare DOI

        # ── Strategy 1: DOI:<doi> in the URL path ─────────────────────────────
        # Most commonly works for well-indexed papers.
        s1_url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/DOI:{bare_doi}"
        logger.info(f"Strategy 1 — DOI prefix lookup: {s1_url}")
        res = self._get_request(s1_url, params=params)
        if res:
            return res

        # ── Strategy 2: Bare DOI directly in the URL path ─────────────────────
        s2_url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{bare_doi}"
        logger.info(f"Strategy 2 — Bare DOI path: {s2_url}")
        res = self._get_request(s2_url, params=params)
        if res:
            return res

        # ── Strategy 3: Full-text search, verify by exact DOI match ───────────
        # S2 may index the paper but not under its DOI in the URL. We search
        # by the DOI string and cross-check the externalIds field.
        logger.info(f"Strategy 3 — Full-text search by DOI string: '{bare_doi}'")
        results = self.search_papers(bare_doi, limit=10)
        for candidate in results:
            # Extract the candidate's DOI, normalised to lowercase for comparison
            c_doi = ((candidate.get("externalIds") or {}).get("DOI") or "").lower().strip()
            if c_doi == bare_doi.lower().strip():
                s2id = candidate.get("paperId")
                if s2id:
                    logger.info(f"Strategy 3 match: '{candidate.get('title')}' ({s2id})")
                    return self._get_request(
                        f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{s2id}", params=params
                    )

        # ── Strategy 4: Search by DOI suffix keyword ──────────────────────────
        # The last component after the final "/" often contains the article slug
        # which is distinctive enough to find the paper via keyword search.
        suffix = bare_doi.split("/")[-1] if "/" in bare_doi else bare_doi
        logger.info(f"Strategy 4 — Search by DOI suffix: '{suffix}'")
        results = self.search_papers(suffix, limit=10)
        for candidate in results:
            c_doi = ((candidate.get("externalIds") or {}).get("DOI") or "").lower().strip()
            if c_doi == bare_doi.lower().strip():
                s2id = candidate.get("paperId")
                if s2id:
                    logger.info(f"Strategy 4 match: '{candidate.get('title')}' ({s2id})")
                    return self._get_request(
                        f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{s2id}", params=params
                    )

        # ── Strategy 5: CrossRef → title → S2 title search ───────────────────
        # CrossRef is a free public DOI metadata registry that reliably handles
        # Nature, Elsevier, and other publisher DOIs that S2 may not index.
        # We resolve the DOI to a title via CrossRef, then search S2 by that title.
        logger.info(f"Strategy 5 — CrossRef resolution + S2 title search: '{bare_doi}'")
        title_from_crossref = self._resolve_title_via_crossref(bare_doi)

        if title_from_crossref:
            logger.info(f"CrossRef resolved title: '{title_from_crossref}'")
            title_results = self.search_papers(title_from_crossref, limit=5)

            for candidate in title_results:
                c_doi = ((candidate.get("externalIds") or {}).get("DOI") or "").lower().strip()
                # Verify match by exact DOI — never accept a candidate with no DOI
                if c_doi and c_doi == bare_doi.lower().strip():
                    s2id = candidate.get("paperId")
                    if s2id:
                        logger.info(
                            f"Strategy 5 DOI-verified match: '{candidate.get('title')}' ({s2id})"
                        )
                        return self._get_request(
                            f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{s2id}", params=params
                        )

            # Last resort: accept by title similarity (≥80% word overlap)
            for candidate in title_results:
                c_title = (candidate.get("title") or "").lower().strip()
                t_lower = title_from_crossref.lower().strip()
                t_words = set(t_lower.split())
                c_words = set(c_title.split())
                overlap = len(t_words & c_words) / max(len(t_words), 1)
                if overlap >= 0.8:
                    s2id = candidate.get("paperId")
                    if s2id:
                        logger.warning(
                            f"Strategy 5 title-match (overlap={overlap:.0%}): "
                            f"'{candidate.get('title')}' — please verify this is correct!"
                        )
                        return self._get_request(
                            f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{s2id}", params=params
                        )

        # All 5 strategies exhausted — log a helpful tip and return None
        logger.error(
            f"All 5 lookup strategies failed for: '{paper_id}'.\n"
            f"  TIP: Find the paper at https://www.semanticscholar.org, copy its CorpusID\n"
            f"       from the URL, and run: python scripts/main.py -a CorpusID:<id>"
        )
        return None

    def _resolve_title_via_crossref(self, doi: str) -> str | None:
        """
        Use the free CrossRef public API to resolve a DOI to its paper title.

        CrossRef is authoritative for publisher DOIs (Nature, Elsevier, Springer)
        that Semantic Scholar may not index under their DOI directly.
        This call does NOT count against the S2 rate limit.

        Args:
            doi: Bare DOI string (without "https://doi.org/" prefix).

        Returns:
            Title string on success, or None if the DOI is not in CrossRef.
        """
        try:
            url = f"https://api.crossref.org/works/{quote(doi, safe='/')}"
            # CrossRef recommends a polite User-Agent with a contact email
            headers = {
                "User-Agent": f"AIResearchStack/1.0 (mailto:{settings.UNPAYWALL_EMAIL})"
            }
            resp = requests.get(url, headers=headers, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                # CrossRef returns titles as a list (usually with one element)
                titles = (data.get("message") or {}).get("title") or []
                if titles:
                    return titles[0].strip()

            elif resp.status_code == 404:
                logger.info(f"CrossRef: DOI '{doi}' not found in the registry.")
            else:
                logger.warning(f"CrossRef returned HTTP {resp.status_code} for DOI '{doi}'")

        except Exception as e:
            logger.warning(f"CrossRef lookup failed for '{doi}': {e}")

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Fetch Papers That Cite a Target Paper
    # ──────────────────────────────────────────────────────────────────────────

    def get_paper_citations(self, paper_id: str, limit: int = 20) -> list[dict]:
        """
        Fetch papers that cite the target paper, including citation context snippets.

        If the provided paper_id is not a canonical S2 ID, it is first resolved
        to one using get_paper_details() to ensure the citations endpoint works.

        Args:
            paper_id: Any valid paper identifier (canonical S2 ID, DOI, arXiv, etc.).
            limit: Maximum number of citing papers to return.

        Returns:
            List of citation entry dicts, each with:
              - "citingPaper": metadata of the paper that cited the target
              - "contexts": list of short text snippets where the citation appears
              - "intents": S2-detected citation intent labels
        """
        # Check if this is already a canonical S2 ID (40 hex chars)
        is_canonical = (
            len(paper_id) == 40 and
            all(c in "0123456789abcdefABCDEF" for c in paper_id)
        )

        if is_canonical:
            s2_paper_id = paper_id
        else:
            # Resolve to a canonical S2 paperId first
            logger.info(f"Resolving paper ID to canonical form: {paper_id}")
            details = self.get_paper_details(paper_id)
            if not details or not details.get("paperId"):
                logger.error(f"Could not resolve canonical paper ID for: {paper_id}")
                return []
            s2_paper_id = details["paperId"]

        # Build the citations endpoint URL using the canonical S2 paper ID
        url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{s2_paper_id}/citations"
        params = {
            "limit": limit,
            # Request citing paper metadata AND the citation context snippets
            "fields": (
                "contexts,intents,"
                "citingPaper.title,citingPaper.authors,citingPaper.venue,"
                "citingPaper.year,citingPaper.externalIds,citingPaper.abstract"
            )
        }

        logger.info(f"Fetching citations for canonical paper ID: {s2_paper_id}")
        data = self._get_request(url, params=params)

        if data and "data" in data:
            return data["data"]
        return []

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Resolve Open-Access PDF URL via Unpaywall
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_open_access_pdf_url(self, doi: str) -> str | None:
        """
        Query the Unpaywall API to find a legal, open-access PDF URL for a DOI.

        Unpaywall aggregates open-access versions from repositories like
        PubMed Central, institutional repositories, and author self-archives.

        Args:
            doi: The paper's DOI string (with or without the https://doi.org/ prefix).

        Returns:
            Direct PDF URL string on success, or None if no OA version exists.
        """
        if not doi:
            logger.warning("No DOI provided — skipping Unpaywall lookup.")
            return None

        # Strip any URL prefix — Unpaywall expects a bare DOI
        doi = doi.replace("https://doi.org/", "").strip()

        url = f"{self.UNPAYWALL_BASE}/{quote(doi)}"
        params = {"email": settings.UNPAYWALL_EMAIL}

        logger.info(f"Querying Unpaywall for DOI: {doi}")
        data = self._get_request(url, params=params)

        if data and data.get("is_oa"):
            # Prefer the "best_oa_location" (Unpaywall's recommended source)
            best_location = data.get("best_oa_location")
            if best_location and best_location.get("url_for_pdf"):
                return best_location["url_for_pdf"]

            # Fall back to any other OA location that has a direct PDF link
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    return loc["url_for_pdf"]

        logger.info(f"No open-access PDF found on Unpaywall for DOI: {doi}")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Download PDF to papers/ Directory
    # ──────────────────────────────────────────────────────────────────────────

    def download_pdf(self, pdf_url: str, save_filename: str) -> Path | None:
        """
        Download a PDF from a URL and save it to the papers/ directory.

        Uses streaming download with tqdm progress bar to handle large files
        without loading the entire file into memory at once.

        A browser-like User-Agent header is sent to avoid bot detection by
        some academic publisher servers (e.g. Springer, arXiv).

        Args:
            pdf_url: Direct URL to the PDF file.
            save_filename: Filename (not full path) to save under in papers/.

        Returns:
            Path object pointing to the saved file on success, or None on failure.
            Partially downloaded files are deleted on failure.
        """
        save_path = settings.PDF_DOWNLOAD_DIR / save_filename

        # Mimic a modern browser to avoid being blocked by publisher servers
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:
            logger.info(f"Downloading PDF: {pdf_url}")
            # stream=True prevents loading the entire response into memory at once
            response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)

            if response.status_code != 200:
                logger.warning(
                    f"PDF download failed (HTTP {response.status_code}): {pdf_url}\n"
                    f"  → '{save_filename}' will fall back to abstract-only ingestion."
                )
                return None

            # Determine the total file size for the progress bar (may be 0 if unknown)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # Read in 1 KiB blocks

            # Write the file in blocks, updating the progress bar after each block
            with open(save_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=save_filename[:20]  # Truncate long names for the progress bar
                ) as bar:
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        f.write(data)

            logger.info(f"PDF saved to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            # Clean up any partially written file to prevent corrupt PDFs
            if save_path.exists():
                save_path.unlink()
            return None

    def fetch_crossref_metadata(self, doi: str) -> dict | None:
        """
        Query the Crossref API for verified publisher metadata of a DOI.
        """
        if not doi:
            return None
        doi = doi.replace("https://doi.org/", "").strip()
        url = f"https://api.crossref.org/works/{quote(doi, safe='/')}"
        headers = {
            "User-Agent": f"AIResearchStack/1.0 (mailto:{settings.UNPAYWALL_EMAIL})"
        }
        try:
            logger.info(f"Querying Crossref for DOI: {doi}")
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                message = data.get("message") or {}
                
                # Title
                titles = message.get("title") or []
                title = titles[0].strip() if titles else "Untitled Paper"
                
                # Format authors: list of dicts with name key
                crossref_authors = message.get("author") or []
                authors = []
                for a in crossref_authors:
                    given = a.get("given", "").strip()
                    family = a.get("family", "").strip()
                    name = f"{given} {family}".strip()
                    if name:
                        authors.append({"name": name})
                
                # Year
                created = message.get("created") or {}
                pub_print = message.get("published-print") or {}
                pub_online = message.get("published-online") or {}
                year = "N/A"
                for date_source in [pub_print, pub_online, created]:
                    date_parts = date_source.get("date-parts")
                    if date_parts and date_parts[0]:
                        year = str(date_parts[0][0])
                        break
                
                # Venue
                container = message.get("container-title") or []
                venue = container[0].strip() if container else "N/A"
                
                return {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "doi": doi
                }
            elif resp.status_code == 404:
                logger.info(f"Crossref: DOI '{doi}' not found.")
            else:
                logger.warning(f"Crossref returned HTTP {resp.status_code} for DOI '{doi}'")
        except Exception as e:
            logger.error(f"Crossref lookup failed for '{doi}': {e}")
        return None

    def fetch_openalex_metadata(self, doi_or_title: str) -> dict | None:
        """
        Query OpenAlex API by DOI or search by title.
        """
        if not doi_or_title:
            return None
            
        doi_or_title = doi_or_title.strip()
        headers = {
            "User-Agent": f"AIResearchStack/1.0 (mailto:{settings.UNPAYWALL_EMAIL})"
        }
        
        # Determine if it's a DOI or a title query
        import re
        is_doi = bool(re.match(r"^(10\.\d{4,9}/[-._;()/:A-Z0-9]+)$", doi_or_title, re.IGNORECASE) or "doi.org" in doi_or_title)
        
        if is_doi:
            bare_doi = doi_or_title.replace("https://doi.org/", "").strip()
            url = f"https://api.openalex.org/works/doi:{bare_doi}"
            params = {}
            logger.info(f"Querying OpenAlex by DOI: {bare_doi}")
        else:
            url = "https://api.openalex.org/works"
            params = {"search": doi_or_title, "per_page": 1}
            logger.info(f"Querying OpenAlex by Title search: '{doi_or_title}'")
            
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                
                # If searching, data is a list under "results"
                if not is_doi:
                    results = data.get("results") or []
                    if not results:
                        logger.info("OpenAlex title search returned no results.")
                        return None
                    work = results[0]
                else:
                    work = data
                
                # Title
                title = work.get("title") or "Untitled Paper"
                
                # Format authors to list of {"name": name}
                authorships = work.get("authorships") or []
                authors = []
                for auth in authorships:
                    author_meta = auth.get("author") or {}
                    name = author_meta.get("display_name", "").strip()
                    if name:
                        authors.append({"name": name})
                
                # Year
                year = str(work.get("publication_year") or "N/A")
                
                # Venue
                primary_loc = work.get("primary_location") or {}
                source = primary_loc.get("source") or {}
                venue = source.get("display_name") or "N/A"
                
                # DOI
                resolved_doi = work.get("doi") or ""
                if resolved_doi:
                    resolved_doi = resolved_doi.replace("https://doi.org/", "").strip()
                else:
                    resolved_doi = bare_doi if is_doi else "N/A"
                    
                # Reconstruct Abstract if inverted index exists
                abstract = ""
                abstract_index = work.get("abstract_inverted_index")
                if abstract_index:
                    try:
                        word_list = []
                        for word, positions in abstract_index.items():
                            for pos in positions:
                                word_list.append((pos, word))
                        word_list.sort()
                        abstract = " ".join([w[1] for w in word_list])
                    except Exception as abs_err:
                        logger.warning(f"Failed to reconstruct abstract from OpenAlex index: {abs_err}")
                
                return {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "doi": resolved_doi,
                    "abstract": abstract
                }
            else:
                logger.warning(f"OpenAlex returned HTTP {resp.status_code} for query '{doi_or_title}'")
        except Exception as e:
            logger.error(f"OpenAlex lookup failed for '{doi_or_title}': {e}")
        return None

