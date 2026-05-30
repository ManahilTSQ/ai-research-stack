"""
citation_analyzer.py — Citation Intent Classification Pipeline.

Orchestrates the full citation analysis workflow for a target academic paper:

  1. Fetch target paper metadata from Semantic Scholar.
  2. Retrieve a list of papers that cite the target.
  3. For each citing paper:
       a. Try to download its open-access PDF via Unpaywall.
       b. Extract pages of text with PyMuPDF.
       c. Search the text for passages referencing the target paper.
       d. Fall back to Semantic Scholar API-provided context snippets if no PDF.
  4. Send each extracted passage to the local Ollama LLM for classification
     into one of: supporting | contrasting | extending | methodological.
  5. Write all results to a CSV report in output/.

The pipeline is invoked via main.py (--analyze-citations) or server.py (API).
"""

import re
import csv
import logging
import requests
from pathlib import Path

# Flat imports — scripts/ directory is on sys.path when this module is loaded
from config import settings
from paper_discovery import PaperDiscoveryService
from pdf_processor import PDFProcessorService
# Reuse the shared Ollama health-check so citation analysis also fails fast
from rag_service import check_ollama_health


# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class CitationAnalyzerService:
    """
    Service to analyse how other papers cite a target paper.

    Uses a two-tier passage extraction strategy:
      Tier A: Download citing paper PDF → extract text → regex search
      Tier B: Fall back to Semantic Scholar API citation context snippets

    Then classifies each passage with the local Ollama LLM into one of four
    citation intent categories defined by academic NLP literature.
    """

    def __init__(self):
        """Initialise sub-services needed by the citation pipeline."""
        logger.info("Initialising Citation Analysis Service...")
        # Paper discovery handles Semantic Scholar API + PDF download
        self.discover_service = PaperDiscoveryService()
        # PDF processor handles text extraction from downloaded citing PDFs
        self.pdf_service = PDFProcessorService()
        logger.info("Citation Analysis Service initialised successfully.")

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Utility
    # ──────────────────────────────────────────────────────────────────────────

    def _slugify(self, text: str) -> str:
        """
        Convert a string to a filesystem-safe slug for use in filenames.

        Replaces anything that isn't alphanumeric, underscore, or hyphen
        with an underscore and lowercases the result.
        """
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", text).lower()

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Passage Extraction from PDF Text
    # ──────────────────────────────────────────────────────────────────────────

    def _strip_references_from_pages(self, pages: list[dict]) -> list[dict]:
        """
        Identify the bibliography/references section page and truncate it and
        all subsequent pages to prevent matching author names in the reference list.
        """
        if not pages:
            return []

        # We only look in the last 40% of the pages list where bibliography sections reside
        num_pages = len(pages)
        start_idx = int(num_pages * 0.60)

        ref_header_pattern = re.compile(
            r'(?:^|\n)\s*(?:'
            r'references'
            r'|bibliography'
            r'|works cited'
            r'|literature cited'
            r'|reference list'
            r'|citations'
            r'|referenzen'
            r'|bibliographie'
            r'|bibliograf[íi]a'
            r')\s*(?:\n|$)',
            re.IGNORECASE
        )

        for i in range(start_idx, num_pages):
            page_text = pages[i]["text"]
            match = ref_header_pattern.search(page_text)
            if match:
                logger.info(
                    f"Found reference section header on page {pages[i]['page_number']} of the citing paper. "
                    f"Truncating pages list here to prevent false author reference matching."
                )
                # Keep pages up to the reference start, copy the references page and truncate it, then drop subsequent pages
                truncated_pages = list(pages[:i])
                last_page_copy = pages[i].copy()
                last_page_copy["text"] = page_text[:match.start()].rstrip()
                truncated_pages.append(last_page_copy)
                return truncated_pages

        return pages

    def _extract_citation_passages_from_text(
        self,
        pages: list[dict],
        author_surnames: list[str],
        target_title: str
    ) -> list[str]:
        """
        Search page texts for sentences that reference the target paper.

        Uses regex patterns built from:
          a) Author surnames (e.g. "Vaswani", "Devlin") — whole-word, case-insensitive.
          b) The first two distinctive words from the target paper title.

        For each matched sentence, captures 1 sentence before and after for context
        (sentence-window extraction), giving the LLM enough surrounding text to
        classify the citation intent accurately.

        Args:
            pages: List of page dicts from PDFProcessorService.extract_text_by_page().
            author_surnames: List of last-name strings extracted from the target paper's authors.
            target_title: Full title of the target paper (used to build a phrase pattern).

        Returns:
            List of unique passage strings found across all pages.
            Returns empty list if no matches are found.
        """
        extracted_passages = []
        if not pages:
            return []

        # ── Build search patterns ─────────────────────────────────────────────
        patterns = []

        # Pattern A: Match any author surname as a full word (not "Attention" in "inattention")
        for surname in author_surnames:
            if len(surname) > 2:  # Skip very short surnames that would cause false matches
                patterns.append(re.compile(r"\b" + re.escape(surname) + r"\b", re.IGNORECASE))

        # Pattern B: Match the first 2 distinctive title words as a phrase
        # Exclude generic stopwords that appear in many paper titles
        stopwords = {
            "attention", "need", "networks", "generation", "augmented",
            "retrieval", "using", "about", "their", "under", "learning",
            "based", "with", "from", "deep", "large"
        }
        title_words = [
            w for w in target_title.split()
            if len(w) > 4 and w.lower() not in stopwords
        ]
        if title_words:
            phrase = " ".join(title_words[:2])  # Use first 2 distinctive words as a phrase
            patterns.append(re.compile(re.escape(phrase), re.IGNORECASE))

        # Fallback: use the first 20 characters of the title if no words survived filtering
        if not patterns:
            patterns.append(re.compile(re.escape(target_title[:20]), re.IGNORECASE))

        # ── Search each page ──────────────────────────────────────────────────
        for page in pages:
            page_text = page["text"]
            if not page_text:
                continue

            # Split page text into sentences using punctuation boundaries.
            # The lookbehind patterns prevent splitting on common abbreviations
            # like "Fig." "et al." "e.g." that end with a period but aren't sentence ends.
            sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", page_text)

            for idx, sentence in enumerate(sentences):
                # Check whether this sentence matches any of our target patterns
                matched = any(pattern.search(sentence) for pattern in patterns)

                if matched:
                    # Capture context window: 1 sentence before + matched + 1 after
                    start_idx = max(0, idx - 1)
                    end_idx = min(len(sentences), idx + 2)
                    passage = " ".join(sentences[start_idx:end_idx]).strip()

                    # De-duplicate and discard trivially short passages (< 20 chars)
                    if passage not in extracted_passages and len(passage) > 20:
                        extracted_passages.append(passage)

        return extracted_passages

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: LLM Classification of a Single Passage
    # ──────────────────────────────────────────────────────────────────────────

    def classify_passage(self, passage: str, target_title: str) -> tuple[str, str]:
        """
        Call the local Ollama LLM to classify a citation passage's intent.

        The LLM is instructed (via a structured system prompt) to return exactly
        one of four citation intent categories defined in academic citation analysis:

          supporting    — The citing text agrees with, validates, or confirms the target.
          contrasting   — The citing text disagrees with or notes limitations of the target.
          extending     — The citing text builds upon, applies, or adapts the target's method.
          methodological — The citing text reuses the target's dataset, tools, or framework.

        Uses temperature=0.0 for fully deterministic, reproducible classification.

        Args:
            passage: The extracted citation passage text (max 2000 chars).
            target_title: Title of the paper being cited.

        Returns:
            Tuple of (category_label, rationale_string).
            Falls back to ("extending", "LLM unavailable.") on error.
        """
        # ── Construct the classification system prompt ─────────────────────────
        system_prompt = (
            "You are an academic NLP researcher specializing in citation intent classification.\n"
            "Your task is to analyze the provided citation passage where a target paper is cited, "
            "and classify the citation relationship into exactly one of these four categories:\n\n"
            "1. supporting: The citing text agrees with, validates, supports, or confirms "
            "the target paper's claims/findings.\n"
            "2. contrasting: The citing text disagrees with, notes limitations of, points out "
            "errors in, or compares negatively to the target paper.\n"
            "3. extending: The citing text builds upon, applies, adapts, or generalizes "
            "the target paper's model/findings/algorithms to a new domain.\n"
            "4. methodological: The citing text uses the target paper's dataset, methodology, "
            "framework, software, or tools directly.\n\n"
            "Response Format — You MUST strictly respond in the following format:\n"
            "Category: <one of: supporting, contrasting, extending, methodological>\n"
            "Rationale: <one sentence explaining the choice based strictly on the text>"
        )

        # ── Construct the user prompt with the actual passage ──────────────────
        user_prompt = (
            f'Target Paper: "{target_title}"\n'
            f"Citation Passage:\n"
            f"{'─' * 80}\n"
            f'"{passage}"\n'
            f"{'─' * 80}\n\n"
            f"Provide your classification below:"
        )

        # ── Call Ollama /api/chat ──────────────────────────────────────────────
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.0}  # Deterministic output for reproducible results
        }

        try:
            response = requests.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT)

            if response.status_code == 200:
                content = response.json()["message"]["content"].strip()

                # Parse the structured "Category: X\nRationale: Y" response
                category_match = re.search(r"Category:\s*(\w+)", content, re.IGNORECASE)
                rationale_match = re.search(r"Rationale:\s*(.*)", content, re.IGNORECASE)

                # Extract or default values
                category = category_match.group(1).lower() if category_match else "extending"
                rationale = (
                    rationale_match.group(1)
                    if rationale_match
                    else content.replace("\n", " ")
                )

                # Validate that the category is one of the four allowed values
                valid_categories = {"supporting", "contrasting", "extending", "methodological"}
                if category not in valid_categories:
                    logger.warning(
                        f"LLM returned unexpected category '{category}' — defaulting to 'extending'"
                    )
                    category = "extending"

                return category, rationale.strip()

        except Exception as e:
            logger.error(f"Error calling Ollama for citation classification: {e}")

        # Safe fallback: return a neutral category with an explanation
        return "extending", "Local LLM classification unavailable."

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Full Citation Analysis Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_citations(self, paper_id: str, limit: int = 5) -> Path | None:
        """
        Execute the complete citation analysis pipeline for a target paper.

        Steps:
          1. Health-check Ollama (fail fast before expensive API calls).
          2. Fetch target paper metadata (title, authors) from Semantic Scholar.
          3. Fetch all papers that cite the target.
          4. For each citing paper:
               a. Attempt Unpaywall PDF download → extract passages.
               b. Fall back to S2 API context snippets if PDF unavailable.
          5. Classify each passage via the local LLM.
          6. Write all results to output/citation_analysis_<id>.csv.

        Args:
            paper_id: DOI, arXiv ID, S2 CorpusID, or canonical 40-hex S2 paper ID.
            limit: Maximum number of citing papers to analyse (default: 5).

        Returns:
            Path to the generated CSV report on success, or None on failure.
        """
        # ── Step 1: Fail fast if Ollama is not running ─────────────────────────
        # check_ollama_health() prints a clear error message if Ollama is offline.
        if not check_ollama_health():
            logger.error("Citation analysis aborted: Ollama is not reachable.")
            return None

        # ── Step 2: Fetch target paper metadata ───────────────────────────────
        logger.info(f"Fetching target paper metadata for: {paper_id}")
        target_paper = self.discover_service.get_paper_details(paper_id)
        if not target_paper:
            logger.error(f"Could not retrieve metadata for target paper: {paper_id}")
            return None

        target_title = target_paper.get("title", "Unknown Target")

        # Extract last names of all authors for use in citation passage search
        authors = target_paper.get("authors", [])
        author_surnames = []
        for author in authors:
            name = author.get("name", "")
            if name:
                # Take the last word of the name as the surname; strip punctuation
                surname = name.split()[-1].strip("*,; ")
                author_surnames.append(surname)

        logger.info(
            f"Target: '{target_title}' | Authors: {author_surnames} | "
            f"Analysing up to {limit} citing papers..."
        )

        # ── Step 3: Fetch citing papers ────────────────────────────────────────
        # Fetch a large batch (up to 50) then filter/sort to prioritise
        # open-access papers so we maximise the number of classifiable entries.
        fetch_limit = max(limit * 6, 50)
        all_citing_entries = self.discover_service.get_paper_citations(
            target_paper.get("paperId", paper_id),
            limit=fetch_limit
        )
        if not all_citing_entries:
            logger.warning(f"No citing papers found for: {paper_id}")
            return None

        # ── Prioritise: arXiv papers first, then papers with API context snippets,
        # then the rest. This maximises classification success rate.
        def oa_priority(entry):
            cp = entry.get("citingPaper", {})
            ext = cp.get("externalIds") or {}
            has_arxiv   = 1 if ext.get("ArXiv") else 0
            has_context = 1 if entry.get("contexts") else 0
            year        = cp.get("year") or 0
            # Score: arXiv > has context > older year (older = more likely OA)
            return (has_arxiv * 100) + (has_context * 10) + (1 if year < 2024 else 0)

        all_citing_entries.sort(key=oa_priority, reverse=True)
        citing_entries = all_citing_entries[:limit]
        logger.info(
            f"Filtered {len(all_citing_entries)} citing papers → "
            f"top {len(citing_entries)} prioritised for open-access."
        )

        # ── Step 4 & 5: Process each citing paper and write CSV ────────────────
        # Create the output/ directory if it doesn't exist (config.py does this,
        # but be defensive in case this service is used standalone).
        reports_dir = settings.BASE_DIR / "output"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate a safe CSV filename from the paper ID
        csv_filename = f"citation_analysis_{self._slugify(paper_id[:25])}.csv"
        csv_path = reports_dir / csv_filename
        logger.info(f"Writing citation analysis report to: {csv_path}")

        try:
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Write the CSV header row
                writer.writerow([
                    "Citing Paper Title", "Year",
                    "Extracted Passage", "LLM Classification", "Rationale"
                ])

                # Process each citing paper entry from the S2 API response
                for entry in citing_entries:
                    citing_paper = entry.get("citingPaper")
                    if not citing_paper:
                        continue  # Skip malformed entries

                    citing_title = citing_paper.get("title", "Untitled Citing Paper")
                    citing_year  = citing_paper.get("year", "N/A")
                    citing_doi   = citing_paper.get("externalIds", {}).get("DOI")
                    # API-provided citation context snippets (short, abstract-level)
                    api_contexts = entry.get("contexts", [])

                    print(f"\nAnalysing: '{citing_title}' (DOI: {citing_doi or 'N/A'})")
                    passages = []

                    # ── Tier A: Download PDF and extract full-text passages ──────
                    # Try Unpaywall first, then arXiv direct if the paper has an ArXiv ID
                    citing_ext    = citing_paper.get("externalIds") or {}
                    citing_arxiv  = citing_ext.get("ArXiv")
                    pdf_url       = None

                    if citing_doi:
                        print("  [+] Querying Unpaywall for open-access PDF...")
                        pdf_url = self.discover_service.fetch_open_access_pdf_url(citing_doi)

                    # ArXiv fallback — works for most CS/ML papers
                    if not pdf_url and citing_arxiv:
                        pdf_url = f"https://arxiv.org/pdf/{citing_arxiv}"
                        print(f"  [+] Unpaywall miss — trying arXiv direct: {pdf_url}")

                    if pdf_url:
                        safe_name = "citing_" + self._slugify(citing_title)[:40] + ".pdf"
                        print(f"  [+] Downloading: {safe_name}")
                        downloaded_path = self.discover_service.download_pdf(pdf_url, safe_name)

                        if downloaded_path and downloaded_path.exists():
                            print("  [+] Extracting text and searching for citation markers...")
                            try:
                                pages = self.pdf_service.extract_text_by_page(downloaded_path)
                                pages = self._strip_references_from_pages(pages)
                                passages = self._extract_citation_passages_from_text(
                                    pages, author_surnames, target_title
                                )
                                downloaded_path.unlink()
                            except Exception as e:
                                logger.error(f"Failed to process citing PDF: {e}")
                                if downloaded_path.exists():
                                    downloaded_path.unlink()

                    # ── Tier B: Fallback to API-provided citation context snippets ─
                    if not passages and api_contexts:
                        logger.warning(
                            f"⚠  PDF unavailable for '{citing_title}' "
                            f"(DOI: {citing_doi or 'N/A'}). "
                            "Using Semantic Scholar API context snippets (abstract-level only)."
                        )
                        print(
                            f"  ⚠  PDF not available. Using S2 API context snippets "
                            f"(abstract-level only) for '{citing_title}'."
                        )
                        passages = api_contexts

                    if not passages:
                        print("  [-] No citation passage context could be extracted. Saving structured placeholder row.")
                        writer.writerow([
                            citing_title,
                            citing_year,
                            "No citation passage could be extracted (paper is paywalled and no API snippet is available).",
                            "unknown",
                            "Skipped due to lack of accessible PDF or context snippet."
                        ])
                        f.flush()
                        continue

                    # ── Classify each extracted passage ───────────────────────────
                    for passage in passages:
                        # Clean up the passage text: collapse newlines, trim whitespace
                        clean_passage = passage.replace("\n", " ").strip()

                        # Truncate extreme-length passages for CSV readability and LLM efficiency
                        if len(clean_passage) > 2000:
                            clean_passage = clean_passage[:2000] + "..."

                        print("  [+] Classifying passage via local LLM...")
                        category, rationale = self.classify_passage(clean_passage, target_title)
                        print(f"      → Classification: {category.upper()}")

                        # Write this passage's classification row to the CSV
                        writer.writerow([
                            citing_title,
                            citing_year,
                            clean_passage,
                            category,
                            rationale
                        ])
                        # Flush after each row so partial results are saved even on crash
                        f.flush()

            print(f"\n[+] Citation Analysis complete! Report saved to: {csv_path}")
            return csv_path

        except Exception as e:
            logger.error(f"Error during citation analysis pipeline: {e}")
            raise  # Re-raise so the caller (main.py / server.py) can log it properly