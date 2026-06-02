"""
pdf_processor.py — PDF Text Extraction and Chunking Service.

Uses PyMuPDF (imported as 'fitz') to open PDF files page-by-page, clean
the extracted raw text, and split it into overlapping chunks suitable for
embedding and storage in the ChromaDB vector database.

This module has NO dependency on config.py — it accepts paths as arguments,
making it easy to test independently.
"""

import re
import logging
from pathlib import Path
import fitz  # PyMuPDF — install with: pip install pymupdf

# ── Logger setup ──────────────────────────────────────────────────────────────
# Uses Python's standard logging so output respects the root logger config
# set up in main.py / server.py.
logger = logging.getLogger(__name__)


class PDFProcessorService:
    """
    Service responsible for two tasks:
      1. Extracting clean text from each page of a PDF file.
      2. Splitting the full document text into overlapping chunks for RAG.

    Typical usage:
        service = PDFProcessorService()
        pages  = service.extract_text_by_page(Path("paper.pdf"))
        chunks = service.chunk_text(pages, chunk_size=1000, chunk_overlap=200)
    """

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Text Cleaning
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize raw text extracted from a PDF page.

        PDF extraction often produces noisy output:
          - Windows-style \\r\\n line endings
          - Words split by hyphens across line breaks (e.g. "atten-\\ntion")
          - Single newlines inside paragraphs (should be spaces)
          - Multiple consecutive spaces and tabs

        Steps applied:
          1. Normalize all line endings to \\n.
          2. Rejoin hyphenated line-break words into a single word.
          3. Split on double-newlines (paragraph boundaries) and clean each
             paragraph's internal single newlines into spaces.
          4. Collapse multiple spaces/tabs into a single space.

        Args:
            text: Raw string from PyMuPDF's page.get_text("text").

        Returns:
            Cleaned, normalized string ready for chunking.
        """
        if not text:
            return ""

        # Step 1: Normalize line endings (\r\n and \r → \n)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Step 2: Rejoin words hyphenated across line breaks
        # Pattern: word characters, literal hyphen, newline, optional whitespace, more word chars
        # e.g. "atten-\n  tion" → "attention"
        text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)

        # Step 3: Process paragraph by paragraph
        # Split on double newlines (true paragraph boundaries)
        paragraphs = text.split("\n\n")
        cleaned_paragraphs = []
        for para in paragraphs:
            # Replace single newlines within the paragraph with a space
            # The negative lookbehind/ahead ensures we only target single \n, not \n\n
            cleaned_para = re.sub(r"(?<!\n)\n(?!\n)", " ", para)
            # Collapse any sequence of spaces or tabs into a single space
            cleaned_para = re.sub(r"[ \t]+", " ", cleaned_para).strip()
            if cleaned_para:  # Discard empty paragraphs
                cleaned_paragraphs.append(cleaned_para)

        # Re-join cleaned paragraphs with the standard double-newline separator
        return "\n\n".join(cleaned_paragraphs)

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Page-Level Text Extraction
    # ──────────────────────────────────────────────────────────────────────────

    def extract_text_by_page(self, pdf_path: Path | str) -> tuple[list[dict], bool]:
        """
        Open a PDF file and extract cleaned text from every page.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            Tuple of (pages, has_full_text):
              - pages: List of dicts, one per page, each containing:
                - "page_number" (int): 1-indexed page number.
                - "text" (str): Cleaned text content of that page.
              - has_full_text: bool, True if sufficient text was extracted (likely full PDF),
                False if text is minimal (likely abstract-only or scan).

        Raises:
            FileNotFoundError: If the PDF does not exist at the given path.
            Exception: Any error from PyMuPDF during open/read is re-raised.
        """
        pdf_path = Path(pdf_path)  # Ensure we have a Path object, not a string

        # Guard: file must exist before we attempt to open it
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Opening PDF for text extraction: {pdf_path.name}")
        pages = []
        total_chars = 0

        try:
            # fitz.open() returns a Document context manager
            # Using 'with' ensures the file handle is closed even on error
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                logger.info(f"Opened '{pdf_path.name}' — {total_pages} pages total")

                # Iterate every page (0-indexed internally, exposed as 1-indexed)
                for page_idx in range(total_pages):
                    page = doc[page_idx]
                    # "text" layout mode extracts text in reading order
                    raw_text = page.get_text("text")
                    # Apply cleaning pipeline to remove PDF extraction noise
                    cleaned_text = self._clean_text(raw_text)
                    total_chars += len(cleaned_text)

                    pages.append({
                        "page_number": page_idx + 1,  # Convert to 1-indexed for humans
                        "text": cleaned_text
                    })

            # Determine if this is likely full text or abstract-only
            # Abstract-only papers typically have < 2000 characters
            # Full papers typically have > 5000 characters
            has_full_text = total_chars >= 2000
            if not has_full_text:
                logger.warning(
                    f"Extracted minimal text ({total_chars} chars) from '{pdf_path.name}' "
                    "- likely abstract-only or scanned PDF. Marking as has_full_text=False."
                )

            logger.info(f"Extracted text from {len(pages)} pages of '{pdf_path.name}' (has_full_text={has_full_text})")
            return pages, has_full_text

        except Exception as e:
            logger.error(f"Error extracting text from '{pdf_path.name}': {e}")
            raise  # Re-raise so the caller can handle or log the failure

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Text Chunking
    # ──────────────────────────────────────────────────────────────────────────

    def _strip_references_section(self, full_text: str) -> str:
        """
        Remove the References/Bibliography section from the end of a document's
        full text BEFORE chunking.

        Why this matters: Academic PDFs end with a reference list. If those
        reference entries are chunked and stored in ChromaDB, the LLM retrieves
        them and falsely cites papers in the reference list as if they are
        separately ingested sources — causing hallucinated citations.

        Strategy:
          - Match common reference section headers in the LAST 40% of the text.
          - Strip everything from that header onwards.
          - Only acts on the last match to avoid removing body text that happens
            to contain the word "references" in a section heading.

        Args:
            full_text: The complete concatenated document text.

        Returns:
            The document text with the reference section removed, or the
            original text unchanged if no reference section header was found.
        """
        if not full_text or len(full_text) < 500:
            return full_text

        # Only scan the last 40% of the document — reference lists are always at the end.
        scan_start = int(len(full_text) * 0.60)
        tail = full_text[scan_start:]

        # Common reference section headers used in academic papers
        ref_header_pattern = re.compile(
            r'\n\s*(?:'
            r'references'
            r'|bibliography'
            r'|works cited'
            r'|literature cited'
            r'|reference list'
            r'|citations'
            r'|referenzen'          # German
            r'|bibliographie'       # French/German
            r'|bibliograf[íi]a'    # Spanish/Portuguese
            r')\s*\n',
            re.IGNORECASE
        )

        match = None
        for m in ref_header_pattern.finditer(tail):
            match = m  # Keep the LAST match (in case "references" appears mid-paper)

        if match:
            cut_pos = scan_start + match.start()
            stripped = full_text[:cut_pos].rstrip()
            logger.info(
                f"Stripped references section starting at char {cut_pos} "
                f"(removed {len(full_text) - cut_pos} chars of bibliography)."
            )
            return stripped

        return full_text

    def chunk_text(
        self,
        pages: list[dict],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> list[dict]:
        """
        Chunk the full document text into overlapping segments.

        Why overlap? Sliding the window with overlap ensures that context
        spanning a chunk boundary is captured in at least one chunk.
        This prevents important passages from being split mid-sentence and
        lost between chunks during retrieval.

        Algorithm:
          1. Concatenate all page texts into one big string.
          2. Strip the references/bibliography section from the end (prevents
             reference list entries from becoming retrievable chunks that the
             LLM falsely cites as ingested papers).
          3. Track which page number each character position belongs to
             (so we can annotate each chunk with its source pages).
          4. Slide a window of 'chunk_size' characters across the full text,
             advancing by (chunk_size - chunk_overlap) each step.

        Args:
            pages: List of page dicts as returned by extract_text_by_page().
            chunk_size: Maximum character length of each chunk (default: 1000).
            chunk_overlap: Characters of overlap between adjacent chunks (default: 200).

        Returns:
            List of chunk dicts, each containing:
              - "chunk_index" (int): Sequential chunk number.
              - "text" (str): The chunk text content.
              - "metadata" (dict): Pages spanned, character offsets, and length.
        """
        if not pages:
            return []

        # ── Step 1: Build the full concatenated text with a character→page map ──
        full_text = ""
        # char_to_page[i] = page_number that character i belongs to
        char_to_page = []

        for page in pages:
            page_num = page["page_number"]
            page_text = page["text"]

            if not page_text:
                continue  # Skip blank pages (e.g., cover pages with only images)

            # Separate pages with a double newline so paragraph context is preserved
            if full_text:
                spacer = "\n\n"
                full_text += spacer
                # Map the spacer characters to the current page (arbitrary but consistent)
                char_to_page.extend([page_num] * len(spacer))

            # Append this page's text and map every char to its page number
            full_text += page_text
            char_to_page.extend([page_num] * len(page_text))

        text_len = len(full_text)
        if text_len == 0:
            logger.warning("No text content found in any page — nothing to chunk.")
            return []

        # ── Step 2: Strip references section to prevent false LLM citations ───
        # Reference list entries, if chunked into ChromaDB, get retrieved and
        # the LLM cites them as if they are separately ingested papers.
        full_text = self._strip_references_section(full_text)
        # Rebuild char_to_page to match the (possibly shortened) full_text
        char_to_page = char_to_page[:len(full_text)]
        text_len = len(full_text)

        if text_len == 0:
            logger.warning("Document was entirely a reference list — nothing to chunk.")
            return []

        # ── Step 3: Validate and sanitize chunking parameters ─────────────────
        if chunk_size <= 0:
            chunk_size = 1000  # Enforce a sensible minimum

        if chunk_overlap >= chunk_size or chunk_overlap < 0:
            # If overlap is invalid (negative or larger than chunk), default to 20%
            chunk_overlap = int(chunk_size * 0.2)
            logger.warning(f"Invalid chunk_overlap — reset to 20% of chunk_size: {chunk_overlap}")

        # ── Step 4: Slide the window across the full text ─────────────────────
        chunks = []
        start = 0       # Current window start position (character index)
        chunk_idx = 0   # Sequential chunk counter

        while start < text_len:
            # Determine the end of this chunk (bounded by the total length)
            end = min(start + chunk_size, text_len)

            # Skip creating a tiny tail chunk (< 100 chars) at the very end
            # to avoid near-empty vectors that add noise to retrieval results
            if len(chunks) > 0 and (end - start) < 100:
                break

            # Extract the chunk text from the full concatenated document
            chunk_text_content = full_text[start:end]

            # Determine which page numbers this chunk spans
            # Using a set ensures each page is listed only once, even if the
            # chunk straddles a page boundary.
            chunk_pages = sorted(set(char_to_page[start:end]))

            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text_content.strip(),  # Remove leading/trailing whitespace
                "metadata": {
                    "pages": chunk_pages,       # List of page numbers this chunk spans
                    "char_start": start,         # Start character offset in the full text
                    "char_end": end,             # End character offset in the full text
                    "length": len(chunk_text_content)  # Actual character count
                }
            })

            chunk_idx += 1

            # If we've consumed the entire document, we're done
            if end == text_len:
                break

            # Advance the start position by (chunk_size - overlap)
            # The overlapping portion will be re-included in the next chunk
            start += (chunk_size - chunk_overlap)

        logger.info(
            f"Chunked document into {len(chunks)} chunks "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )
        return chunks

