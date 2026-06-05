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
            # Abstract-only papers typically have < 8000 characters
            # Full papers typically have > 8000 characters
            has_full_text = total_chars >= 8000
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
            r'\n\s*(?:\d+[\.\s]*)?\s*(?:'
            r'references'
            r'|bibliography'
            r'|works cited'
            r'|literature cited'
            r'|reference list'
            r'|citations'
            r'|referenzen'          # German
            r'|bibliographie'       # French/German
            r'|bibliograf[íi]a'    # Spanish/Portuguese
            r')\b[\.:\s]*\n',
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

    def _detect_section_boundaries(self, full_text: str) -> list[tuple[int, str]]:
        """
        Detect section headings in the document text.

        Returns a list of (position, section_name) tuples for detected sections.
        Common academic section headings are detected.
        """
        section_patterns = [
            r'\n\s*(?:Abstract|ABSTRACT)\s*\n',
            r'\n\s*(?:Introduction|INTRODUCTION)\s*\n',
            r'\n\s*(?:Background|BACKGROUND)\s*\n',
            r'\n\s*(?:Related Work|RELATED WORK|Literature Review|LITERATURE REVIEW)\s*\n',
            r'\n\s*(?:Methodology|METHODOLOGY|Methods|METHODS)\s*\n',
            r'\n\s*(?:Method|METHOD)\s*\n',
            r'\n\s*(?:Experimental Setup|EXPERIMENTAL SETUP)\s*\n',
            r'\n\s*(?:Results|RESULTS)\s*\n',
            r'\n\s*(?:Discussion|DISCUSSION)\s*\n',
            r'\n\s*(?:Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\s*\n',
            r'\n\s*(?:Future Work|FUTURE WORK)\s*\n',
            r'\n\s*(?:References|REFERENCES)\s*\n',
        ]
        
        boundaries = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                section_name = match.group(0).strip()
                boundaries.append((match.start(), section_name))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        return boundaries

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """
        Split text into paragraphs based on double newlines.
        Filters out empty paragraphs and very short ones.
        """
        paragraphs = text.split('\n\n')
        cleaned = []
        for para in paragraphs:
            para = para.strip()
            if len(para) >= 50:  # Keep only substantial paragraphs
                cleaned.append(para)
        return cleaned

    def chunk_text(
        self,
        pages: list[dict],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_structure_aware: bool = True
    ) -> list[dict]:
        """
        Chunk the full document text into overlapping segments.

        Structure-aware mode (default):
          - Detects section headings (Abstract, Introduction, Methodology, etc.)
          - Splits by sections first to preserve semantic boundaries
          - Within sections, splits by paragraphs (not fixed character count)
          - Overlap preserves last complete paragraph/sentence, not just characters

        Legacy character-based mode (use_structure_aware=False):
          - Original sliding window approach by character count
          - Kept for backward compatibility

        Args:
            pages: List of page dicts as returned by extract_text_by_page().
            chunk_size: Target chunk size in characters (soft limit in structure-aware mode).
            chunk_overlap: Overlap in characters (used in legacy mode).
            use_structure_aware: If True, use semantic section-aware chunking.

        Returns:
            List of chunk dicts, each containing:
              - "chunk_index" (int): Sequential chunk number.
              - "text" (str): The chunk text content.
              - "metadata" (dict): Pages spanned, section name, character offsets, and length.
        """
        if not pages:
            return []

        # ── Step 1: Build the full concatenated text with a character→page map ──
        full_text = ""
        char_to_page = []

        for page in pages:
            page_num = page["page_number"]
            page_text = page["text"]

            if not page_text:
                continue

            if full_text:
                spacer = "\n\n"
                full_text += spacer
                char_to_page.extend([page_num] * len(spacer))

            full_text += page_text
            char_to_page.extend([page_num] * len(page_text))

        text_len = len(full_text)
        if text_len == 0:
            logger.warning("No text content found in any page — nothing to chunk.")
            return []

        # ── Step 2: Strip references section to prevent false LLM citations ───
        full_text = self._strip_references_section(full_text)
        char_to_page = char_to_page[:len(full_text)]
        text_len = len(full_text)

        if text_len == 0:
            logger.warning("Document was entirely a reference list — nothing to chunk.")
            return []

        # ── Step 3: Choose chunking strategy ────────────────────────────────
        if use_structure_aware:
            return self._chunk_structure_aware(full_text, char_to_page, chunk_size)
        else:
            return self._chunk_character_based(full_text, char_to_page, chunk_size, chunk_overlap)

    def _chunk_structure_aware(
        self,
        full_text: str,
        char_to_page: list[int],
        target_chunk_size: int = 1000
    ) -> list[dict]:
        """
        Structure-aware chunking that respects document sections.

        Algorithm:
          1. Detect section boundaries (Abstract, Introduction, etc.)
          2. Split document into sections
          3. Within each section, split into paragraphs
          4. Group paragraphs into chunks that respect target size
          5. Add semantic overlap (last complete paragraph/sentence)
        """
        # Detect section boundaries
        section_boundaries = self._detect_section_boundaries(full_text)
        
        # If no sections detected, fall back to paragraph-based chunking
        if not section_boundaries:
            logger.info("No clear section boundaries detected, using paragraph-based chunking")
            return self._chunk_paragraph_based(full_text, char_to_page, target_chunk_size)
        
        # Build sections
        sections = []
        prev_pos, prev_section = 0, "Introduction"
        
        for pos, section_name in section_boundaries:
            if pos > prev_pos:
                section_text = full_text[prev_pos:pos].strip()
                if section_text:
                    sections.append({
                        "name": prev_section,
                        "start": prev_pos,
                        "end": pos,
                        "text": section_text
                    })
            prev_pos = pos
            prev_section = section_name
        
        # Add final section
        if prev_pos < len(full_text):
            section_text = full_text[prev_pos:].strip()
            if section_text:
                sections.append({
                    "name": prev_section,
                    "start": prev_pos,
                    "end": len(full_text),
                    "text": section_text
                })
        
        # Chunk each section
        chunks = []
        chunk_idx = 0
        
        for section in sections:
            section_chunks = self._chunk_section(
                section["text"],
                char_to_page,
                section["start"],
                section["name"],
                target_chunk_size,
                chunk_idx
            )
            chunks.extend(section_chunks)
            chunk_idx = len(chunks)
        
        logger.info(f"Structure-aware chunking: {len(chunks)} chunks across {len(sections)} sections")
        return chunks

    def _chunk_section(
        self,
        section_text: str,
        char_to_page: list[int],
        section_start: int,
        section_name: str,
        target_size: int,
        start_chunk_idx: int
    ) -> list[dict]:
        """
        Chunk a single section by paragraphs with semantic overlap.
        """
        paragraphs = self._split_into_paragraphs(section_text)
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = start_chunk_idx
        
        # Track paragraph positions for metadata
        para_start = 0
        # Track actual character positions of all paragraphs
        para_positions = [0]
        for i, para in enumerate(paragraphs):
            if i > 0:
                para_positions.append(para_positions[-1] + len(paragraphs[i-1]) + 2)  # +2 for \n\n
        
        for i, para in enumerate(paragraphs):
            para_len = len(para)
            
            # If adding this paragraph would exceed target size and we have content,
            # create a chunk
            if current_length + para_len > target_size and current_chunk:
                # Create chunk with current paragraphs
                chunk_text = "\n\n".join(current_chunk)
                chunk_end = section_start + para_start
                
                # Get pages for this chunk
                chunk_pages = sorted(set(char_to_page[max(0, chunk_end - current_length):chunk_end]))
                
                chunks.append({
                    "chunk_index": chunk_idx,
                    "text": chunk_text.strip(),
                    "metadata": {
                        "section": section_name,
                        "pages": chunk_pages,
                        "char_start": chunk_end - current_length,
                        "char_end": chunk_end,
                        "length": current_length
                    }
                })
                
                chunk_idx += 1
                
                # Semantic overlap: keep last 1-2 paragraphs for context
                overlap_paras = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_paras
                current_length = sum(len(p) for p in overlap_paras)
                
                # Reset para_start to the start of the first overlap paragraph
                if overlap_paras:
                    # Find the index of the first overlap paragraph in original paragraphs
                    overlap_start_idx = i - len(overlap_paras)
                    para_start = para_positions[overlap_start_idx]
            
            # Add current paragraph
            current_chunk.append(para)
            current_length += para_len
            para_start = para_positions[i] + para_len + 2  # +2 for \n\n
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_end = section_start + para_start
            chunk_pages = sorted(set(char_to_page[max(0, chunk_end - current_length):chunk_end]))
            
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text.strip(),
                "metadata": {
                    "section": section_name,
                    "pages": chunk_pages,
                    "char_start": chunk_end - current_length,
                    "char_end": chunk_end,
                    "length": current_length
                }
            })
        
        return chunks

    def _chunk_paragraph_based(
        self,
        full_text: str,
        char_to_page: list[int],
        target_size: int
    ) -> list[dict]:
        """
        Fallback paragraph-based chunking when no clear sections are detected.
        """
        paragraphs = self._split_into_paragraphs(full_text)
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        para_start = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_length + para_len > target_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_end = para_start
                chunk_pages = sorted(set(char_to_page[max(0, chunk_end - current_length):chunk_end]))
                
                chunks.append({
                    "chunk_index": chunk_idx,
                    "text": chunk_text.strip(),
                    "metadata": {
                        "section": "Unknown",
                        "pages": chunk_pages,
                        "char_start": chunk_end - current_length,
                        "char_end": chunk_end,
                        "length": current_length
                    }
                })
                
                chunk_idx += 1
                
                # Keep last paragraph for overlap
                overlap_paras = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_paras
                current_length = sum(len(p) for p in overlap_paras)
            
            current_chunk.append(para)
            current_length += para_len
            para_start += para_len + 2
        
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_end = para_start
            chunk_pages = sorted(set(char_to_page[max(0, chunk_end - current_length):chunk_end]))
            
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text.strip(),
                "metadata": {
                    "section": "Unknown",
                    "pages": chunk_pages,
                    "char_start": chunk_end - current_length,
                    "char_end": chunk_end,
                    "length": current_length
                }
            })
        
        logger.info(f"Paragraph-based chunking: {len(chunks)} chunks")
        return chunks

    def _chunk_character_based(
        self,
        full_text: str,
        char_to_page: list[int],
        chunk_size: int,
        chunk_overlap: int
    ) -> list[dict]:
        """
        Legacy character-based chunking (original implementation).
        Kept for backward compatibility.
        """
        text_len = len(full_text)
        
        if chunk_size <= 0:
            chunk_size = 1000
        
        if chunk_overlap >= chunk_size or chunk_overlap < 0:
            chunk_overlap = int(chunk_size * 0.2)
            logger.warning(f"Invalid chunk_overlap — reset to 20% of chunk_size: {chunk_overlap}")
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            if len(chunks) > 0 and (end - start) < 100:
                break
            
            chunk_text_content = full_text[start:end]
            chunk_pages = sorted(set(char_to_page[start:end]))
            
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text_content.strip(),
                "metadata": {
                    "section": "Unknown",
                    "pages": chunk_pages,
                    "char_start": start,
                    "char_end": end,
                    "length": len(chunk_text_content)
                }
            })
            
            chunk_idx += 1
            
            if end == text_len:
                break
            
            start += (chunk_size - chunk_overlap)
        
        logger.info(f"Character-based chunking: {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
        return chunks

