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
        full_text, char_to_page = service.extract_text_by_page(Path("paper.pdf"))
        # Step 6b: Standardize default chunk sizes to 2000/400
        chunks = service.chunk_text(full_text, char_to_page, chunk_size=2000, chunk_overlap=400)
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

    def extract_text_by_page(self, pdf_path: Path | str) -> tuple[str, list[int]]:
        """
        Extracts text page-by-page and ensures perfect character-to-page alignment
        by cleaning page content before calculating indexing boundaries.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        full_text = ""
        char_to_page = []

        try:
            doc = fitz.open(pdf_path)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                raw_text = page.get_text("text") or ""
                
                # Clean the text immediately BEFORE measuring its string length
                page_text = self._clean_text(raw_text)
                if not page_text:
                    continue
                
                text_to_append = page_text + "\n"
                append_len = len(text_to_append)
                
                # Accumulate master text string
                full_text += text_to_append
                
                # Add exactly one page integer value per character added to full_text
                char_to_page.extend([page_idx] * append_len)
                
            doc.close()
        except Exception as e:
            logger.error(f"Failed text extraction from PDF {pdf_path}: {e}")
            
        return full_text, char_to_page

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

    def chunk_text(self, full_text: str, char_to_page: list[int], chunk_size: int = 1500, chunk_overlap: int = 300) -> list[dict]:
        """
        Slices the fully synchronized text string into overlapping chunks.
        
        FIX: Strip references section before chunking to prevent citation bleed.
        """
        # FIX: Strip references/bibliography section to prevent citation chunk bleed
        full_text = self._strip_references_section(full_text)
        
        text_len = len(full_text)
        map_len = len(char_to_page)
        chunks = []
        start = 0
        chunk_idx = 0

        if text_len == 0 or map_len == 0:
            return chunks

        # Enforce baseline boundary constraints
        if chunk_size <= 0:
            chunk_size = 1500
        if chunk_overlap >= chunk_size or chunk_overlap < 0:
            chunk_overlap = int(chunk_size * 0.2)

        # Slide completely to the end of the text string length without premature breaking
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            chunk_text_content = full_text[start:end]
            
            # Slice tracking array safely within limits
            slice_end = min(end, map_len)
            chunk_pages_slice = char_to_page[start:slice_end]
            
            if chunk_pages_slice:
                # Filter out None, empty, and invalid values, then deduplicate
                valid_pages = [p for p in chunk_pages_slice if p is not None and isinstance(p, int)]
                unique_pages = sorted(list(set(valid_pages)))
                # Format as human readable 1-indexed comma separated string
                if unique_pages:
                    pages_metadata_str = ",".join(str(p + 1) for p in unique_pages)
                else:
                    pages_metadata_str = "1"
            else:
                pages_metadata_str = "1"

            if not pages_metadata_str:
                pages_metadata_str = "1"

            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text_content.strip(),
                "metadata": {
                    "section": "Unknown",
                    "pages": pages_metadata_str,
                    "char_start": start,
                    "char_end": end,
                    "length": len(chunk_text_content)
                }
            })
            
            chunk_idx += 1
            if end == text_len:
                break
                
            start += (chunk_size - chunk_overlap)

        logger.info(f"Successfully generated {len(chunks)} synchronized document slices.")
        return chunks

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
        Page metadata stored as comma-separated string with dual range debris defense.
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
                
                # DUAL RANGE DEBRIS DEFENSE: Safely wrap page index slice
                safe_start = max(0, chunk_end - current_length)
                safe_end = min(len(char_to_page), chunk_end)
                
                if safe_end > safe_start:
                    # Get pages for this chunk and convert to comma-separated string
                    chunk_pages_list = sorted(list(set(char_to_page[safe_start:safe_end])))
                    chunk_pages_str = ",".join(str(p) for p in chunk_pages_list if p is not None)
                    if not chunk_pages_str:
                        chunk_pages_str = "1"
                else:
                    chunk_pages_str = "1"
                
                chunks.append({
                    "chunk_index": chunk_idx,
                    "text": chunk_text.strip(),
                    "metadata": {
                        "section": section_name,
                        "pages": chunk_pages_str,
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
            
            # DUAL RANGE DEBRIS DEFENSE: Safely wrap page index slice
            safe_start = max(0, chunk_end - current_length)
            safe_end = min(len(char_to_page), chunk_end)
            
            if safe_end > safe_start:
                chunk_pages_list = sorted(list(set(char_to_page[safe_start:safe_end])))
                chunk_pages_str = ",".join(str(p) for p in chunk_pages_list if p is not None)
                if not chunk_pages_str:
                    chunk_pages_str = "1"
            else:
                chunk_pages_str = "1"
            
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text.strip(),
                "metadata": {
                    "section": section_name,
                    "pages": chunk_pages_str,
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
        Page metadata stored as comma-separated string with dual range debris defense.
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
                
                # DUAL RANGE DEBRIS DEFENSE: Safely wrap page index slice
                safe_start = max(0, chunk_end - current_length)
                safe_end = min(len(char_to_page), chunk_end)
                
                if safe_end > safe_start:
                    chunk_pages_list = sorted(list(set(char_to_page[safe_start:safe_end])))
                    chunk_pages_str = ",".join(str(p) for p in chunk_pages_list if p is not None)
                    if not chunk_pages_str:
                        chunk_pages_str = "1"
                else:
                    chunk_pages_str = "1"
                
                chunks.append({
                    "chunk_index": chunk_idx,
                    "text": chunk_text.strip(),
                    "metadata": {
                        "section": "Unknown",
                        "pages": chunk_pages_str,
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
            
            # DUAL RANGE DEBRIS DEFENSE: Safely wrap page index slice
            safe_start = max(0, chunk_end - current_length)
            safe_end = min(len(char_to_page), chunk_end)
            
            if safe_end > safe_start:
                chunk_pages_list = sorted(list(set(char_to_page[safe_start:safe_end])))
                chunk_pages_str = ",".join(str(p) for p in chunk_pages_list if p is not None)
                if not chunk_pages_str:
                    chunk_pages_str = "1"
            else:
                chunk_pages_str = "1"
            
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text.strip(),
                "metadata": {
                    "section": "Unknown",
                    "pages": chunk_pages_str,
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
        Character-based sliding window chunking with dual range debris defense.

        FIX SLIDING WINDOW DEPLETION:
          - Sliding loop runs completely (while start < text_len)
          - Chunk size: 1500 characters
          - Chunk overlap: 300 characters

        DUAL RANGE DEBRIS DEFENSE & CLEAN METADATA STRING:
          - Safely wrap page index slice to avoid index errors or empty entries
          - Convert to 1-indexed for reader clarity
          - Filter None values

        Args:
            full_text: Complete concatenated document text.
            char_to_page: List mapping each character to its page number (0-indexed).
            chunk_size: Target chunk size in characters (default: 1500).
            chunk_overlap: Overlap in characters (default: 300).

        Returns:
            List of validated chunk dicts with page metadata as comma-separated string.
        """
        text_len = len(full_text)

        # Validate parameters
        if chunk_size <= 0:
            chunk_size = 1500
            logger.warning(f"Invalid chunk_size — reset to default: {chunk_size}")

        if chunk_overlap >= chunk_size or chunk_overlap < 0:
            chunk_overlap = int(chunk_size * 0.2)
            logger.warning(f"Invalid chunk_overlap — reset to 20% of chunk_size: {chunk_overlap}")

        # Validate char_to_page mapping
        if len(char_to_page) != text_len:
            logger.error(f"CRITICAL: char_to_page length ({len(char_to_page)}) != full_text length ({text_len})")
            return []

        chunks = []
        start = 0
        chunk_idx = 0

        # Sliding window iteration through ENTIRE document - NO EARLY BREAKS
        while start < text_len:
            end = min(start + chunk_size, text_len)

            # DUAL RANGE DEBRIS DEFENSE: Safely wrap page index slice
            safe_start = max(0, start)
            safe_end = min(len(char_to_page), end)
            
            if safe_end <= safe_start:
                logger.debug(f"Skipping chunk {chunk_idx} (invalid range: {safe_start} to {safe_end})")
                start += (chunk_size - chunk_overlap)
                continue

            # Get the exact slice of char_to_page for this chunk
            chunk_pages_slice = char_to_page[safe_start:safe_end]
            
            # CLEAN METADATA STRING: Convert to 1-indexed, filter None values, join with commas
            chunk_pages_list = sorted(list(set(chunk_pages_slice)))
            pages_str = ",".join(str(p) for p in chunk_pages_list if p is not None)
            
            # Fallback if pages_str is empty
            if not pages_str:
                pages_str = "1"

            # Extract chunk text
            chunk_text_content = full_text[start:end].strip()
            chunk_length = len(chunk_text_content)

            # Validation: drop empty or near-empty chunks
            if chunk_length < 100:
                logger.debug(f"Skipping chunk {chunk_idx} (too short: {chunk_length} chars at position {start})")
                start += (chunk_size - chunk_overlap)
                continue

            # Create chunk with clean page metadata
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text_content,
                "metadata": {
                    "section": "Unknown",
                    "pages": pages_str,
                    "char_start": start,
                    "char_end": end,
                    "length": chunk_length
                }
            })

            chunk_idx += 1

            # Move to next window with overlap
            if end == text_len:
                logger.debug(f"Reached end of document at position {text_len}")
                break

            start += (chunk_size - chunk_overlap)

        # Log validation results
        logger.info(
            f"Character-based chunking: Generated {len(chunks)} valid chunks "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap}, total_text={text_len} chars)"
        )

        return chunks

