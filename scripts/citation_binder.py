"""
citation_binder.py — Citation Binding Service for RAG.

Enforces per-sentence traceability to ensure every answer sentence
can be traced to at least one retrieved chunk. This forces the model
to stay grounded and prevents filler reasoning.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CitationBinder:
    """
    Binds answer sentences to source chunks for traceability.
    """

    def __init__(self):
        """Initialize the citation binder."""
        logger.info("Citation binder initialized")

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using simple heuristics.

        Args:
            text: Input text.

        Returns:
            List of sentences.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def extract_citations_from_sentence(self, sentence: str) -> list[str]:
        """
        Extract APA7-style citations from a sentence.

        Args:
            sentence: Input sentence.

        Returns:
            List of citation strings (e.g., "(Smith, 2020)").
        """
        # APA7 citation patterns
        patterns = [
            r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})(?:,\s*p\.\s*\d+)?\)',
            r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})(?:,\s+pp\.\s*\d+)?\)',
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        
        return citations

    def check_sentence_grounding(
        self,
        sentence: str,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Check if a sentence is grounded in the retrieved chunks.

        Args:
            sentence: The sentence to check.
            chunks: Retrieved chunks.

        Returns:
            Dict with:
                - is_grounded: bool
                - citations: list of citations found
                - matching_chunks: list of chunk indices that support the sentence
        """
        citations = self.extract_citations_from_sentence(sentence)
        
        if not citations:
            # No citations found - check if sentence has any content from chunks
            sentence_lower = sentence.lower()
            matching_chunks = []
            
            for idx, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "").lower()
                # Check for significant overlap (at least 3 words)
                sentence_words = set(re.findall(r'\b[a-z]{3,}\b', sentence_lower))
                chunk_words = set(re.findall(r'\b[a-z]{3,}\b', chunk_text))
                overlap = sentence_words & chunk_words
                
                if len(overlap) >= 3:
                    matching_chunks.append(idx)
            
            return {
                "is_grounded": len(matching_chunks) > 0,
                "citations": citations,
                "matching_chunks": matching_chunks
            }
        
        # Has citations - verify they match chunk metadata
        matching_chunks = []
        for citation in citations:
            # Extract author and year from citation
            match = re.search(r'\(([A-Z][a-zA-Z]+)(?:\s+et\s+al\.)?,\s*(\d{4})', citation)
            if match:
                author = match.group(1).lower()
                year = match.group(2)
                
                for idx, chunk in enumerate(chunks):
                    meta = chunk.get("metadata", {})
                    chunk_authors = meta.get("authors", "").lower()
                    chunk_year = str(meta.get("year", ""))
                    
                    if author in chunk_authors and year in chunk_year:
                        matching_chunks.append(idx)
        
        return {
            "is_grounded": len(matching_chunks) > 0,
            "citations": citations,
            "matching_chunks": matching_chunks
        }

    def analyze_answer_grounding(
        self,
        answer: str,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze the grounding of an entire answer.

        Args:
            answer: The LLM-generated answer.
            chunks: Retrieved chunks.

        Returns:
            Dict with:
                - total_sentences: int
                - grounded_sentences: int
                - ungrounded_sentences: int
                - grounding_ratio: float
                - sentence_analysis: list of per-sentence analysis
        """
        sentences = self.split_into_sentences(answer)
        
        if not sentences:
            return {
                "total_sentences": 0,
                "grounded_sentences": 0,
                "ungrounded_sentences": 0,
                "grounding_ratio": 0.0,
                "sentence_analysis": []
            }
        
        sentence_analysis = []
        grounded_count = 0
        
        for idx, sentence in enumerate(sentences):
            analysis = self.check_sentence_grounding(sentence, chunks)
            analysis["sentence"] = sentence
            analysis["sentence_index"] = idx
            sentence_analysis.append(analysis)
            
            if analysis["is_grounded"]:
                grounded_count += 1
        
        grounding_ratio = grounded_count / len(sentences) if sentences else 0.0
        
        result = {
            "total_sentences": len(sentences),
            "grounded_sentences": grounded_count,
            "ungrounded_sentences": len(sentences) - grounded_count,
            "grounding_ratio": grounding_ratio,
            "sentence_analysis": sentence_analysis
        }
        
        logger.info(
            f"Answer grounding analysis: {grounded_count}/{len(sentences)} "
            f"sentences grounded ({grounding_ratio:.2%})"
        )
        
        return result

    def enforce_citation_binding(
        self,
        answer: str,
        chunks: list[dict[str, Any]],
        min_grounding_ratio: float = 0.7,
        strict_mode: bool = True
    ) -> tuple[str, bool]:
        """
        Enforce citation binding by checking grounding and flagging issues.

        Args:
            answer: The LLM-generated answer.
            chunks: Retrieved chunks.
            min_grounding_ratio: Minimum acceptable grounding ratio.
            strict_mode: If True, rejects answers below threshold instead of adding warning.

        Returns:
            Tuple of (answer, is_acceptable).
            If strict_mode=True and grounding is below threshold, returns refusal message.
            If strict_mode=False and grounding is below threshold, adds a warning to the answer.
        """
        analysis = self.analyze_answer_grounding(answer, chunks)
        
        if analysis["grounding_ratio"] >= min_grounding_ratio:
            return answer, True
        
        logger.warning(
            f"Citation binding failed: grounding ratio {analysis['grounding_ratio']:.2%} "
            f"below threshold {min_grounding_ratio:.2%}"
        )
        
        if strict_mode:
            # HARD CONSTRAINT: Return refusal instead of ungrounded answer
            refusal = (
                f"I cannot provide a complete answer because {analysis['ungrounded_sentences']} "
                f"of {analysis['total_sentences']} sentences could not be traced to the provided sources. "
                f"This indicates the answer would contain ungrounded claims. "
                f"Please try a more specific question or ingest more relevant papers."
            )
            return refusal, False
        else:
            # Soft constraint: add warning
            warning = (
                f"\n\n[Note: {analysis['ungrounded_sentences']} of {analysis['total_sentences']} "
                f"sentences could not be traced to the provided sources. "
                f"Please verify these claims against the original papers.]"
            )
            return answer + warning, False

    def add_missing_citations(
        self,
        answer: str,
        chunks: list[dict[str, Any]]
    ) -> str:
        """
        Attempt to add missing citations to ungrounded sentences.

        This is a best-effort approach - it adds citations based on
        chunk metadata if the sentence content matches chunk content.

        Args:
            answer: The LLM-generated answer.
            chunks: Retrieved chunks.

        Returns:
            Answer with added citations where possible.
        """
        sentences = self.split_into_sentences(answer)
        enhanced_sentences = []
        
        for sentence in sentences:
            # Check if sentence already has citations
            existing_citations = self.extract_citations_from_sentence(sentence)
            
            if existing_citations:
                enhanced_sentences.append(sentence)
                continue
            
            # Try to find matching chunks
            sentence_lower = sentence.lower()
            best_match = None
            best_overlap = 0
            
            for chunk in chunks:
                chunk_text = chunk.get("text", "").lower()
                sentence_words = set(re.findall(r'\b[a-z]{3,}\b', sentence_lower))
                chunk_words = set(re.findall(r'\b[a-z]{3,}\b', chunk_text))
                overlap = sentence_words & chunk_words
                
                if len(overlap) > best_overlap and len(overlap) >= 3:
                    best_overlap = len(overlap)
                    best_match = chunk
            
            if best_match:
                # Add citation
                meta = best_match.get("metadata", {})
                authors = meta.get("authors", "Unknown")
                year = meta.get("year", "N/A")
                
                # Extract first author's last name
                author_parts = authors.split()
                if author_parts:
                    first_author = author_parts[0].split()[-1]  # Last name of first author
                    citation = f"({first_author}, {year})"
                    
                    # Add citation at end of sentence
                    if sentence.endswith('.'):
                        enhanced_sentence = sentence[:-1] + f" {citation}."
                    else:
                        enhanced_sentence = f"{sentence} {citation}."
                    
                    enhanced_sentences.append(enhanced_sentence)
                else:
                    enhanced_sentences.append(sentence)
            else:
                enhanced_sentences.append(sentence)
        
        return " ".join(enhanced_sentences)
