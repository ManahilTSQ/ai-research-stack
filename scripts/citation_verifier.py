"""
citation_verifier.py — Citation Verification Service for RAG.

Verifies that every claim in the LLM-generated answer is supported by
retrieved chunks. This is the final accuracy guard that prevents
hallucinated or unsupported claims.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CitationVerifier:
    """
    Verifies that answer claims are supported by retrieved chunks.
    """

    def __init__(self):
        """Initialize the citation verifier."""
        logger.info("Citation verifier initialized")

    def split_into_claims(self, text: str) -> list[str]:
        """
        Split text into individual claims (sentences).

        Args:
            text: Input text.

        Returns:
            List of claim sentences.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def extract_claim_entities(self, claim: str) -> set[str]:
        """
        Extract key entities from a claim for verification.

        Args:
            claim: The claim sentence.

        Returns:
            Set of key entities (words).
        """
        # Define standard English and academic stopwords to ignore
        stopwords = {
            "that", "this", "these", "those", "have", "has", "had", "having",
            "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "from", "down", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "only", "own",
            "same", "so", "than", "too", "very", "can", "will", "just", "should", 
            "now", "their", "theirs", "them", "themselves", "they", "were", "what", 
            "which", "who", "whom", "been", "being", "does", "doing", "would", 
            "could", "should", "brought", "also", "using", "used", "uses", "show", 
            "shows", "shown", "find", "finds", "found", "based", "presents", 
            "present", "study", "paper", "article", "author", "authors", "many",
            "from", "into", "onto", "upon", "within", "without", "throughout"
        }
        # Extract significant words (nouns, numbers, technical terms) of 4+ characters
        words = re.findall(r'\b[a-z]{4,}\b|\d+(?:\.\d+)?', claim.lower())
        return {w for w in words if w not in stopwords}

    def verify_claim_against_chunks(
        self,
        claim: str,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Verify if a claim is supported by retrieved chunks.

        Args:
            claim: The claim to verify.
            chunks: Retrieved chunks.

        Returns:
            Dict with:
                - is_supported: bool
                - supporting_chunks: list of chunk indices
                - evidence: list of supporting text snippets
        """
        claim_entities = self.extract_claim_entities(claim)
        
        if not claim_entities:
            # No entities to verify, assume supported
            return {
                "is_supported": True,
                "supporting_chunks": [],
                "evidence": []
            }
        
        supporting_chunks = []
        evidence = []
        
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "").lower()
            
            # Check for entity overlap
            chunk_words = set(re.findall(r'\b[a-z]{4,}\b|\d+(?:\.\d+)?', chunk_text))
            overlap = claim_entities & chunk_words
            
            # Overlap criteria:
            # For short claims (1-2 entities): must match all entities
            # For medium/long claims: must match at least 3 entities OR at least 40% of the entities (with a minimum of 2)
            min_overlap = len(claim_entities) if len(claim_entities) <= 2 else max(2, min(3, int(len(claim_entities) * 0.4)))
            
            if len(overlap) >= min_overlap:
                supporting_chunks.append(idx)
                # Extract relevant snippet
                snippet = self._extract_relevant_snippet(claim, chunk_text)
                if snippet:
                    evidence.append(snippet)
        
        return {
            "is_supported": len(supporting_chunks) > 0,
            "supporting_chunks": supporting_chunks,
            "evidence": evidence
        }

    def _extract_relevant_snippet(self, claim: str, chunk_text: str) -> str:
        """
        Extract the most relevant snippet from chunk text for a claim.

        Args:
            claim: The claim.
            chunk_text: The chunk text.

        Returns:
            Relevant text snippet.
        """
        claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim.lower()))
        
        # Find sentences in chunk with highest overlap
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
        
        best_sentence = ""
        best_overlap = 0
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b[a-z]{4,}\b', sentence.lower()))
            overlap = len(claim_words & sentence_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence
        
        return best_sentence if best_overlap > 0 else chunk_text[:200]

    def verify_answer(
        self,
        answer: str,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Verify an entire answer against retrieved chunks.

        Args:
            answer: The LLM-generated answer.
            chunks: Retrieved chunks.

        Returns:
            Dict with:
                - total_claims: int
                - supported_claims: int
                - unsupported_claims: int
                - support_ratio: float
                - claim_verification: list of per-claim verification
        """
        claims = self.split_into_claims(answer)
        
        if not claims:
            return {
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "support_ratio": 1.0,
                "claim_verification": []
            }
        
        claim_verification = []
        supported_count = 0
        
        for idx, claim in enumerate(claims):
            verification = self.verify_claim_against_chunks(claim, chunks)
            verification["claim"] = claim
            verification["claim_index"] = idx
            claim_verification.append(verification)
            
            if verification["is_supported"]:
                supported_count += 1
        
        support_ratio = supported_count / len(claims) if claims else 1.0
        
        result = {
            "total_claims": len(claims),
            "supported_claims": supported_count,
            "unsupported_claims": len(claims) - supported_count,
            "support_ratio": support_ratio,
            "claim_verification": claim_verification
        }
        
        logger.info(
            f"Answer verification: {supported_count}/{len(claims)} claims supported "
            f"({support_ratio:.2%})"
        )
        
        return result

    def flag_unsupported_claims(
        self,
        verification: dict[str, Any]
    ) -> list[str]:
        """
        Extract unsupported claims from verification results.

        Args:
            verification: Verification result from verify_answer.

        Returns:
            List of unsupported claim texts.
        """
        unsupported = []
        
        for claim_verif in verification.get("claim_verification", []):
            if not claim_verif.get("is_supported", False):
                unsupported.append(claim_verif.get("claim", ""))
        
        return unsupported

    def regenerate_or_remove_unsupported(
        self,
        answer: str,
        verification: dict[str, Any],
        action: str = "remove"
    ) -> str:
        """
        Handle unsupported claims by either removing them or flagging them.

        Args:
            answer: Original answer.
            verification: Verification result.
            action: "remove" or "flag" unsupported claims.

        Returns:
            Modified answer.
        """
        unsupported_claims = self.flag_unsupported_claims(verification)
        
        if not unsupported_claims:
            return answer
        
        if action == "remove":
            # Remove unsupported claims
            claims = self.split_into_claims(answer)
            supported_claims = [
                claim for claim in claims
                if claim not in unsupported_claims
            ]
            return " ".join(supported_claims)
        
        elif action == "flag":
            # Add warning about unsupported claims
            warning = (
                f"\n\n[Verification Warning: {len(unsupported_claims)} claim(s) "
                f"could not be verified against the provided sources. "
                f"Please verify these claims independently.]"
            )
            return answer + warning
        
        return answer

    def check_citation_alignment(
        self,
        answer: str,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Check if citations in the answer align with chunk metadata.

        Args:
            answer: The LLM-generated answer.
            chunks: Retrieved chunks.

        Returns:
            Dict with:
                - total_citations: int
                - aligned_citations: int
                - misaligned_citations: int
                - alignment_ratio: float
        """
        # Extract citations from answer
        citation_pattern = r'\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?,\s*\d{4})\)'
        citations = re.findall(citation_pattern, answer)
        
        if not citations:
            return {
                "total_citations": 0,
                "aligned_citations": 0,
                "misaligned_citations": 0,
                "alignment_ratio": 1.0
            }
        
        aligned_count = 0
        
        # Build set of valid (author, year) pairs from chunks
        valid_pairs = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            authors = meta.get("authors", "")
            year = str(meta.get("year", ""))
            
            # Extract first author's last name
            author_parts = authors.split()
            if author_parts:
                first_author = author_parts[0].split()[-1]
                valid_pairs.add((first_author.lower(), year))
        
        # Check each citation
        for citation in citations:
            match = re.search(r'([A-Z][a-zA-Z]+)(?:\s+et\s+al\.)?,\s*(\d{4})', citation)
            if match:
                author = match.group(1).lower()
                year = match.group(2)
                
                if (author, year) in valid_pairs:
                    aligned_count += 1
        
        alignment_ratio = aligned_count / len(citations) if citations else 1.0
        
        result = {
            "total_citations": len(citations),
            "aligned_citations": aligned_count,
            "misaligned_citations": len(citations) - aligned_count,
            "alignment_ratio": alignment_ratio
        }
        
        logger.info(
            f"Citation alignment: {aligned_count}/{len(citations)} citations aligned "
            f"({alignment_ratio:.2%})"
        )
        
        return result

    def full_verification(
        self,
        answer: str,
        chunks: list[dict[str, Any]],
        min_support_ratio: float = 0.8,
        min_alignment_ratio: float = 0.9
    ) -> dict[str, Any]:
        """
        Perform full verification including claim support and citation alignment.

        Args:
            answer: The LLM-generated answer.
            chunks: Retrieved chunks.
            min_support_ratio: Minimum acceptable claim support ratio.
            min_alignment_ratio: Minimum acceptable citation alignment ratio.

        Returns:
            Dict with verification results and overall pass/fail.
        """
        # Verify claims
        claim_verification = self.verify_answer(answer, chunks)
        
        # Check citation alignment
        citation_alignment = self.check_citation_alignment(answer, chunks)
        
        # Overall pass/fail
        claims_pass = claim_verification["support_ratio"] >= min_support_ratio
        citations_pass = citation_alignment["alignment_ratio"] >= min_alignment_ratio
        overall_pass = claims_pass and citations_pass
        
        result = {
            "claim_verification": claim_verification,
            "citation_alignment": citation_alignment,
            "overall_pass": overall_pass,
            "claims_pass": claims_pass,
            "citations_pass": citations_pass
        }
        
        logger.info(
            f"Full verification: {'PASS' if overall_pass else 'FAIL'} "
            f"(claims: {claim_verification['support_ratio']:.2%}, "
            f"citations: {citation_alignment['alignment_ratio']:.2%})"
        )
        
        return result

    def strip_unverified_citations(
        self,
        answer: str,
        chunks: list[dict[str, Any]]
    ) -> str:
        """
        Remove citations that don't match retrieved chunk metadata to prevent hallucination.
        Also cleans up broken parentheticals and ensures whole parentheticals are stripped
        if any of their cited sources are unverified.
        """
        # Build set of valid (author, year) pairs from chunks
        valid_pairs = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            authors = meta.get("authors", "")
            year = str(meta.get("year", ""))
            
            # Extract author name segments and get their last names/surnames
            for part in re.split(r"[,;&]| and ", authors.lower()):
                words = [w for w in re.findall(r"[a-z\u00C0-\u017F]+", part) if w not in {"et", "al"}]
                if words:
                    valid_pairs.add((words[-1], year))
                    
        logger.info(f"Valid citation pairs from chunks: {valid_pairs}")
        
        # Find all parentheticals in the answer
        parentheticals = re.findall(r'\(([^()]+)\)', answer)
        
        modified_answer = answer
        
        for content in parentheticals:
            # Check if this parenthetical looks like a citation (contains a 4-digit year)
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', content)
            if not year_match:
                continue
                
            year = year_match.group(1)
            
            # Extract potential author surnames: capitalized words
            # (ignore common citation markers like 'et', 'al', 'and', '&', etc.)
            author_candidates = re.findall(r'\b([A-Z][a-zA-Z\u00C0-\u017F\-’\']+)\b', content)
            author_candidates = [a.lower() for a in author_candidates if a.lower() not in {"et", "al", "and"}]
            
            if not author_candidates:
                continue
                
            # Verify if all extracted author surnames matched with the year in valid_pairs
            citation_valid = True
            for author in author_candidates:
                if (author, year) not in valid_pairs:
                    # Check if surname is present in the chunk text to be safe
                    surname_in_text = False
                    for chunk in chunks:
                        if author in chunk.get("text", "").lower():
                            surname_in_text = True
                            break
                    if not surname_in_text:
                        citation_valid = False
                        logger.warning(f"Unverified citation author: {author} with year {year}")
                        break
            
            if not citation_valid:
                # Remove the entire parenthetical citation
                full_citation = f"({content})"
                modified_answer = modified_answer.replace(full_citation, "")
                logger.warning(f"Removed unverified parenthetical citation: {full_citation}")
                
        # Clean up any leftover empty parentheticals, double spaces, or leading/trailing punctuation spacing
        modified_answer = re.sub(r'\(\s*\)', '', modified_answer)
        modified_answer = re.sub(r'[ \t]+', ' ', modified_answer)
        modified_answer = re.sub(r'\s+([.,;:])', r'\1', modified_answer)
        
        return modified_answer
