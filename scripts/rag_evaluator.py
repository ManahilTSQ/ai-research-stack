"""
rag_evaluator.py — RAG Evaluation Harness.

Measures RAG system performance across multiple dimensions:
- Retrieval precision: How relevant are retrieved chunks?
- Context faithfulness: Does the answer stay grounded in context?
- Citation correctness: Are citations accurate and traceable?
- Contradiction rate: Does the answer contain internal contradictions?
- Answer completeness: Does the answer fully address the query?

This is the #1 missing component - without evaluation, you're guessing
improvements instead of verifying them.
"""

import logging
import re
from typing import Any
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of RAG evaluation."""
    query: str
    answer: str
    chunks: list[dict]
    
    # Metrics
    retrieval_precision: float
    context_faithfulness: float
    citation_correctness: float
    contradiction_rate: float
    answer_completeness: float
    
    # Overall score
    overall_score: float
    
    # Details
    details: dict[str, Any]


class RAGEvaluator:
    """
    Evaluates RAG system performance across multiple dimensions.
    """
    
    def __init__(self):
        """Initialize the RAG evaluator."""
        logger.info("RAG evaluator initialized")
    
    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list[dict[str, Any]],
        expected_answer: str | None = None
    ) -> EvaluationResult:
        """
        Run full evaluation on a RAG query-response pair.
        
        Args:
            query: The original query.
            answer: The generated answer.
            chunks: The retrieved chunks used as context.
            expected_answer: Optional ground truth answer for completeness check.
        
        Returns:
            EvaluationResult with all metrics.
        """
        # Calculate individual metrics
        retrieval_precision = self._calculate_retrieval_precision(query, chunks)
        context_faithfulness = self._calculate_context_faithfulness(answer, chunks)
        citation_correctness = self._calculate_citation_correctness(answer, chunks)
        contradiction_rate = self._calculate_contradiction_rate(answer)
        answer_completeness = self._calculate_answer_completeness(
            query, answer, expected_answer
        )
        
        # Overall score (weighted average)
        overall_score = (
            0.25 * retrieval_precision +
            0.25 * context_faithfulness +
            0.20 * citation_correctness +
            0.15 * (1.0 - contradiction_rate) +  # Lower contradiction is better
            0.15 * answer_completeness
        )
        
        details = {
            "retrieval_precision": retrieval_precision,
            "context_faithfulness": context_faithfulness,
            "citation_correctness": citation_correctness,
            "contradiction_rate": contradiction_rate,
            "answer_completeness": answer_completeness,
            "chunk_count": len(chunks),
            "answer_length": len(answer),
        }
        
        result = EvaluationResult(
            query=query,
            answer=answer,
            chunks=chunks,
            retrieval_precision=retrieval_precision,
            context_faithfulness=context_faithfulness,
            citation_correctness=citation_correctness,
            contradiction_rate=contradiction_rate,
            answer_completeness=answer_completeness,
            overall_score=overall_score,
            details=details
        )
        
        logger.info(
            f"Evaluation complete: overall_score={overall_score:.3f}, "
            f"retrieval={retrieval_precision:.3f}, "
            f"faithfulness={context_faithfulness:.3f}"
        )
        
        return result
    
    def _calculate_retrieval_precision(
        self,
        query: str,
        chunks: list[dict[str, Any]]
    ) -> float:
        """
        Calculate retrieval precision: how relevant are retrieved chunks?
        
        Measures:
        - Semantic distance (lower is better)
        - Lexical overlap with query
        - Section relevance
        """
        if not chunks:
            return 0.0
        
        # Extract query tokens
        query_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", query.lower()))
        
        precision_scores = []
        
        for chunk in chunks:
            # Semantic score (from distance)
            distance = chunk.get("distance", 1.0)
            semantic_score = max(0.0, 1.0 - (distance / 2.0))
            
            # Lexical overlap
            chunk_text = chunk.get("text", "").lower()
            chunk_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", chunk_text))
            if query_tokens:
                overlap = len(query_tokens & chunk_tokens)
                lexical_score = overlap / len(query_tokens)
            else:
                lexical_score = 0.0
            
            # Combined precision for this chunk
            chunk_precision = 0.6 * semantic_score + 0.4 * lexical_score
            precision_scores.append(chunk_precision)
        
        # Average precision across all chunks
        return sum(precision_scores) / len(precision_scores)
    
    def _calculate_context_faithfulness(
        self,
        answer: str,
        chunks: list[dict[str, Any]]
    ) -> float:
        """
        Calculate context faithfulness: does the answer stay grounded?
        
        Measures:
        - What percentage of answer sentences have supporting evidence in chunks?
        - How much of the answer content is traceable to retrieved context?
        """
        if not chunks:
            return 0.0
        
        # Split answer into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Build corpus from chunks
        chunk_corpus = " ".join(chunk.get("text", "") for chunk in chunks).lower()
        
        grounded_sentences = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Extract significant words from sentence
            sentence_words = set(re.findall(r"\b[a-z0-9]{4,}\b", sentence_lower))
            
            if not sentence_words:
                continue
            
            # Check how many sentence words appear in chunk corpus
            matching_words = sum(1 for word in sentence_words if word in chunk_corpus)
            
            # Sentence is grounded if >50% of significant words appear in context
            if matching_words / len(sentence_words) > 0.5:
                grounded_sentences += 1
        
        faithfulness = grounded_sentences / len(sentences)
        return faithfulness
    
    def _calculate_citation_correctness(
        self,
        answer: str,
        chunks: list[dict[str, Any]]
    ) -> float:
        """
        Calculate citation correctness: are citations accurate?
        
        Measures:
        - Do citations match chunk metadata?
        - Are citation formats correct?
        - Is there proper attribution?
        """
        # Extract citations from answer
        citation_patterns = [
            r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})',
            r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?',
        ]
        
        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, answer))
        
        if not citations:
            # No citations - check if answer has any claims that need citations
            # If answer is factual but uncited, score is low
            return 0.3
        
        correct_citations = 0
        
        # Build metadata lookup
        chunk_metadata = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            chunk_metadata.append({
                "authors": meta.get("authors", "").lower(),
                "year": str(meta.get("year", "")),
                "title": meta.get("title", "").lower()
            })
        
        for citation in citations:
            # Extract author and year
            match = re.search(r'\(([A-Z][a-zA-Z]+)(?:\s+et\s+al\.)?,\s*(\d{4})', citation)
            if not match:
                continue
            
            author = match.group(1).lower()
            year = match.group(2)
            
            # Check if this citation matches any chunk
            for meta in chunk_metadata:
                if author in meta["authors"] and year in meta["year"]:
                    correct_citations += 1
                    break
        
        if not citations:
            return 0.0
        
        return correct_citations / len(citations)
    
    def _calculate_contradiction_rate(self, answer: str) -> float:
        """
        Calculate contradiction rate: does the answer contradict itself?
        
        Looks for:
        - Explicit contradiction markers (however, but, although)
        - Numerical contradictions
        - Logical inconsistencies
        """
        if not answer:
            return 0.0
        
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        contradictions = 0
        
        # Check for explicit contradiction markers
        contradiction_markers = [
            "however", "but", "although", "despite", "conversely",
            "on the other hand", "in contrast", "contrary to"
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(marker in sentence_lower for marker in contradiction_markers):
                # This might indicate a contradiction - check if it's a valid contrast
                # or an actual contradiction
                contradictions += 0.5  # Partial weight for potential contradiction
        
        # Check for numerical contradictions
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', answer)
        if len(numbers) >= 2:
            # If same number appears with different qualifiers, might be contradiction
            number_counts = Counter(numbers)
            for num, count in number_counts.items():
                if count > 1:
                    contradictions += 0.3
        
        # Normalize by sentence count
        contradiction_rate = min(contradictions / len(sentences), 1.0)
        return contradiction_rate
    
    def _calculate_answer_completeness(
        self,
        query: str,
        answer: str,
        expected_answer: str | None = None
    ) -> float:
        """
        Calculate answer completeness: does the answer fully address the query?
        
        Measures:
        - Does answer address all parts of the query?
        - Is answer length appropriate?
        - If expected_answer provided, measures overlap
        """
        if not answer:
            return 0.0
        
        # Extract key terms from query
        query_terms = set(re.findall(r"\b[a-z0-9]{4,}\b", query.lower()))
        
        if not query_terms:
            return 0.5  # Neutral if query has no significant terms
        
        # Check how many query terms are addressed in answer
        answer_lower = answer.lower()
        addressed_terms = sum(1 for term in query_terms if term in answer_lower)
        
        term_coverage = addressed_terms / len(query_terms)
        
        # Length appropriateness (not too short, not too long)
        answer_length = len(answer.split())
        optimal_length = len(query.split()) * 3  # Heuristic
        
        if answer_length < optimal_length * 0.5:
            length_score = 0.5  # Too short
        elif answer_length > optimal_length * 3:
            length_score = 0.7  # Too long
        else:
            length_score = 1.0  # Good length
        
        # If expected answer provided, measure overlap
        if expected_answer:
            expected_terms = set(re.findall(r"\b[a-z0-9]{4,}\b", expected_answer.lower()))
            answer_terms = set(re.findall(r"\b[a-z0-9]{4,}\b", answer.lower()))
            
            if expected_terms:
                overlap = len(expected_terms & answer_terms)
                expected_coverage = overlap / len(expected_terms)
                completeness = 0.5 * term_coverage + 0.5 * expected_coverage
            else:
                completeness = term_coverage
        else:
            completeness = 0.7 * term_coverage + 0.3 * length_score
        
        return completeness
    
    def evaluate_batch(
        self,
        evaluations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Evaluate a batch of query-response pairs.
        
        Args:
            evaluations: List of dicts with keys: query, answer, chunks, expected_answer (optional)
        
        Returns:
            Aggregate metrics across all evaluations.
        """
        results = []
        
        for eval_data in evaluations:
            result = self.evaluate(
                query=eval_data["query"],
                answer=eval_data["answer"],
                chunks=eval_data["chunks"],
                expected_answer=eval_data.get("expected_answer")
            )
            results.append(result)
        
        # Calculate aggregate metrics
        avg_overall = sum(r.overall_score for r in results) / len(results)
        avg_retrieval = sum(r.retrieval_precision for r in results) / len(results)
        avg_faithfulness = sum(r.context_faithfulness for r in results) / len(results)
        avg_citation = sum(r.citation_correctness for r in results) / len(results)
        avg_contradiction = sum(r.contradiction_rate for r in results) / len(results)
        avg_completeness = sum(r.answer_completeness for r in results) / len(results)
        
        aggregate = {
            "num_evaluations": len(results),
            "avg_overall_score": avg_overall,
            "avg_retrieval_precision": avg_retrieval,
            "avg_context_faithfulness": avg_faithfulness,
            "avg_citation_correctness": avg_citation,
            "avg_contradiction_rate": avg_contradiction,
            "avg_answer_completeness": avg_completeness,
            "individual_results": results
        }
        
        logger.info(
            f"Batch evaluation complete: {len(results)} queries, "
            f"avg_overall={avg_overall:.3f}"
        )
        
        return aggregate
    
    def generate_report(self, result: EvaluationResult) -> str:
        """
        Generate a human-readable evaluation report.
        """
        report_lines = [
            f"RAG Evaluation Report",
            f"=" * 50,
            f"",
            f"Query: {result.query}",
            f"",
            f"Metrics:",
            f"  Retrieval Precision: {result.retrieval_precision:.3f}",
            f"  Context Faithfulness: {result.context_faithfulness:.3f}",
            f"  Citation Correctness: {result.citation_correctness:.3f}",
            f"  Contradiction Rate: {result.contradiction_rate:.3f}",
            f"  Answer Completeness: {result.answer_completeness:.3f}",
            f"",
            f"Overall Score: {result.overall_score:.3f}",
            f"",
            f"Details:",
            f"  Chunks used: {result.details['chunk_count']}",
            f"  Answer length: {result.details['answer_length']} chars",
        ]
        
        return "\n".join(report_lines)
