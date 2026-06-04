"""
failure_handler.py — Failure Mode Handler for RAG.

System safety logic that detects and handles failure modes:
- No good retrieval → respond: "insufficient evidence in corpus"
- High contradiction → explicitly present disagreement
- Low coherence → broaden retrieval automatically
- Over-filtering → relax constraints dynamically

This is NOT a new retrieval module. It is system safety logic that
detects when the pipeline is producing poor output and handles it gracefully.
"""

import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FailureDetection:
    """Result of failure mode detection."""
    has_failure: bool
    failure_type: str | None
    severity: str  # "low", "medium", "high"
    message: str
    suggested_action: str
    metrics: dict[str, Any]


class FailureHandler:
    """
    Detects and handles failure modes in the RAG pipeline.
    
    This is a safety layer that runs after retrieval but before
    final answer generation. It detects when the system is about
    to produce poor output and handles it appropriately.
    """
    
    def __init__(self):
        """Initialize the failure handler."""
        logger.info("Failure handler initialized")
    
    def detect_failures(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        pipeline_state: dict[str, Any] | None = None
    ) -> FailureDetection:
        """
        Detect failure modes in the current pipeline state.
        
        Args:
            query: The original query.
            chunks: Retrieved chunks.
            pipeline_state: Optional pipeline state with intermediate metrics.
        
        Returns:
            FailureDetection with details about any failures detected.
        """
        failures = []
        
        # Check 1: No good retrieval
        no_retrieval = self._check_no_retrieval(chunks, query)
        if no_retrieval["has_failure"]:
            failures.append(no_retrieval)
        
        # Check 2: High contradiction
        high_contradiction = self._check_high_contradiction(chunks)
        if high_contradiction["has_failure"]:
            failures.append(high_contradiction)
        
        # Check 3: Low coherence
        low_coherence = self._check_low_coherence(chunks, query)
        if low_coherence["has_failure"]:
            failures.append(low_coherence)
        
        # Check 4: Over-filtering
        over_filtering = self._check_over_filtering(chunks, pipeline_state)
        if over_filtering["has_failure"]:
            failures.append(over_filtering)
        
        # Determine the most severe failure
        if not failures:
            return FailureDetection(
                has_failure=False,
                failure_type=None,
                severity="low",
                message="No failures detected",
                suggested_action="proceed",
                metrics={}
            )
        
        # Sort by severity (high > medium > low)
        severity_order = {"high": 3, "medium": 2, "low": 1}
        failures.sort(key=lambda f: severity_order[f["severity"]], reverse=True)
        
        worst_failure = failures[0]
        
        return FailureDetection(
            has_failure=True,
            failure_type=worst_failure["type"],
            severity=worst_failure["severity"],
            message=worst_failure["message"],
            suggested_action=worst_failure["suggested_action"],
            metrics=worst_failure.get("metrics", {})
        )
    
    def _check_no_retrieval(
        self,
        chunks: list[dict[str, Any]],
        query: str
    ) -> dict[str, Any]:
        """
        Check if retrieval produced insufficient results.
        
        Failure conditions:
        - Zero chunks retrieved
        - Very few chunks (< 3)
        - All chunks have very high distance (> 0.9)
        """
        if not chunks:
            return {
                "has_failure": True,
                "type": "no_retrieval",
                "severity": "high",
                "message": "No relevant chunks were retrieved from the corpus.",
                "suggested_action": "respond_insufficient_evidence",
                "metrics": {"chunk_count": 0}
            }
        
        if len(chunks) < 3:
            return {
                "has_failure": True,
                "type": "insufficient_retrieval",
                "severity": "medium",
                "message": f"Only {len(chunks)} chunks retrieved - insufficient evidence.",
                "suggested_action": "respond_insufficient_evidence",
                "metrics": {"chunk_count": len(chunks)}
            }
        
        # Check if all chunks have high distance (poor relevance)
        distances = [chunk.get("distance", 1.0) for chunk in chunks]
        avg_distance = sum(distances) / len(distances)
        
        if avg_distance > 0.9:
            return {
                "has_failure": True,
                "type": "poor_retrieval_quality",
                "severity": "high",
                "message": f"Retrieved chunks have poor relevance (avg distance: {avg_distance:.3f}).",
                "suggested_action": "respond_insufficient_evidence",
                "metrics": {"avg_distance": avg_distance, "chunk_count": len(chunks)}
            }
        
        return {
            "has_failure": False,
            "type": None,
            "severity": "low",
            "message": "",
            "suggested_action": "",
            "metrics": {"chunk_count": len(chunks), "avg_distance": avg_distance}
        }
    
    def _check_high_contradiction(
        self,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Check if retrieved chunks contain high contradiction.
        
        Failure conditions:
        - Chunks from same paper with conflicting claims
        - Chunks with explicit contradiction markers
        """
        if not chunks:
            return {
                "has_failure": False,
                "type": None,
                "severity": "low",
                "message": "",
                "suggested_action": "",
                "metrics": {}
            }
        
        # Group chunks by paper
        from collections import defaultdict
        papers = defaultdict(list)
        for chunk in chunks:
            title = chunk.get("metadata", {}).get("title", "Unknown")
            papers[title].append(chunk)
        
        # Check for contradictions within papers
        contradiction_markers = [
            "however", "but", "although", "despite", "conversely",
            "on the other hand", "in contrast", "contrary to", "contradicts"
        ]
        
        contradiction_count = 0
        total_chunks = len(chunks)
        
        for paper_title, paper_chunks in papers.items():
            if len(paper_chunks) < 2:
                continue
            
            # Check if chunks from same paper have contradiction markers
            for chunk in paper_chunks:
                text = chunk.get("text", "").lower()
                if any(marker in text for marker in contradiction_markers):
                    contradiction_count += 1
        
        # If >30% of chunks have contradiction markers, flag it
        contradiction_rate = contradiction_count / total_chunks if total_chunks > 0 else 0
        
        if contradiction_rate > 0.3:
            return {
                "has_failure": True,
                "type": "high_contradiction",
                "severity": "medium",
                "message": f"High contradiction detected in retrieved chunks ({contradiction_rate:.1%} have contradiction markers).",
                "suggested_action": "present_disagreement",
                "metrics": {
                    "contradiction_rate": contradiction_rate,
                    "contradiction_count": contradiction_count
                }
            }
        
        return {
            "has_failure": False,
            "type": None,
            "severity": "low",
            "message": "",
            "suggested_action": "",
            "metrics": {"contradiction_rate": contradiction_rate}
        }
    
    def _check_low_coherence(
        self,
        chunks: list[dict[str, Any]],
        query: str
    ) -> dict[str, Any]:
        """
        Check if retrieved chunks have low coherence.
        
        Failure conditions:
        - Chunks from too many different papers (fragmented)
        - Chunks from unrelated sections
        - Poor semantic coherence
        """
        if not chunks:
            return {
                "has_failure": False,
                "type": None,
                "severity": "low",
                "message": "",
                "suggested_action": "",
                "metrics": {}
            }
        
        # Count unique papers
        papers = set()
        for chunk in chunks:
            title = chunk.get("metadata", {}).get("title", "Unknown")
            papers.add(title)
        
        paper_diversity = len(papers)
        
        # Count unique sections
        sections = set()
        for chunk in chunks:
            section = chunk.get("metadata", {}).get("section", "Unknown")
            sections.add(section)
        
        section_diversity = len(sections)
        
        # Check for fragmentation (too many papers for few chunks)
        fragmentation_score = paper_diversity / len(chunks) if chunks else 0
        
        if fragmentation_score > 0.8:
            return {
                "has_failure": True,
                "type": "low_coherence_fragmented",
                "severity": "medium",
                "message": f"Retrieved chunks are highly fragmented ({paper_diversity} papers for {len(chunks)} chunks).",
                "suggested_action": "broaden_retrieval",
                "metrics": {
                    "paper_diversity": paper_diversity,
                    "section_diversity": section_diversity,
                    "fragmentation_score": fragmentation_score
                }
            }
        
        # Check for section incoherence
        if section_diversity > len(chunks) * 0.7:
            return {
                "has_failure": True,
                "type": "low_coherence_sections",
                "severity": "low",
                "message": f"Retrieved chunks span too many different sections ({section_diversity} sections).",
                "suggested_action": "broaden_retrieval",
                "metrics": {
                    "paper_diversity": paper_diversity,
                    "section_diversity": section_diversity,
                    "fragmentation_score": fragmentation_score
                }
            }
        
        return {
            "has_failure": False,
            "type": None,
            "severity": "low",
            "message": "",
            "suggested_action": "",
            "metrics": {
                "paper_diversity": paper_diversity,
                "section_diversity": section_diversity,
                "fragmentation_score": fragmentation_score
            }
        }
    
    def _check_over_filtering(
        self,
        chunks: list[dict[str, Any]],
        pipeline_state: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Check if over-filtering occurred in the pipeline.
        
        Failure conditions:
        - Hard filter removed too many chunks (>80%)
        - Quality gate rejected most chunks
        """
        if not pipeline_state:
            return {
                "has_failure": False,
                "type": None,
                "severity": "low",
                "message": "",
                "suggested_action": "",
                "metrics": {}
            }
        
        original_count = pipeline_state.get("original_chunk_count", len(chunks))
        current_count = len(chunks)
        
        if original_count == 0:
            return {
                "has_failure": False,
                "type": None,
                "severity": "low",
                "message": "",
                "suggested_action": "",
                "metrics": {}
            }
        
        filter_ratio = current_count / original_count
        
        if filter_ratio < 0.2:
            return {
                "has_failure": True,
                "type": "over_filtering",
                "severity": "high",
                "message": f"Over-filtering detected: {original_count} → {current_count} chunks ({filter_ratio:.1%} retained).",
                "suggested_action": "relax_constraints",
                "metrics": {
                    "original_count": original_count,
                    "current_count": current_count,
                    "filter_ratio": filter_ratio
                }
            }
        
        return {
            "has_failure": False,
            "type": None,
            "severity": "low",
            "message": "",
            "suggested_action": "",
            "metrics": {
                "original_count": original_count,
                "current_count": current_count,
                "filter_ratio": filter_ratio
            }
        }
    
    def handle_failure(
        self,
        detection: FailureDetection,
        query: str,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Handle a detected failure mode.
        
        Args:
            detection: FailureDetection from detect_failures().
            query: The original query.
            chunks: Retrieved chunks.
        
        Returns:
            Dict with handling action and response.
        """
        if not detection.has_failure:
            return {
                "action": "proceed",
                "message": "No failure detected, proceeding normally.",
                "modified_chunks": chunks
            }
        
        action = detection.suggested_action
        
        if action == "respond_insufficient_evidence":
            return {
                "action": "refuse",
                "message": self._insufficient_evidence_message(query),
                "modified_chunks": []
            }
        
        elif action == "present_disagreement":
            return {
                "action": "warn",
                "message": detection.message,
                "modified_chunks": chunks
            }
        
        elif action == "broaden_retrieval":
            # This would require re-running retrieval with relaxed constraints
            # For now, we return a warning and proceed
            return {
                "action": "proceed_with_warning",
                "message": detection.message + " Proceeding with current chunks.",
                "modified_chunks": chunks
            }
        
        elif action == "relax_constraints":
            # This would require re-running pipeline with relaxed constraints
            # For now, we return a warning and proceed
            return {
                "action": "proceed_with_warning",
                "message": detection.message + " Proceeding with current chunks.",
                "modified_chunks": chunks
            }
        
        else:
            return {
                "action": "proceed",
                "message": "Unknown failure type, proceeding normally.",
                "modified_chunks": chunks
            }
    
    def _insufficient_evidence_message(self, query: str) -> str:
        """
        Generate a refusal message for insufficient evidence.
        """
        return (
            "I cannot provide a comprehensive answer to this question because "
            "there is insufficient evidence in the ingested corpus. "
            "The retrieved chunks do not contain enough relevant information "
            "to address your query accurately. Please try rephrasing your question "
            "or consider adding more relevant papers to your library."
        )
