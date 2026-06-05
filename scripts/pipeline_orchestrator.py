"""
pipeline_orchestrator.py — Unified Decision Orchestrator for RAG.

Eliminates signal duplication by establishing a clear decision hierarchy:
- Hard filters: metadata_filter + domain filter (only constraints)
- Soft signals: reranker (ALL scoring happens here)
- Validation: quality_gate + citation_verifier (only validation)

Everything else only contributes signals, not decisions.

Now includes three critical layers:
- Evaluation: RAG evaluation harness for measuring system performance
- Failure Handling: System safety logic for detecting and handling failure modes
- Weight Calibration: Empirical tuning of scoring weights
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """State tracking through the RAG pipeline."""
    query: str
    original_chunks: list[dict]
    hard_filtered_chunks: list[dict]
    reranked_chunks: list[dict]
    quality_checked_chunks: list[dict]
    final_chunks: list[dict]
    decisions: dict[str, Any]
    
    # New layers
    failure_detection: dict[str, Any] | None = None
    evaluation_result: dict[str, Any] | None = None
    weight_config: dict[str, Any] | None = None
    
    # Debugging
    trace_mode: bool = False


class PipelineOrchestrator:
    """
    Unified orchestrator that coordinates all RAG components without
    overlapping decision logic.
    """

    def __init__(self, trace_mode: bool = False):
        """Initialize the pipeline orchestrator."""
        self.trace_mode = trace_mode
        logger.info(f"Pipeline orchestrator initialized (trace_mode={trace_mode})")
        
        # Initialize new layers (lazy loading to avoid import errors if modules not available)
        self.evaluator = None
        self.failure_handler = None
        self.weight_calibrator = None
        
        try:
            from rag_evaluator import RAGEvaluator
            self.evaluator = RAGEvaluator()
            logger.info("RAG evaluator loaded")
        except ImportError:
            logger.warning("RAG evaluator not available")
        
        try:
            from failure_handler import FailureHandler
            self.failure_handler = FailureHandler()
            logger.info("Failure handler loaded")
        except ImportError:
            logger.warning("Failure handler not available")
        
        try:
            from weight_calibrator import WeightCalibrator
            self.weight_calibrator = WeightCalibrator()
            logger.info("Weight calibrator loaded")
        except ImportError:
            logger.warning("Weight calibrator not available")

    def _validate_chunk_schema(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Validate and enforce chunk schema consistency.
        
        Every chunk must have: chunk_id, paper_id, section, year, authors, text
        If any field is missing, this is a data contract violation.
        """
        required_fields = ["chunk_id", "paper_id", "section", "year", "authors", "text"]
        validated_chunks = []
        
        for idx, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            
            # Check for missing required fields
            missing_fields = []
            for field in required_fields:
                if field not in metadata and field != "text":
                    missing_fields.append(field)
                if field == "text" and field not in chunk:
                    missing_fields.append(field)
            
            if missing_fields:
                logger.error(
                    f"Chunk schema violation at index {idx}: missing fields {missing_fields}. "
                    f"This will cause unpredictable pipeline behavior."
                )
                # Skip invalid chunks
                continue
            
            validated_chunks.append(chunk)
        
        if len(validated_chunks) < len(chunks):
            logger.warning(
                f"Filtered {len(chunks) - len(validated_chunks)} chunks due to schema violations. "
                f"Only {len(validated_chunks)} valid chunks remain."
            )
        
        return validated_chunks

    def execute_pipeline(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        *,
        enable_hard_filters: bool = True,
        enable_reranking: bool = True,
        enable_quality_gate: bool = True,
        enable_failure_handling: bool = True,
        enable_evaluation: bool = False,
        answer: str | None = None
    ) -> PipelineState:
        """
        Execute the unified RAG pipeline with clear decision hierarchy.

        Pipeline stages:
          0. Schema validation - enforce data contract
          1. Hard filters (metadata + domain) - only constraints
          2. Reranking - ALL scoring happens here (ONLY ordering authority)
          3. Quality gate - REAL gate (refuse when quality is low)
          4. Failure detection - safety logic
          5. Final selection
          6. Evaluation - performance measurement (optional)

        Args:
            query: User's research question.
            chunks: Retrieved chunks from vector search.
            enable_hard_filters: Enable metadata/domain filtering.
            enable_reranking: Enable reranking.
            enable_quality_gate: Enable quality gate validation.
            enable_failure_handling: Enable failure mode detection.
            enable_evaluation: Enable RAG evaluation (requires answer).
            answer: Generated answer for evaluation (required if enable_evaluation=True).

        Returns:
            PipelineState with all intermediate results and decisions.
        """
        state = PipelineState(
            query=query,
            original_chunks=chunks,
            hard_filtered_chunks=[],
            reranked_chunks=[],
            quality_checked_chunks=[],
            final_chunks=[],
            decisions={},
            failure_detection=None,
            evaluation_result=None,
            weight_config=None,
            trace_mode=self.trace_mode
        )
        
        # Stage 0: Schema validation (enforce data contract)
        if self.trace_mode:
            logger.info(f"[TRACE] Stage 0: Schema validation - {len(chunks)} chunks")
        
        validated_chunks = self._validate_chunk_schema(chunks)
        state.original_chunks = validated_chunks  # Use validated chunks
        
        if self.trace_mode:
            logger.info(f"[TRACE] After schema validation: {len(validated_chunks)} chunks")

        # Stage 1: Hard filters (only constraints, no scoring)
        if self.trace_mode:
            logger.info(f"[TRACE] Stage 1: Hard filters - {len(validated_chunks)} chunks")
        
        if enable_hard_filters:
            state.hard_filtered_chunks = self._apply_hard_filters(
                query, validated_chunks
            )
            state.decisions["hard_filter_applied"] = True
            state.decisions["hard_filter_count"] = len(validated_chunks) - len(state.hard_filtered_chunks)
        else:
            state.hard_filtered_chunks = validated_chunks
            state.decisions["hard_filter_applied"] = False
        
        if self.trace_mode:
            logger.info(f"[TRACE] After hard filters: {len(state.hard_filtered_chunks)} chunks")

        # Stage 2: Weight calibration (if available)
        if self.trace_mode:
            logger.info(f"[TRACE] Stage 2: Weight calibration")
        
        if self.weight_calibrator and enable_reranking:
            weight_config = self.weight_calibrator.get_weights_for_query(query)
            state.weight_config = {
                "semantic": weight_config.semantic_weight,
                "lexical": weight_config.lexical_weight,
                "section": weight_config.section_weight,
                "diversity": weight_config.diversity_weight
            }
            state.decisions["weight_calibration_applied"] = True
        else:
            state.weight_config = None
            state.decisions["weight_calibration_applied"] = False

        # Stage 3: Reranking (ALL scoring happens here - ONLY ordering authority)
        if self.trace_mode:
            logger.info(f"[TRACE] Stage 3: Reranking (ONLY ordering authority) - {len(state.hard_filtered_chunks)} chunks")
        
        if enable_reranking:
            state.reranked_chunks = self._apply_reranking(
                query, state.hard_filtered_chunks,
                weight_config=state.weight_config
            )
            state.decisions["reranking_applied"] = True
        else:
            state.reranked_chunks = state.hard_filtered_chunks
            state.decisions["reranking_applied"] = False
        
        if self.trace_mode:
            logger.info(f"[TRACE] After reranking: {len(state.reranked_chunks)} chunks")
            if state.reranked_chunks:
                top_score = state.reranked_chunks[0].get("rerank_score", 0)
                logger.info(f"[TRACE] Top rerank score: {top_score:.3f}")

        # Stage 4: Quality gate (REAL gate - re-retrieve, relax filters, or refuse)
        if self.trace_mode:
            logger.info(f"[TRACE] Stage 4: Quality gate (REAL gate) - {len(state.reranked_chunks)} chunks")
        
        if enable_quality_gate:
            quality_result = self._apply_quality_gate(
                query, state.reranked_chunks
            )
            state.decisions["quality_gate_passed"] = quality_result["passed"]
            state.decisions["quality_score"] = quality_result["score"]
            
            # REAL gate behavior: if quality is too low, take explicit action
            if not quality_result["passed"]:
                logger.warning(
                    f"Quality gate FAILED (score: {quality_result['score']:.3f} < threshold). "
                    f"Taking explicit action: REFUSE answer"
                )
                state.quality_checked_chunks = []  # Clear chunks to refuse answer
                state.decisions["quality_gate_action"] = "refuse"
            else:
                state.quality_checked_chunks = state.reranked_chunks  # Preserve reranker order
                state.decisions["quality_gate_action"] = "proceed"
        else:
            state.quality_checked_chunks = state.reranked_chunks
            state.decisions["quality_gate_passed"] = True
            state.decisions["quality_score"] = 1.0
            state.decisions["quality_gate_action"] = "proceed"
        
        if self.trace_mode:
            logger.info(f"[TRACE] After quality gate: {len(state.quality_checked_chunks)} chunks")
            logger.info(f"[TRACE] Quality gate action: {state.decisions['quality_gate_action']}")

        # Stage 5: Failure detection (safety logic)
        if self.trace_mode:
            logger.info(f"[TRACE] Stage 5: Failure detection - {len(state.quality_checked_chunks)} chunks")
        
        if enable_failure_handling and self.failure_handler:
            pipeline_state_for_failure = {
                "original_chunk_count": len(state.original_chunks),
                "current_chunk_count": len(state.quality_checked_chunks)
            }
            failure_detection = self.failure_handler.detect_failures(
                query, state.quality_checked_chunks, pipeline_state_for_failure
            )
            state.failure_detection = {
                "has_failure": failure_detection.has_failure,
                "failure_type": failure_detection.failure_type,
                "severity": failure_detection.severity,
                "message": failure_detection.message,
                "suggested_action": failure_detection.suggested_action,
                "metrics": failure_detection.metrics
            }
            state.decisions["failure_detected"] = failure_detection.has_failure
            
            # Handle failure if detected
            if failure_detection.has_failure:
                handling = self.failure_handler.handle_failure(
                    failure_detection, query, state.quality_checked_chunks
                )
                state.decisions["failure_handling_action"] = handling["action"]
                
                # If action is to refuse, clear chunks
                if handling["action"] == "refuse":
                    state.final_chunks = []
                    state.decisions["refusal_reason"] = handling["message"]
                else:
                    state.final_chunks = handling["modified_chunks"]
            else:
                state.final_chunks = state.quality_checked_chunks
        else:
            state.failure_detection = None
            state.decisions["failure_detected"] = False
            state.final_chunks = state.quality_checked_chunks
        
        if self.trace_mode:
            logger.info(f"[TRACE] After failure detection: {len(state.final_chunks)} chunks")
            if state.decisions.get("failure_detected"):
                logger.info(f"[TRACE] Failure type: {state.failure_detection.get('failure_type') if state.failure_detection else 'N/A'}")

        # Stage 6: Final selection (if not refused) - preserves reranker order
        if self.trace_mode:
            logger.info(f"[TRACE] Stage 6: Final selection - {len(state.final_chunks)} chunks")
        
        if state.final_chunks:
            state.final_chunks = self._select_final_chunks(
                state.final_chunks,
                limit=8
            )
        
        if self.trace_mode:
            logger.info(f"[TRACE] After final selection: {len(state.final_chunks)} chunks")

        # Stage 7: Evaluation (optional, requires answer)
        if enable_evaluation and self.evaluator and answer:
            if self.trace_mode:
                logger.info(f"[TRACE] Stage 7: Evaluation")
            
            evaluation_result = self.evaluator.evaluate(
                query=query,
                answer=answer,
                chunks=state.final_chunks
            )
            state.evaluation_result = {
                "retrieval_precision": evaluation_result.retrieval_precision,
                "context_faithfulness": evaluation_result.context_faithfulness,
                "citation_correctness": evaluation_result.citation_correctness,
                "contradiction_rate": evaluation_result.contradiction_rate,
                "answer_completeness": evaluation_result.answer_completeness,
                "overall_score": evaluation_result.overall_score,
                "details": evaluation_result.details
            }
            state.decisions["evaluation_performed"] = True
        else:
            state.evaluation_result = None
            state.decisions["evaluation_performed"] = False

        logger.info(
            f"Pipeline executed: {len(chunks)} → {len(state.final_chunks)} chunks. "
            f"Decisions: {state.decisions}"
        )

        return state

    def _apply_hard_filters(
        self,
        query: str,
        chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Apply hard filters (metadata + domain constraints only).

        This stage only applies constraints, no scoring or ranking.
        """
        filtered = chunks.copy()

        # Domain filtering (if query is domain-specific)
        try:
            from topic_classifier import TopicClassifier
            classifier = TopicClassifier()
            filter_domain = classifier.get_domain_filter(query)
            if filter_domain:
                # Strict domain filtering: only keep chunks with matching domain metadata
                filtered = [
                    c for c in filtered
                    if c.get("metadata", {}).get("domain") == filter_domain
                ]
                logger.debug(f"Domain filter applied: {filter_domain}")
                
                # Additional cross-domain contamination check
                # Remove chunks that contain keywords from other domains
                domain_keywords = {
                    "medical_imaging": ["chain of thought", "alignment", "latent reasoning", "ai safety", "model alignment"],
                    "smart_city_cyber": ["medical", "diagnosis", "clinical", "patient", "treatment"],
                    "coffee_landscape": ["malware", "intrusion", "cybersecurity", "network security"],
                }
                
                if filter_domain in domain_keywords:
                    forbidden_keywords = domain_keywords[filter_domain]
                    for kw in forbidden_keywords:
                        filtered = [
                            c for c in filtered
                            if kw not in c.get("text", "").lower()
                        ]
                        if len(filtered) < len(chunks):
                            logger.debug(f"Cross-domain contamination filter removed chunks with keyword: {kw}")
        except ImportError:
            pass

        # Metadata filtering (year, venue constraints)
        try:
            from metadata_filter import MetadataFilter
            metadata_filter = MetadataFilter()
            if metadata_filter.should_apply_metadata_filtering(query):
                year_constraint = metadata_filter.extract_year_constraint(query)
                venue_constraint = metadata_filter.extract_venue_constraint(query)

                if year_constraint:
                    filtered = [
                        c for c in filtered
                        if str(c.get("metadata", {}).get("year", "")) == str(year_constraint)
                    ]
                if venue_constraint:
                    filtered = [
                        c for c in filtered
                        if venue_constraint.lower() in c.get("metadata", {}).get("venue", "").lower()
                    ]
        except ImportError:
            pass

        return filtered

    def _apply_reranking(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        weight_config: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Apply reranking (ALL scoring happens here).

        This is the ONLY place where scoring decisions happen.
        
        Args:
            query: The query string.
            chunks: Retrieved chunks.
            weight_config: Optional weight configuration from calibrator.
        """
        if not chunks:
            return chunks

        try:
            from reranker import RerankerService
            reranker = RerankerService()
            
            # Use calibrated weights if available
            if weight_config:
                weights = {
                    "semantic": weight_config.get("semantic", 0.4),
                    "lexical": weight_config.get("lexical", 0.3),
                    "section": weight_config.get("section", 0.2),
                    "diversity": weight_config.get("diversity", 0.1)
                }
                reranked = reranker.rerank(chunks, query, top_k=len(chunks), weights=weights)
            else:
                reranked = reranker.rerank(chunks, query, top_k=len(chunks))
            
            return reranked
        except ImportError:
            logger.warning("Reranker not available, skipping")
            return chunks

    def _apply_quality_gate(
        self,
        query: str,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Apply quality gate (validation only, no filtering).

        This stage only validates quality, does not filter.
        """
        if not chunks:
            return {"passed": False, "score": 0.0}

        try:
            from context_shaper import ContextShaper
            shaper = ContextShaper()
            metrics = shaper.estimate_context_quality(chunks, query)
            
            # Quality threshold
            passed = metrics["quality_score"] >= 0.3
            return {
                "passed": passed,
                "score": metrics["quality_score"],
                "metrics": metrics
            }
        except ImportError:
            return {"passed": True, "score": 1.0}

    def _select_final_chunks(
        self,
        chunks: list[dict[str, Any]],
        limit: int = 8
    ) -> list[dict[str, Any]]:
        """
        Select final chunks based on reranking scores.

        Simple selection based on pre-computed scores from reranking.
        """
        if not chunks:
            return chunks

        # Chunks should already be sorted by reranker scores
        return chunks[:limit]

    def get_pipeline_summary(self, state: PipelineState) -> str:
        """
        Get a human-readable summary of pipeline decisions.
        """
        summary_parts = []
        summary_parts.append(f"Original chunks: {len(state.original_chunks)}")

        if state.decisions.get("hard_filter_applied"):
            summary_parts.append(
                f"Hard filter removed: {state.decisions.get('hard_filter_count', 0)} chunks"
            )

        if state.decisions.get("weight_calibration_applied"):
            summary_parts.append("Weight calibration applied")

        if state.decisions.get("reranking_applied"):
            summary_parts.append("Reranking applied")

        if state.decisions.get("quality_gate_passed"):
            summary_parts.append("Quality gate: PASSED")
        else:
            summary_parts.append(
                f"Quality gate: FAILED (score: {state.decisions.get('quality_score', 0):.2f})"
            )

        if state.decisions.get("failure_detected"):
            failure_type = state.failure_detection.get("failure_type", "unknown") if state.failure_detection else "unknown"
            summary_parts.append(f"Failure detected: {failure_type}")

        if state.decisions.get("evaluation_performed"):
            eval_score = state.evaluation_result.get("overall_score", 0) if state.evaluation_result else 0
            summary_parts.append(f"Evaluation score: {eval_score:.3f}")

        summary_parts.append(f"Final chunks: {len(state.final_chunks)}")

        return " → ".join(summary_parts)
