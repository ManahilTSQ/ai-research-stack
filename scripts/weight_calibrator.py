"""
weight_calibrator.py — Weight Calibration + Decision Tuning Layer.

Empirically tunes scoring weights through grid search on sample queries.
Tracks grounding scores and adjusts weights per query type.

Right now the system has many scoring systems with uncalibrated weights:
- Reranker weights (semantic, lexical, section, diversity)
- Coherence weights
- Metadata filtering strength

This layer provides empirical calibration instead of guesswork.
"""

import logging
import json
from typing import Any
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import product

logger = logging.getLogger(__name__)


@dataclass
class WeightConfig:
    """Configuration for scoring weights."""
    # Reranker weights
    semantic_weight: float = 0.4
    lexical_weight: float = 0.3
    section_weight: float = 0.2
    diversity_weight: float = 0.1
    
    # Quality gate thresholds
    quality_threshold: float = 0.3
    coherence_threshold: float = 0.4
    
    # Metadata filtering strength
    metadata_filter_strictness: float = 1.0  # 0.0 = loose, 1.0 = strict


@dataclass
class CalibrationResult:
    """Result of weight calibration."""
    weight_config: WeightConfig
    avg_grounding_score: float
    avg_retrieval_precision: float
    avg_context_faithfulness: float
    num_queries_tested: int
    query_type: str


class WeightCalibrator:
    """
    Calibrates scoring weights through empirical testing.
    
    Uses grid search on sample queries to find optimal weight configurations
    for different query types (methodology, results, overview, etc.).
    """
    
    def __init__(self, calibration_file: str | None = None):
        """
        Initialize the weight calibrator.
        
        Args:
            calibration_file: Path to JSON file storing calibrated weights.
        """
        self.calibration_file = calibration_file or "output/weight_calibration.json"
        self.calibrated_weights: dict[str, WeightConfig] = {}
        
        # Load existing calibrations if available
        self._load_calibrations()
        
        logger.info("Weight calibrator initialized")
    
    def _load_calibrations(self):
        """Load calibrated weights from file."""
        try:
            path = Path(self.calibration_file)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    for query_type, config_dict in data.items():
                        self.calibrated_weights[query_type] = WeightConfig(**config_dict)
                logger.info(f"Loaded {len(self.calibrated_weights)} calibrated weight configs")
        except Exception as e:
            logger.warning(f"Failed to load calibrations: {e}")
    
    def _save_calibrations(self):
        """Save calibrated weights to file."""
        try:
            path = Path(self.calibration_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                query_type: asdict(config)
                for query_type, config in self.calibrated_weights.items()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.calibrated_weights)} calibrated weight configs")
        except Exception as e:
            logger.warning(f"Failed to save calibrations: {e}")
    
    def classify_query_type(self, query: str) -> str:
        """
        Classify query into type for weight selection.
        
        Returns one of: methodology, results, overview, comparison, general
        """
        query_lower = query.lower()
        
        # Methodology queries
        if any(word in query_lower for word in ["method", "approach", "technique", "algorithm", "how", "implement"]):
            return "methodology"
        
        # Results queries
        elif any(word in query_lower for word in ["result", "finding", "outcome", "performance", "accuracy", "achieve"]):
            return "results"
        
        # Overview queries
        elif any(word in query_lower for word in ["overview", "summary", "introduction", "what is", "describe", "explain"]):
            return "overview"
        
        # Comparison queries
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs", "better", "contrast"]):
            return "comparison"
        
        # Default
        else:
            return "general"
    
    def get_weights_for_query(self, query: str) -> WeightConfig:
        """
        Get calibrated weights for a query type.
        
        If no calibration exists for the query type, returns default weights.
        """
        query_type = self.classify_query_type(query)
        
        if query_type in self.calibrated_weights:
            logger.debug(f"Using calibrated weights for query type: {query_type}")
            return self.calibrated_weights[query_type]
        
        logger.debug(f"No calibration for query type '{query_type}', using defaults")
        return WeightConfig()  # Default weights
    
    def grid_search_calibration(
        self,
        sample_queries: list[dict[str, Any]],
        evaluator,  # RAGEvaluator instance
        reranker,  # RerankerService instance
        query_type: str = "general"
    ) -> CalibrationResult:
        """
        Perform grid search to find optimal weights for a query type.
        
        Args:
            sample_queries: List of dicts with keys: query, chunks, answer, expected_answer (optional)
            evaluator: RAGEvaluator instance for scoring
            reranker: RerankerService instance for reranking with weights
            query_type: Query type being calibrated
        
        Returns:
            CalibrationResult with best weight configuration.
        """
        logger.info(f"Starting grid search calibration for query type: {query_type}")
        
        # Define grid search space
        semantic_values = [0.3, 0.4, 0.5]
        lexical_values = [0.2, 0.3, 0.4]
        section_values = [0.1, 0.2, 0.3]
        diversity_values = [0.05, 0.1, 0.15]
        
        best_config = None
        best_score = 0.0
        
        total_combinations = len(semantic_values) * len(lexical_values) * len(section_values) * len(diversity_values)
        tested_combinations = 0
        
        for semantic, lexical, section, diversity in product(
            semantic_values, lexical_values, section_values, diversity_values
        ):
            # Normalize weights to sum to 1.0
            total = semantic + lexical + section + diversity
            semantic_norm = semantic / total
            lexical_norm = lexical / total
            section_norm = section / total
            diversity_norm = diversity / total
            
            weights = {
                "semantic": semantic_norm,
                "lexical": lexical_norm,
                "section": section_norm,
                "diversity": diversity_norm
            }
            
            # Test this configuration on sample queries
            total_grounding = 0.0
            total_retrieval = 0.0
            total_faithfulness = 0.0
            
            for sample in sample_queries:
                # Rerank chunks with these weights
                reranked_chunks = reranker.rerank(
                    sample["chunks"],
                    sample["query"],
                    top_k=len(sample["chunks"]),
                    weights=weights
                )
                
                # Evaluate with reranked chunks
                result = evaluator.evaluate(
                    query=sample["query"],
                    answer=sample["answer"],
                    chunks=reranked_chunks,
                    expected_answer=sample.get("expected_answer")
                )
                
                total_grounding += result.context_faithfulness
                total_retrieval += result.retrieval_precision
                total_faithfulness += result.context_faithfulness
            
            # Calculate average scores
            avg_grounding = total_grounding / len(sample_queries)
            avg_retrieval = total_retrieval / len(sample_queries)
            avg_faithfulness = total_faithfulness / len(sample_queries)
            
            # Combined score (prioritize faithfulness)
            combined_score = 0.5 * avg_grounding + 0.3 * avg_retrieval + 0.2 * avg_faithfulness
            
            tested_combinations += 1
            
            if tested_combinations % 10 == 0:
                logger.info(
                    f"Tested {tested_combinations}/{total_combinations} combinations, "
                    f"best score so far: {best_score:.3f}"
                )
            
            if combined_score > best_score:
                best_score = combined_score
                best_config = WeightConfig(
                    semantic_weight=semantic_norm,
                    lexical_weight=lexical_norm,
                    section_weight=section_norm,
                    diversity_weight=diversity_norm
                )
        
        # Store the best configuration
        self.calibrated_weights[query_type] = best_config
        self._save_calibrations()
        
        result = CalibrationResult(
            weight_config=best_config,
            avg_grounding_score=best_score,
            avg_retrieval_precision=avg_retrieval,
            avg_context_faithfulness=avg_faithfulness,
            num_queries_tested=len(sample_queries),
            query_type=query_type
        )
        
        logger.info(
            f"Grid search complete for {query_type}: best score={best_score:.3f}, "
            f"weights: semantic={best_config.semantic_weight:.2f}, "
            f"lexical={best_config.lexical_weight:.2f}, "
            f"section={best_config.section_weight:.2f}, "
            f"diversity={best_config.diversity_weight:.2f}"
        )
        
        return result
    
    def calibrate_all_query_types(
        self,
        sample_queries_by_type: dict[str, list[dict[str, Any]]],
        evaluator,
        reranker
    ) -> dict[str, CalibrationResult]:
        """
        Calibrate weights for all query types.
        
        Args:
            sample_queries_by_type: Dict mapping query types to sample query lists
            evaluator: RAGEvaluator instance
            reranker: RerankerService instance
        
        Returns:
            Dict mapping query types to CalibrationResults
        """
        results = {}
        
        for query_type, sample_queries in sample_queries_by_type.items():
            if not sample_queries:
                logger.warning(f"No sample queries for query type: {query_type}")
                continue
            
            try:
                result = self.grid_search_calibration(
                    sample_queries=sample_queries,
                    evaluator=evaluator,
                    reranker=reranker,
                    query_type=query_type
                )
                results[query_type] = result
            except Exception as e:
                logger.error(f"Calibration failed for {query_type}: {e}")
        
        logger.info(f"Calibration complete for {len(results)} query types")
        return results
    
    def adaptive_tuning(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        evaluator,
        reranker,
        current_weights: WeightConfig | None = None
    ) -> WeightConfig:
        """
        Adaptively tune weights for a specific query based on chunk characteristics.
        
        This is a lightweight tuning that adjusts weights based on the specific
        characteristics of the retrieved chunks, without full grid search.
        
        Args:
            query: The query being processed
            chunks: Retrieved chunks
            evaluator: RAGEvaluator instance
            reranker: RerankerService instance
            current_weights: Current weight configuration (if None, uses defaults)
        
        Returns:
            Adjusted WeightConfig
        """
        if current_weights is None:
            current_weights = self.get_weights_for_query(query)
        
        # Analyze chunk characteristics
        if not chunks:
            return current_weights
        
        # Count unique papers
        papers = set(chunk.get("metadata", {}).get("title", "Unknown") for chunk in chunks)
        paper_diversity = len(papers)
        
        # Count unique sections
        sections = set(chunk.get("metadata", {}).get("section", "Unknown") for chunk in chunks)
        section_diversity = len(sections)
        
        # Average distance
        distances = [chunk.get("distance", 1.0) for chunk in chunks]
        avg_distance = sum(distances) / len(distances)
        
        # Adaptive adjustments
        adjusted_weights = WeightConfig(
            semantic_weight=current_weights.semantic_weight,
            lexical_weight=current_weights.lexical_weight,
            section_weight=current_weights.section_weight,
            diversity_weight=current_weights.diversity_weight
        )
        
        # If high paper diversity, increase diversity penalty weight
        if paper_diversity > 3:
            adjusted_weights.diversity_weight = min(0.2, current_weights.diversity_weight + 0.05)
            # Renormalize
            total = (adjusted_weights.semantic_weight + 
                    adjusted_weights.lexical_weight + 
                    adjusted_weights.section_weight + 
                    adjusted_weights.diversity_weight)
            adjusted_weights.semantic_weight /= total
            adjusted_weights.lexical_weight /= total
            adjusted_weights.section_weight /= total
            adjusted_weights.diversity_weight /= total
        
        # If high average distance (poor semantic match), increase lexical weight
        if avg_distance > 0.7:
            adjusted_weights.lexical_weight = min(0.5, current_weights.lexical_weight + 0.1)
            # Renormalize
            total = (adjusted_weights.semantic_weight + 
                    adjusted_weights.lexical_weight + 
                    adjusted_weights.section_weight + 
                    adjusted_weights.diversity_weight)
            adjusted_weights.semantic_weight /= total
            adjusted_weights.lexical_weight /= total
            adjusted_weights.section_weight /= total
            adjusted_weights.diversity_weight /= total
        
        # If high section diversity, increase section weight
        if section_diversity > 4:
            adjusted_weights.section_weight = min(0.3, current_weights.section_weight + 0.05)
            # Renormalize
            total = (adjusted_weights.semantic_weight + 
                    adjusted_weights.lexical_weight + 
                    adjusted_weights.section_weight + 
                    adjusted_weights.diversity_weight)
            adjusted_weights.semantic_weight /= total
            adjusted_weights.lexical_weight /= total
            adjusted_weights.section_weight /= total
            adjusted_weights.diversity_weight /= total
        
        logger.debug(
            f"Adaptive tuning: paper_diversity={paper_diversity}, "
            f"section_diversity={section_diversity}, avg_distance={avg_distance:.3f}"
        )
        
        return adjusted_weights
    
    def get_calibration_summary(self) -> dict[str, Any]:
        """
        Get a summary of all calibrated weights.
        """
        return {
            "num_calibrated_types": len(self.calibrated_weights),
            "calibrated_types": list(self.calibrated_weights.keys()),
            "weights": {
                query_type: asdict(config)
                for query_type, config in self.calibrated_weights.items()
            }
        }
