"""
query_understanding.py — Query Understanding Service for RAG.

Normalizes and interprets research questions before retrieval.
Extracts intent, constraints, and builds structured query representations.
"""

import re
import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Structured analysis of a user query."""
    original_query: str
    normalized_query: str
    intent: str
    constraints: dict[str, Any]
    key_terms: list[str]
    question_type: str


class QueryUnderstanding:
    """
    Understands and classifies user queries for better retrieval.
    """

    # Intent patterns
    INTENT_PATTERNS = {
        "definition": [
            r"\b(?:what\s+is|define|explain|describe)\b",
            r"\b(?:meaning\s+of|definition\s+of)\b",
        ],
        "comparison": [
            r"\b(?:compare|contrast|difference|versus|vs\.?)\b",
            r"\b(?:better|worse)\s+than\b",
            r"\b(?:similar|different)\s+(?:to|from)\b",
        ],
        "method_explanation": [
            r"\b(?:how|method|approach|technique|algorithm)\b",
            r"\b(?:implementation|pipeline|process)\b",
            r"\b(?:step|procedure)\b",
        ],
        "result_lookup": [
            r"\b(?:result|finding|outcome|performance|accuracy)\b",
            r"\b(?:achieve|obtain|report)\b",
            r"\b(?:score|metric|benchmark)\b",
        ],
        "overview": [
            r"\b(?:overview|summary|introduction|background)\b",
            r"\b(?:what\s+do|what\s+are)\s+(?:they|the)\b",
            r"\b(?:main|key)\s+(?:idea|concept|point)\b",
        ],
        "listing": [
            r"\b(?:list|show|enumerate|tabulate)\b",
            r"\b(?:all|every)\s+(?:paper|article|study)\b",
            r"\b(?:papers?|articles?|studies?)\s+(?:by|from|of)\b",
        ],
    }

    def __init__(self):
        """Initialize the query understanding service."""
        logger.info("Query understanding service initialized")

    def normalize_query(self, query: str) -> str:
        """
        Normalize the query into a search-optimized form.

        Removes filler words, standardizes phrasing, and extracts core intent.
        """
        # Lowercase
        normalized = query.lower().strip()
        
        # Remove common filler phrases
        filler_phrases = [
            "i want to know", "i would like to know", "can you tell me",
            "please tell me", "i need to know", "help me understand",
        ]
        for phrase in filler_phrases:
            normalized = normalized.replace(phrase, "")
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def classify_intent(self, query: str) -> str:
        """
        Classify the query intent.

        Returns one of: definition, comparison, method_explanation, 
        result_lookup, overview, listing, unknown.
        """
        query_lower = query.lower()
        
        # Check each intent pattern
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.debug(f"Query intent classified as: {intent}")
                    return intent
        
        logger.debug("Query intent classified as: unknown")
        return "unknown"

    def extract_constraints(self, query: str) -> dict[str, Any]:
        """
        Extract constraints from the query.

        Returns dict with:
            - paper_title: Specific paper mentioned
            - author: Author name mentioned
            - year: Year constraint
            - venue: Venue constraint
            - domain: Research domain
        """
        constraints = {}
        query_lower = query.lower()
        
        # Extract paper title (in quotes)
        title_match = re.search(r'"([^"]{10,200})"', query)
        if title_match:
            constraints["paper_title"] = title_match.group(1)
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            constraints["year"] = int(year_match.group(0))
        
        # Extract venue
        venues = ["CVPR", "ICCV", "NeurIPS", "ICML", "AAAI", "IJCAI", "KDD", "SIGIR", "WWW", "ACL"]
        for venue in venues:
            if venue.lower() in query_lower:
                constraints["venue"] = venue
                break
        
        # Extract author (simple heuristic - capitalized words before "papers" or "work")
        author_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:papers?|work|research)', query)
        if author_match:
            constraints["author"] = author_match.group(1)
        
        logger.debug(f"Extracted constraints: {constraints}")
        return constraints

    def extract_key_terms(self, query: str) -> list[str]:
        """
        Extract key terms from the query for retrieval.

        Removes stop words and returns significant terms.
        """
        stop_words = {
            "what", "which", "when", "where", "does", "about", "from", "with",
            "that", "this", "have", "into", "your", "their", "paper", "papers",
            "author", "authors", "say", "says", "line", "summarize", "summary",
            "brief", "explain", "describe", "tell", "give", "please", "would",
            "thoughts", "views", "opinions", "ideas", "main", "the", "a", "an",
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "shall", "can", "need", "how", "why",
        }
        
        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', query.lower())
        
        # Filter stop words
        key_terms = [w for w in words if w not in stop_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        logger.debug(f"Extracted key terms: {unique_terms}")
        return unique_terms

    def classify_question_type(self, query: str) -> str:
        """
        Classify the question type for retrieval optimization.

        Returns one of: factual, analytical, comparative, procedural, listing.
        """
        query_lower = query.lower()
        
        # Factual questions
        if re.search(r'\b(?:what|who|when|where|which)\b', query_lower):
            if not re.search(r'\b(?:why|how)\b', query_lower):
                return "factual"
        
        # Analytical questions
        if re.search(r'\b(?:analyze|analysis|evaluate|assess|critique)\b', query_lower):
            return "analytical"
        
        # Comparative questions
        if re.search(r'\b(?:compare|contrast|difference|versus|vs\.?)\b', query_lower):
            return "comparative"
        
        # Procedural questions
        if re.search(r'\b(?:how|method|approach|technique|algorithm|process)\b', query_lower):
            return "procedural"
        
        # Listing questions
        if re.search(r'\b(?:list|show|enumerate|tabulate|all)\b', query_lower):
            return "listing"
        
        return "factual"  # Default

    def understand_query(self, query: str) -> QueryAnalysis:
        """
        Perform complete query understanding.

        Args:
            query: User's research question.

        Returns:
            QueryAnalysis with normalized query, intent, constraints, etc.
        """
        normalized = self.normalize_query(query)
        intent = self.classify_intent(query)
        constraints = self.extract_constraints(query)
        key_terms = self.extract_key_terms(query)
        question_type = self.classify_question_type(query)
        
        analysis = QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            constraints=constraints,
            key_terms=key_terms,
            question_type=question_type
        )
        
        logger.info(
            f"Query understood: intent={intent}, type={question_type}, "
            f"constraints={len(constraints)}, terms={len(key_terms)}"
        )
        
        return analysis

    def build_search_query(self, analysis: QueryAnalysis) -> str:
        """
        Build an optimized search query from the analysis.

        Combines key terms and constraints into a search-optimized string.
        """
        parts = []
        
        # Add key terms
        if analysis.key_terms:
            parts.append(" ".join(analysis.key_terms))
        
        # Add constraints
        if "paper_title" in analysis.constraints:
            parts.append(f'"{analysis.constraints["paper_title"]}"')
        
        if "author" in analysis.constraints:
            parts.append(analysis.constraints["author"])
        
        if "venue" in analysis.constraints:
            parts.append(analysis.constraints["venue"])
        
        # If no parts, use normalized query
        if not parts:
            return analysis.normalized_query
        
        return " ".join(parts)

    def get_pipeline_routing(self, analysis: QueryAnalysis) -> dict[str, Any]:
        """
        Determine pipeline routing based on query analysis.

        Returns routing configuration that controls:
          - Section boosting (which sections to prioritize)
          - Diversity requirements
          - Metadata filtering strictness
          - Retrieval limit adjustments

        Args:
            analysis: Query analysis result.

        Returns:
            Dict with routing configuration.
        """
        routing = {
            "boost_sections": [],
            "increase_diversity": False,
            "strict_metadata_filter": False,
            "retrieval_limit_multiplier": 1.0,
            "rerank_weights": None
        }
        
        # Definition queries → boost intro sections
        if analysis.intent == "definition" or analysis.question_type == "factual":
            routing["boost_sections"] = ["Abstract", "Introduction", "Background"]
        
        # Method queries → boost methodology sections
        elif analysis.intent == "method_explanation" or analysis.question_type == "procedural":
            routing["boost_sections"] = ["Methodology", "Methods", "Method", "Experimental Setup"]
        
        # Comparison queries → increase diversity
        elif analysis.intent == "comparison" or analysis.question_type == "comparative":
            routing["increase_diversity"] = True
            routing["retrieval_limit_multiplier"] = 1.5  # Retrieve more for comparison
        
        # Specific paper constraints → strict metadata filter
        if "paper_title" in analysis.constraints or "author" in analysis.constraints:
            routing["strict_metadata_filter"] = True
        
        # Listing queries → increase retrieval limit
        if analysis.intent == "listing" or analysis.question_type == "listing":
            routing["retrieval_limit_multiplier"] = 2.0
        
        # Adjust reranker weights based on intent
        if analysis.intent == "result_lookup":
            routing["rerank_weights"] = {
                "semantic": 0.5,
                "lexical": 0.2,
                "section": 0.3,
                "diversity": 0.0
            }
        elif analysis.intent == "comparison":
            routing["rerank_weights"] = {
                "semantic": 0.3,
                "lexical": 0.3,
                "section": 0.2,
                "diversity": 0.2
            }
        
        logger.info(
            f"Pipeline routing: intent={analysis.intent}, "
            f"boost_sections={routing['boost_sections']}, "
            f"diversity={routing['increase_diversity']}"
        )
        
        return routing
