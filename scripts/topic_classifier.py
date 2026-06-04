"""
topic_classifier.py — Academic Domain/Topic Classification Service.

Detects the research domain of papers to enable multi-topic separation.
This prevents cross-domain contamination during retrieval (e.g., biology
queries pulling physics papers just because they're semantically similar).
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TopicClassifier:
    """
    Classifies academic papers into research domains based on title, abstract,
    and content metadata.
    """

    # Domain profiles with distinctive keywords
    DOMAINS = {
        "computer_science": {
            "keywords": [
                "algorithm", "machine learning", "deep learning", "neural network",
                "artificial intelligence", "data mining", "natural language",
                "computer vision", "cybersecurity", "cryptography", "blockchain",
                "distributed system", "cloud computing", "internet of things",
                "software engineering", "database", "information retrieval",
                "optimization", "computational", "programming", "code",
                "network", "protocol", "wireless", "5g", "iot", "federated",
            ],
            "venue_patterns": [
                r"\bIEEE\b", r"\bACM\b", r"arxiv", r"CVPR", r"ICCV", r"NeurIPS",
                r"ICML", r"AAAI", r"IJCAI", r"KDD", r"SIGIR", r"WWW", r"ACL",
            ]
        },
        "medical_imaging": {
            "keywords": [
                "medical imaging", "radiology", "mri", "ct scan", "x-ray",
                "histopathology", "dermoscopy", "retinopathy", "melanoma",
                "brain tumor", "cancer detection", "diagnosis", "clinical",
                "patient", "disease", "biomedical", "healthcare", "medical",
                "ultrasound", "mammography", "lesion", "tissue", "cell",
            ],
            "venue_patterns": [
                r"\bRadiology\b", r"\bMedical Imaging\b", r"\bMICCAI\b",
                r"\bSPIE\b", r"\bMedical Physics\b",
            ]
        },
        "biology": {
            "keywords": [
                "gene", "protein", "dna", "rna", "cell", "molecular", "genomics",
                "bioinformatics", "evolution", "ecosystem", "species", "organism",
                "bacteria", "virus", "mutation", "sequencing", "pathway",
                "metabolism", "enzyme", "receptor", "antibody", "vaccine",
            ],
            "venue_patterns": [
                r"\bNature\b", r"\bScience\b", r"\bCell\b", r"\bPNAS\b",
                r"\bGenome Research\b", r"\bBioinformatics\b",
            ]
        },
        "physics": {
            "keywords": [
                "quantum", "particle", "atom", "photon", "electron", "wave",
                "energy", "force", "gravity", "relativity", "thermodynamics",
                "mechanics", "optics", "electromagnetic", "nuclear", "plasma",
                "condensed matter", "astrophysics", "cosmology", "string",
            ],
            "venue_patterns": [
                r"\bPhysical Review\b", r"\bPhysics\b", r"\bNature Physics\b",
                r"\bScience\b", r"\bPRL\b", r"\bPRB\b",
            ]
        },
        "chemistry": {
            "keywords": [
                "molecule", "reaction", "catalyst", "synthesis", "polymer",
                "crystal", "bond", "oxidation", "reduction", "acid", "base",
                "solvent", "compound", "element", "spectroscopy", "chromatography",
                "nanomaterial", "nanoparticle", "chemical", "organic",
            ],
            "venue_patterns": [
                r"\bJournal of.*Chemistry\b", r"\bChemical\b", r"\bACS\b",
                r"\bNature Chemistry\b", r"\bAngewandte\b",
            ]
        },
        "engineering": {
            "keywords": [
                "design", "manufacturing", "materials", "structural", "mechanical",
                "electrical", "civil", "industrial", "robotics", "automation",
                "control", "sensor", "actuator", "fabrication", "processing",
                "quality", "reliability", "maintenance", "optimization",
            ],
            "venue_patterns": [
                r"\bIEEE\b", r"\bEngineering\b", r"\bASME\b", r"\bASCE\b",
            ]
        },
    }

    def __init__(self):
        """Initialize the topic classifier."""
        logger.info("Topic classifier initialized")

    def _score_domain(self, text: str, venue: str, domain: str) -> float:
        """
        Score how well a paper matches a specific domain.

        Returns a score between 0 and 1 based on keyword matches and venue patterns.
        """
        if not text and not venue:
            return 0.0

        domain_config = self.DOMAINS.get(domain, {})
        keywords = domain_config.get("keywords", [])
        venue_patterns = domain_config.get("venue_patterns", [])

        text_lower = text.lower()
        venue_lower = venue.lower()

        # Keyword matching (weighted 70%)
        keyword_score = 0.0
        if keywords and text:
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            keyword_score = matches / len(keywords) if keywords else 0.0

        # Venue pattern matching (weighted 30%)
        venue_score = 0.0
        if venue_patterns and venue:
            pattern_matches = sum(
                1 for pattern in venue_patterns
                if re.search(pattern, venue_lower, re.IGNORECASE)
            )
            venue_score = pattern_matches / len(venue_patterns) if venue_patterns else 0.0

        # Combined score
        total_score = 0.7 * keyword_score + 0.3 * venue_score
        return total_score

    def classify_paper(
        self,
        title: str,
        abstract: str = "",
        venue: str = "",
        content_sample: str = ""
    ) -> dict[str, Any]:
        """
        Classify a paper into research domains.

        Args:
            title: Paper title.
            abstract: Paper abstract (optional).
            venue: Publication venue (optional).
            content_sample: Sample of paper content (optional).

        Returns:
            Dict with:
                - "primary_domain": str - The best-matching domain
                - "all_scores": dict - Scores for all domains
                - "confidence": float - Confidence in primary classification
        """
        # Combine text for classification
        combined_text = " ".join([title, abstract, content_sample])

        # Score each domain
        scores = {}
        for domain in self.DOMAINS.keys():
            score = self._score_domain(combined_text, venue, domain)
            scores[domain] = score

        # Find primary domain
        if not scores:
            return {
                "primary_domain": "unknown",
                "all_scores": {},
                "confidence": 0.0
            }

        primary_domain = max(scores, key=scores.get)
        primary_score = scores[primary_domain]

        # Calculate confidence (normalized by second-best)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            confidence = primary_score / (sorted_scores[1] + 0.01)
        else:
            confidence = primary_score

        # If confidence is too low, mark as unknown
        if primary_score < 0.1 or confidence < 1.2:
            primary_domain = "unknown"
            confidence = 0.0

        result = {
            "primary_domain": primary_domain,
            "all_scores": scores,
            "confidence": confidence
        }

        logger.debug(
            f"Classified paper '{title[:50]}...' as {primary_domain} "
            f"(confidence: {confidence:.2f})"
        )

        return result

    def get_domain_filter(self, query: str) -> str | None:
        """
        Detect if a query is domain-specific and return the domain filter.

        Returns the domain name if the query clearly targets a specific domain,
        otherwise returns None (no filter).
        """
        query_lower = query.lower()

        # Domain-specific query patterns
        domain_indicators = {
            "medical_imaging": [
                "medical imaging", "radiology", "mri", "ct scan", "x-ray",
                "histopathology", "dermoscopy", "retinopathy", "melanoma",
                "brain tumor", "cancer detection", "clinical", "patient",
                "biomedical", "healthcare",
            ],
            "computer_science": [
                "algorithm", "machine learning", "deep learning", "neural network",
                "artificial intelligence", "data mining", "natural language",
                "computer vision", "cybersecurity", "cryptography", "blockchain",
                "distributed system", "cloud computing", "internet of things",
                "software engineering", "database", "information retrieval",
            ],
            "biology": [
                "gene", "protein", "dna", "rna", "cell", "molecular", "genomics",
                "bioinformatics", "evolution", "ecosystem", "species", "organism",
                "bacteria", "virus", "mutation", "sequencing",
            ],
            "physics": [
                "quantum", "particle", "atom", "photon", "electron", "wave",
                "energy", "force", "gravity", "relativity", "thermodynamics",
                "mechanics", "optics", "electromagnetic", "nuclear", "plasma",
            ],
            "chemistry": [
                "molecule", "reaction", "catalyst", "synthesis", "polymer",
                "crystal", "bond", "oxidation", "reduction", "acid", "base",
                "solvent", "compound", "element", "spectroscopy",
            ],
        }

        # Check for domain indicators
        for domain, indicators in domain_indicators.items():
            matches = sum(1 for ind in indicators if ind in query_lower)
            if matches >= 2:  # Require at least 2 matches for confidence
                logger.debug(f"Query detected as domain-specific: {domain}")
                return domain

        return None
