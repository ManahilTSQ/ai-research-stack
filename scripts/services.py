"""
services.py — Module-level singletons for all stateless RAG services.

This file instantiates all stateless service classes once at import time,
eliminating repeated initialization overhead and log spam on every request.
"""

from query_expansion import QueryExpansion
from query_router import QueryRouter
from query_understanding import QueryUnderstanding
from topic_classifier import TopicClassifier
from metadata_filter import MetadataFilter
from citation_verifier import CitationVerifier
from chunk_grouper import ChunkGrouper
from context_builder import ContextBuilder
from failure_handler import FailureHandler

# Instantiate once at import time — these are all stateless classifiers
query_expansion   = QueryExpansion()
query_router      = QueryRouter()
query_understand  = QueryUnderstanding()
topic_classifier  = TopicClassifier()
metadata_filter   = MetadataFilter()
citation_verifier = CitationVerifier()
chunk_grouper     = ChunkGrouper()
context_builder   = ContextBuilder()
failure_handler   = FailureHandler()
