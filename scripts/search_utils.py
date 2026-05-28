"""
search_utils.py — Query parsing and post-filtering for Paper Discovery.

Semantic Scholar's search API does not support strict Boolean operators or
exact author-name matching. This module compensates client-side by:
  - Extracting quoted phrases (e.g. "Manahil Shahid") for exact author matching
  - Optional exact-author mode: every token must appear as a whole word in an author name
  - Preventing partial surname matches (e.g. "Hammoud" must not match "Hammouda")
"""

from __future__ import annotations

import re
from typing import Any


# Matches double-quoted strings; supports escaped quotes inside (rare in author names).
# Support both straight quotes and smart quotes pasted from rich text sources.
_QUOTED_PHRASE_RE = re.compile(
    r'(?:"([^"\\]*(?:\\.[^"\\]*)*)"|“([^”\\]*(?:\\.[^”\\]*)*)”)'
)


def extract_quoted_phrases(query: str) -> tuple[str, list[str]]:
    """
    Pull quoted exact-match phrases out of the query string.

    Returns:
        (remaining_query, list_of_exact_phrases)
        Example: 'theorizing "Manahil Shahid" IS' → ('theorizing  IS', ['Manahil Shahid'])
    """
    phrases: list[str] = []

    def _collect(match: re.Match) -> str:
        # group(1)=straight-quote capture, group(2)=smart-quote capture
        phrase = (match.group(1) or match.group(2) or "").strip()
        if phrase:
            phrases.append(phrase)
        return " "

    remainder = _QUOTED_PHRASE_RE.sub(_collect, query).strip()
    remainder = re.sub(r"\s+", " ", remainder)
    return remainder, phrases


def _author_tokens(name: str) -> list[str]:
    """Split an author name into lowercase word tokens (letters/digits only)."""
    return re.findall(r"[a-z0-9]+", (name or "").lower())


def author_name_exact_match(author_name: str, required: str) -> bool:
    """True when the full author string equals the required name (case-insensitive)."""
    clean_author = re.sub(r"\s+", " ", author_name.strip().lower())
    clean_required = re.sub(r"\s+", " ", required.strip().lower())
    return clean_author == clean_required


def author_name_has_all_tokens_as_words(author_name: str, tokens: list[str]) -> bool:
    """
    True when every token appears as its own word in the author name.
    'hammoud' matches 'Mahmoud Hammoud' but not 'Hammouda' (different token).
    """
    if not tokens:
        return False
    words = set(_author_tokens(author_name))
    return all(t in words for t in tokens)


def paper_matches_exact_author_phrase(paper: dict[str, Any], phrase: str) -> bool:
    """True if any author on the paper exactly matches the phrase (full name)."""
    authors = paper.get("authors") or []
    for author in authors:
        name = author.get("name") if isinstance(author, dict) else str(author)
        if name and author_name_exact_match(name, phrase):
            return True
    return False


def paper_matches_token_author_filter(paper: dict[str, Any], author_query: str) -> bool:
    """
    True if any author name contains every whitespace-separated token as a whole word.
    Used for queries like 'Mahmoud Hammoud' without accidental substring matches.
    """
    tokens = _author_tokens(author_query)
    if not tokens:
        return True
    authors = paper.get("authors") or []
    for author in authors:
        name = author.get("name") if isinstance(author, dict) else str(author)
        if name and author_name_has_all_tokens_as_words(name, tokens):
            return True
    return False


def filter_papers_for_precision(
    papers: list[dict[str, Any]],
    *,
    exact_phrases: list[str] | None = None,
    exact_author_mode: bool = False,
    author_query: str | None = None,
) -> list[dict[str, Any]]:
    """
    Post-filter Semantic Scholar results for author precision.

    Args:
        papers: Raw API result list.
        exact_phrases: From quoted strings in the search box.
        exact_author_mode: When True, treat the whole query (or author_query) as an author filter.
        author_query: Optional dedicated author string (unused when phrases are present).
    """
    if not papers:
        return papers

    filtered = papers

    # Quoted phrases always require exact full-name author match.
    if exact_phrases:
        for phrase in exact_phrases:
            filtered = [p for p in filtered if paper_matches_exact_author_phrase(p, phrase)]

    # Exact-author checkbox: require whole-word token match on every author token in query.
    if exact_author_mode:
        q = (author_query or "").strip()
        if q:
            filtered = [p for p in filtered if paper_matches_token_author_filter(p, q)]

    return filtered


def build_api_query_string(query: str) -> str:
    """
    Build the string sent to Semantic Scholar (quotes removed; filtering is post-hoc).
    """
    remainder, _ = extract_quoted_phrases(query)
    return remainder if remainder else query.strip()
