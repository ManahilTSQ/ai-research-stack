"""
paper_labels.py — Consistent Author, Year sidebar labels for the Knowledge Base UI.

The UI must show compact bibliographic labels (e.g. "Hassan et al., 2019"), not long
paper titles or raw filenames. This module centralises that formatting for API + ingest.
"""

from __future__ import annotations

import re
from pathlib import Path

UNKNOWN_AUTHORS = "Unknown Authors"


def _extract_year(text: str) -> str:
    if not text:
        return ""
    # Underscores break \\b boundaries (e.g. file_2018_topic), so match years directly.
    match = re.search(r"(?<!\d)(19[5-9]\d|20[0-2]\d)(?!\d)", text)
    return match.group(1) if match else ""


def _last_names_from_authors(authors_str: str) -> list[str]:
    """Parse manifest/Chroma author strings into surname tokens for display."""
    cleaned = re.sub(r"\s*et al\.?$", "", (authors_str or "").strip(), flags=re.I)
    if not cleaned or cleaned == UNKNOWN_AUTHORS:
        return []

    parts = re.split(r"\s*;\s*", cleaned)
    if len(parts) == 1:
        parts = re.split(r",\s*(?=[A-Z])", cleaned)

    surnames: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # "Khamsani, A." or "Smith, John" → surname is before the comma.
        if "," in part:
            surname = part.split(",", 1)[0].strip().replace(".", "")
            if surname and len(surname) > 1:
                surnames.append(surname)
            continue
        # "John Smith" or "Smith J." → surname is the last non-initial token
        tokens = [t for t in part.split() if t]
        if not tokens:
            continue
        # Filter out initials (single letters with optional period)
        non_initials = [t for t in tokens if len(t) > 1 and not re.match(r"^[A-Z]\.?$", t, re.I)]
        # If we have non-initial tokens, use the last one as surname
        if non_initials:
            surname = non_initials[-1].replace(".", "")
        else:
            # Fallback: use the last token even if it's an initial
            surname = tokens[-1].replace(".", "")
        if surname and surname.lower() not in {"unknown", "none", "null"}:
            surnames.append(surname)
    return surnames[:3]


def _author_year_from_filename(filename: str | None) -> tuple[str, str]:
    """
    Best-effort author + year from PDF filename patterns:
      Author_2021_topic.pdf, 2021_Author_topic.pdf, Author-et-al-2019.pdf
    """
    if not filename:
        return "", ""
    stem = Path(filename).stem
    year = _extract_year(stem)
    tokens = [t for t in re.split(r"[_\-\s]+", stem) if t]
    junk = {
        "pdf", "paper", "final", "draft", "preprint", "manuscript", "copy",
        "v1", "v2", "v3", "the", "and", "for", "of", "on", "in",
    }
    name_tokens = []
    for t in tokens:
        tl = t.lower()
        if tl in junk or re.fullmatch(r"\d+", t):
            continue
        if re.fullmatch(r"(19|20)\d{2}", t):
            continue
        if len(t) >= 2 and re.search(r"[A-Za-z]", t):
            name_tokens.append(t)

    author = ""
    if name_tokens:
        # Prefer token immediately before year in filename, else first token.
        if year:
            for i, t in enumerate(tokens):
                if t == year and i > 0:
                    prev = tokens[i - 1]
                    if prev.lower() not in junk and re.search(r"[A-Za-z]", prev):
                        author = prev
                        break
        if not author:
            author = name_tokens[0]

    if author:
        author = author[:1].upper() + author[1:]
    return author, year


def _embedded_author_in_title(title: str) -> tuple[str, str]:
    """Parse leading 'Smith et al., 2021' patterns sometimes stored as title."""
    if not title:
        return "", ""
    match = re.match(
        r"^([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\s+et\s+al\.?,?\s*(\d{4})?",
        title.strip(),
        flags=re.I,
    )
    if not match:
        return "", ""
    author = f"{match.group(1).strip()} et al."
    year = match.group(2) or ""
    return author, year


def format_sidebar_label(
    authors: str | None,
    year: str | int | None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """
    Build the sidebar label shown in the Knowledge Base list.
    Never returns a long paper title — only Author, Year style (or minimal unknown).
    """
    authors_str = (authors or "").strip()
    year_str = str(year or "").strip()
    if year_str in ("N/A", "None", "null"):
        year_str = ""

    has_authors = bool(authors_str and authors_str != UNKNOWN_AUTHORS)
    if not has_authors and title:
        emb_author, emb_year = _embedded_author_in_title(title)
        if emb_author:
            authors_str = emb_author
            has_authors = True
            if not year_str and emb_year:
                year_str = emb_year

    if not year_str and filename:
        year_str = _extract_year(Path(filename).stem)
    if not year_str and title:
        year_str = _extract_year(title)

    if has_authors:
        last_names = _last_names_from_authors(authors_str)
        has_et_al = bool(re.search(r"\bet\s+al\.?", authors_str, re.I))
        if last_names:
            if len(last_names) == 1:
                name_part = f"{last_names[0]} et al." if has_et_al else last_names[0]
            elif len(last_names) == 2:
                name_part = f"{last_names[0]} & {last_names[1]}"
            else:
                name_part = f"{last_names[0]} et al."
            if year_str:
                return f"{name_part}, {year_str}"
            return name_part

    # Filename heuristic when Semantic Scholar metadata is missing.
    if not has_authors and filename:
        fn_author, fn_year = _author_year_from_filename(filename)
        if fn_author and fn_year:
            return f"{fn_author}, {fn_year}"
        if fn_year:
            return f"Unknown, {fn_year}"
        # If we have filename but couldn't extract author/year, use first token as author
        stem = Path(filename).stem
        tokens = [t for t in re.split(r"[_\-\s]+", stem) if t and len(t) > 1]
        if tokens:
            first_token = tokens[0][:1].upper() + tokens[0][1:]
            if year_str:
                return f"{first_token}, {year_str}"
            return f"{first_token}, N/A"

    if year_str:
        return f"Unknown, {year_str}"

    # Pending ingest — short status, not a long title string.
    if filename:
        stem = Path(filename).stem
        tokens = [t for t in re.split(r"[_\-\s]+", stem) if t and len(t) > 1]
        if tokens:
            first_token = tokens[0][:1].upper() + tokens[0][1:]
            return f"{first_token}, N/A"
        return "Unknown, N/A"

    return "Unknown, N/A"
