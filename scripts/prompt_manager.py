"""
prompt_manager.py — Load, validate, save, and format RAG prompt templates.

On-disk format (required for templates created via UI or hand-edited files):

    ## SYSTEM PROMPT — Human-readable title

    (system instructions…)

    ---
    ## USER PROMPT TEMPLATE

    (user message with {context}, {title}, and other placeholders…)
"""

from __future__ import annotations

import re
from pathlib import Path

# Section markers used across the app — keep in sync with README and the Prompts UI.
USER_SECTION_DIVIDER = "## USER PROMPT TEMPLATE"
SYSTEM_SECTION_MARKER = "## SYSTEM PROMPT"

# Safe filename stem: lowercase letter first, then letters, digits, underscores.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Built-in templates that ship with the repo — cannot be deleted from the UI.
PROTECTED_TEMPLATE_NAMES = frozenset({
    "summarize",
    "linkedin_draft",
    "comparative_analysis",
    "article_draft",
    "hassanian_article",
})


class PromptValidationError(ValueError):
    """Raised when a template name or body fails validation."""


def validate_prompt_name(name: str) -> str:
    """Return a sanitised template stem or raise PromptValidationError."""
    stem = (name or "").strip().lower().replace("-", "_")
    stem = re.sub(r"[^a-z0-9_]", "", stem)
    if not _NAME_RE.match(stem):
        raise PromptValidationError(
            "Template name must start with a letter and use only lowercase letters, "
            "digits, and underscores (max 64 characters)."
        )
    return stem


def parse_prompt_file(content: str) -> tuple[str, str, str]:
    """
    Parse a .txt template into (display_title, system_prompt, user_template).

    display_title is taken from the first line after the SYSTEM PROMPT marker.
    """
    text = (content or "").strip()
    if not text:
        raise PromptValidationError("Template file is empty.")

    if USER_SECTION_DIVIDER in text:
        system_part, user_part = text.split(USER_SECTION_DIVIDER, 1)
        user_template = user_part.strip()
    else:
        system_part = text
        user_template = "{context}"

    system_raw = system_part.replace(SYSTEM_SECTION_MARKER, "").strip()
    lines = [ln for ln in system_raw.split("\n") if ln.strip()]
    display_title = lines[0].strip() if lines else "Custom Template"
    # Strip leading "—" or "-" decoration from title line
    display_title = re.sub(r"^[—\-:\s]+", "", display_title).strip()
    system_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else system_raw

    if not system_body:
        system_body = system_raw

    if not user_template:
        raise PromptValidationError("USER PROMPT TEMPLATE section cannot be empty.")

    return display_title, system_body, user_template


def build_prompt_file_content(display_title: str, system_body: str, user_template: str) -> str:
    """Serialize a template to the canonical on-disk format."""
    title_line = display_title.strip() or "Custom Template"
    system = (system_body or "").strip()
    user = (user_template or "{context}").strip()
    return (
        f"{SYSTEM_SECTION_MARKER} — {title_line}\n\n"
        f"{system}\n\n"
        f"---\n"
        f"{USER_SECTION_DIVIDER}\n\n"
        f"{user}\n"
    )


def list_prompt_files(prompts_dir: Path) -> list[Path]:
    """Return sorted .txt paths under prompts_dir."""
    if not prompts_dir.exists():
        prompts_dir.mkdir(parents=True, exist_ok=True)
    return sorted(prompts_dir.glob("*.txt"))


def load_prompt_metadata(prompt_path: Path) -> dict:
    """Load one template's metadata for API listing."""
    content = prompt_path.read_text(encoding="utf-8").strip()
    lines = [ln for ln in content.split("\n") if ln.strip()]
    try:
        title, _, _ = parse_prompt_file(content)
    except PromptValidationError:
        raw_title = lines[0] if lines else prompt_path.stem
        title = re.sub(r"^#+\s*", "", raw_title).strip()

    desc_lines = [
        ln for ln in lines[1:]
        if ln.strip() and not ln.startswith("#") and not ln.startswith("---")
    ]
    description = desc_lines[0].strip() if desc_lines else f"Prompt template: {prompt_path.stem}"

    return {
        "name": prompt_path.stem,
        "title": title[:200],
        "description": description[:300],
        "content": content,
        "protected": prompt_path.stem in PROTECTED_TEMPLATE_NAMES,
    }


def save_prompt(prompts_dir: Path, name: str, display_title: str, system_body: str, user_template: str) -> Path:
    """Write or overwrite a template file; returns the path written."""
    stem = validate_prompt_name(name)
    path = prompts_dir / f"{stem}.txt"
    path.write_text(
        build_prompt_file_content(display_title, system_body, user_template),
        encoding="utf-8",
    )
    return path


def delete_prompt(prompts_dir: Path, name: str) -> None:
    """Delete a template file; raises if protected or missing."""
    stem = validate_prompt_name(name)
    if stem in PROTECTED_TEMPLATE_NAMES:
        raise PromptValidationError(f"Template '{stem}' is built-in and cannot be deleted.")
    path = prompts_dir / f"{stem}.txt"
    if not path.exists():
        raise PromptValidationError(f"Template '{stem}' not found.")
    path.unlink()


def substitute_placeholders(template: str, variables: dict[str, str]) -> str:
    """
    Replace {key} placeholders in the user template.
    Unknown placeholders are left unchanged so custom fields can be added later.
    """
    result = template
    for key, value in variables.items():
        result = result.replace("{" + key + "}", value or "")
    return result
