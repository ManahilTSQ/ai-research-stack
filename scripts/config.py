"""
config.py — Central configuration loader for the AI Research Stack.

Reads all settings from the .env file located at the project root and exposes
them as a typed Settings singleton used by every other script in the project.

Usage (in any sibling script):
    from config import settings
    print(settings.OLLAMA_MODEL)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project Root Resolution ────────────────────────────────────────────────────
# __file__              → scripts/config.py
# .resolve()            → absolute path, resolves any symlinks
# .parent               → scripts/
# .parent.parent        → AI Research Stack/  (the true project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Load .env file from the project root ──────────────────────────────────────
# python-dotenv reads KEY=VALUE pairs from the .env file and injects them
# into os.environ so os.getenv() picks them up below.
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """
    Typed container for all runtime configuration values.

    All values are read from environment variables that were injected by .env.
    Default values are provided so the project still runs without a .env file,
    though a proper .env is strongly recommended for production use.

    Directories referenced by the settings are auto-created on instantiation
    so the user never has to manually mkdir anything.
    """

    # ── Semantic Scholar API Key ───────────────────────────────────────────────
    # Optional. Without a key the public rate limit applies (≈1 request/second).
    # Get a free key at: https://www.semanticscholar.org/product/api
    SEMANTIC_SCHOLAR_API_KEY: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    # ── Unpaywall Email ────────────────────────────────────────────────────────
    # Required by the Unpaywall API. Any valid email is accepted.
    # Unpaywall uses this for their polite-pool rate limiting policy.
    UNPAYWALL_EMAIL: str = os.getenv("UNPAYWALL_EMAIL", "researcher@example.com")

    # ── PDF Storage Directory ──────────────────────────────────────────────────
    # Where downloaded PDFs are saved on disk.
    # Default: AI Research Stack/papers/
    PDF_DOWNLOAD_DIR: Path = BASE_DIR / os.getenv("PDF_DOWNLOAD_DIR", "papers")

    # ── ChromaDB Vector Database Directory ────────────────────────────────────
    # Where ChromaDB persists its HNSW index and chunk metadata.
    # Do NOT delete this folder — it is your entire vector knowledge base.
    # Default: AI Research Stack/vectordb/
    VECTOR_DB_DIR: Path = BASE_DIR / os.getenv("VECTOR_DB_DIR", "vectordb")

    # ── Project Root ──────────────────────────────────────────────────────────
    # Exposed so other modules can build arbitrary paths relative to the root.
    # Example: settings.BASE_DIR / "output" / "my_report.csv"
    BASE_DIR: Path = BASE_DIR

    # ── Ollama LLM Server URL ─────────────────────────────────────────────────
    # HTTP base URL of the running Ollama instance.
    # Default assumes Ollama is running locally on the standard port.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── Ollama Model Name ─────────────────────────────────────────────────────
    # Which model Ollama will use for generation. The model must be pulled first:
    #   ollama pull llama3
    # Recommended lightweight alternatives: mistral, phi3
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

    # ── Ollama Request Timeout ────────────────────────────────────────────────
    # Seconds to wait for a single LLM response before timing out.
    # Large models on CPU can be slow — 300 s (5 min) is a safe default.
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))

    # ── HTTP Basic Authentication ─────────────────────────────────────────────
    # Credentials for the web UI and API (set strong values in production .env).
    BASIC_AUTH_USER: str = os.getenv("BASIC_AUTH_USER", "admin")
    BASIC_AUTH_PASS: str = os.getenv("BASIC_AUTH_PASS", "Aitawfiq26!!!")

    # ── RAG Relevance Threshold ───────────────────────────────────────────────
    # Maximum ChromaDB cosine distance for a chunk to count as "relevant".
    # Lower = stricter. Cosine distance range: 0 (identical) to 2 (opposite).
    # 0.85 → only ~57% similarity required (too loose, lets off-topic chunks through)
    # 0.72 → ~64% similarity required (tighter, blocks weak semantic matches)
    # This is the FIRST line of defense before the reranker.
    RAG_MAX_DISTANCE: float = float(os.getenv("RAG_MAX_DISTANCE", "0.72"))

    # ── RAG Query Term Guard ──────────────────────────────────────────────────
    # If True, retrieved chunks must include at least one significant query term.
    # Helps block off-topic answers when vectors are weakly similar.
    RAG_REQUIRE_QUERY_TERM_MATCH: bool = os.getenv("RAG_REQUIRE_QUERY_TERM_MATCH", "true").lower() == "true"

    # Strict RAG: entity gates, scoped retrieval, post-answer verification (recommended).
    RAG_STRICT_MODE: bool = os.getenv("RAG_STRICT_MODE", "true").lower() == "true"

    # ── Citation Report Retention ───────────────────────────────────────────────
    # Auto-delete CSV reports older than this many days on list/load (0 = keep forever).
    REPORT_RETENTION_DAYS: int = int(os.getenv("REPORT_RETENTION_DAYS", "0"))

    def __init__(self):
        """
        Auto-create all required storage directories on first use.

        This means the user never has to manually run:
            mkdir papers vectordb output
        The directories are created with parents=True so nested paths also work.
        exist_ok=True prevents errors if the directories already exist.
        """
        # Create the PDF download directory (papers/)
        self.PDF_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # Create the ChromaDB persistence directory (vectordb/)
        self.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

        # Create the output directory for CSV reports and runtime state files
        (self.BASE_DIR / "output").mkdir(parents=True, exist_ok=True)


# ── Module-level singleton ────────────────────────────────────────────────────
# Instantiate once at import time. All other modules simply do:
#   from config import settings
# and use the already-initialized settings object.
settings = Settings()
