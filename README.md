# AI Research Stack

A fully **self-hosted, offline academic research assistant** that runs entirely on your local machine. No cloud subscriptions, no data leaving your system.

Built with Python (FastAPI) + ChromaDB + Ollama + a glassmorphic web UI.

---

## Project Structure

```
AI Research Stack/             ← project root
├── papers/                    ← downloaded & manually added PDFs
├── vectordb/                  ← ChromaDB persistent vector index (DO NOT DELETE)
├── scripts/                   ← all Python source code
│   ├── config.py              ← central settings loader (.env → typed Settings object)
│   ├── paper_discovery.py     ← Semantic Scholar search + Unpaywall PDF resolution
│   ├── pdf_processor.py       ← PyMuPDF text extraction + overlapping chunk splitting
│   ├── vector_store.py        ← ChromaDB read/write + ONNX embedding
│   ├── rag_service.py         ← RAG orchestration: retrieve → prompt → Ollama → answer
│   ├── citation_analyzer.py   ← Citation intent classification pipeline (LLM-powered)
│   ├── manifest_manager.py    ← PDF ingestion state tracker (output/ingestion_manifest.json)
│   ├── main.py                ← CLI entry point (argparse)
│   └── server.py              ← FastAPI web server
├── web/                       ← browser-based SPA (glassmorphic dark UI)
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── prompts/                   ← system prompt templates for structured LLM output
│   ├── summarize.txt
│   ├── linkedin_draft.txt
│   ├── article_draft.txt
│   └── comparative_analysis.txt
├── output/                    ← citation analysis CSV reports + runtime state
├── .env                       ← your local configuration (never commit this)
├── .env.example               ← configuration template
├── requirements.txt           ← Python dependencies
└── README.md                  ← this file
```

---

## Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| **Python 3.11 or 3.12** | Runtime (NOT 3.14 — PyMuPDF incompatible) | [python.org](https://python.org) |
| **Ollama** | Local LLM server | [ollama.com](https://ollama.com) |
| A pulled LLM model | Language model for RAG + citation classification | `ollama pull llama3` |

---

## Quick Start

### 1. Install Python dependencies

```powershell
# From the project root (AI Research Stack/)
pip install -r requirements.txt
```

### 2. Configure your environment

```powershell
# Copy the example config and edit it
copy .env.example .env
```

Open `.env` and set at minimum:
- `UNPAYWALL_EMAIL` — any valid email address
- `OLLAMA_MODEL` — the model name you pulled (e.g. `llama3`, `mistral`, `phi3`)

### 3. Start Ollama

```powershell
ollama serve
```

### 4. Start the web server

```powershell
cd scripts
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

---

## Using the Web Interface

### Paper Discovery Tab
- Enter keywords → searches Semantic Scholar
- Click **Ingest offline** on any result → downloads PDF (if open-access) + ingests into ChromaDB

### RAG Knowledge Base Tab
- Shows all ingested papers + vector chunk counts
- **Scan & Ingest Folder** → batch-ingest any PDFs you manually dropped into `papers/`
- Chat interface → ask research questions → grounded answers with inline [Source N] citations
- Select a **Prompt Template** for structured output (summarization, LinkedIn post, etc.)

### Citation Analysis Tab
- Enter a DOI or arXiv ID → fetches citing papers → downloads their PDFs → classifies citation intent with the local LLM
- Downloads a CSV report with: Citing Paper, Year, Passage, Classification, Rationale
- **Classifications**: `supporting` | `contrasting` | `extending` | `methodological`

### Prompt Templates Tab
- View and preview all templates in `prompts/`
- Click **Use Template** → switches to RAG tab with that template pre-selected

---

## Using the CLI

All commands are run from the **project root** (`AI Research Stack/`):

```powershell
# Interactive wizard — search and select papers to ingest
python scripts/main.py --interactive

# Auto-ingest the top result for a keyword search
python scripts/main.py --query "attention mechanism transformers"

# Batch-ingest all PDFs already in papers/
python scripts/main.py --ingest-all

# Ask a grounded research question (RAG)
python scripts/main.py --query-rag "What is multi-head attention?"

# Grounded question with a structured prompt template
python scripts/main.py --query-rag "Summarize BERT" --prompt summarize

# Citation analysis for a paper (by DOI or arXiv ID)
python scripts/main.py --analyze-citations "10.48550/arXiv.1706.03762" --limit 5
```

Full options:

```
  -q, --query TEXT            Search + auto-ingest top result
  -i, --interactive           Interactive paper selection wizard
  -g, --ingest-all            Batch-ingest all PDFs in papers/
  -r, --query-rag TEXT        RAG question answering
  -p, --prompt TEMPLATE       Prompt template for RAG (use with -r)
  -a, --analyze-citations ID  Citation intent analysis
  -l, --limit N               Result/chunk/citation limit (default: 5)
  -c, --chunk-size N          Chunk character length (default: 1000)
  -o, --chunk-overlap N       Overlap between chunks (default: 200)
```

---

## How It Works

```
Search Query
     │
     ▼
Semantic Scholar API  ──► Paper Metadata + DOI + arXiv ID
     │
     ▼
Unpaywall / arXiv  ──────► PDF Download → papers/
     │
     ▼
PyMuPDF (fitz)  ──────────► Page-level text extraction + cleaning
     │
     ▼
PDFProcessorService  ─────► Overlapping text chunks (default: 1000 chars, 200 overlap)
     │
     ▼
ChromaDB (ONNX MiniLM)  ──► Vector embeddings stored in vectordb/
     │
 RAG Query
     │
     ▼
Cosine similarity search  ► Top-K relevant chunks retrieved
     │
     ▼
Ollama /api/chat  ─────────► Grounded answer with [Source N] citations
```

---

## Recommended Models

| Model | RAM Required | Best For |
|---|---|---|
| `llama3` (8B) | ~8 GB | General RAG, citation analysis |
| `mistral` (7B) | ~5 GB | Fast responses, lower RAM |
| `phi3` (3.8B) | ~4 GB | Ultra-lightweight systems |
| `llama3:70b` | ~40 GB | Highest quality (needs GPU) |

Pull a model: `ollama pull <model-name>`

---

## Adding Papers Manually

1. Copy any PDF into the `papers/` directory.
2. In the web UI → **RAG Knowledge Base** tab → click **Scan & Ingest Folder**.
3. Or via CLI: `python scripts/main.py --ingest-all`

---

## Prompt Templates

Templates in `prompts/*.txt` follow this format:

```
## SYSTEM PROMPT — Template Name

<system instructions for the LLM>

---
## USER PROMPT TEMPLATE

Context from ingested research paper(s):
...
{context}
...
Paper title: {title}
```

Available placeholders: `{context}`, `{title}`, `{authors}`, `{year}`, `{venue}`, `{context_a}`, `{context_b}`, `{title_a}`, `{title_b}`

---

## Troubleshooting

**Ollama connection refused**
```
ollama serve       # Start the Ollama server
ollama pull llama3 # Pull the model if not already pulled
```

**ChromaDB error on startup**
- Ensure `vectordb/` directory is not corrupted — if in doubt, delete it and re-ingest papers.

**PDF download fails (403 / paywall)**
- The paper is behind a paywall. Manually download the PDF and place it in `papers/`, then run **Scan & Ingest Folder**.

**No results from Semantic Scholar**
- Add `SEMANTIC_SCHOLAR_API_KEY` to `.env` to increase rate limits.

---

## License

MIT License — free to use, modify, and distribute for personal and research purposes.
