# AI Research Stack

A fully **self-hosted academic research assistant** running on Ubuntu Server.
No cloud subscriptions. No data leaving your system. No per-query fees.

**Stack:** Python · FastAPI · ChromaDB · Ollama · Cloudflare Tunnel

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Server Details](#server-details)
4. [Project Structure](#project-structure)
5. [Features](#features)
6. [How to Add Papers](#how-to-add-papers)
7. [Using the RAG Knowledge Base](#using-the-rag-knowledge-base)
8. [APA7 Citations](#apa7-citations)
9. [Deleting Papers](#deleting-papers)
10. [Citation Analysis](#citation-analysis)
11. [Prompt Templates](#prompt-templates)
12. [Security & Authentication](#security--authentication)
13. [Deployment & Updates](#deployment--updates)
14. [Service Management](#service-management)
15. [Configuration Reference](#configuration-reference)
16. [Troubleshooting](#troubleshooting)
17. [Known Limitations](#known-limitations)

---

## Overview

The AI Research Stack is a private, offline-capable research tool designed for a single professional researcher. It ingests academic PDF papers into a local vector database and allows you to query them using a locally-running large language model (LLM) — with full APA7-formatted citations in every answer.

**Key capabilities:**

- 🔍 **Paper Discovery** — search Semantic Scholar, resolve open-access PDFs, ingest in one click
- 🧠 **RAG Knowledge Base** — ask questions grounded strictly in your ingested literature
- 📝 **APA7 Citations** — every answer ends with a full reference list in APA7 format
- 🗑️ **Paper Deletion** — remove any paper and its vector chunks instantly
- 📊 **Citation Analysis** — classify how papers cite a target paper (supporting, contrasting, etc.)
- 📁 **Recursive Folder Scanning** — drop PDFs into `papers/` or any subfolder and ingest them
- 🔐 **HTTP Basic Auth** — protects the entire UI with a username/password

---

## Architecture

```
Browser (cite.aitawfiq.com)
        │
        ▼ HTTPS
Cloudflare Tunnel  ──────────────────────────────────────┐
        │                                                 │
        ▼ HTTP (localhost)                                │
FastAPI Server (port 8000)                               │
        │                                                 │
        ├── /api/search      → Semantic Scholar API       │
        ├── /api/download    → Unpaywall / arXiv PDF      │
        ├── /api/pdfs        → Ingestion Manifest         │
        ├── /api/ingest-pending → Bulk folder scan        │
        ├── /api/query-rag   → ChromaDB → Ollama → Answer │
        ├── /api/papers/{id} → Delete paper + chunks      │
        └── /api/analyze-citations → Citation classifier  │
                │                                         │
                ├── ChromaDB (vectordb/)                  │
                │     └── ONNX MiniLM-L6-v2 embeddings   │
                ├── Ollama (port 11434)                   │
                │     └── llama3:70b (CPU, Q4)            │
                └── ingestion_manifest.json               │
```

---

## Server Details

| Item | Value |
|---|---|
| Public URL | https://cite.aitawfiq.com |
| Server IP (LAN) | 192.168.68.60 |
| FastAPI (local) | http://localhost:8000 |
| Project Root | `/home/researcher/ai-research-stack` |
| Papers Folder | `/home/researcher/ai-research-stack/papers/` |
| Vector DB | `/home/researcher/ai-research-stack/vectordb/` |
| Citation Reports | `/home/researcher/ai-research-stack/output/` |
| Ingestion Manifest | `/home/researcher/ai-research-stack/output/ingestion_manifest.json` |
| Prompt Templates | `/home/researcher/ai-research-stack/prompts/` |
| LLM Model | llama3:70b (Q4 quantized, CPU) |
| Embedding Model | ONNX MiniLM-L6-v2 (local, no cloud) |
| Vector Database | ChromaDB (persistent on disk) |
| PDF Extraction | PyMuPDF (fitz) |
| Chunk Size | 1000 characters, 200 overlap |

---

## Project Structure

```
/home/researcher/ai-research-stack/
├── papers/                    ← PDFs to ingest (supports subfolders)
│   ├── mypaper.pdf
│   └── group1/
│       └── another.pdf
├── vectordb/                  ← ChromaDB vector index (DO NOT DELETE)
├── output/                    ← Citation CSV reports + manifest
│   └── ingestion_manifest.json
├── prompts/                   ← System prompt templates (.txt)
├── scripts/                   ← Python source code
│   ├── config.py              ← Settings loader (.env → typed Settings)
│   ├── server.py              ← FastAPI web server + all API endpoints
│   ├── manifest_manager.py    ← PDF ingestion tracker (JSON manifest)
│   ├── paper_discovery.py     ← Semantic Scholar + Unpaywall integration
│   ├── pdf_processor.py       ← PyMuPDF extraction + text chunking
│   ├── vector_store.py        ← ChromaDB read/write + ONNX embedding
│   ├── rag_service.py         ← RAG pipeline: retrieve → prompt → answer
│   ├── citation_analyzer.py   ← Citation intent classification (LLM)
│   ├── main.py                ← CLI entry point
│   └── batch_reingest.py      ← Re-ingest all papers from scratch
├── web/                       ← Single-page browser UI
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── venv/                      ← Python virtual environment
├── .env                       ← Your local config (never commit)
├── .env.example               ← Config template
├── requirements.txt           ← Python dependencies
└── README.md                  ← This file
```

---

## Features

### 1. Paper Discovery

Search the global Semantic Scholar index by keywords, title, or author. Each result shows:
- Full title, authors, year, venue
- Citation count
- Open Access badge (PDF available vs abstract only)
- One-click **Download & Ingest** button

Papers with open-access PDFs are downloaded, chunked, and embedded automatically. Abstract-only papers store the abstract as a searchable chunk.

### 2. RAG Knowledge Base

The RAG (Retrieval-Augmented Generation) tab lets you ask natural-language questions answered strictly from your ingested literature.

**Left sidebar shows:**
- Total papers ingested and total vector chunks
- Each paper listed as `Author, Year` (e.g. `Shahid & Hammoud, 2026`) — hover for full title
- Trash icon to remove any paper instantly (no confirmation required)
- "Scan & Ingest Folder" button to pick up any PDFs dropped into `papers/`

**Chat area:**
- Type any research question
- Adjust context chunks (3–10) via the slider
- Select a system prompt template (literature review, comparative analysis, etc.)
- Every answer ends with full APA7 references

### 3. APA7 Citations

Every RAG answer automatically includes:
- In-text citations in the body: `(Shahid & Hammoud, 2026)`
- A **References** section at the end of every response with full APA7 entries:

```
References

Shahid, M., & Hammoud, A. (2026). Title of the Paper. Journal Name, 12(3), 45–67.
  https://doi.org/10.xxxx/xxxxx
```

The metadata (author names, year, DOI, journal) is pulled from Semantic Scholar during ingestion and stored in both ChromaDB and `ingestion_manifest.json`.

### 4. Recursive Folder Scanning

The system scans `papers/` **and all its subfolders** recursively. You can organise papers into topic folders:

```
papers/
├── cryptography/
│   └── Cryptography-09-00017.pdf
├── ai-ethics/
│   └── 21091121.pdf
└── general/
    └── attention_is_all_you_need.pdf
```

Click **Scan & Ingest Folder** and all new PDFs across all subfolders are detected and queued.

Manifest keys are stored as relative paths (e.g. `cryptography/Cryptography-09-00017.pdf`) to avoid name collisions between subfolders.

### 5. Metadata Auto-Resolution

When a PDF has a generic filename (e.g. `21091121.pdf`), the system:
1. Extracts the first page text and looks for a DOI pattern
2. Queries Semantic Scholar with the DOI or the extracted title
3. Backfills author names, year, DOI into the manifest and ChromaDB

This runs in a background thread and updates the sidebar automatically.

### 6. Citation Analysis

Enter any paper's DOI or Semantic Scholar ID and the system:
1. Fetches the top N citing papers from Semantic Scholar
2. Downloads open-access PDFs of the citing papers
3. Uses the local LLM to classify each citation context:
   - **Supporting** — cites the paper approvingly
   - **Contrasting** — challenges or contradicts the paper
   - **Background** — mentions it as context
   - **Methodological** — uses its methods
4. Exports results as a CSV report

---

## How to Add Papers

### Option A — SFTP upload (recommended for bulk)

Upload PDFs directly into the papers folder using any SFTP client (FileZilla, WinSCP):

```
Host:     cite.aitawfiq.com (or 192.168.68.60)
Port:     22
User:     root1 (or researcher)
Path:     /home/researcher/ai-research-stack/papers/
```

You may create subfolders to organise by topic. Then click **Scan & Ingest Folder** in the UI.

### Option B — Paper Discovery UI

1. Go to the **Paper Discovery** tab
2. Search by keywords (e.g. `"blockchain security IoT"`)
3. Click **Download & Ingest PDF** on any result
4. The paper is downloaded and ingested automatically

### Option C — CLI (server terminal)

```bash
cd /home/researcher/ai-research-stack
source venv/bin/activate
python scripts/main.py --ingest-all
```

### Option D — scp from your machine

```bash
scp myarticle.pdf root1@<server-ip>:/home/researcher/ai-research-stack/papers/
```

Then trigger **Scan & Ingest Folder** in the UI.

> **Note:** The system skips already-ingested papers automatically based on `output/ingestion_manifest.json`.

---

## Using the RAG Knowledge Base

1. Open https://cite.aitawfiq.com and log in
2. Navigate to the **RAG Knowledge Base** tab
3. Ensure at least one paper shows **Ingested** status in the sidebar
4. Type a research question in the chat input, for example:
   - *"What are the main arguments presented in the core papers?"*
   - *"List the first 3 article titles, authors, and year of publication"*
   - *"Compare the research methodologies across my papers"*
   - *"What are the key limitations outlined by the authors?"*
5. The answer will be grounded in your papers with APA7 in-text citations and a full reference list

**Context chunks slider:** Higher values (up to 10) retrieve more text passages and produce more thorough answers at the cost of longer processing time.

**System prompt templates:** Switch between Standard Chat, Literature Review, Comparative Analysis, etc. using the dropdown.

---

## APA7 Citations

The system is configured to always produce APA7 output. To get citations:

- Simply ask a question — every answer includes them automatically
- For a full reference list of all papers, ask: *"List all ingested papers in APA7 format"*
- For specific papers: *"Give me the APA7 reference for the Shahid paper"*

**Example APA7 output format:**

```
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in
  Neural Information Processing Systems, 30. https://doi.org/10.48550/arXiv.1706.03762
```

If a paper was ingested without full metadata (e.g. from a poorly named file), the system will attempt to resolve it from Semantic Scholar in the background. The sidebar will update once resolved.

---

## Deleting Papers

To remove a paper from the knowledge base:

1. Go to **RAG Knowledge Base** tab
2. Find the paper in the left sidebar
3. Click the 🗑️ trash icon — the paper and all its vector chunks are removed instantly
4. The sidebar refreshes automatically

No confirmation dialog is shown. This is by design — the tool is for a single responsible professional user.

To delete via API:

```bash
curl -u admin:PASSWORD -X DELETE http://localhost:8000/api/papers/filename.pdf
```

For papers in subfolders:

```bash
curl -u admin:PASSWORD -X DELETE http://localhost:8000/api/papers/subgroup/filename.pdf
```

---

## Citation Analysis

### Via Web UI

1. Go to the **Citation Analysis** tab
2. Enter a DOI (e.g. `10.17705/1JAIS.00730`) or arXiv ID
3. Select the number of citing papers to fetch (3 / 5 / 10 / 20)
4. Click **Analyze Citations** — the pipeline runs in the background
5. Results appear as a classified table; download as CSV

### Via API

```bash
curl -u admin:PASSWORD \
  -X POST http://localhost:8000/api/analyze-citations \
  -H "Content-Type: application/json" \
  -d '{"paper_id": "10.17705/1JAIS.00730", "limit": 10}'
```

### CSV Reports

Reports are saved to `output/` and can be downloaded from the **Reports Registry** in the Citation Analysis tab. They contain:

| Column | Description |
|---|---|
| Citing Paper | Title of the citing paper |
| Year | Publication year |
| Extracted Citation Passage | The sentence(s) containing the citation |
| Classification | supporting / contrasting / background / methodological |
| Local LLM Rationale | One-sentence explanation of the classification |

---

## Prompt Templates

Templates are stored in `prompts/` as `.txt` files. They appear automatically in the web UI.

| Template | File | Best for |
|---|---|---|
| Standard Chat RAG | *(built-in)* | General Q&A |
| Paper Summarizer | `summarize.txt` | Structured summary of a paper |
| LinkedIn Post | `linkedin_draft.txt` | Professional post from research |
| Comparative Analysis | `comparative_analysis.txt` | Comparing two papers or concepts |
| Literature Review | `article_draft.txt` | Cohesive literature review paragraph |

### Add a New Template

1. Create `prompts/mytemplate.txt` on the server
2. Follow the format of existing templates (SYSTEM PROMPT section at the top)
3. It appears immediately in the UI — no restart needed

```bash
nano /home/researcher/ai-research-stack/prompts/mytemplate.txt
```

---

## Security & Authentication

The application is protected with **HTTP Basic Authentication**.

- All API endpoints and the UI require a valid username and password
- Credentials are set in `.env` — see [Configuration Reference](#configuration-reference)
- The tunnel (cite.aitawfiq.com) runs over HTTPS via Cloudflare — traffic is encrypted in transit

### Default credentials (change these!)

```env
BASIC_AUTH_USER=admin
BASIC_AUTH_PASS=aitawfiq2026
```

To change credentials:

```bash
nano /home/researcher/ai-research-stack/.env
# Edit BASIC_AUTH_USER and BASIC_AUTH_PASS
sudo systemctl restart research-stack
```

---

## Deployment & Updates

### Push changes from local machine

```bash
cd "C:\Users\PMLS\OneDrive\Desktop\AI Research Stack"

git add .
git commit -m "Describe your changes"
git push origin main
```

### Pull and restart on the server

```bash
cd /home/researcher/ai-research-stack
git pull origin main
sudo systemctl restart research-stack
```

### First-time server setup

```bash
# Clone repo
git clone https://github.com/ManahilTSQ/ai-research-stack.git
cd ai-research-stack

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env

# Install systemd service (see research-stack.service)
sudo cp research-stack.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable research-stack
sudo systemctl start research-stack
```

---

## Service Management

### Research Stack (FastAPI — port 8000)

```bash
sudo systemctl start research-stack      # Start
sudo systemctl stop research-stack       # Stop
sudo systemctl restart research-stack    # Restart
sudo systemctl status research-stack     # Check status
journalctl -u research-stack -f          # Live logs
journalctl -u research-stack -n 50 --no-pager   # Last 50 log lines
```

### Ollama (LLM — port 11434)

```bash
ollama list             # List available models
ollama serve            # Start manually if needed
ollama pull llama3:8b   # Pull a faster/smaller model
```

### Cloudflare Tunnel

```bash
# Check if tunnel is running
sudo systemctl status cloudflared

# Restart tunnel
sudo systemctl restart cloudflared
```

---

## Configuration Reference

All configuration is in `/home/researcher/ai-research-stack/.env`:

```env
# LLM
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3:70b

# Authentication (HTTP Basic Auth — protects all endpoints)
BASIC_AUTH_USER=admin
BASIC_AUTH_PASS=aitawfiq2026

# Storage paths (relative to project root)
PDF_DOWNLOAD_DIR=papers
VECTORDB_DIR=vectordb
PROMPTS_DIR=prompts
REPORTS_DIR=output

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

To switch to a faster model (for testing):

```bash
nano .env
# Change: OLLAMA_MODEL=llama3:8b
sudo systemctl restart research-stack
```

---

## Troubleshooting

### Server not responding (curl: Could not connect)

```bash
sudo systemctl status research-stack
journalctl -u research-stack -n 30 --no-pager
sudo systemctl restart research-stack
```

### Port 8000 already in use

```bash
sudo fuser -k 8000/tcp
sudo systemctl start research-stack
```

### RAG returns "I could not find any information"

- Verify papers are ingested: `curl -u admin:PASS http://localhost:8000/api/health`
- Check `total_chunks > 0` in the response
- If 0 chunks, re-run **Scan & Ingest Folder** in the UI
- Check logs: `journalctl -u research-stack -f`

### Papers show "Unknown Authors" or no year

The background metadata resolver is still working. Wait 1–2 minutes and refresh the page. If it persists:
- The PDF may not have a DOI embedded in its text
- The title may not have been matched by Semantic Scholar
- Check logs for `[WARNING] Could not resolve metadata for...`

### Sidebar shows filename instead of Author, Year

This means the paper's metadata has not yet been resolved from Semantic Scholar. It will update automatically in the background. You can also re-ingest via **Scan & Ingest Folder**.

### ChromaDB error on startup

```bash
ls /home/researcher/ai-research-stack/vectordb/
# If corrupted:
rm -rf /home/researcher/ai-research-stack/vectordb/*
sudo systemctl restart research-stack
# Then re-ingest all papers:
source venv/bin/activate
python scripts/batch_reingest.py
```

### Semantic Scholar rate limiting (429 errors in logs)

Normal behaviour. The system automatically retries with exponential backoff (3s → 6s → 12s). No action needed.

### Git merge conflict on pull

```bash
# If rebase conflict:
git rebase --abort
git pull origin main --no-rebase
# Resolve conflicts manually, then:
git add .
git commit -m "Resolve merge conflicts"
```

---

## Known Limitations

| Limitation | Detail |
|---|---|
| No GPU | llama3:70b runs on CPU only. Expect 2–5 min per RAG response. Switch to llama3:8b for speed. |
| Paywalled citing papers | Citation analysis only works on open-access citing papers. Paywalled ones are listed as "not analyzed". |
| Scanned PDFs | PDFs that are scanned images (no text layer) are skipped with a warning. Use OCR pre-processing first. |
| Single user | No multi-user support. Designed for one researcher. |
| Metadata resolution | If a PDF filename is a random code and the first-page text gives no clues, metadata cannot be auto-resolved. Rename the file to a meaningful title and re-ingest. |
| S2 rate limits | Semantic Scholar API has rate limits. Bulk ingestion may be slow if many papers need metadata lookup. |

---

## Data Locations Summary

| Data | Server Path |
|---|---|
| PDF papers | `/home/researcher/ai-research-stack/papers/` (+ subfolders) |
| Vector embeddings | `/home/researcher/ai-research-stack/vectordb/` |
| Ingestion manifest | `/home/researcher/ai-research-stack/output/ingestion_manifest.json` |
| Citation CSV reports | `/home/researcher/ai-research-stack/output/` |
| Prompt templates | `/home/researcher/ai-research-stack/prompts/` |
| Environment config | `/home/researcher/ai-research-stack/.env` |
| Ollama LLM models | `/usr/share/ollama/.ollama/models/` |

---

*© 2026 AI Research Stack — Self-Hosted Academic Intelligence. Running entirely locally.*
