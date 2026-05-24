# AI Research Stack — Server Deployment Guide

A fully **self-hosted academic research assistant** running on Ubuntu Server. No cloud subscriptions, no data leaving your system.

**Stack:** Python (FastAPI) + ChromaDB + Ollama (llama3:70b) + Open WebUI + Docker

---

## Server Details

| Item | Value |
|---|---|
| Server IP | 192.168.68.60 |
| Research Stack UI | http://192.168.68.60:8000 |
| Open WebUI (Chat) | http://192.168.68.60:8080 |
| Project Root | /home/researcher/ai-research-stack |
| Papers Folder | /home/researcher/ai-research-stack/papers |
| Vector DB | /home/researcher/ai-research-stack/vectordb |
| Citation Reports | /home/researcher/ai-research-stack/output |
| LLM Model | llama3:70b (Q4 quantized, CPU) |
| Embedding Model | ONNX MiniLM-L6-v2 (local, no cloud) |
| Vector Database | ChromaDB (persistent on disk) |
| PDF Extraction | PyMuPDF (fitz) |
| Chunk Size | 1000 characters, 200 overlap |

---

## Technical Decisions

| Decision | Chosen | Reason |
|---|---|---|
| LLM serving | Ollama | Simple setup, supports quantized models |
| Chat frontend | Open WebUI | Full-featured, Docker deployment |
| LLM model | llama3:70b (Q4) | No GPU available; 60GB RAM sufficient for CPU inference |
| Embedding | ONNX MiniLM-L6-v2 | Local, fast, no cloud API needed |
| Vector DB | ChromaDB | Simple setup, persistent disk storage |
| PDF extraction | PyMuPDF | Best two-column academic layout handling |
| Web framework | FastAPI + HTML | Lightweight, async, easy to maintain |

**Note on response times:** With no GPU, llama3:70b on CPU takes 2–5 minutes per RAG response. For faster responses during testing, switch to llama3:8b (see model switching below).

---

## Project Structure

```
/home/researcher/ai-research-stack/
├── papers/                    ← downloaded & manually added PDFs
├── vectordb/                  ← ChromaDB persistent vector index (DO NOT DELETE)
├── scripts/                   ← all Python source code
│   ├── config.py              ← settings loader (.env → typed Settings object)
│   ├── paper_discovery.py     ← Semantic Scholar + Unpaywall PDF resolution
│   ├── pdf_processor.py       ← PyMuPDF text extraction + chunk splitting
│   ├── vector_store.py        ← ChromaDB read/write + ONNX embedding
│   ├── rag_service.py         ← RAG: retrieve → prompt → Ollama → answer
│   ├── citation_analyzer.py   ← Citation intent classification (LLM-powered)
│   ├── manifest_manager.py    ← PDF ingestion tracker (output/ingestion_manifest.json)
│   ├── main.py                ← CLI entry point
│   └── server.py              ← FastAPI web server
├── web/                       ← browser-based UI
├── prompts/                   ← system prompt templates
├── output/                    ← citation analysis CSV reports
├── venv/                      ← Python virtual environment
├── .env                       ← your local configuration (never commit this)
├── .env.example               ← configuration template
├── requirements.txt           ← Python dependencies
└── README.md                  ← this file
```

---

## Starting and Stopping Services

### Research Stack (FastAPI — port 8000)

The research stack runs as a systemd service and **starts automatically on boot**.

```bash
# Start
sudo systemctl start research-stack

# Stop
sudo systemctl stop research-stack

# Restart
sudo systemctl restart research-stack

# Check status
sudo systemctl status research-stack

# View logs
journalctl -u research-stack -f
```

### Open WebUI (Docker — port 8080)

Open WebUI runs as a Docker container and **starts automatically on boot**.

```bash
# Start
docker start open-webui

# Stop
docker stop open-webui

# Restart
docker restart open-webui

# Check status
docker ps

# View logs
docker logs open-webui --tail 50
```

### Ollama (LLM server — port 11434)

```bash
# Check if running
ollama list

# Start manually if needed
ollama serve

# Check which models are available
ollama list
```

---

## How to Add New Papers to the Knowledge Base

### Option A — Via Web UI (recommended)
1. Copy your PDF into `/home/researcher/ai-research-stack/papers/`
2. Open http://192.168.68.60:8000 in your browser
3. Go to the **RAG Knowledge Base** tab
4. Click **Scan & Ingest Folder**
5. The new paper will be ingested and queryable immediately

### Option B — Via CLI
```bash
cd /home/researcher/ai-research-stack
source venv/bin/activate
python scripts/main.py --ingest-all
```

### Option C — Via Paper Discovery
1. Search for a paper in the **Paper Discovery** tab
2. Click **Ingest** on any result — downloads PDF + ingests automatically

**Note:** The system tracks ingested papers in `output/ingestion_manifest.json`. Already-ingested papers are skipped automatically.

---

## How to Update or Switch the LLM Model

```bash
# Pull a new model
ollama pull llama3:8b       # faster, less accurate
ollama pull llama3:70b      # slower, higher quality (current default)
ollama pull mistral         # alternative

# Switch the active model
nano /home/researcher/ai-research-stack/.env
# Change: OLLAMA_MODEL=llama3:8b

# Restart the research stack to apply
sudo systemctl restart research-stack
```

---

## How to Run Citation Analysis

### Via Web UI
1. Open http://192.168.68.60:8000
2. Go to **Citation Analysis** tab
3. Enter a DOI (e.g. `10.17705/1JAIS.00730`)
4. Click **Analyze** — results saved to `output/` as CSV

### Via API
```bash
curl -X POST http://localhost:8000/api/analyze-citations \
  -H "Content-Type: application/json" \
  -d '{"paper_id": "10.17705/1JAIS.00730", "limit": 50}'
```

### View CSV Reports
```bash
ls /home/researcher/ai-research-stack/output/*.csv
cat /home/researcher/ai-research-stack/output/citation_analysis_FILENAME.csv
```

---

## Prompt Templates

Templates are stored in `/home/researcher/ai-research-stack/prompts/`:

| Template | File | Use |
|---|---|---|
| Paper Summarizer | `summarize.txt` | Structured summary of a paper |
| LinkedIn Post | `linkedin_draft.txt` | Professional LinkedIn post from research topic |
| Comparative Analysis | `comparative_analysis.txt` | Compare two papers or concepts |
| Literature Review | `article_draft.txt` | Cohesive literature review paragraph |

### To Add a New Template
1. Create a new `.txt` file in `prompts/`
2. Follow the format in existing templates (SYSTEM PROMPT + USER PROMPT TEMPLATE sections)
3. The template appears automatically in the web UI — no restart needed

### To Modify a Template
```bash
nano /home/researcher/ai-research-stack/prompts/linkedin_draft.txt
```

---

## Where Data Is Stored

| Data | Location |
|---|---|
| Downloaded PDFs | `/home/researcher/ai-research-stack/papers/` |
| Vector embeddings | `/home/researcher/ai-research-stack/vectordb/` |
| Citation CSV reports | `/home/researcher/ai-research-stack/output/` |
| Ingestion manifest | `/home/researcher/ai-research-stack/output/ingestion_manifest.json` |
| Prompt templates | `/home/researcher/ai-research-stack/prompts/` |
| Environment config | `/home/researcher/ai-research-stack/.env` |
| Ollama models | `/usr/share/ollama/.ollama/models/` |
| Open WebUI data | Docker volume `open-webui` |

---

## Known Limitations

- **No GPU:** llama3:70b runs on CPU only. Expect 2–5 min per RAG response. Switch to llama3:8b for faster responses.
- **Citation analysis — paywalled papers:** Only open-access citing papers are analyzed. Paywalled papers are listed as "not analyzed" in the CSV.
- **Scanned PDFs:** PDFs that are scanned images (no extractable text) are skipped during ingestion with a warning logged.
- **Local network only:** The system is accessible only on the local network (192.168.68.60). No SSL or domain name configured by design.
- **Single user:** No authentication. Designed for single-user access only.

---

## Troubleshooting

**Research stack not responding:**
```bash
sudo systemctl status research-stack
sudo systemctl restart research-stack
```

**Port 8000 already in use:**
```bash
sudo fuser -k 8000/tcp
sudo systemctl start research-stack
```

**Ollama not responding:**
```bash
ollama list
ollama serve
```

**ChromaDB error on startup:**
```bash
# Check vectordb folder
ls /home/researcher/ai-research-stack/vectordb/
# If corrupted, delete and re-ingest
rm -rf /home/researcher/ai-research-stack/vectordb/*
# Then re-ingest all papers via web UI or CLI
```

**Open WebUI not loading:**
```bash
docker ps
docker restart open-webui
docker logs open-webui --tail 20
```

**RAG returns no results:**
- Check that papers are ingested: `curl http://localhost:8000/api/health`
- Look for `total_chunks > 0` in the response

