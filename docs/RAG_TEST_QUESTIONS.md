# RAG Test Question Bank (any author / any paper)

Production: **cite.aitawfiq.com** — ingest via **Upload** or **Paper Discovery**.  
The server reads **live ChromaDB metadata** on every question (all papers currently ingested, including new uploads).

After **code deploy**: `git pull` + restart API. No code change needed when only adding PDFs.

---

## How author/paper matching works (not name-specific)

1. **Ingest** stores `authors`, `title`, `year`, `venue`, `doi` on every chunk.  
2. **Each question** rebuilds an **author catalog** from all papers in ChromaDB.  
3. Your question is matched to **author strings exactly as stored** (e.g. `Noor Zaman Jhanjhi`, `Walaa Gouda`, `Muhammad Sajid Khan`).  
4. The system loads **text chunks only from those papers**, then Ollama answers from that evidence + the inventory list.  
5. **Common surnames only** (e.g. `Khan` alone with many different Khans) → refuse or ask for a **full name**.

---

## 1. Catalog (deterministic)

| Question | Expected |
|----------|----------|
| How many papers are in my knowledge base? | Matches sidebar count |
| List all ingested papers | Every paper, no truncation |
| List all authors in my library | All distinct author lines from metadata |
| List papers by *[any surname in your library]* | Only that author’s papers |

---

## 2. Author synthesis (pick names from **your** author list)

Replace `NAME` with a real author from “List all authors”:

| Question | Expected |
|----------|----------|
| What are the main contributions of NAME? | Only NAME’s papers cited |
| What does NAME research? | Same scope |
| Papers by NAME — summarize their themes | List + synthesis, scoped |
| Describe the research of NAME | Scoped chunks only |

**Negative tests**

| Question | Expected |
|----------|----------|
| Contributions of NAME_NOT_IN_LIBRARY | Not in library |
| Papers by Khan *(if many Khans in KB)* | Refuse or require full name |
| What does N. Hassan research *(if no Hassan in KB)* | Not in library — **not** another author |

---

## 3. Topics (cross-corpus)

| Question | Expected |
|----------|----------|
| What do my papers say about deep learning in medical imaging? | Imaging papers only; no phishing/barcode/traffic |
| What do my papers say about smart cities and cybersecurity? | IoT/smart-city papers |
| Federated learning in my library | Papers that actually discuss federated learning |

---

## 4. Single paper

| Question | Expected |
|----------|----------|
| Summarize "*exact title from your list*" | That paper only |
| Summarize "Completely Fake Title 2099" | Not in library |
| **Focus on Paper** dropdown + main contribution? | One paper |

---

## 5. After new ingest

1. Sidebar paper count increases.  
2. `How many papers…` — new total.  
3. `List papers by <new author>`.  
4. One synthesis question about that author.

---

## 6. Pass rubric

- **Retrieved chunks** belong to the named author/paper only.  
- **Inventory** line for each cited paper shows that author.  
- **New ingest** appears without redeploying code.  
- **Wrong surname** (Riskhan vs Hassan) never matches.

---

## Deploy

Validate on the **server**, not only a sparse local ChromaDB. Local laptop may have fewer papers than production.
