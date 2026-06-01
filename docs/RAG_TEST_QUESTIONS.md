# RAG Test Question Bank

Use this checklist after ingesting papers or changing `rag_context.py` / `rag_strict.py`.  
Restart the API server, then run questions in the chat UI. Mark **Pass** / **Fail** / **Notes**.

**Current library snapshot (local):** ~31 papers, ~2k chunks — authors include Hassan, Zhang, Paszke, Sharma, Khan, Joshi, He, Isensee, and others.  
If you ingest the full smart-city / Jhanjhi corpus (~86 papers), use **Section 8** as well.

---

## 1. Catalog & inventory (deterministic — no LLM)

These should return exact counts/lists from metadata, fast, with no hallucinated papers.

| # | Question | Expected behavior |
|---|----------|-------------------|
| 1.1 | How many papers are in my knowledge base? | Exact paper count |
| 1.2 | List all ingested papers | Full numbered list, every paper |
| 1.3 | List all authors in my library | Author strings + paper counts |
| 1.4 | List papers by Hassan | Only papers with Hassan in author field |
| 1.5 | List papers by Zhang | Only Zhang co-authored papers |
| 1.6 | Papers by Muhammad Sajid Khan | Single-paper or exact match list |
| 1.7 | List papers by Unknown Authors | Papers with missing author metadata (if any) |

---

## 2. Author-scoped synthesis (LLM + locked retrieval)

Should scope to that author’s papers only; must **not** refuse if author exists; must **not** cite unrelated authors’ work as theirs.

| # | Question | Expected behavior |
|---|----------|-------------------|
| 2.1 | What are the main contributions of Hassan? | Synthesis from Hassan papers only |
| 2.2 | What does N. Hassan research? | Scoped to Hassan corpus |
| 2.3 | Summarize the work of Satyadhar Joshi | Financial cybersecurity review paper(s) |
| 2.4 | What are the thoughts of Zhang on deep learning? | Review-of-deep-learning / related papers |
| 2.5 | Papers by Paszke — what framework do they describe? | PyTorch paper content |
| 2.6 | What are Md Mehedi Hassan's contributions? | Cryptography / IoT paper scope |
| 2.7 | Describe the research of Muhammad Sajid Khan | European cybersecurity & AI framework |
| 2.8 | What are his main contributions? *(after 2.1 in same chat)* | Follow-up stays on Hassan if history enabled |

---

## 3. Single-paper focus

| # | Question | Expected behavior |
|---|----------|-------------------|
| 3.1 | Summarize the paper titled "PyTorch: An Imperative Style, High-Performance Deep Learning Library" | Summary only from that paper |
| 3.2 | Summarize "Deep Residual Learning for Image Recognition" | ResNet / He et al. content |
| 3.3 | What methodology does nnU-Net use? | Biomedical segmentation paper |
| 3.4 | Summarize the paper titled "Completely Fake Title 2099" | Clear “not in library” (not a random real paper summary) |
| 3.5 | What are the main findings of "European Cybersecurity And Ai Framework"? | Khan 2025 paper only |

*Tip:* Use the **paper filter dropdown** plus: “What is the core contribution of this paper?”

---

## 4. Tables & structured extraction

| # | Question | Expected behavior |
|---|----------|-------------------|
| 4.1 | Give me a table of all papers with title, year, and venue | Markdown table; one row per paper; no `...` truncation |
| 4.2 | For each paper by Hassan, extract title, year, author | Per-paper table; authors consistent with metadata |
| 4.3 | You have a corpus of Hassan's articles. For each paper, extract: (1) title (2) year (3) venue | Numbered-column parse; scoped to Hassan |
| 4.4 | Table of all papers with title, year, venue for Joshi | Single- or few-paper table |

---

## 5. Topic & cross-corpus questions

Should search broadly (not locked to one author unless named).

| # | Question | Expected behavior |
|---|----------|-------------------|
| 5.1 | What do my papers say about deep learning in medical imaging? | Relevant papers (Litjens survey, nnU-Net, DeepLabCut, etc.) |
| 5.2 | Papers on cybersecurity and artificial intelligence | Multiple matches; grounded answers |
| 5.3 | What is information systems theorizing? | Hassan / Burton-Jones / Jarvenpaa papers |
| 5.4 | Compare digital transformation and cybersecurity themes in my library | Thematic comparison with citations |
| 5.5 | What challenges of CNN architectures are discussed? | Alzubaidi et al. review content |

---

## 6. Refusal & safety gates (must refuse correctly)

| # | Question | Expected behavior |
|---|----------|-------------------|
| 6.1 | Give me a recipe for pasta | Off-topic refusal (not in knowledge base) |
| 6.2 | What is the weather in London today? | Off-topic refusal |
| 6.3 | List papers by Zzzzzzznonexistent | Not in library / no matching author |
| 6.4 | Summarize the paper titled "Completely Fake Title 2099" | Not in library (no fabricated summary) |
| 6.5 | What are the contributions of Einstein on relativity? | Not in library OR explicit gap if no Einstein papers |

---

## 7. Regression checks (bugs we fixed)

| # | Question | Expected behavior |
|---|----------|-------------------|
| 7.1 | what are the thoughts of Jhanjhi, what are his main contributions | **If Jhanjhi in KB:** scoped answer, no false “out of scope” refusal |
| 7.2 | List papers by Jhanjhi | **If in KB:** full list (e.g. 25+ papers); same set as 7.1 scope |
| 7.3 | Ask 7.1 then a broad IoT question | Second answer may use wider corpus; not stuck on Jhanjhi only |

---

## 8. Full smart-city / Jhanjhi corpus (~86 papers)

Run this block when that collection is ingested (metadata shows Jhanjhi / Humayun / smart cities).

| # | Question | Expected behavior |
|---|----------|-------------------|
| 8.1 | How many papers are in my knowledge base? | ~86 |
| 8.2 | List papers by Jhanjhi | ~25 papers; consistent author variants |
| 8.3 | What are Jhanjhi's main contributions to IoT and smart cities? | Synthesis; citations only from Jhanjhi papers |
| 8.4 | List papers by Humayun | Humayun co-authored subset |
| 8.5 | Give me a table of all papers with title, year, and venue | 86 rows; venue N/A OK if metadata missing |
| 8.6 | Corpus of Noor Zaman Jhanjhi's articles on smart-city cybersecurity. For each paper: (1) title (2) year (3) author | Author-scoped extraction table |
| 8.7 | What do papers say about federated learning in smart cities? | Topic search across corpus |
| 8.8 | Summarize "Hybrid TCP SYN attack detection model in SDN" | Single-paper scope |
| 8.9 | Compare ransomware IoT papers in my library | Multi-paper, grounded comparison |
| 8.10 | List papers by Kumar | **Careful:** common surname — should only match if Kumar is in author fields, not every “Kumar” in text |

---

## 9. After **new ingest** (repeat every time)

1. `How many papers are in my knowledge base?` — count increased  
2. `List all authors in my library` — new author appears  
3. `List papers by <new author surname>` — new paper listed  
4. `What are the main contributions of <new author>?` — answer uses new paper chunks  
5. Re-run **one** question from sections 2–4 that failed before ingest  

---

## 10. Pass criteria (quick rubric)

- **Catalog:** Counts and lists match UI / ChromaDB stats exactly.  
- **Author scope:** No citations to papers outside that author’s list (check “Show retrieved chunks”).  
- **Paper scope:** Answer only from named or filtered paper.  
- **Refusal:** Off-topic and missing entities refused; in-library entities never refused with “out of retrieved scope”.  
- **Tables:** No truncated rows, no mixed columns (DOI in year column).  
- **New ingest:** Visible in catalog within same server session (no code deploy needed).

---

## Authors in current local KB (use for section 2)

| Surname / name | Example ask |
|----------------|-------------|
| Hassan | contributions, papers by Hassan, IS theorizing |
| Zhang | deep learning review, papers by Zhang |
| Paszke / PyTorch | papers by Paszke, summarize PyTorch paper |
| Sharma / Hassan (Md Mehedi) | cryptography IoT paper |
| Khan | European cybersecurity AI framework |
| Joshi | Gen AI financial cybersecurity |
| He | ResNet paper |
| Isensee | nnU-Net |
| Mathis | DeepLabCut |
| Burton-Jones | next-generation IS theorizing |
| Sarker | machine learning survey |
| Alzubaidi | deep learning review |
| Ashrafi | digital transformation |

*Replace or extend this table after each bulk ingest.*
