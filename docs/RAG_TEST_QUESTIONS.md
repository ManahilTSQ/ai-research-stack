# RAG Test Question Bank

Use after deploying code changes or ingesting papers on **https://cite.aitawfiq.com** (production).  
Your client ingests via **Upload PDF** and **Paper Discovery** — data lives in **ChromaDB on the server**, not in the GitHub `/papers` folder.

**Production corpus (example):** ~86 papers, ~1.3k chunks — Jhanjhi, Humayun, Gouda, medical imaging, smart-city IoT, etc.

Restart the API after deploy, then run tests in the chat UI.

---

## 1. Catalog & inventory (deterministic)

| # | Question | Expected |
|---|----------|----------|
| 1.1 | How many papers are in my knowledge base? | **86** (or current manifest count) |
| 1.2 | List all ingested papers | All 86, no truncation |
| 1.3 | List all authors in my library | Full author list |
| 1.4 | List papers by Jhanjhi | ~25 papers (all Jhanjhi variants) |
| 1.5 | List papers by Humayun | Humayun co-authored papers only |
| 1.6 | List papers by Gouda | Gouda / Walaa Gouda papers |

---

## 2. Author-scoped synthesis (your real corpus)

Use authors **in your library**. Do **not** use Hassan unless that name exists on ingested papers.

| # | Question | Expected |
|---|----------|----------|
| 2.1 | What are the main contributions of Jhanjhi? | IoT, smart cities, security, ML — **only Jhanjhi papers** |
| 2.2 | What does Noor Zaman Jhanjhi research? | Same corpus as 2.1 |
| 2.3 | What are the thoughts of Jhanjhi, what are his main contributions | No false “out of scope” refusal |
| 2.4 | What does M. Humayun research? | Humayun papers only |
| 2.5 | Describe the research of Walaa Gouda | Medical imaging / COVID / skin papers |
| 2.6 | Papers by Jhanjhi — summarize their themes | Scoped list + synthesis |
| 2.7 | What are the main contributions of Hassan? | **Refusal** if no author named Hassan in KB (not Riskhan/phishing) |
| 2.8 | What does N. Hassan research? | **Refusal** or “not in library” — must **not** answer about Jhanjhi |
| 2.9 | Describe the research of Muhammad Sajid Khan | Only if that exact author exists; must **not** attribute N. A. Khan / Fida Khan papers to him |

---

## 3. Topic questions (medical imaging — critical regression)

| # | Question | Expected |
|---|----------|----------|
| 3.1 | What do my papers say about deep learning in medical imaging? | DR, melanoma, MRI brain, breast cancer, colon histopathology, etc. **No** phishing, barcode UAV, or traffic-only papers |
| 3.2 | What papers discuss diabetic retinopathy or skin cancer imaging? | Gouda / Alwakid imaging papers |
| 3.3 | Papers on smart cities and cybersecurity | IoT / smart-city corpus (broad topic) |
| 3.4 | Federated learning in smart cities | Federated-learning smart-city papers |

---

## 4. Single-paper focus

| # | Question | Expected |
|---|----------|----------|
| 4.1 | Summarize "Automated Deception Detection in Videos Using 3DCNN" | That paper only |
| 4.2 | Summarize "Completely Fake Title 2099" | Not in library |
| 4.3 | Use **Focus on Paper** dropdown + "What is the main contribution?" | Single-paper answer |

---

## 5. Tables

| # | Question | Expected |
|---|----------|----------|
| 5.1 | Give me a table of all papers with title, year, and venue | 86 rows; venue may be N/A |
| 5.2 | For each paper by Jhanjhi, extract: (1) title (2) year (3) author | Jhanjhi scope only |

---

## 6. Refusals

| # | Question | Expected |
|---|----------|----------|
| 6.1 | Give me a recipe for pasta | Off-topic refusal |
| 6.2 | List papers by Zzzzzzznonexistent | Not in library |
| 6.3 | What are Einstein's contributions to relativity? | Not in library / no evidence |

---

## 7. After client ingests new PDFs (upload or discovery)

1. Manifest count increases (sidebar).  
2. `How many papers are in my knowledge base?` — new total.  
3. `List papers by <new author surname>`.  
4. One synthesis question about that author.  

No GitHub pull required for ingest — only **deploy** when you change Python code.

---

## 8. Pass rubric

- **Chunks panel:** For author questions, retrieved chunks should be from that author’s papers only.  
- **Topic 3.1:** No phishing / barcode / pure traffic papers in the answer body.  
- **Wrong author:** Never attribute Khan A’s paper to Khan B.  
- **Substring names:** “Hassan” must not match “Riskhan”.  

---

## Deploy note

Local dev machines may show **fewer papers** than production if ChromaDB on your laptop is not the same as the server. Always validate on **cite.aitawfiq.com** after `git pull` and server restart.
