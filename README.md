# Research-RAG

A **hybrid Retrieval-Augmented Generation** system for multi-paper academic question answering.

Combines **dense semantic search** (FAISS), **sparse BM25 retrieval**, and **knowledge-graph traversal** (NetworkX) to answer questions across multiple research PDFs — with built-in hallucination detection.

![Architecture](paper/architecture.png)

---

## Features

- 📄 **Multi-PDF ingestion** — drop any research papers into `papers/` and the system indexes them automatically
- 🔍 **Hybrid retrieval** — fuses dense (all-MiniLM-L6-v2 + FAISS) and sparse (BM25) signals for best-of-both-worlds recall
- 🕸️ **Knowledge graph** — LLM extracts structured entities (methods, datasets, metrics, results) into a NetworkX graph with 11 node types and 9 edge types
- 🧠 **Structured JSON answers** — responses include reasoning summary, narrative, evidence citations, and a cross-paper comparison block
- ✅ **Grounding verification** — a second LLM pass checks every claim against retrieved context and flags hallucinations
- 🌐 **Web UI** — FastAPI backend + static HTML/CSS/JS frontend for paper upload, chat, comparison, and analysis views
- 📊 **Paper comparison** — side-by-side metric comparison across artifacts via `compare_cli.py`
- 🔬 **Evaluation harness** — `eval/eval_rag.py` benchmarks Simple vs Graph vs Hybrid RAG on faithfulness and completeness

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Xalibur1/research_rag.git
cd research_rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
```

### 3. Add papers

```bash
cp your_papers/*.pdf papers/
```

### 4. Run the web UI

```bash
uvicorn api:app --reload
# Open http://localhost:8000
```

### 5. Or use the CLI

```bash
# Interactive multi-paper Q&A
python multi_rag.py --pdf-dir ./papers

# Generate a comparison report
python multi_rag.py --pdf-dir ./papers --compare --output-md report.md
```

---

## Project Structure

```
research_rag/
│
├── Core Engine
│   ├── parser.py            # PDF → section-aware chunks (PyMuPDF)
│   ├── schemas.py           # Pydantic data models (PaperExtraction)
│   ├── multi_extractor.py   # LLM extractor (Groq / Gemini backends)
│   ├── graph_builder.py     # NetworkX knowledge graph construction
│   ├── graph_index.py       # FAISS + BM25 + graph hybrid index
│   └── multi_generator.py   # Multi-paper answer generation + grounding check
│
├── Entry Points
│   ├── api.py               # FastAPI web server
│   ├── multi_rag.py         # Multi-paper interactive CLI
│   ├── batch_extract.py     # Bulk PDF → JSON extraction
│   └── compare_cli.py       # Side-by-side paper comparison
│
├── compare/                 # Comparison engine module
├── eval/
│   └── eval_rag.py          # Experiment harness (LLM-as-judge)
├── paper/
│   └── research_rag_ieee.tex  # IEEE research paper (LaTeX)
├── static/                  # Frontend (HTML / CSS / JS)
├── schemas/                 # JSON schema definitions
├── tests/                   # Unit tests
│
├── papers/                  # ← drop your PDFs here
├── artifacts/               # ← auto-generated extraction cache
├── .env.example             # API key template
└── requirements.txt
```

---

## How It Works

### Indexing Pipeline (one-time per paper set)

```
PDF Files
  │
  ▼ parser.py          — section-aware chunking (1500 chars, 200 overlap)
  ▼ multi_extractor.py — LLM → structured PaperExtraction JSON
  ├─► graph_builder.py — NetworkX DiGraph (methods, datasets, metrics…)
  └─► graph_index.py   — FAISS dense index + BM25 index + graph node index
```

### Query Pipeline (every question)

```
User Query
  ├─ Dense retrieval   (FAISS cosine over chunk embeddings)
  ├─ Sparse retrieval  (BM25 term frequency)
  └─ Graph retrieval   (FAISS over graph nodes → ego-graph traversal)
       │
       ▼ Hybrid fusion:  score = 0.5·dense + 0.5·BM25  +  graph triples
       ▼ Context builder (chunks + triples, ≤ 8000 chars)
       ▼ LLM generate   (structured JSON answer)
       ▼ Grounding check (second LLM pass — detects hallucinations)
       ▼ Answer
```

---

## Experimental Results

Evaluated on 12 questions across 4 papers using `llama-3.3-70b-versatile` (Groq) as both generator and judge.

| System | Faithfulness | Completeness | **Combined** |
|---|---|---|---|
| Simple RAG (dense only) | 0.983 | 0.708 | **0.846** |
| Graph RAG (graph only)  | 0.983 | 0.483 | **0.733** |
| Hybrid RAG (all three)  | 0.908 | 0.683 | **0.796** |

**By question type (Combined score):**

| Type | Simple RAG | Graph RAG | Hybrid RAG |
|---|---|---|---|
| Factual     | 0.850 | 0.800 | 0.775 |
| Reasoning   | 0.875 | 0.613 | **0.850** |
| Comparison  | 0.813 | 0.788 | 0.763 |

Key findings:
- **Graph RAG** is perfect on structured-fact queries (Q2, Q12: Combined = 1.00) where answers are explicit graph edges
- **Hybrid RAG** outperforms Graph RAG by +0.24 on reasoning questions that require prose context
- All three systems maintain faithfulness ≥ 0.90 — the "answer only from context" instruction holds

Full results: see `eval/eval_rag.py` and the IEEE paper in `paper/research_rag_ieee.tex`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/papers` | List all loaded papers |
| `GET` | `/api/papers/{id}` | Full extraction for one paper |
| `POST` | `/api/chat` | Ask a question (hybrid RAG answer) |
| `POST` | `/api/verify` | Check an answer for grounding |
| `POST` | `/api/upload` | Upload a new PDF and re-index |
| `GET` | `/api/dashboard/stats` | Aggregate stats |
| `GET/POST/PUT/DELETE` | `/api/collections` | Manage named paper groups |

---

## LLM Backends

| Backend | Model | Set in `.env` |
|---|---|---|
| **Groq** (default) | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Gemini | `gemini-2.5-flash` | `GOOGLE_API_KEY` |

Switch backend with `--backend groq` or `--backend gemini` on any CLI script.

---

## Requirements

- Python 3.11+
- `faiss-cpu`, `sentence-transformers`, `rank-bm25`, `networkx`
- `fastapi`, `uvicorn`, `pymupdf`, `pydantic`
- `groq` and/or `google-genai` (depending on backend)

See `requirements.txt` for pinned versions.

---

## Research Paper

An IEEE-format research paper documenting the architecture and experimental results is available at [`paper/research_rag_ieee.tex`](paper/research_rag_ieee.tex).
