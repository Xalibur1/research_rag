#!/usr/bin/env python3
"""
eval/eval_rag.py – Real experiment harness for Research-RAG.

Uses Groq (llama-3.3-70b-versatile, 30 RPM / 6000 TPM free tier)
for both answer generation and LLM-as-judge grading.

Compares:
  1. Simple RAG  – dense-only FAISS retrieval
  2. Graph RAG   – graph-neighbourhood retrieval only
  3. Hybrid RAG  – dense + BM25 + graph

Saves results to eval/results.json and eval/results_summary.txt.
"""

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from parser import PDFParser, DocumentChunk
from graph_builder import GraphBuilder
from graph_index import HybridIndex

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
PAPERS_DIR    = ROOT / "papers"
ARTIFACTS_DIR = ROOT / "artifacts"
EVAL_DIR      = ROOT / "eval"
EVAL_DIR.mkdir(exist_ok=True)
RESULTS_JSON  = EVAL_DIR / "results.json"
RESULTS_TXT   = EVAL_DIR / "results_summary.txt"

GROQ_MODEL   = "llama-3.3-70b-versatile"
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 1500
OVERLAP      = 200
TOP_K        = 5
SLEEP_SECS   = 4   # 4s between calls → ~15 calls/min, well under 30 RPM

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_groq_client: Groq | None = None

def get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

# ─────────────────────────────────────────
# Questions
# ─────────────────────────────────────────
QUESTIONS = [
    # factual
    {"id": "Q01", "q": "What is the core idea of Meta Faster R-CNN?",
     "type": "factual", "paper_ids": ["2104.07719v4"]},
    {"id": "Q02", "q": "What datasets does YOLO-IOD use for evaluation?",
     "type": "factual", "paper_ids": ["2512.22973v2"]},
    {"id": "Q03", "q": "What evaluation metrics are reported in the deep learning image detection comparison paper?",
     "type": "factual", "paper_ids": ["s40537-021-00434-w"]},
    {"id": "Q04", "q": "What image enhancement model is used in the C4PS pipeline?",
     "type": "factual", "paper_ids": ["DL_image_caption_gen (2) (1)"]},
    # reasoning
    {"id": "Q05", "q": "What are the stated limitations of Meta Faster R-CNN?",
     "type": "reasoning", "paper_ids": ["2104.07719v4"]},
    {"id": "Q06", "q": "What problem does YOLO-IOD solve compared to standard YOLO detectors?",
     "type": "reasoning", "paper_ids": ["2512.22973v2"]},
    {"id": "Q07", "q": "Why do the authors of the deep learning comparison paper use Microsoft COCO specifically?",
     "type": "reasoning", "paper_ids": ["s40537-021-00434-w"]},
    {"id": "Q08", "q": "How does C4PS handle multilingual caption generation?",
     "type": "reasoning", "paper_ids": ["DL_image_caption_gen (2) (1)"]},
    # comparison
    {"id": "Q09", "q": "Which paper among Meta Faster R-CNN and YOLO-IOD addresses catastrophic forgetting, and how?",
     "type": "comparison", "paper_ids": ["2104.07719v4", "2512.22973v2"]},
    {"id": "Q10", "q": "Compare the object detection architectures used across all four papers.",
     "type": "comparison", "paper_ids": "all"},
    {"id": "Q11", "q": "Which papers use COCO as an evaluation dataset and what metrics do they report?",
     "type": "comparison", "paper_ids": "all"},
    {"id": "Q12", "q": "What are the inputs and outputs of the methods in YOLO-IOD and Meta Faster R-CNN?",
     "type": "comparison", "paper_ids": ["2104.07719v4", "2512.22973v2"]},
]

# ─────────────────────────────────────────
# LLM helpers (Groq)
# ─────────────────────────────────────────
def groq_call(prompt: str, max_tokens: int = 1500, max_retries: int = 6) -> str:
    client = get_groq()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err = str(exc)
            # Groq rate-limit errors contain "rate_limit_exceeded" or 429
            if "rate_limit" in err.lower() or "429" in err:
                m = re.search(r"try again in ([\d.]+)s", err, re.IGNORECASE)
                wait = float(m.group(1)) + 2 if m else min(60, 10 * (2 ** attempt))
                print(f"    [rate-limit] sleeping {wait:.0f}s (attempt {attempt+1})…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Groq failed after {max_retries} retries")


def groq_json(prompt: str) -> dict:
    # Groq doesn't have JSON-mode for all models; ask explicitly and parse
    full_prompt = prompt + "\n\nIMPORTANT: Output ONLY valid JSON, no markdown, no explanation."
    raw = groq_call(full_prompt, max_tokens=800)
    # strip any fence
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


# ─────────────────────────────────────────
# Retrieval classes
# ─────────────────────────────────────────
class SimpleRAG:
    def __init__(self, chunks: List[DocumentChunk], embed: SentenceTransformer):
        self.chunks = chunks
        self.embed  = embed
        texts = [c.text for c in chunks]
        embs  = embed.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        q = self.embed.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        _, I = self.index.search(q, k=top_k)
        ctx = "--- TEXT CHUNKS (Dense Only) ---\n"
        for idx in I[0]:
            if idx < 0: continue
            c = self.chunks[int(idx)]
            ctx += (f"[Paper: {c.paper_id} | Pages {c.page_start}-{c.page_end}"
                    f" | Section: {c.section}]\n{c.text[:700]}\n\n")
        return ctx


class GraphRAG:
    def __init__(self, graph, node_ids, node_index, embed: SentenceTransformer):
        self.graph      = graph
        self.node_ids   = node_ids
        self.node_index = node_index
        self.embed      = embed

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        import networkx as nx
        q = self.embed.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        _, I = self.node_index.search(q, k=top_k)
        ctx = "--- KNOWLEDGE GRAPH (Graph Only) ---\n"
        for idx in I[0]:
            idx = int(idx)
            if idx < 0 or idx >= len(self.node_ids): continue
            nid = self.node_ids[idx]
            ego = nx.ego_graph(self.graph.to_undirected(), nid, radius=1)
            ctx += f"Node: {nid}\n"
            for u, v, data in ego.edges(data=True):
                ctx += f"  {u} --[{data.get('relation','related')}]--> {v}\n"
            ctx += "\n"
        return ctx


class HybridRAG:
    def __init__(self, hidx: HybridIndex):
        self.hidx = hidx

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        chunks_scored = self.hidx.search_chunks(query, top_k=top_k, dense_weight=0.5)
        graph_results = self.hidx.search_graph(query, top_k=top_k)
        ctx = "--- TEXT CHUNKS (Dense+BM25 Hybrid) ---\n"
        for c, score in chunks_scored:
            ctx += (f"[Paper: {c.paper_id} | Pages {c.page_start}-{c.page_end}"
                    f" | Section: {c.section} | Score: {score:.3f}]\n{c.text[:700]}\n\n")
        ctx += "\n--- KNOWLEDGE GRAPH ---\n"
        for nid, subgraph in graph_results.items():
            ctx += f"Node: {nid}\n"
            for link in subgraph.get("links", [])[:8]:
                ctx += (f"  {link.get('source','')} --[{link.get('relation','related')}]-->"
                        f" {link.get('target','')}\n")
            ctx += "\n"
        return ctx


# ─────────────────────────────────────────
# Generation & judging
# ─────────────────────────────────────────
def generate_answer(query: str, context: str, paper_titles: Dict[str, str]) -> str:
    titles = "\n".join(f"  - {pid}: {t}" for pid, t in paper_titles.items())
    prompt = (
        "You are a research assistant. Answer the question using ONLY the provided context.\n"
        "If the context lacks information, say so explicitly.\n\n"
        f"Papers:\n{titles}\n\n"
        f"QUESTION: \"{query}\"\n\n"
        f"CONTEXT:\n{context[:5000]}\n\n"
        "Answer in 100-200 words. Reference specific paper names. "
        "Do NOT invent facts not in the context."
    )
    return groq_call(prompt, max_tokens=600)


JUDGE_PROMPT = """\
You are a strict academic grader.

QUESTION: "{question}"

SOURCE CONTEXT:
{context}

SYSTEM ANSWER:
{answer}

Score on two axes (0.0 to 1.0):
- faithfulness_score: every claim traceable to the SOURCE CONTEXT (1.0=perfect, 0.0=hallucinated)
- completeness_score: question fully answered (1.0=complete, 0.0=not answered)

One-sentence reasoning.

Output only JSON: {{"faithfulness_score": <float>, "completeness_score": <float>, "reasoning": "<str>"}}"""


def judge_answer(question: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context[:3500],
        answer=answer,
    )
    try:
        result = groq_json(prompt)
        f = float(result.get("faithfulness_score", 0))
        c = float(result.get("completeness_score", 0))
        result["faithfulness_score"] = round(f, 2)
        result["completeness_score"] = round(c, 2)
        result["combined"] = round((f + c) / 2, 3)
        return result
    except Exception as exc:
        print(f"    [judge error] {exc}")
        return {"faithfulness_score": 0.0, "completeness_score": 0.0,
                "combined": 0.0, "reasoning": str(exc)}


def mean_key(records: list, key: str) -> float:
    vals = [float(r.get(key, 0)) for r in records]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    W = 68
    print("=" * W)
    print("  Research-RAG — Actual Experiment (Groq backend)")
    print("=" * W)

    # 1. Parse
    print("\n[1/4] Parsing PDFs …")
    parser = PDFParser(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    all_chunks: List[DocumentChunk] = []
    paper_ids_found: List[str] = []
    for pdf in sorted(PAPERS_DIR.glob("*.pdf")):
        pid = pdf.stem
        try:
            chunks = parser.parse(str(pdf))
            for c in chunks:
                c.paper_id = pid
            all_chunks.extend(chunks)
            paper_ids_found.append(pid)
            print(f"  ✓ {pid} → {len(chunks)} chunks")
        except Exception as exc:
            print(f"  ✗ {pid}: {exc}")
    print(f"  Total: {len(all_chunks)} chunks from {len(paper_ids_found)} papers")

    # 2. Titles
    print("\n[2/4] Loading titles …")
    paper_titles: Dict[str, str] = {}
    for pid in paper_ids_found:
        art = ARTIFACTS_DIR / f"{pid}.json"
        title = pid
        if art.exists():
            with open(art) as f:
                title = json.load(f).get("title") or pid
        paper_titles[pid] = title
        print(f"  {pid[:30]}: {title[:55]}")

    # 3. Index
    print("\n[3/4] Building indices …")
    embed = SentenceTransformer(EMBED_MODEL)
    simple_rag = SimpleRAG(all_chunks, embed)
    print("  ✓ Simple RAG index")

    gb = GraphBuilder()
    for pid in paper_ids_found:
        art = ARTIFACTS_DIR / f"{pid}.json"
        if art.exists():
            from schemas import PaperExtraction
            with open(art) as f:
                data = json.load(f)
            try:
                gb.build_from_extraction(PaperExtraction(**data))
            except Exception as exc:
                print(f"  ⚠  {pid}: {exc}")
    graph = gb.graph
    print(f"  ✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    hidx = HybridIndex(model_name=EMBED_MODEL)
    hidx.build_chunk_index(all_chunks)
    hidx.build_graph_index(graph)
    print("  ✓ Hybrid index")

    graph_rag  = GraphRAG(graph, hidx.node_ids, hidx.node_index, embed)
    hybrid_rag = HybridRAG(hidx)

    SYSTEMS = {
        "simple_rag": simple_rag,
        "graph_rag":  graph_rag,
        "hybrid_rag": hybrid_rag,
    }

    # 4. Run experiments
    print(f"\n[4/4] {len(QUESTIONS)} questions × {len(SYSTEMS)} systems "
          f"via Groq ({GROQ_MODEL})\n")

    all_results = []
    for qi, qblock in enumerate(QUESTIONS, 1):
        qid   = qblock["id"]
        query = qblock["q"]
        qtype = qblock["type"]
        scope = qblock["paper_ids"]

        active = paper_titles if scope == "all" else {
            pid: paper_titles[pid] for pid in scope if pid in paper_titles
        }

        print(f"  [{qi:02d}/{len(QUESTIONS)}] {qid} [{qtype}]")
        print(f"       {query[:85]}")

        q_res = {"id": qid, "q": query, "type": qtype,
                 "scope": scope, "systems": {}}

        for sys_name, sys_obj in SYSTEMS.items():
            try:
                ctx    = sys_obj.retrieve(query, top_k=TOP_K)
                time.sleep(SLEEP_SECS)
                answer = generate_answer(query, ctx, active)
                time.sleep(SLEEP_SECS)
                scores = judge_answer(query, ctx, answer)
                time.sleep(SLEEP_SECS)
                q_res["systems"][sys_name] = {
                    "answer": answer,
                    "context_length": len(ctx),
                    "scores": scores,
                }
                print(f"    {sys_name:<12}  F={scores['faithfulness_score']:.2f}"
                      f"  C={scores['completeness_score']:.2f}"
                      f"  Comb={scores['combined']:.2f}")
            except Exception as exc:
                print(f"    {sys_name}: ERROR → {exc}")
                traceback.print_exc()
                q_res["systems"][sys_name] = {
                    "answer": f"ERROR: {exc}",
                    "scores": {"faithfulness_score": 0.0,
                               "completeness_score": 0.0,
                               "combined": 0.0,
                               "reasoning": str(exc)},
                }
        all_results.append(q_res)
        print()

    # 5. Aggregate
    aggregates: Dict[str, dict] = {}
    for sys_name in SYSTEMS:
        all_sc = [r["systems"][sys_name]["scores"]
                  for r in all_results if sys_name in r["systems"]]
        agg: Dict[str, Any] = {
            "faithfulness": mean_key(all_sc, "faithfulness_score"),
            "completeness": mean_key(all_sc, "completeness_score"),
            "combined":     mean_key(all_sc, "combined"),
        }
        for qt in ("factual", "reasoning", "comparison"):
            sub = [r["systems"][sys_name]["scores"]
                   for r in all_results
                   if r["type"] == qt and sys_name in r["systems"]]
            agg[f"{qt}_faithfulness"] = mean_key(sub, "faithfulness_score")
            agg[f"{qt}_completeness"] = mean_key(sub, "completeness_score")
            agg[f"{qt}_combined"]     = mean_key(sub, "combined")
        aggregates[sys_name] = agg

    output = {
        "meta": {
            "papers": list(paper_titles.keys()),
            "n_questions": len(QUESTIONS),
            "embed_model": EMBED_MODEL,
            "llm_model": GROQ_MODEL,
        },
        "aggregates": aggregates,
        "per_question": all_results,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved → {RESULTS_JSON}")

    # 6. Print summary
    lines = []
    lines.append("=" * W)
    lines.append("  Research-RAG — Experiment Results Summary")
    lines.append("=" * W)
    lines.append(f"\nPapers ({len(paper_titles)}):")
    for pid, title in paper_titles.items():
        lines.append(f"  • {title[:60]} [{pid}]")
    lines.append(f"\nQuestions: {len(QUESTIONS)}  "
                 f"(4 factual | 4 reasoning | 4 comparison)")
    lines.append(f"LLM: {GROQ_MODEL} | Embed: {EMBED_MODEL}")

    for label, key in [("OVERALL", None), ("FACTUAL", "factual"),
                        ("REASONING", "reasoning"), ("COMPARISON", "comparison")]:
        lines.append(f"\n{'─'*W}\n  {label}\n{'─'*W}")
        lines.append(f"  {'System':<14}{'Faithfulness':>14}{'Completeness':>14}{'Combined':>12}")
        lines.append(f"  {'─'*52}")
        for sn, agg in aggregates.items():
            f_ = agg["faithfulness"] if key is None else agg[f"{key}_faithfulness"]
            c_ = agg["completeness"] if key is None else agg[f"{key}_completeness"]
            m_ = agg["combined"]     if key is None else agg[f"{key}_combined"]
            lines.append(f"  {sn:<14}{f_:>14.4f}{c_:>14.4f}{m_:>12.4f}")

    lines.append(f"\n{'─'*W}\n  PER-QUESTION\n{'─'*W}")
    for r in all_results:
        lines.append(f"\n  {r['id']} [{r['type']}]  {r['q'][:65]}")
        for sn in SYSTEMS:
            sc = r["systems"].get(sn, {}).get("scores", {})
            reason = sc.get("reasoning", "")[:72]
            lines.append(
                f"    {sn:<12}  F={sc.get('faithfulness_score',0):.2f}"
                f"  C={sc.get('completeness_score',0):.2f}"
                f"  Comb={sc.get('combined',0):.2f}  | {reason}"
            )

    summary = "\n".join(lines)
    with open(RESULTS_TXT, "w") as f:
        f.write(summary)
    print(f"✓ Saved → {RESULTS_TXT}\n")
    print(summary)
    return output


if __name__ == "__main__":
    main()
