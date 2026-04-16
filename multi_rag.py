#!/usr/bin/env python3
"""
multi_rag.py — Multi-paper Research RAG system.

Loads multiple PDFs, builds a combined knowledge graph + hybrid index,
optionally generates a comparison report, and enters interactive Q&A.

Usage:
    python multi_rag.py --pdfs paper1.pdf paper2.pdf paper3.pdf
    python multi_rag.py --pdf-dir ./papers
    python multi_rag.py --pdf-dir ./papers --compare --output-md report.md
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from parser import PDFParser
from multi_extractor import MultiBackendExtractor
from graph_builder import GraphBuilder
from graph_index import HybridIndex
from multi_generator import MultiPaperRAGGenerator


def main():
    ap = argparse.ArgumentParser(
        description="Multi-paper Research RAG: ask questions across multiple papers.",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdfs", nargs="+", help="PDF file paths.")
    group.add_argument("--pdf-dir", help="Directory containing PDF files.")
    ap.add_argument("--backend", choices=["groq", "gemini"], default="groq",
                     help="LLM backend (default: groq)")
    ap.add_argument("--model", default=None, help="Override model name.")
    ap.add_argument("--compare", action="store_true",
                     help="Also generate a comparison report from extracted data.")
    ap.add_argument("--output-md", default=None, help="Path for comparison MD report.")
    ap.add_argument("--output-json", default=None, help="Path for comparison JSON report.")
    ap.add_argument("--artifacts-dir", default="./artifacts",
                     help="Directory to save extracted JSON artifacts.")
    args = ap.parse_args()

    # Collect PDFs
    if args.pdfs:
        pdf_paths = [Path(p) for p in args.pdfs]
    else:
        pdf_dir = Path(args.pdf_dir)
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            print(f"No PDF files found in {pdf_dir}")
            sys.exit(1)

    n = len(pdf_paths)
    print(f"\n{'='*60}")
    print(f"  Research RAG — Multi-Paper Mode")
    print(f"{'='*60}")
    print(f"  Papers:  {n}")
    print(f"  Backend: {args.backend}")
    print(f"{'='*60}\n")

    # --- Phase 1: Parse all PDFs ---
    print(f"--- Phase 1/{4}: Parsing {n} PDF(s) ---")
    doc_parser = PDFParser(chunk_size=1500, overlap=200)
    all_chunks = []       # combined chunks for the hybrid index
    paper_chunks = {}     # per-paper chunks
    for i, pdf in enumerate(pdf_paths, 1):
        print(f"  [{i}/{n}] Parsing {pdf.name}...", end=" ", flush=True)
        try:
            chunks = doc_parser.parse(str(pdf))
            paper_chunks[pdf.stem] = chunks
            all_chunks.extend(chunks)
            print(f"✓ {len(chunks)} chunks")
        except Exception as e:
            print(f"✗ {e}")
    print(f"  Total chunks: {len(all_chunks)}\n")

    # --- Phase 2: Extract entities ---
    print(f"--- Phase 2/{4}: Extracting entities (this may take a while) ---")
    extractor = MultiBackendExtractor(backend=args.backend, model_name=args.model)
    extractions = {}
    paper_titles = {}
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for i, (paper_id, chunks) in enumerate(paper_chunks.items(), 1):
        print(f"  [{i}/{len(paper_chunks)}] Extracting {paper_id}...", end=" ", flush=True)
        try:
            extraction = extractor.extract_from_chunks(chunks, paper_id)
            extractions[paper_id] = extraction
            paper_titles[paper_id] = extraction.title or paper_id
            print(f"✓ {extraction.title}")

            # Save artifact JSON
            artifact_path = artifacts_dir / f"{paper_id}.json"
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(extraction.model_dump(), f, indent=2, ensure_ascii=False)

            # Rate limit pause for Groq free tier
            if args.backend == "groq" and i < len(paper_chunks):
                print("  ⏳ Waiting 15s for rate limit...", flush=True)
                time.sleep(15)

        except Exception as e:
            print(f"✗ {e}")
    print(f"  Successfully extracted: {len(extractions)}/{len(paper_chunks)}\n")

    if not extractions:
        print("No papers were successfully extracted. Exiting.")
        sys.exit(1)

    # --- Phase 3: Build combined knowledge graph ---
    print(f"--- Phase 3/{4}: Building combined knowledge graph ---")
    gb = GraphBuilder()
    for paper_id, extraction in extractions.items():
        gb.build_from_extraction(extraction)
    graph = gb.graph
    print(f"  Combined graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n")

    # --- Phase 4: Build combined hybrid index ---
    print(f"--- Phase 4/{4}: Building hybrid index ---")
    index = HybridIndex()
    index.build_chunk_index(all_chunks)
    index.build_graph_index(graph)
    print(f"  Indexing complete: {len(all_chunks)} chunks, {len(index.node_ids)} graph nodes\n")

    # --- Optional: Comparison report ---
    if args.compare:
        print("--- Generating comparison report ---")
        try:
            from compare.api import compare_papers
            result = compare_papers(
                input_dir=str(artifacts_dir),
                output_json=args.output_json,
                output_md=args.output_md,
                summary=True,
            )
            if args.output_md:
                print(f"  ✓ MD report: {args.output_md}")
            if args.output_json:
                print(f"  ✓ JSON report: {args.output_json}")
        except Exception as e:
            print(f"  ⚠ Comparison failed: {e}")
        print()

    # --- Interactive Q&A ---
    print("=" * 60)
    print("  Multi-Paper Research RAG — Interactive Q&A")
    print("=" * 60)
    print(f"  Papers loaded ({len(paper_titles)}):")
    for pid, title in paper_titles.items():
        print(f"    • {title} ({pid})")
    print()
    print("  Ask questions across all papers. Type 'quit' to exit.")
    print("  Tips:")
    print("    - 'compare the methods' — cross-paper comparison")
    print("    - 'what datasets are used?' — aggregated answer")
    print("    - 'what are the limitations of [paper]?' — specific paper")
    print("=" * 60)

    rag = MultiPaperRAGGenerator(
        index=index,
        paper_titles=paper_titles,
        backend=args.backend,
        model_name=args.model,
    )

    while True:
        try:
            query = input("\n📝 Your Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ["quit", "exit", "q"]:
            break
        if not query:
            continue

        print("🔍 Searching across all papers...")

        # Rate limit for Groq
        answer = rag.generate_answer(query, top_k_chunks=5, top_k_graph=5)

        # Display answer
        print(f"\n{'─'*50}")
        print("💡 REASONING")
        print(f"{'─'*50}")
        print(answer.get("reasoning_summary", "N/A"))

        print(f"\n{'─'*50}")
        print("📖 ANSWER")
        print(f"{'─'*50}")
        print(answer.get("narrative_answer", "N/A"))

        # Structured comparison if present
        ss = answer.get("structured_summary", {})
        if ss:
            comparisons = ss.get("comparisons", [])
            if comparisons:
                print(f"\n{'─'*50}")
                print("⚖️  COMPARISONS")
                print(f"{'─'*50}")
                for c in comparisons:
                    print(f"  • {c}")

            agreements = ss.get("agreements", [])
            if agreements:
                print(f"\n✅ Agreements:")
                for a in agreements:
                    print(f"  • {a}")

            disagreements = ss.get("disagreements", [])
            if disagreements:
                print(f"\n❌ Disagreements:")
                for d in disagreements:
                    print(f"  • {d}")

        # Evidence
        evidence = answer.get("evidence", [])
        if evidence:
            print(f"\n{'─'*50}")
            print("📎 EVIDENCE")
            print(f"{'─'*50}")
            for ev in evidence:
                pid = ev.get("paper_id", "?")
                pages = ev.get("page_numbers", [])
                snippet = ev.get("snippet", "")
                print(f"  [{pid}, p.{pages}] {snippet[:150]}")

        # Insufficiency warning
        insuff = answer.get("insufficiency", {})
        if insuff.get("flag"):
            print(f"\n⚠️  WARNING: Insufficient evidence.")
            missing = insuff.get("missing_fields", [])
            if missing:
                print(f"  Missing: {', '.join(missing)}")

    print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()
