#!/usr/bin/env python3
"""
batch_extract.py — Extract structured JSON artifacts from research paper PDFs.

Supports Groq (default) and Gemini backends.

Usage:
    python batch_extract.py --pdfs paper1.pdf paper2.pdf --output-dir ./artifacts
    python batch_extract.py --pdf-dir ./papers --output-dir ./artifacts
    python batch_extract.py --pdfs paper.pdf --output-dir ./artifacts --backend gemini
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from parser import PDFParser

load_dotenv()


EXTRACTION_PROMPT = """You are an expert academic researcher. Analyze the following research paper text and extract ALL available information into the JSON structure below.

Extract as much as you can — leave fields as null only if the information is truly not in the paper.

PAPER ID: {paper_id}

PAPER CONTENT:
{content}

OUTPUT FORMAT — return ONLY valid JSON matching this structure exactly:
{{
  "paper_id": "{paper_id}",
  "title": "...",
  "authors": ["..."],
  "venue": "...",
  "year": 2025,
  "abstract": "...",
  "doi_or_arxiv": "...",

  "domain": "e.g. NLP, CV, OCR, RAG, legal tech, etc.",
  "research_question": "The core problem or question addressed.",
  "task": "e.g. classification, qa, summarization, detection, etc.",
  "novelty": "Key contribution or claim of novelty in 1-2 sentences.",

  "input_type": "e.g. text, image, PDF, table, multimodal",
  "output_type": "e.g. labels, spans, embeddings, generated text, JSON",

  "methodology": {{
    "name": "Name of proposed method/model",
    "summary": "2-3 sentence summary of the approach",
    "pipeline_steps": ["Step 1", "Step 2", "..."],
    "architecture": "Architecture details if applicable",
    "preprocessing": "Preprocessing steps if mentioned",
    "hyperparameters": {{"learning_rate": 0.001, "...": "..."}},
    "assumptions": ["Assumption 1", "..."]
  }},

  "dataset": ["Dataset1", "Dataset2"],
  "data_details": {{
    "size": "e.g. 10k samples",
    "split": "e.g. 80/10/10 train/val/test",
    "gold_standard": "e.g. human-annotated",
    "sample_size": "e.g. 1000 test examples",
    "public": true
  }},

  "model": "Primary model name",
  "baselines": ["Baseline1", "Baseline2"],

  "evaluation": {{
    "setup": "Description of evaluation setup",
    "ablations": ["Ablation 1: removed X, result Y", "..."]
  }},

  "metrics": {{
    "accuracy": 0.92,
    "f1": 0.89,
    "precision": 0.91,
    "recall": 0.87
  }},

  "main_results": [
    {{"dataset": "...", "metric": "...", "value": 0.92, "baseline": "...", "delta": "+2.3%"}}
  ],

  "limitations": ["Limitation 1", "..."],
  "robustness": "Notes on robustness, failure cases, edge cases",
  "future_work": ["Future direction 1", "..."],
  "runtime_efficiency": "Runtime/efficiency/scalability notes",

  "artifacts": {{
    "code_url": "https://github.com/...",
    "data_url": "...",
    "model_url": "..."
  }},

  "sections": {{
    "problem": "1-2 sentence summary of the problem section",
    "approach": "1-2 sentence summary of the approach",
    "data": "1-2 sentence summary of data used",
    "experiments": "1-2 sentence summary of experimental setup",
    "results": "1-2 sentence summary of key results",
    "discussion": "1-2 sentence summary of discussion points",
    "limitations": "1-2 sentence summary of limitations",
    "future_work": "1-2 sentence summary of future work"
  }}
}}

IMPORTANT:
- For "metrics", use numeric values only (floats or ints). Use the best/primary result.
- For "year", use an integer.
- If a field is not found in the paper, set it to null (not "N/A" or empty string).
- For arrays, use an empty array [] if nothing found.
- Return ONLY the JSON, no markdown fences or extra text."""


def _call_groq(prompt: str, model: str) -> str:
    """Call Groq API and return the response text."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError("GROQ_API_KEY not set in .env file")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=8000,
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, model: str) -> str:
    """Call Gemini API and return the response text."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return response.text


BACKENDS = {
    "groq": {"call": _call_groq, "default_model": "llama-3.3-70b-versatile"},
    "gemini": {"call": _call_gemini, "default_model": "gemini-2.5-flash"},
}


def extract_paper(
    pdf_path: str,
    output_dir: str,
    backend: str = "groq",
    model: str | None = None,
) -> dict | None:
    """Extract a single PDF into a canonical JSON artifact."""
    path = Path(pdf_path)
    paper_id = path.stem.replace(" ", "-").lower()
    be = BACKENDS[backend]
    model = model or be["default_model"]

    print(f"\n{'─'*60}")
    print(f"  Processing: {path.name}")
    print(f"  Paper ID:   {paper_id}")
    print(f"  Backend:    {backend} ({model})")
    print(f"{'─'*60}")

    # 1. Parse PDF
    print("  [1/3] Parsing PDF...")
    parser = PDFParser(chunk_size=2000, overlap=200)
    try:
        chunks = parser.parse(str(path))
    except Exception as e:
        print(f"  ✗ Failed to parse PDF: {e}")
        return None
    print(f"  ✓ Extracted {len(chunks)} chunks")

    # 2. Build content — smart selection to fit within token limits
    #    Prioritize: abstract/intro (first chunks), methods, results, conclusion
    #    Groq free tier: 12k TPM, so aim for ~6k tokens (~24k chars) of content
    max_chars = 12_000  # ~3k tokens of content, leaves room for prompt + response

    # Categorize chunks by section importance
    priority_keywords = ["abstract", "introduction", "method", "approach",
                         "result", "experiment", "evaluation", "conclusion",
                         "discussion", "limitation", "related"]
    high_priority = []
    low_priority = []
    for i, chunk in enumerate(chunks):
        section_lower = chunk.section.lower()
        # First 3 chunks always high priority (title, abstract, intro)
        if i < 3 or any(kw in section_lower for kw in priority_keywords):
            high_priority.append((i, chunk))
        else:
            low_priority.append((i, chunk))

    # Build content from high priority first, then fill with low priority
    content = ""
    for i, chunk in high_priority:
        entry = f"\n--- CHUNK {i+1} | PAGES {chunk.page_start}-{chunk.page_end} | SECTION: {chunk.section} ---\n"
        entry += chunk.text + "\n"
        if len(content) + len(entry) > max_chars:
            break
        content += entry

    # Fill remaining space with low priority
    for i, chunk in low_priority:
        entry = f"\n--- CHUNK {i+1} | PAGES {chunk.page_start}-{chunk.page_end} | SECTION: {chunk.section} ---\n"
        entry += chunk.text + "\n"
        if len(content) + len(entry) > max_chars:
            break
        content += entry

    used_chars = len(content)
    total_chunks = len(chunks)
    used_chunks = content.count("--- CHUNK")
    if used_chunks < total_chunks:
        print(f"  ⚠ Using {used_chunks}/{total_chunks} chunks ({used_chars} chars) to fit token limit")

    # 3. Extract with LLM
    print("  [2/3] Extracting with LLM...")
    prompt = EXTRACTION_PROMPT.format(paper_id=paper_id, content=content)

    try:
        resp_text = be["call"](prompt, model).strip()
        # Clean markdown fences if present
        if resp_text.startswith("```json"):
            resp_text = resp_text[7:]
        if resp_text.startswith("```"):
            resp_text = resp_text[3:]
        if resp_text.endswith("```"):
            resp_text = resp_text[:-3]
        data = json.loads(resp_text)
    except Exception as e:
        print(f"  ✗ LLM extraction failed: {e}")
        return None

    # 4. Inject provenance
    data["provenance"] = {
        "source_path": str(path.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # 5. Save
    print("  [3/3] Saving artifact...")
    out_path = Path(output_dir) / f"{paper_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    title = data.get("title", "Unknown")
    n_metrics = len(data.get("metrics") or {})
    print(f"  ✓ Saved: {out_path}")
    print(f"  ✓ Title: {title}")
    print(f"  ✓ Metrics extracted: {n_metrics}")
    return data


def main():
    ap = argparse.ArgumentParser(
        description="Extract structured JSON from research paper PDFs.",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdfs", nargs="+", help="PDF file paths.")
    group.add_argument("--pdf-dir", help="Directory of PDFs.")
    ap.add_argument("--output-dir", default="./artifacts")
    ap.add_argument("--backend", choices=list(BACKENDS), default="groq",
                     help="LLM backend (default: groq)")
    ap.add_argument("--model", default=None, help="Override model name.")
    args = ap.parse_args()

    if args.pdfs:
        pdf_paths = args.pdfs
    else:
        pdf_paths = sorted(str(p) for p in Path(args.pdf_dir).glob("*.pdf"))
        if not pdf_paths:
            print(f"No PDFs found in {args.pdf_dir}")
            sys.exit(1)

    be = BACKENDS[args.backend]
    model = args.model or be["default_model"]

    print(f"\n{'='*60}")
    print(f"  Research RAG — Batch PDF Extraction")
    print(f"{'='*60}")
    print(f"  Papers:   {len(pdf_paths)}")
    print(f"  Backend:  {args.backend} ({model})")
    print(f"  Output:   {args.output_dir}")

    results, errors = [], []
    for pdf in pdf_paths:
        r = extract_paper(pdf, args.output_dir, args.backend, args.model)
        (results if r else errors).append(pdf)

    print(f"\n{'='*60}")
    print(f"  Extraction Complete")
    print(f"{'='*60}")
    print(f"  ✓ Successful: {len(results)}")
    if errors:
        print(f"  ✗ Failed:     {len(errors)}")
        for e in errors:
            print(f"    - {e}")
    print(f"  Output: {args.output_dir}")
    print(f"\n  Next step:")
    print(f"    python compare_cli.py --input-dir {args.output_dir} --output-md report.md --summary")
    print()


if __name__ == "__main__":
    main()
