"""
Multi-backend extractor — supports Groq and Gemini for entity extraction.

Drop-in replacement for extractor.py that works with multiple LLM backends.
"""

import json
import os
from typing import List

from dotenv import load_dotenv

from parser import DocumentChunk
from schemas import PaperExtraction

load_dotenv()


EXTRACTION_PROMPT = """You are an expert academic researcher. Analyze the following research paper text and extract structured information precisely matching the JSON schema below.

Extract:
- title, authors, venue, year
- method (name, purpose, inputs, outputs, core idea, algorithm steps, assumptions, limitations, hyperparameters, code/data links)
- evaluation (datasets, metrics, results)
- comparisons and applications
- glossary of terms
- citations
- an evidence map mapping fields to page numbers and snippets.

PAPER ID: {paper_id}

PAPER CONTENT:
{full_text}

Output valid JSON according to this structure:
{{
    "paper_id": "{paper_id}",
    "title": "...",
    "authors": ["..."],
    "venue": "...",
    "year": "...",
    "doi_or_arxiv": "...",
    "method": {{
        "name": "...",
        "purpose_one_sentence": "...",
        "inputs": [{{"name": "...", "type": "...", "shape_or_example": "..."}}],
        "outputs": [{{"name": "...", "type": "...", "shape_or_example": "..."}}],
        "core_idea": "...",
        "algorithm_steps": ["..."],
        "assumptions": ["..."],
        "limitations": ["..."],
        "hyperparameters": ["..."],
        "code_or_data_links": ["..."]
    }},
    "evaluation": {{
        "datasets": ["..."],
        "metrics": ["..."],
        "results": [{{"dataset": "...", "metric": "...", "value": "...", "baseline_name": "...", "delta_if_stated": "..."}}]
    }},
    "comparisons": ["..."],
    "applications": ["..."],
    "glossary": [{{"term": "...", "definition": "..."}}],
    "citations": ["..."],
    "evidence_map": [{{"field": "...", "page_numbers": [1], "snippet": "..."}}]
}}

Return ONLY valid JSON, no markdown fences."""


class MultiBackendExtractor:
    """Extractor supporting Groq and Gemini backends."""

    def __init__(self, backend: str = "groq", model_name: str | None = None):
        self.backend = backend
        if backend == "groq":
            self.model_name = model_name or "llama-3.3-70b-versatile"
        else:
            self.model_name = model_name or "gemini-2.5-flash"

    def _call_groq(self, prompt: str) -> str:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=8000,
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        from google import genai
        from google.genai import types
        api_key = os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return response.text

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "groq":
            return self._call_groq(prompt)
        return self._call_gemini(prompt)

    def extract_from_chunks(
        self, chunks: List[DocumentChunk], paper_id: str
    ) -> PaperExtraction:
        """Extract structured paper data from chunks using the configured backend."""

        # Smart content selection for token limits
        max_chars = 12_000 if self.backend == "groq" else 80_000

        priority_kw = ["abstract", "introduction", "method", "approach",
                       "result", "experiment", "evaluation", "conclusion",
                       "discussion", "limitation"]
        high, low = [], []
        for i, c in enumerate(chunks):
            sec = c.section.lower()
            if i < 3 or any(k in sec for k in priority_kw):
                high.append((i, c))
            else:
                low.append((i, c))

        full_text = ""
        for i, c in high + low:
            entry = (f"\n--- CHUNK {i+1} | PAGES {c.page_start}-{c.page_end}"
                     f" | SECTION: {c.section} ---\n{c.text}\n")
            if len(full_text) + len(entry) > max_chars:
                break
            full_text += entry

        prompt = EXTRACTION_PROMPT.format(paper_id=paper_id, full_text=full_text)
        resp_text = self._call_llm(prompt).strip()

        if resp_text.startswith("```json"):
            resp_text = resp_text[7:]
        if resp_text.startswith("```"):
            resp_text = resp_text[3:]
        if resp_text.endswith("```"):
            resp_text = resp_text[:-3]

        data = json.loads(resp_text)
        return PaperExtraction(**data)
