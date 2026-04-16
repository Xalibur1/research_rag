"""
Multi-paper RAG generator — answers questions across multiple papers.

Uses Groq (default) or Gemini to generate answers with context from
multiple papers' chunks and knowledge graph neighborhoods.
"""

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv

from graph_index import HybridIndex

load_dotenv()


class MultiPaperRAGGenerator:
    """Generate answers across multiple papers using hybrid retrieval."""

    def __init__(
        self,
        index: HybridIndex,
        paper_titles: dict[str, str] | None = None,
        backend: str = "groq",
        model_name: str | None = None,
    ):
        self.index = index
        self.paper_titles = paper_titles or {}
        self.backend = backend
        if backend == "groq":
            self.model_name = model_name or "llama-3.3-70b-versatile"
        else:
            self.model_name = model_name or "gemini-2.5-flash"

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "groq":
            return self._call_groq(prompt)
        return self._call_gemini(prompt)

    def _call_groq(self, prompt: str) -> str:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4000,
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

    def generate_answer(
        self, query: str, top_k_chunks: int = 5, top_k_graph: int = 5,
        paper_ids: list[str] | None = None
    ) -> Dict[str, Any]:
        """Retrieve evidence from all papers and generate an answer.
        
        If paper_ids is provided, restrict the context and prompt to those papers only.
        """

        # 1. Retrieve evidence
        text_results = self.index.search_chunks(query, top_k=top_k_chunks)
        graph_results = self.index.search_graph(query, top_k=top_k_graph)

        # 2. Filter by paper_ids if specified
        active_titles = self.paper_titles
        if paper_ids:
            text_results = [(c, s) for c, s in text_results if c.paper_id in paper_ids]
            graph_results = {k: v for k, v in graph_results.items()
                            if any(pid in k for pid in paper_ids)}
            active_titles = {pid: title for pid, title in self.paper_titles.items()
                            if pid in paper_ids}

        # 3. Build context (keep it compact for Groq token limits)
        context_str = "--- TEXT EVIDENCE ---\n"
        for chunk, score in text_results:
            context_str += (
                f"[Pages {chunk.page_start}-{chunk.page_end}, "
                f"Section: {chunk.section}]\n"
                f"{chunk.text[:800]}\n\n"
            )

        context_str += "\n--- KNOWLEDGE GRAPH ---\n"
        for node_id, subgraph in graph_results.items():
            context_str += f"Node: {node_id}\n"
            for link in subgraph.get("links", [])[:10]:
                src = link.get("source", "")
                tgt = link.get("target", "")
                rel = link.get("relation", "related")
                context_str += f"  {src} --[{rel}]--> {tgt}\n"
            context_str += "\n"

        # Truncate context for Groq
        max_context = 6000 if self.backend == "groq" else 30000
        if len(context_str) > max_context:
            context_str = context_str[:max_context] + "\n[...truncated]"

        # 4. Papers info
        papers_info = ""
        if active_titles:
            papers_info = f"Papers in scope ({len(active_titles)}):\n"
            for pid, title in active_titles.items():
                papers_info += f"  - {pid}: {title}\n"

        # 5. Prompt
        scope_note = (f"Focus ONLY on the {len(active_titles)} paper(s) listed above."
                      if paper_ids else "Draw on ALL available papers.")
        prompt = f"""You are a multi-paper research assistant. {scope_note}
When comparing papers, be specific about which paper says what.

{papers_info}

USER QUESTION: "{query}"

RETRIEVED CONTEXT:
{context_str}

OUTPUT FORMAT (JSON):
{{
    "reasoning_summary": "2-4 sentences explaining how evidence from different papers supports the answer.",
    "narrative_answer": "Clear, student-friendly answer (150-300 words). Reference specific papers by name when relevant. Compare and contrast if the question involves multiple papers.",
    "papers_cited": ["paper_id_1", "paper_id_2"],
    "structured_summary": {{
        "key_findings": ["Finding 1", "Finding 2"],
        "comparisons": ["Paper A does X while Paper B does Y"],
        "agreements": ["Both papers agree that..."],
        "disagreements": ["Paper A claims X but Paper B claims Y"]
    }},
    "evidence": [
        {{"paper_id": "...", "page_numbers": [1, 2], "snippet": "..."}}
    ],
    "insufficiency": {{
        "flag": false,
        "missing_fields": []
    }}
}}"""

        try:
            resp_text = self._call_llm(prompt).strip()
            if resp_text.startswith("```json"):
                resp_text = resp_text[7:]
            if resp_text.startswith("```"):
                resp_text = resp_text[3:]
            if resp_text.endswith("```"):
                resp_text = resp_text[:-3]
            return json.loads(resp_text)
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {"narrative_answer": f"Error: {e}", "insufficiency": {"flag": True}}

    def check_grounding(self, answer: str, context: str) -> Dict[str, Any]:
        """Fact-check the generated answer against the retrieved context."""
        prompt = f"""You are a fact-checker. Given a CONTEXT (the only truth) and an ANSWER, 
rate how well the answer is grounded in the context.

CONTEXT:
{context[:12000]}  # Ensure it fits context limits

ANSWER:
{answer}

OUTPUT FORMAT (JSON):
{{
  "grounding_score": 0.0 to 1.0 (float),
  "hallucinated_claims": ["claim A was not found in context", "..."],
  "grounded_claims": ["claim B is directly supported", "..."]
}}"""
        try:
            resp_text = self._call_llm(prompt).strip()
            if resp_text.startswith("```json"):
                resp_text = resp_text[7:]
            if resp_text.startswith("```"):
                resp_text = resp_text[3:]
            if resp_text.endswith("```"):
                resp_text = resp_text[:-3]
            return json.loads(resp_text)
        except Exception as e:
            print(f"Error checking grounding: {e}")
            return {"grounding_score": 0.0, "hallucinated_claims": [f"Error: {e}"], "grounded_claims": []}
