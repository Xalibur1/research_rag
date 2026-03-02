import os
import json
from google import genai
from google.genai import types
from typing import Dict, Any, List
from graph_index import HybridIndex
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

class RAGGenerator:
    def __init__(self, index: HybridIndex, model_name="gemini-2.5-flash"):
        self.index = index
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def generate_answer(self, query: str, top_k_chunks: int = 5, top_k_graph: int = 3) -> Dict[str, Any]:
        # 1. Retrieve evidence
        text_results = self.index.search_chunks(query, top_k=top_k_chunks)
        graph_results = self.index.search_graph(query, top_k=top_k_graph)
        
        # 2. Build Context
        context_str = "--- TEXT CHUNKS ---\n"
        for chunk, score in text_results:
            context_str += f"[Pages {chunk.page_start}-{chunk.page_end}, Section: {chunk.section}]\n"
            context_str += f"{chunk.text}\n\n"
            
        context_str += "\n--- KNOWLEDGE GRAPH NEIGHBORHOODS ---\n"
        for node_id, subgraph in graph_results.items():
            context_str += f"Graph matches around {node_id}:\n"
            for link in subgraph.get("links", []):
                source = link.get("source")
                target = link.get("target")
                rel = link.get("relation", "connected_to")
                context_str += f"  {source} --[{rel}]--> {target}\n"
            context_str += "\n"

        # 3. Prompt for the final structured output
        prompt = f"""
        You are a research-graph assistant for university students. 
        Your goal is to answer the user's question about the uploaded research paper(s).
        Prioritize explaining the implemented method and its inputs/outputs in plain English.
        Provide a concise reasoning summary first, then the final narrative, then structured JSON facts, and finally an evidence map citing exact pages.
        If the context does not contain enough information, mark 'insufficiency.flag' as true and list missing fields.
        
        USER QUESTION: "{query}"
        
        RETRIEVED CONTEXT (Text Chunks with Page Numbers & Knowledge Graph fragments):
        {context_str}
        
        OUTPUT FORMAT (JSON):
        {{
            "reasoning_summary": "2-4 sentences explaining how retrieved evidence supports the answer.",
            "narrative_answer": "Student-friendly explanation; 150-250 words. Avoid jargon or define it briefly.",
            "structured_summary": {{
                "paper_id": "...",
                "title": "...",
                "method": {{
                    "name": "...",
                    "inputs": [{{"name": "...", "type": "...", "example": "..."}}],
                    "outputs": [{{"name": "...", "type": "...", "example": "..."}}],
                    "core_idea": "...",
                    "algorithm_steps": ["..."]
                }},
                "evaluation": {{
                    "datasets": ["..."],
                    "metrics": ["..."],
                    "results": [{{"dataset": "...", "metric": "...", "value": "...", "baseline": "..."}}]
                }},
                "assumptions": ["..."],
                "limitations": ["..."],
                "glossary": [{{"term": "...", "definition": "..."}}]
            }},
            "evidence": [
                {{"paper_title": "...", "page_numbers": [1, 2], "snippet": "..."}}
            ],
            "insufficiency": {{
                "flag": false,
                "missing_fields": []
            }}
        }}
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        resp_text = response.text.strip()
        if resp_text.startswith("```json"):
            resp_text = resp_text[7:-3]
        elif resp_text.startswith("```"):
            resp_text = resp_text[3:-3]
            
        try:
            return json.loads(resp_text)
        except json.JSONDecodeError:
            print("Failed to parse LLM Output as JSON. Raw output:")
            print(resp_text)
            return {}
