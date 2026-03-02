import os
import json
from google import genai
from google.genai import types
from typing import List
from schemas import PaperExtraction
from parser import DocumentChunk
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

class Extractor:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def extract_from_chunks(self, chunks: List[DocumentChunk], paper_id: str) -> PaperExtraction:
        full_text = ""
        for i, chunk in enumerate(chunks):
            full_text += f"\n--- CHUNK {i+1} | PAGES {chunk.page_start}-{chunk.page_end} | SECTION: {chunk.section} ---\n"
            full_text += chunk.text + "\n"

        prompt = f"""
        You are an expert academic researcher. Please analyze the following research paper text and extract the structured information precisely matching the JSON schema provided below.
        
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
            "paper_id": "...",
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
            
        data = json.loads(resp_text)
        return PaperExtraction(**data)
