from pydantic import BaseModel, Field
from typing import List, Optional

class PaperInput(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    shape_or_example: Optional[str] = None

class PaperOutput(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    shape_or_example: Optional[str] = None

class Result(BaseModel):
    dataset: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[str] = None
    baseline_name: Optional[str] = None
    delta_if_stated: Optional[str] = None

class MethodExtractions(BaseModel):
    name: Optional[str] = None
    purpose_one_sentence: Optional[str] = None
    inputs: List[PaperInput] = Field(default_factory=list)
    outputs: List[PaperOutput] = Field(default_factory=list)
    core_idea: Optional[str] = None
    algorithm_steps: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    hyperparameters: List[str] = Field(default_factory=list)
    code_or_data_links: List[str] = Field(default_factory=list)

class Evaluation(BaseModel):
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    results: List[Result] = Field(default_factory=list)

class GlossaryTerm(BaseModel):
    term: Optional[str] = None
    definition: Optional[str] = None

class Evidence(BaseModel):
    field: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    snippet: Optional[str] = None

class PaperExtraction(BaseModel):
    paper_id: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    venue: Optional[str] = None
    year: Optional[str] = None
    doi_or_arxiv: Optional[str] = None
    method: Optional[MethodExtractions] = None
    evaluation: Optional[Evaluation] = None
    comparisons: List[str] = Field(default_factory=list)
    applications: List[str] = Field(default_factory=list)
    glossary: List[GlossaryTerm] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    evidence_map: List[Evidence] = Field(default_factory=list)
