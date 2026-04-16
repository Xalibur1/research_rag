import json
from typing_extensions import Annotated
from pydantic import BaseModel, Field, BeforeValidator
from typing import List, Optional

def coerce_str(v):
    if v is None: return None
    if isinstance(v, (dict, list)): return json.dumps(v)
    return str(v)

RobustStr = Annotated[str, BeforeValidator(coerce_str)]

class PaperInput(BaseModel):
    name: Optional[RobustStr] = None
    type: Optional[RobustStr] = None
    shape_or_example: Optional[RobustStr] = None

class PaperOutput(BaseModel):
    name: Optional[RobustStr] = None
    type: Optional[RobustStr] = None
    shape_or_example: Optional[RobustStr] = None

class Result(BaseModel):
    dataset: Optional[RobustStr] = None
    metric: Optional[RobustStr] = None
    value: Optional[RobustStr] = None
    baseline_name: Optional[RobustStr] = None
    delta_if_stated: Optional[RobustStr] = None

class MethodExtractions(BaseModel):
    name: Optional[RobustStr] = None
    purpose_one_sentence: Optional[RobustStr] = None
    inputs: List[PaperInput] = Field(default_factory=list)
    outputs: List[PaperOutput] = Field(default_factory=list)
    core_idea: Optional[RobustStr] = None
    algorithm_steps: List[RobustStr] = Field(default_factory=list)
    assumptions: List[RobustStr] = Field(default_factory=list)
    limitations: List[RobustStr] = Field(default_factory=list)
    hyperparameters: List[RobustStr] = Field(default_factory=list)
    code_or_data_links: List[RobustStr] = Field(default_factory=list)

class Evaluation(BaseModel):
    datasets: List[RobustStr] = Field(default_factory=list)
    metrics: List[RobustStr] = Field(default_factory=list)
    results: List[Result] = Field(default_factory=list)

class GlossaryTerm(BaseModel):
    term: Optional[RobustStr] = None
    definition: Optional[RobustStr] = None

class Evidence(BaseModel):
    field: Optional[RobustStr] = None
    page_numbers: List[int] = Field(default_factory=list)
    snippet: Optional[RobustStr] = None

class PaperExtraction(BaseModel):
    paper_id: Optional[RobustStr] = None
    title: Optional[RobustStr] = None
    authors: List[RobustStr] = Field(default_factory=list)
    venue: Optional[RobustStr] = None
    year: Optional[RobustStr] = None
    doi_or_arxiv: Optional[RobustStr] = None
    method: Optional[MethodExtractions] = None
    evaluation: Optional[Evaluation] = None
    comparisons: List[RobustStr] = Field(default_factory=list)
    applications: List[RobustStr] = Field(default_factory=list)
    glossary: List[GlossaryTerm] = Field(default_factory=list)
    citations: List[RobustStr] = Field(default_factory=list)
    evidence_map: List[Evidence] = Field(default_factory=list)

