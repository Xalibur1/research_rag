from typing import Annotated, List, Optional
from pydantic import BaseModel, BeforeValidator, Field
import json

def coerce_str(v):
    if v is None: return None
    if isinstance(v, (dict, list)): return json.dumps(v)
    return str(v)

RobustStr = Annotated[str, BeforeValidator(coerce_str)]

class TestModel(BaseModel):
    name: Optional[RobustStr] = None
    items: List[RobustStr] = Field(default_factory=list)

try:
    m = TestModel(**{"name": {"first": "John"}, "items": [{"id": 1}, "two", 3]})
    print("Parsed:", m)
except Exception as e:
    print("Error:", e)
