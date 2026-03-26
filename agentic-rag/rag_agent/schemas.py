from typing import List
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(description="Indicates if the user's question is clear and answerable")
    questions: List[str] = Field(description="List of rewritten, self-contained queries")
    clarification_needed: str = Field(description="Explanation of why the question is not clear and answerable")
