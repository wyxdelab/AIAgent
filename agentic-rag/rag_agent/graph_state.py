from langgraph.graph import MessagesState
from typing import List, Annotated, Set
import operator

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

def accumulate_or_reset(a: List[dict], b: List[dict]) -> List[dict]:
    if b and any(item.get("__reset__") for item in b):
        return []
    return a + b

class State(MessagesState):
    conversation_summary: str = ""
    questionIsClear: bool = False
    rewrittenQuestions: List[str] = []
    originalQuery: str = "" 
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    

class AgentState(MessagesState):
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    retrieval_keys: Annotated[Set[str], set_union] = set()
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0