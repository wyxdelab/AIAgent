from typing import Literal
from rag_agent.graph_state import State
import config
from langgraph.types import Send

def route_after_orchestrator(state: State) -> Literal["tools", "collect_answer"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= config.MAX_ITERATIONS or tool_count > config.MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"
    
    return "tools"

def route_after_rewrite_query(state: State) -> Literal["agent", "request_clarification"]:
    if not state.get("questionIsClear", False):
        return "request_clarification"
    return [Send("agent", {"question": query, "question_index": idx, "messages":[]})
    for idx, query in enumerate(state["rewrittenQuestions"])]