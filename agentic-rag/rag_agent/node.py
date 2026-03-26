from rag_agent.graph_state import State, AgentState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage, ToolMessage
import config
from rag_agent.prompts import *
from rag_agent.schemas import QueryAnalysis
from langgraph.types import Command
from typing import Literal, Set
from utils import estimate_context_tokens

FINAL_ANSWER_ON_LIMIT = "已达到工具调用或迭代上限，请开启新的聊天窗口。"

def summarize_history(state: State, llm):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    relevant_msgs = [
        msg for msg in state["messages"][:-1] 
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    conversion = "Conversation history:\n"
    for msg in relevant_msgs[-config.HISTORY_WINDOW:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversion += f"{role}: {msg.content}\n"
    
    summary_response = llm.with_config(temperature=config.LLM_SUMMARY_TEMPERATURE).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversion)])


    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State, llm):
    last_msg = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_msg.content}\n"
    llm_with_structure = llm.with_config(temperature=config.LLM_REWRITE_QUERY_TEMPERATURE).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])
    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {"questionIsClear": response.is_clear, "rewrittenQuestions": response.questions, "messages": delete_all, "originalQuery": last_msg.content}

    clarification = response.clarification_needed if response.clarification_needed else "I need more information to understand your question."
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    return {}

def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        question = state.get("question", "").strip()
        human_msg = HumanMessage(content=question)
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}
    
def fallback_response(state: AgentState, llm):
    seen = set()
    unique_contents = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.content not in seen:
            seen.add(msg.content)
            unique_contents.append(msg.content)
    
    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}


def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "retrieve_parent_chunks":
                    raw = tool_call["args"].get("parent_id") or tool_call["args"].get("id") or tool_call["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{id}" for id in raw)
                elif tool_call["name"] == "search_child_chunks":
                    query = tool_call["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens(HumanMessage(content=state.get("context_summary", "")))
    current_token = current_token_messages + current_token_summary

    max_allowed = config.BASE_TOKEN_THRESHOLD + int(current_token_summary * config.TOKEN_GROWTH_FACTOR)
    goto = "compress_context" if current_token > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState, llm):
    messages = state["messages"]
    if not messages:
        return {}
    
    existing_summary = state.get("context_summary", "").strip()

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"
    
    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT]{tool_calls_info}\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT - {tool_name}]\n{msg.content or '(tool response only)'}\n\n"
        
    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do not repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q.replace('search::', '')}" for q in search_queries) + "\n"
        conversation_text += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}


def collect_answer(state: AgentState):
    last = state["messages"][-1] if state["messages"] else None
    is_valid_answer = isinstance(last, AIMessage) and not getattr(last, "tool_calls", None)
    answer = last.content if is_valid_answer else "抱歉，我暂时无法基于文档生成答案。"
    return {
        "agent_answers": [{"index": state.get("question_index", 0), "answer": answer}],
    }

def aggregate_answers(state: State, llm):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers collected.")]}
    
    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])
    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}