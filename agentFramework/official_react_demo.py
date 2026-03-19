"""
LangChain 官方风格：ReAct Agent 示例（同一个 prompt）。

运行：
python -m agentFramework.official_react_demo
"""

from __future__ import annotations

try:
    from .common_task import DEFAULT_QUESTION, build_llm, pretty_print
    from .official_tools import rag_cheatsheet, simple_fact_lookup
except ImportError:  # pragma: no cover
    from common_task import DEFAULT_QUESTION, build_llm, pretty_print
    from official_tools import rag_cheatsheet, simple_fact_lookup


def _build_react_executor():
    """
    适配 langchain==1.2.12：
    - 传统的 AgentExecutor/create_react_agent/initialize_agent 在 v1 中不再作为公开入口
    - 推荐用 `langchain.agents.create_agent`（底层是 LangGraph 状态机）
    """
    from langchain.agents import create_agent

    llm = build_llm()
    tools = [rag_cheatsheet, simple_fact_lookup]

    system_prompt = (
        "你是一个使用 ReAct（Reason + Act）风格的智能体，会在需要时调用工具。\n"
        "要求：最终回答用中文 3-4 句话解释清楚，并给一个非常简单的实际使用场景示例。\n"
        "如果你调用工具，请先明确你的意图，再基于工具结果组织最终回答。"
    )

    return create_agent(model=llm, tools=tools, system_prompt=system_prompt, debug=True)


def run(question: str = DEFAULT_QUESTION) -> None:
    agent = _build_react_executor()
    from langchain_core.messages import HumanMessage

    state = agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = state.get("messages", []) if isinstance(state, dict) else []
    final_text = getattr(messages[-1], "content", None) if messages else str(state)
    pretty_print("Official ReAct 输出", final_text)


if __name__ == "__main__":
    run()

