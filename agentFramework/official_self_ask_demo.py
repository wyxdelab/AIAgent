"""
LangChain 官方风格：Self-Ask with Search 示例（同一个 prompt）。

运行：
python -m agentFramework.official_self_ask_demo
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from .common_task import DEFAULT_QUESTION, build_llm, pretty_print
    from .official_tools import rag_cheatsheet
except ImportError:  # pragma: no cover
    from common_task import DEFAULT_QUESTION, build_llm, pretty_print
    from official_tools import rag_cheatsheet


def _build_self_ask_executor():
    """
    适配 langchain==1.2.12：
    - v1 不再暴露旧版 `SelfAskWithSearchAgent` / `create_self_ask_with_search_agent`
    - 用 LCEL 实现 Self-Ask 控制流：分解子问题 →（工具）查资料 → 综合回答
    """

    llm = build_llm()
    from langchain_core.prompts import ChatPromptTemplate

    decompose = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是 Self-Ask Agent。面对问题时，先提出 2-4 个你需要弄清的子问题。"
                    "只输出子问题列表，每行一个，不要直接回答。"
                ),
            ),
            ("human", "{question}"),
        ]
    )

    synth = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你将看到子问题与工具返回资料，请综合成 3-4 句中文回答，并给一个简单应用例子。",
            ),
            ("human", "原问题：{question}\n\n子问题与资料：\n{notes}"),
        ]
    )

    def invoke(payload: Dict[str, Any]) -> Dict[str, str]:
        question = payload["input"]
        sub_q_text = (decompose | llm).invoke({"question": question}).content
        sub_qs = [s.strip("- ").strip() for s in sub_q_text.splitlines() if s.strip()]
        notes_parts = []
        for sq in sub_qs[:6]:
            # `rag_cheatsheet` 在 v1 中是 Tool（单字符串入参），直接传字符串即可
            ref = rag_cheatsheet.invoke(sq)
            notes_parts.append(f"子问题：{sq}\n资料：\n{ref}")
        notes = "\n\n".join(notes_parts) if notes_parts else rag_cheatsheet.invoke(question)
        final = (synth | llm).invoke({"question": question, "notes": notes}).content
        return {"output": final}

    class _Wrapper:
        def invoke(self, payload: Dict[str, Any]) -> Dict[str, str]:
            return invoke(payload)

    return _Wrapper()


def run(question: str = DEFAULT_QUESTION) -> None:
    executor = _build_self_ask_executor()
    result = executor.invoke({"input": question})
    text = result.get("output") if isinstance(result, dict) else str(result)
    pretty_print("Official Self-Ask 输出", text)


if __name__ == "__main__":
    run()

