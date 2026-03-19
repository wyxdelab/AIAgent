"""
LangChain 官方风格：Thinking + Reflection（思考→反思→重写）示例（同一个 prompt）。

说明：
- Thinking/Reflection 更像一种“控制流/提示词编排”而不是内置 Agent 类；
  这里用 LCEL（prompt | llm）的方式做一个标准三段式：Draft -> Critique -> Rewrite。

运行：
python -m agentFramework.official_thinking_reflection_demo
"""

from __future__ import annotations

import os
import sys

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "_vendor")
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

try:
    from .common_task import DEFAULT_QUESTION, build_llm, pretty_print
except ImportError:  # pragma: no cover
    from common_task import DEFAULT_QUESTION, build_llm, pretty_print

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def run(question: str = DEFAULT_QUESTION) -> None:
    llm = build_llm()
    parser = StrOutputParser()

    # 1) Thinking / Draft（先写得更全，允许啰嗦）
    draft_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你处于思考阶段：请先写一份较完整的草稿来解释问题，"
                    "可以多一些要点、类比、流程描述。不要担心长度。"
                ),
            ),
            ("human", "{question}"),
        ]
    )

    # 2) Reflection / Critique（从“面向非技术同事、3-4 句、给例子”角度做批评）
    critique_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是审稿人，请针对下面草稿做批评改进建议。"
                    "关注：是否通俗、是否准确、是否 3-4 句话可表达、是否包含一个简单例子。"
                    "请用要点列出 3-6 条改进建议。"
                ),
            ),
            ("human", "原问题：{question}\n\n草稿：\n{draft}"),
        ]
    )

    # 3) Rewrite（严格按最终输出要求重写）
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "请基于草稿与改进建议，重写最终答案：中文 3-4 句话，通俗易懂，"
                    "并在最后给一个非常简单的实际使用场景示例。"
                ),
            ),
            ("human", "原问题：{question}\n\n草稿：\n{draft}\n\n改进建议：\n{critique}"),
        ]
    )

    draft = (draft_prompt | llm | parser).invoke({"question": question})
    critique = (critique_prompt | llm | parser).invoke({"question": question, "draft": draft})
    final = (rewrite_prompt | llm | parser).invoke(
        {"question": question, "draft": draft, "critique": critique}
    )

    text = (
        "Draft（思考草稿）：\n"
        f"{draft}\n\n"
        "Critique（反思建议）：\n"
        f"{critique}\n\n"
        "Final Answer（最终输出）：\n"
        f"{final}\n"
    )
    pretty_print("Official Thinking + Reflection 输出", text)


if __name__ == "__main__":
    run()

