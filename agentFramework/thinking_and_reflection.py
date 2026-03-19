"""
“Thinking & Reflection” 框架示例。

思想：
1. 第一次调用：只让模型“思考/推理草稿”（Thought），不要输出给用户。
2. 第二次调用：把前面的思考草稿当作材料，让模型进行“反思与润色”，输出更清晰的最终答案。
"""

from langchain_core.prompts import ChatPromptTemplate

try:
    from .common_task import (
        build_llm,
        build_task_prompt,
        DEFAULT_QUESTION,
        pretty_print,
    )
except ImportError:
    from common_task import (
        build_llm,
        build_task_prompt,
        DEFAULT_QUESTION,
        pretty_print,
    )


def thinking_phase(question: str) -> str:
    """
    让模型先写出一份“只给自己看的”思考草稿。
    """
    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你处于【思考阶段】。\n"
                    "请详细写出你如何理解并解释 RAG 的思考过程，可以包含要点、结构、例子等。\n"
                    "注意：这些内容是给你自己看的草稿，不会直接展示给用户。"
                ),
            ),
            ("human", "问题是：{question}\n请写出你的详细思考草稿。"),
        ]
    )
    resp = (prompt | llm).invoke({"question": question})
    return resp.content


def reflection_phase(question: str, draft: str) -> str:
    """
    让模型对草稿进行反思、取舍和重写，得到最终更清晰的回答。
    """
    llm = build_llm()
    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你现在进入【反思与润色阶段】。\n"
                    "下面是你刚才写的思考草稿，请基于其中的要点进行筛选、重组和简化，"
                    "给非技术同事一个更清晰友好的解释。"
                ),
            ),
            (
                "human",
                "原始问题：{question}\n\n下面是你的思考草稿：\n{draft}\n\n"
                "现在请输出最终给用户看的回答，依然控制在 3-4 句话，用中文。",
            ),
        ]
    )
    resp = (final_prompt | llm).invoke({"question": question, "draft": draft})
    return resp.content


def run_thinking_and_reflection(question: str = DEFAULT_QUESTION) -> None:
    draft = thinking_phase(question)
    final_answer = reflection_phase(question, draft)
    pretty_print("Thinking & Reflection 输出", final_answer)


if __name__ == "__main__":
    run_thinking_and_reflection()

