"""
使用 “Self-Ask with Search” 风格的 Agent 完成同一任务。

思想：让 Agent 在回答前，可以主动提出若干子问题（自问自答），
再把这些中间结论整合成最终答案。这里用简单的两阶段示例代替真实的“搜索”。
"""

from typing import List

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


def generate_sub_questions(question: str) -> List[str]:
    """
    通过 LLM 生成若干“自问”的子问题。
    """
    llm = build_llm()
    q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你现在在做【自问分解】。\n"
                    "请把原问题拆成 2-4 个更小的子问题，用于帮助理解和解释。\n"
                    "只输出子问题列表，每行一个，不要回答内容。"
                ),
            ),
            ("human", "原问题是：{question}\n请列出子问题。"),
        ]
    )
    text = (q_prompt | llm).invoke({"question": question}).content
    subs = [s.strip() for s in text.split("\n") if s.strip()]
    return subs


def answer_sub_questions(sub_questions: List[str]) -> List[str]:
    """
    对每个子问题进行“自问自答”。
    """
    llm = build_llm()
    task_prompt = build_task_prompt()
    answers: List[str] = []
    for sq in sub_questions:
        resp = (task_prompt | llm).invoke({"question": sq})
        answers.append(f"子问题：{sq}\n回答：{resp.content}")
    return answers


def synthesize_final_answer(question: str, sub_qa: List[str]) -> str:
    """
    让 LLM 读取所有自问自答的中间结论，再产出一个最终整合回答。
    """
    llm = build_llm()
    synth_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你拿到了若干关于同一主题的“子问题-回答”对，请基于这些内容综合出一个最终答案。",
            ),
            (
                "human",
                "原始问题：{question}\n\n下面是若干子问题及其回答：\n{sub_qa}\n\n"
                "请综合它们，用 3-4 句话、中文，给出最终解释。",
            ),
        ]
    )
    text_block = "\n\n".join(sub_qa)
    resp = (synth_prompt | llm).invoke(
        {"question": question, "sub_qa": text_block}
    )
    return resp.content


def run_self_ask(question: str = DEFAULT_QUESTION) -> None:
    subs = generate_sub_questions(question)
    sub_qa = answer_sub_questions(subs)
    final_answer = synthesize_final_answer(question, sub_qa)
    pretty_print("Self-Ask 输出", final_answer)


if __name__ == "__main__":
    run_self_ask()

