"""
使用“Plan-and-Execute”框架完成统一的 RAG 讲解任务。

思想：先让模型产出一个【计划】，再严格按照计划逐步执行，每一步都可以再次调用 LLM。
"""

from typing import List

from langchain_core.prompts import ChatPromptTemplate

try:
    # 作为包运行：python -m agentFramework.plan_and_execute
    from .common_task import (
        build_llm,
        build_task_prompt,
        DEFAULT_QUESTION,
        pretty_print,
    )
except ImportError:
    # 作为脚本在当前目录运行：python plan_and_execute.py
    from common_task import (
        build_llm,
        build_task_prompt,
        DEFAULT_QUESTION,
        pretty_print,
    )


def plan_phase(question: str) -> List[str]:
    """
    规划阶段：只让模型思考“要怎么讲解”，而不是直接给最终答案。
    """
    llm = build_llm()
    plan_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你现在处于【规划阶段】。\n"
                    "不要直接回答问题内容，而是列出一个清晰的讲解计划（2-4 步即可）。\n"
                    "每一步说明这一段要讲什么，用简短的中文说明。"
                ),
            ),
            ("human", "问题是：{question}\n请给出讲解计划。"),
        ]
    )
    plan = (plan_prompt | llm).invoke({"question": question}).content
    # 简单切分为多步，这里只做示例，不做复杂解析
    steps = [s.strip() for s in plan.split("\n") if s.strip()]
    return steps


def execute_phase(question: str, steps: List[str]) -> str:
    """
    执行阶段：针对每个计划步骤再次调用 LLM，最后合并为完整回答。
    """
    llm = build_llm()
    task_prompt = build_task_prompt()

    sub_answers: List[str] = []
    for idx, step in enumerate(steps, start=1):
        sub_question = (
            f"总问题是：{question}\n"
            f"当前是计划中的第 {idx} 步：{step}\n"
            "请根据当前这一步的目标，输出本段内容。"
        )
        msg = task_prompt | llm
        resp = msg.invoke({"question": sub_question})
        sub_answers.append(resp.content)

    return "\n\n".join(sub_answers)


def run_plan_and_execute(question: str = DEFAULT_QUESTION) -> None:
    steps = plan_phase(question)
    answer = execute_phase(question, steps)
    pretty_print("Plan & Execute 输出", answer)


if __name__ == "__main__":
    run_plan_and_execute()

