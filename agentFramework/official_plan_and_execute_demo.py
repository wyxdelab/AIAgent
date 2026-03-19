"""
LangChain 官方风格：Plan-and-Execute 示例（同一个 prompt）。

运行：
python -m agentFramework.official_plan_and_execute_demo
"""

from __future__ import annotations

from typing import List

try:
    # 作为包运行：python -m agentFramework.official_plan_and_execute_demo
    from .common_task import DEFAULT_QUESTION, build_llm, build_task_prompt, pretty_print
    from .official_tools import rag_cheatsheet, simple_fact_lookup
except ImportError:  # pragma: no cover
    # 作为脚本运行：python official_plan_and_execute_demo.py
    from common_task import DEFAULT_QUESTION, build_llm, build_task_prompt, pretty_print
    from official_tools import rag_cheatsheet, simple_fact_lookup


def _build_plan_and_execute_chain():
    """
    尽量走官方 `langchain_experimental.plan_and_execute` 路径；
    若当前环境未安装 experimental 组件，则回退到“Plan -> 执行器”的等价 LCEL 组合。
    """

    llm = build_llm()
    tools = [rag_cheatsheet, simple_fact_lookup]

    try:
        from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor
        from langchain_experimental.plan_and_execute import load_chat_planner

        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        return PlanAndExecute(planner=planner, executor=executor, verbose=True)
    except Exception:
        # 回退：仍然保持 “先计划再执行”，只是不用 experimental 封装类
        from langchain_core.prompts import ChatPromptTemplate

        plan_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个规划器。请把任务拆成 2-4 步，步骤要具体、可执行，并可能使用工具。",
                ),
                ("human", "{question}"),
            ]
        )

        def plan(question: str) -> List[str]:
            text = (plan_prompt | llm).invoke({"question": question}).content
            steps = [s.strip("- ").strip() for s in text.splitlines() if s.strip()]
            return steps[:6]

        task_prompt = build_task_prompt()

        def execute(question: str) -> str:
            steps = plan(question)
            chunks: List[str] = []
            for i, step in enumerate(steps, start=1):
                tool_hint = rag_cheatsheet.invoke({"query": step})
                sub_q = (
                    f"总问题：{question}\n"
                    f"计划第 {i} 步：{step}\n\n"
                    f"可用工具返回的信息：\n{tool_hint}\n\n"
                    "请完成这一步对应的一段解释。"
                )
                chunks.append((task_prompt | llm).invoke({"question": sub_q}).content)
            return "\n\n".join(chunks)

        return execute


def run(question: str = DEFAULT_QUESTION) -> None:
    chain_or_callable = _build_plan_and_execute_chain()

    if hasattr(chain_or_callable, "invoke"):
        result = chain_or_callable.invoke({"input": question})
        # PlanAndExecute 的返回在不同版本里结构不一，这里做最小兼容
        text = (
            result.get("output") if isinstance(result, dict) else getattr(result, "output", None)
        ) or str(result)
    else:
        text = chain_or_callable(question)

    pretty_print("Official Plan & Execute 输出", text)


if __name__ == "__main__":
    run()

