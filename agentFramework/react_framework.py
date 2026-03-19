"""
ReAct（Reason + Act）风格示例。

思想：在一次对话中交替出现：
- Thought（思考）
- Action（调用工具，如检索 / 查询）
- Observation（工具返回结果）
最终再输出 Answer。

这里为了简单起见，只模拟一个“知识库检索”工具，并用字符串拼接展示 ReAct 轨迹。
"""

from typing import Dict
import os

from langchain_core.prompts import ChatPromptTemplate

# Tavily 在 langchain-community 不同版本里导入路径可能不同；
# 同时也提供直接使用官方 tavily-python 的兜底实现。
TavilySearchAPIWrapper = None
TavilySearchResults = None
try:  # 新/常见路径
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper  # type: ignore
except Exception:
    try:  # 另一种版本提供的是 Tool
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
    except Exception:
        TavilySearchAPIWrapper = None
        TavilySearchResults = None

try:
    from .common_task import (
        build_llm,
        DEFAULT_QUESTION,
        pretty_print,
    )
except ImportError:
    from common_task import (
        build_llm,
        DEFAULT_QUESTION,
        pretty_print,
    )


def fake_search_tool(query: str) -> str:
    """
    使用 Tavily 的检索工具进行实时搜索。
    需要环境变量 TAVILY_API_KEY。若缺失或调用失败，会给出友好提示。
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "[检索失败] 未检测到环境变量 TAVILY_API_KEY。"
            "请到 https://tavily.com 获取 API Key，并执行：\n"
            "export TAVILY_API_KEY='your_key_here'"
        )
    try:
        # 优先使用 langchain-community 中的 Wrapper（如果可用）
        if TavilySearchAPIWrapper is not None:
            # 注意：不同版本的 TavilySearchAPIWrapper 对构造参数支持不同；
            # 你当前版本不允许在构造函数里传 max_results（会触发 pydantic extra_forbidden）。
            tavily = TavilySearchAPIWrapper(tavily_api_key=api_key)

            # 优先调用 results(query, max_results=...)（如果该方法存在）
            results_fn = getattr(tavily, "results", None)
            if callable(results_fn):
                result = results_fn(query, max_results=5)
                return f"[Tavily 检索结果]\n{result}"

            # 否则回退到 run(query)（部分版本只提供 run）
            result_text = tavily.run(query)
            return f"[Tavily 检索结果]\n{result_text}"

        # 兼容某些版本：提供的是 TavilySearchResults 工具
        if TavilySearchResults is not None:
            tool = TavilySearchResults(max_results=5, tavily_api_key=api_key)
            result = tool.invoke({"query": query})
            return f"[Tavily 检索结果]\n{result}"

        # 兜底：直接使用官方 tavily-python（pip install tavily-python）
        try:
            from tavily import TavilyClient  # type: ignore
        except Exception:
            return (
                "[检索不可用] 当前环境的 langchain-community 未包含 Tavily 相关封装，"
                "且未安装官方 tavily-python。\n"
                "请执行：pip install -U langchain-community tavily-python\n"
                "然后重试。"
            )

        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=5)
        return f"[Tavily 检索结果]\n{resp}"
    except Exception as e:
        return f"[检索异常] {type(e).__name__}: {e}"


def build_react_prompt() -> ChatPromptTemplate:
    """
    让 LLM 以 ReAct 的格式思考：Thought / Action / Observation / Answer。
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是一个使用 ReAct 模式的 Agent，需要显式写出：\n"
                    "Thought: ...\nAction: search[查询内容]\nObservation: ...\nThought: ...\nAnswer: ...\n\n"
                    "当你决定使用 Action 搜索时，动作格式必须是：Action: search[查询内容]\n"
                    "注意：这只是轨迹演示，最终 Answer 仍然要用中文、3-4 句话解释清楚 RAG。"
                ),
            ),
            ("human", "用户问题：{question}\n请按 ReAct 轨迹先思考再回答。"),
        ]
    )


def run_react(question: str = DEFAULT_QUESTION) -> None:
    llm = build_llm()
    prompt = build_react_prompt()

    # 第一次调用：让 LLM 决定是否要调用工具（并给出 Action 格式）
    first_step = (prompt | llm).invoke({"question": question}).content

    # 简单解析 Action: search[...] 这一行，现实中可以做更健壮的解析
    action_query = None
    for line in first_step.split("\n"):
        line = line.strip()
        if line.startswith("Action:") and "search" in line:
            # 例：Action: search[什么是 RAG]
            start = line.find("search[")
            end = line.rfind("]")
            if start != -1 and end != -1 and end > start + len("search["):
                action_query = line[start + len("search[") : end]
                break

    observation = ""
    if action_query:
        observation = fake_search_tool(action_query)

    # 第二次调用：把第一步思考 + 工具 Observation 一起喂给 LLM，让它给最终 Answer
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你将看到之前的 ReAct 轨迹以及工具返回结果，请在最后补全清晰的 Answer。",
            ),
            (
                "human",
                "之前的轨迹：\n{trajectory}\n\n"
                "工具 Observation：\n{observation}\n\n"
                "请基于这些信息，输出最终的 Answer（中文 3-4 句话）。",
            ),
        ]
    )

    trajectory = first_step
    second_step = (answer_prompt | llm).invoke(
        {"trajectory": trajectory, "observation": observation}
    ).content

    # 为了演示，我们把两次调用的内容都打印出来，便于理解 ReAct 的中间过程。
    combined: Dict[str, str] = {
        "first_step": first_step,
        "observation": observation,
        "final": second_step,
    }
    text = (
        "First Step (LLM 轨迹草稿):\n"
        f"{combined['first_step']}\n\n"
        "Observation (工具返回):\n"
        f"{combined['observation']}\n\n"
        "Final Answer:\n"
        f"{combined['final']}\n"
    )
    pretty_print("ReAct 输出", text)


if __name__ == "__main__":
    run_react()

