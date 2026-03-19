"""
给多个官方 Agent 示例复用的工具集（真实外部 API）。

工具约定：
- `rag_cheatsheet`: 使用 Tavily 做检索（需要环境变量 `TAVILY_API_KEY`）
- `simple_fact_lookup`: 使用 Wikipedia 做通用事实查询（通常不需要 Key）
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import Tool


def _tavily_search(query: str, *, max_results: int = 5) -> Any:
    """
    真实 Tavily 检索。
    兼容不同版本的 `langchain-community` 导入路径与调用方式。
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("缺少环境变量 TAVILY_API_KEY（Tavily 检索需要）。")

    # 优先：utilities wrapper（常见）
    try:
        from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper  # type: ignore

        wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)
        results_fn = getattr(wrapper, "results", None)
        if callable(results_fn):
            return results_fn(query, max_results=max_results)
        return wrapper.run(query)
    except Exception:
        pass

    # 回退：tools 版本
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore

        tool_obj = TavilySearchResults(max_results=max_results, tavily_api_key=api_key)
        return tool_obj.invoke({"query": query})
    except Exception:
        pass

    # 兜底：官方 tavily-python
    try:
        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=api_key)
        return client.search(query=query, max_results=max_results)
    except Exception as e:
        raise RuntimeError(f"Tavily 工具不可用：{type(e).__name__}: {e}") from e


def _wikipedia_summary(topic: str, *, lang: str = "zh", top_k: int = 3) -> str:
    """
    真实 Wikipedia 查询（无 Key）。
    """
    try:
        from langchain_community.utilities.wikipedia import WikipediaAPIWrapper  # type: ignore

        wiki = WikipediaAPIWrapper(top_k_results=top_k, lang=lang)
        return wiki.run(topic)
    except Exception as e:
        raise RuntimeError(f"Wikipedia 工具不可用：{type(e).__name__}: {e}") from e

def _rag_cheatsheet_impl(query: str) -> str:
    """用 Tavily 检索“RAG”相关资料，返回可直接引用的要点。"""
    q = (query or "").strip() or "RAG 检索增强生成 工作原理 应用"
    results = _tavily_search(q, max_results=5)
    return f"[Tavily 检索结果]\n{results}"


def _simple_fact_lookup_impl(topic: str) -> str:
    """用 Wikipedia 做通用事实查询（优先中文）。"""
    t = (topic or "").strip()
    if not t:
        raise ValueError("topic 不能为空")

    # 对常见中文术语做一次轻量 normalize
    normalized: str = t
    if t.lower() in {"rag", "retrieval augmented generation", "retrieval-augmented generation"}:
        normalized = "檢索增強生成"

    summary = _wikipedia_summary(normalized, lang="zh", top_k=3)
    # 某些条目中文可能缺失，再回退英文
    if not summary or "No good Wikipedia Search Result" in summary:
        summary = _wikipedia_summary(normalized, lang="en", top_k=3)
    return f"[Wikipedia]\n{summary}"


# 用 Tool（单字符串入参）而非 @tool（结构化入参），避免不同 Agent/Executor 的 schema 兼容问题。
rag_cheatsheet = Tool.from_function(
    name="rag_cheatsheet",
    description="使用 Tavily 搜索并返回与查询相关的网页要点/摘要（需要 TAVILY_API_KEY）。输入为查询字符串。",
    func=_rag_cheatsheet_impl,
)

simple_fact_lookup = Tool.from_function(
    name="simple_fact_lookup",
    description="使用 Wikipedia 查询某个概念/术语的摘要（通常无需 Key）。输入为主题字符串。",
    func=_simple_fact_lookup_impl,
)

