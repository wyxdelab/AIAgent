from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    """
    创建一个统一的 LLM，用于四种 Agent 框架示例复用同一个模型和 prompt。
    """
    return ChatOpenAI(model_name=model, temperature=temperature)


def build_task_prompt() -> ChatPromptTemplate:
    """
    统一任务说明：
    - 领域：RAG（Retrieval-Augmented Generation）
    - 目标：用 3-4 句话给非技术同事讲明白什么是 RAG，并给一个简单示例。
    这样四种 Agent 只是“思考方式/控制流”不同，任务本身保持一致。
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是一名资深 AI 工程师，正在给非技术背景的产品同事解释 RAG。\n"
                    "请用简洁、通俗的方式回答下面的问题，回答长度控制在 3-4 句话。\n"
                    "使用中文回答，并在最后给出一个非常简单的实际使用场景示例。"
                ),
            ),
            ("human", "{question}"),
        ]
    )


DEFAULT_QUESTION = "什么是 RAG（检索增强生成），它在实际应用中是怎么工作的？"


def pretty_print(title: str, content: str) -> None:
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(content.strip())
    print("=" * 80)
