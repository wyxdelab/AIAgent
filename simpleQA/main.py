from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import tiktoken

# 初始化 Embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 文档内容
docs = ["Deep learning is a subset of machine learning that uses neural networks to model complex patterns in data. RAG stands for Retrieval-Augmented Generation."]

# 建立向量数据库
vectorstore = Chroma.from_texts(docs, embedding=embeddings)

# 初始化 LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 从向量数据库创建检索器
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 使用 LCEL 构建 RAG 管道（不依赖 langchain.chains）
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based only on the following context:\n\n{context}"),
    ("human", "{input}"),
])
qa = (
    RunnablePassthrough.assign(context=lambda x: retriever.invoke(x["input"]))
    | RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | llm
    | StrOutputParser()
)

# 用户问题
user_question = "What is RAG in AI?"

enc = tiktoken.encoding_for_model("gpt-4o-mini")

# 构建输入文本（用于 token 计数）
retrieved_docs = retriever.invoke(user_question)
retrieved_docs_text = " ".join([d.page_content for d in retrieved_docs])
full_prompt_text = f"Question: {user_question}\nContext: {retrieved_docs_text}"

tokens = enc.encode(full_prompt_text)
print(f"Token count: {len(tokens)}")

# 假设 gpt-4o-mini context window = 8192
MAX_CONTEXT = 8192
if len(tokens) > MAX_CONTEXT:
    print("Warning: context window exceeded, need to truncate or summarize")

result = qa.invoke({"input": user_question})

print("Answer:")
print(result)