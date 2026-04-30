# 🔹 Ollama 기반 임베딩 (OpenAI 제거)
from langchain_ollama import OllamaEmbeddings, ChatOllama

# 🔹 벡터 DB
from langchain_chroma import Chroma

# 🔹 최신 체인
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# 1. Embedding (로컬)
# =========================
embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

# =========================
# 2. Vector DB 로드
# =========================
print("Loading existing Chroma store")

persist_directory = 'C:/Aiprojects/ch09/chroma_store'

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# =========================
# 3. Retriever
# =========================
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================
# 4. LLM (DeepSeek)
# =========================
llm = ChatOllama(
    model="deepseek-r1:latest"
)

# =========================
# 5. Prompt
# =========================
prompt = ChatPromptTemplate.from_template("""
너는 도시 정책 전문가야.
아래 문서를 기반으로 사용자 질문에 답변해라.

문서:
{context}

질문:
{input}
""")

# =========================
# 6. RAG Chain (핵심)
# =========================
chain = create_retrieval_chain(
    retriever,
    prompt | llm | StrOutputParser()
)

# https://visualstudio.microsoft.com/visual-cpp-build-tools/ 
# python.exe -m pip install --upgrade pip
# pip uninstall langchain langchain-core langchain-community langchain-ollama langsmith -y
# pip install langchain==0.2.16 langchain-core==0.2.38 langchain-community==0.2.16 langchain-ollama==0.1.3 langchain-chroma==0.1.2 chromadb==0.4.24
# streamlit run rag_deepseek.py              