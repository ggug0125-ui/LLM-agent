import os
import re
import sqlite3
from pathlib import Path

from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama


ROOT_DIR = Path(__file__).resolve().parents[2]
CHROMA_SQLITE = ROOT_DIR / "ch09" / "chroma_store" / "chroma.sqlite3"


def _load_documents():
    con = sqlite3.connect(CHROMA_SQLITE)
    rows = con.execute(
        """
        select doc.string_value as document,
               source.string_value as source,
               page.int_value as page
        from embeddings e
        join embedding_metadata doc on doc.id = e.id and doc.key = 'chroma:document'
        left join embedding_metadata source on source.id = e.id and source.key = 'source'
        left join embedding_metadata page on page.id = e.id and page.key = 'page'
        order by e.id
        """
    ).fetchall()
    con.close()

    return [
        Document(
            page_content=document,
            metadata={"source": source or "", "page": page if page is not None else ""},
        )
        for document, source, page in rows
        if document
    ]


DOCUMENTS = _load_documents()


def _terms(text):
    words = re.findall(r"[A-Za-z0-9_]+|[가-힣]+", text.lower())
    stopwords = {"나는", "안녕", "이야", "입니다", "해줘", "알려줘"}
    return [word for word in words if len(word) > 1 and word not in stopwords]


def _retrieve(input_text):
    query = input_text["input"] if isinstance(input_text, dict) else str(input_text)
    terms = _terms(query)

    if not terms:
        return []

    scored = []
    for doc in DOCUMENTS:
        content = doc.page_content.lower()
        score = sum(content.count(term) for term in terms)
        if score:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:3]]


retriever = RunnableLambda(_retrieve)

llm = ChatOllama(
    model=os.getenv("OLLAMA_CHAT_MODEL", "deepseek-r1:7b")
)

prompt = ChatPromptTemplate.from_template(
    """
You are a document-based assistant.

If the context is empty and the user is greeting you, reply with a short friendly greeting.
If the context is empty and the user asks about documents, say that you could not find relevant document content.
Otherwise, answer using the context below.

Context:
{context}

Question:
{input}
"""
)

chain = create_retrieval_chain(
    retriever,
    prompt | llm | StrOutputParser(),
)
