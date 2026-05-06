import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


ROOT_DIR = Path(__file__).resolve().parents[1]
PERSIST_DIRECTORY = ROOT_DIR / "chroma_store_suwon"

load_dotenv(ROOT_DIR.parent / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Check Aiprojects/.env.")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key,
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
)

print(f"Loading Chroma store: {PERSIST_DIRECTORY}")
vectorstore = Chroma(
    persist_directory=str(PERSIST_DIRECTORY),
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an AI research assistant that answers based on the provided "
                "Suwon 2040 urban master plan context. Answer in Korean. "
                "If the answer is not in the context, say that it cannot be found "
                "in the document.\n\nContext:\n{context}"
            ),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = (
    create_stuff_documents_chain(llm, question_answering_prompt)
    | StrOutputParser()
)

query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            (
                "Rewrite the user's latest question as a clear Korean search query "
                "for retrieving passages from the Suwon 2040 urban master plan. "
                "Return only the rewritten query.\n\nQuestion:\n{query}"
            ),
        ),
    ]
)

query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()
