import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import retriever


def get_ai_response(messages, docs):
    return retriever.document_chain.stream(
        {
            "messages": messages,
            "context": docs,
        }
    )


st.title("수원 2040 도시기본계획 RAG 챗봇")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content="문서에 기반해 답변하는 AI 연구원입니다."),
        AIMessage(content="수원 2040 도시기본계획에 대해 무엇이 궁금하신가요?"),
    ]

for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        st.chat_message("system").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input("질문을 입력하세요"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.spinner("관련 문서를 검색 중입니다..."):
        augmented_query = retriever.query_augmentation_chain.invoke(
            {
                "messages": st.session_state["messages"],
                "query": prompt,
            }
        )
        docs = retriever.retriever.invoke(f"{prompt}\n{augmented_query}")

    st.caption(f"검색어: {augmented_query}")

    for doc in docs:
        source = doc.metadata.get("source", "문서")
        page = doc.metadata.get("page", "")
        with st.expander(f"근거 문서: {source} / page {page}"):
            st.write(doc.page_content)

    with st.spinner("AI가 문서를 기반으로 답변을 생성 중입니다..."):
        response = get_ai_response(st.session_state["messages"], docs)
        result = st.chat_message("assistant").write_stream(response)

    st.session_state["messages"].append(AIMessage(content=result))
