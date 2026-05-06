import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import retriever


st.title("DeepSeek-R1 LangChain RAG Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content="You are a helpful assistant that answers based on documents."),
        AIMessage(content="무엇을 도와드릴까요?"),
    ]

for msg in st.session_state.messages:
    role = "assistant"
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, SystemMessage):
        role = "system"

    st.chat_message(role).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    docs = retriever.retriever.invoke(prompt)

    for doc in docs:
        source = doc.metadata.get("source", "문서")
        with st.expander(f"Document: {source}"):
            st.write(doc.page_content)

    with st.spinner("AI가 답변을 생성 중입니다..."):
        response = retriever.chain.stream({"input": prompt})

        result = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in response:
                if "answer" in chunk:
                    result += chunk["answer"]
                    placeholder.write(result)

    st.session_state.messages.append(AIMessage(content=result))
