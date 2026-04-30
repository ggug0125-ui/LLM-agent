import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import retriever

# 모델 초기화
llm = ChatOllama(model="deepseek-r1:latest")

# Streamlit 앱
st.title("💬 DeepSeek-R1 Langchain Chat")

# 세션 메시지 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content="너는 문서 기반으로 답변하는 도시 정책 전문가야."),
        AIMessage(content="무엇을 도와드릴까요?")
    ]

# 메시지 출력
for msg in st.session_state.messages:
    role = "assistant"
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, SystemMessage):
        role = "system"

    st.chat_message(role).write(msg.content)

# 사용자 입력
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    # 🔍 문서 검색
    docs = retriever.retriever.invoke(prompt)

    for doc in docs:
        with st.expander(f"📄 {doc.metadata.get('source', '문서')}"):
            st.write(doc.page_content)

    # 🤖 RAG 실행
    with st.spinner("AI가 답변 생성 중..."):
        response = retriever.chain.stream({
            "input": prompt
        })

        result = ""
        with st.chat_message("assistant"):
            for chunk in response:
                if "answer" in chunk:
                    st.write(chunk["answer"], end="")
                    result += chunk["answer"]

    st.session_state.messages.append(AIMessage(content=result))