import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import retriever  # 추가 270
from dotenv import load_dotenv
import os

load_dotenv()
# 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini")

# 0. 사용자의 메시지 처리하기 위한 함수 (문서와 관계없는 결과가 나온다.)
# def get_ai_response(messages):
#     response = llm.stream(messages)

#     for chunk in response:
#         yield chunk

# 1. 문서를 활용하는 리트리버를 추가한다. (문서에 관련된 결과가 나온다.) -> retriever.py 수정
def get_ai_response(messages, docs):    
    response = retriever.document_chain.stream({
        "messages": messages,
        "context": docs
    })

    for chunk in response:
        yield chunk

# Streamlit 앱
st.title("💬 GPT-4o Langchain Chat")

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 문서에 기반해 답변하는 도시 정책 전문가야 "),  
        AIMessage("How can I help you?")
    ]

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    st.session_state.messages.append(HumanMessage(prompt)) # 사용자 메시지 저장

    augmented_query = retriever.query_augmentation_chain.invoke({
        "messages": st.session_state["messages"],
        "query": prompt,
    })
    print("augmented_query\t", augmented_query)

    # 관련 문서 검색 p272추가
    print("관련 문서 검색")
    docs = retriever.retriever.invoke(f"{prompt}\n{augmented_query}")

    for doc in docs:
        print('---------------')
        print(doc)
        #p274 추가 (리트리버로 검색한 문서 출처 표시용)
        with st.expander(f"**문서:** {doc.metadata.get('source', '알 수 없음')}"):
            # 파일명과 페이지 정보 표시
            st.write(f"**page:**{doc.metadata.get('page', '')}")
            st.write(doc.page_content)
    print("===============")
    # 관련 문서 검색 p272추가 끝

    with st.spinner(f"AI가 답변을 준비 중입니다... '{augmented_query}'"):
        # response = get_ai_response(st.session_state["messages"]) p272 수정
        response = get_ai_response(st.session_state["messages"], docs)
        result = st.chat_message("assistant").write_stream(response) # AI 메시지 출력
    st.session_state["messages"].append(AIMessage(result)) # AI 메시지 저장    


# (ch09_env) PS C:\Aiprojects\ch09> cd .\ch09-3\
# (ch09_env) PS C:\Aiprojects\ch09\ch09-3> ls


#     디렉터리: C:\Aiprojects\ch09\ch09-3


# Mode                 LastWriteTime         Length Name
# ----                 -------------         ------ ----
# -a----      2026-04-21  오전 11:44           2513 rag.py
# -a----      2026-04-21  오전 11:37              0 retriever.py

# (ch09_env) PS C:\Aiprojects\ch09\ch09-3> pip install streamlit
# (ch09_env) PS C:\Aiprojects\ch09\ch09-3> streamlit run rag.py