# 랭체인의 메모리 기능을 활용하기 보다 대화 내용을 리스트로 지접관리하는 것이 더 간편함
# 대화내용을 데이터베이스에 저장하거나 히스로리를 수정해야 할 경우 활용

import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI 

from langchain_openai import ChatOpenAI  # 오픈AI 모델을 사용하는 랭체인 챗봇 클래스
from langchain_core.chat_history import (
    BaseChatMessageHistory,  # 기본 대화 기록 클래스
    InMemoryChatMessageHistory,  # 메모리에 대화 기록을 저장하는 클래스
)
from langchain_core.runnables.history import RunnableWithMessageHistory  # 메시지 기록을 활용해 실행 가능한 wrapper 클래스
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기
client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자의 질문에 친절이 답하는 AI챗봇이다.")
    ]

# 세션별 대화 기록을 저장할 딕셔너리 대신 session_state 사용
if "store" not in st.session_state:
    st.session_state["store"] = {}

# InMemoryChatMessageHistory 랭체인 메모리가 세션 히스트로 사용
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = InMemoryChatMessageHistory()
    return st.session_state["store"][session_id]

llm = ChatOpenAI(model="gpt-4o-mini")
with_message_history = RunnableWithMessageHistory(llm, get_session_history)

config = {"configurable": {"session_id": "abc2"}}

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

if prompt := st.chat_input():
    print('user:', prompt)  
    st.session_state.messages.append(HumanMessage(prompt))
    st.chat_message("user").write(prompt)

    response = with_message_history.invoke([HumanMessage(prompt)], config=config)

    msg = response.content
    st.session_state.messages.append(response)
    st.chat_message("assistant").write(msg)
    print('assistant:', msg)

# c:\Aiprojects\venv\Scripts\Activate.ps1
# pip install streamlit
## streamlit run c:/Aiprojects/ch08/ch08-5/2.langchain_simple_chat_streamlit_NoMemory.py