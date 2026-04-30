import streamlit as st  # pip install streamlit==1.41.1
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# (0) 사이드바에서 api_key 입력하는 부분 
# 스트림릿은 사이드바에 내용을 추가하는 기능인 st.sidebar를 제공함
# st.session_state() 사용자의 세션을 관리하는 기능(웹브라우져에서 상호작용하는 동안 상태 유지)
with st.sidebar:
    openai_api_key = os.getenv('OPENAI_API_KEY') 
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")

# (1) st.session_state에 "messages"가 없으면 초기값을 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# (2) 대화 기록을 출력 : st.chat_message() 채팅 인터페이스에서 메시지를 출력하는 용도
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# (3) 사용자 입력을 받아 대화 기록에 추가하고 AI 응답을 생성
if prompt := st.chat_input():
    
    # api키가 없을때 예외처리용
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()


    client = OpenAI(api_key=openai_api_key)
    
    # 사용자가 채팅 창에 질문을 입력하면 해당 내용을 st.session_state.messages의 dict 형태로 추가
    st.session_state.messages.append({"role": "user", "content": prompt}) 
    st.chat_message("user").write(prompt)  # 사용자의 입력 내용을 출력

    # gpt의 답변을 받아와서 다시  st.session_state.messages 에 추가하고 답변을 화면에 출력
    response = client.chat.completions.create(model="gpt-4o", messages=st.session_state.messages) 
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg}) 
    st.chat_message("assistant").write(msg)

# 실행 결과
# (venv) PS C:\Aiprojects\ch04> streamlit run .\streamlit_basic.py

#       Welcome to Streamlit!

#       If you’d like to receive helpful onboarding emails, news, offers, promotions,
#       and the occasional swag, please enter your email address below. Otherwise,
#       leave this field blank.

#       Email: lonen@nate.com

#   You can find our privacy policy at https://streamlit.io/privacy-policy

#   Summary:
#   - This open source library collects usage statistics.
#   - We cannot see and do not store information contained inside Streamlit apps,
#     such as text, charts, images, etc.
#   - Telemetry data is stored in servers in the United States.
#   - If you'd like to opt out, add the following to %userprofile%/.streamlit/config.toml,
#     creating that file if necessary:

#     [browser]
#     gatherUsageStats = false


#   You can now view your Streamlit app in your browser.

#   Local URL: http://localhost:8501
#   Network URL: http://192.168.0.20:8501