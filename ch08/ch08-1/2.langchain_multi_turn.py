from openai import OpenAI  # 주석처리
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기
client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

llm = ChatOpenAI(model="gpt-4o")  # ChatOpenAI 클래스의 인스턴스 생성


# def get_ai_response(messages):
#     response = client.chat.completions.create(
#         model="gpt-4o",  # 응답 생성에 사용할 모델 지정
#         temperature=0.9,  # 응답 생성에 사용할 temperature 설정
#         messages=messages,  # 대화 기록을 입력으로 전달
#     )
#     return response.choices[0].message.content  # 생성된 응답의 내용 반환

messages = [
    # {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},  # 초기 시스템 메시지
    SystemMessage("너는 사용자를 도와주는 상담사야."),  # 초기 시스템 메시지
]

while True:
    user_input = input("사용자: ")  # 사용자 입력 받기

    if user_input == "exit":  # ② 사용자가 대화를 종료하려는지 확인인
        break
    
    messages.append(
        # {"role": "user", "content": user_input} # 주석처리
        HumanMessage(user_input)
    )  # 사용자 메시지를 대화 기록에 추가 
    
    # ai_response = get_ai_response(messages)  # 주석처리
    ai_response = llm.invoke(messages)  # 대화 기록을 기반으로 AI 응답 가져오기
    messages.append(
        # {"role": "assistant", "content": ai_response} # 주석처리
        ai_response
    )  # AI 응답 대화 기록에 추가하기

    print("AI: " + ai_response.content)  # AI 응답 출력


# (venv) PS C:\Aiprojects> & c:\Aiprojects\venv\Scripts\python.exe c:/Aiprojects/ch08/ch08-1/2.langchain_multi_turn.py
# 사용자: 나는 김기원이야
# AI: 김기원님, 만나서 반갑습니다! 어떻게 도와드릴까요?
# 사용자: 내이름 기억해?
# AI: 대화 중에는 기억할 수 있지만, 세션이 종료되면 개인정보는 저장되지 않습니다. 그래서 김기원님께서 다시 말씀해 주시면 그때 기억할 수 있습니다. 다른 질문이나 도움이 필요하시면 언제든지 말
# 씀해 주세요!
# 사용자: 아까 나는 김기원이야 에서 김기원이 이름이자나
# AI: 맞습니다, 김기원님. 김기원이란 이름을 말씀해 주셨습니다. 다른 도움이 필요하시면 언제든지 말씀해 주세요!
# 사용자: 내이름 기억해?
# AI: 네, 이 대화 동안은 김기원님이라는 것을 기억하고 있습니다. 다른 궁금한 점이나 필요하신 것이 있으면 말씀해 주세요!
# 사용자: exit
# (venv) PS C:\Aiprojects> 