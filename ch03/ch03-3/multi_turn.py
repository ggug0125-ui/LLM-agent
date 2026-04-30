from openai import OpenAI  # 오픈AI 라이브러리를 가져오기
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

# ① api로 답변을 받아오는 함수로 생성 (messages로 매개변수로 받음)
def get_ai_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o",  # 응답 생성에 사용할 모델 지정
        temperature=0.9,  # 응답 생성에 사용할 temperature 설정
        messages=messages,  # 대화 기록을 입력으로 전달
    )
    return response.choices[0].message.content  # 생성된 응답의 내용 반환

messages = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},  # 초기 시스템 메시지
]

while True:
    user_input = input("사용자: ")  # 사용자 입력 받기

    if user_input == "exit":  # ② 사용자가 대화를 종료하려는지 확인인
        break
    
    messages.append({"role": "user", "content": user_input})  # 사용자 메시지를 대화 기록에 추가 
    ai_response = get_ai_response(messages)  # 대화 기록을 기반으로 AI 응답 가져오기
    messages.append({"role": "assistant", "content": ai_response})  # AI 응답 대화 기록에 추가하기

    print("AI: " + ai_response)  # AI 응답 출력

# (venv) PS C:\Aiprojects\ch04> py .\multi_turn.py
# 사용자: 안녕 내 이름은 김기원이야
# AI: 안녕하세요, 기원이 씨! 만나서 반가워요. 어떻게 도와드릴까요?
# 사용자: 내가 누구게~
# AI: 음, 힌트를 조금 더 주시면 맞춰볼 수 있을지도 모르겠네요! 어떤 힌트라도 좋으니 말씀해 주세요. 😊
# 사용자: 내 이름을 알아봐
# AI: 이름은 김기원이시라고 하셨는데, 혹시 다른 의미로 말씀하신 걸까요? 아니면 제가 놓친 부분이 있는지 궁금하네요. 조금 더 설명해 주실 수 있을까요?
# 사용자: 너와 나의 관계는?
# AI: 저는 당신의 디지털 비서이자 상담사로서, 궁금한 점을 해결하거나 정보를 제공하고, 대화를 통해 도움을 드리는 역할을 하고 있어요. 언제든지 궁금한 점이 있거나 도움이 필요할 때 말씀해 주세요! 😊
# 사용자: exit