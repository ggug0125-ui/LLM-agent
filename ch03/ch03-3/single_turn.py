from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=[
            {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
            {"role": "user", "content": user_input},
        ],
    )
    print("AI: " + response.choices[0].message.content)

# 결과보기
# (venv) PS C:\Aiprojects\ch03> cd ..
# (venv) PS C:\Aiprojects> cd ch04
# (venv) PS C:\Aiprojects\ch04> py .\single_turn.py
# 사용자: 너는 누구야?
# AI: 나는 개인 맞춤형 조언과 정보를 제공하기 위해 개발된 인공지능 상담사야. 질문이 있거나 도움이 필요하면 언제든지 말해줘!
# 사용자: exit

# (venv) PS C:\Aiprojects\ch04> py .\single_turn.py
# 사용자: 안녕? 내 이름은 김기원이야
# AI: 안녕하세요, 기원이님! 만나서 반갑습니다. 오늘 어떻게 도와드릴까요?
# 사용자: 내 이름이 모라고?
# AI: 죄송하지만, 제게는 사용자의 이름에 대한 정보가 없습니다. 이름을 알려주시면 기쁘게 대화를 이어가겠습니다.
# 사용자: 김기원이라고
# AI: 안녕하세요, 김기원님! 어떻게 도와드릴까요?
# 사용자: 내 이름이 모라고?
# AI: 죄송하지만 사용자의 이름을 알 수 있는 정보는 제공받지 않았습니다. 알려주실 수 있나요?
# 사용자: exit