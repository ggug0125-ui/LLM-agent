# 인공지능 더 안전하게 활용하기

# 지금까지 언어 모델에 기반한 인공지능 프로그램을 개발하는 기술에 대해 배웠습니다. 
# 하지만 이 기술을 어떻게 사용하느냐에 따라 우리를 돕는 도구가 될 수도 있고 해를 끼칠 위험 요소가 될 수도 있습니다. 
# 이번 장에서는 인공지능에 기반한 프로그램을 개발할 때 주의해 야 할 요소를 알아보고 로컬에서 언어 모델을 활용하는 방법도 살펴보겠습니다

# 대규모 언어 모델의 API를 활용하면 해당 기업의 서버에서 계산을 처리하고 결과만 받아 개인 컴퓨터에서 실행됩니다. 
# 그래서 고성능 컴퓨터가 없어도 인터넷만 연결되면 어디서든 대규모 언어 모델을 활용할 수 있습니다. 
# 하지만 API를 활용하는 방식이 적합하지 않은 상황도 있습니다. 
# 인터넷이 연결되지 않는 환경에서도 실행되는 언어 모델 기반 프로그램을 개발해 야 할 수도 있고, API 호출 비용이 부담스러울 수도 있습니다. 
# 또한 RAG 기반 프로그램을 개발할 때 회사의 기밀 문서를 다룰 경우 문제가 발생할 수도 있습니다. 
# 오픈AI의 임베딩 모델을 사용하면 해당 문서가 인터넷을 통해 오픈AI의 서버로 전송됩니다. 
# 비록 오픈AI가 사용자의 데이터를 사용하지 않는다고 하더라도 회사의 보안 규정상 자료를 외부로 전송하는 것을 금지하고 있다면 그에 따른 문제가 발생할 수 있습니다. 
# 이런 경우 로컬 환경에서 언어 모델을 구동할 수 있다면 좋겠죠.
# API를 사용하지 않고 로컬 컴퓨터에서 챗봇이나 RAG 기반 프로그램을 개발하는 방법을 알아 보겠습니다. 여기서는 소규모 언어 모델로 자주 활용되는 메타의 라마 모델을 사용합니다.


# 메타의 라마 모델을 로컬에서 구동하기
# 메타는 API로 활용할 수 있는 대규모 언어 모델과 로컬에서 구동할 수 있는 소규모 언어 모델을 모두 제공합니다. 
# 이 책에서는 로컬에 설치하여 사용할 수 있는 소규모 언어 모델의 구동법을 알아 보겠습니다.
# 라마 API 사용법은 공식 문서(https://docs.lama-api.com/quickstart)에서 확인하세요

# 메타의 라마 언어 모델을 로컬 환경에 설치하는 방법은 여러 가지 있습니다. 
# 허깅페이스에서 모델을 직접 내려받거나 메타에서 제공하는 올라마이ama를 통해 내려받을 수 있습니다. 
# 여기서는 올라마를 활용해 설치하겠습니다.

# 올라마 웹 사이트(https://ollama.com)에 접속해 상단의 [Models] 버튼을 클릭하고 사용할 모델을 선택합니다. 
# 최신 모델은 llama 3.3인데, 이 모델은 43GB의 저장 공간과 24GB 이상의 메모리, 최신 그래픽 카드가 필요하므로 
# 이번 실습에서는 llama 3.2의 3b 모델을 사용하겠습니다. 
# 여기서 3b는 이 모델의 매개변수를 의미합니다. 같은 모델이라도 매개변수가 많을 수록 성능이 좋고 계산 자원이 많이 필요합니다. 
# llama 3.2의 3b 모델은 2GB로 용량이 비교적 작아서 개인 컴퓨터에서도 구동할 수 있습니다.


# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama # pip install langchain-ollama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 모델 초기화
# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOllama(model="llama3.2:3b")

messages = [
    SystemMessage("You are a helpful assistant."),
]

while True:
    user_input = input("You\t: ").strip()

    if user_input in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    messages.append(HumanMessage(user_input))

    response = llm.invoke(messages)
    print("Bot\t: ", response.content)

    messages.append(AIMessage(response.content))
