# 랭그래프로 목차를 작성하는 멀티에이전트 만들기

# 크고 복잡한 일을 할 때는 여러 전문가가 작업 결과를 서로 리뷰하고 회의를 하며 진행합니다. 
# 인공지능도 마찬가지입니다. 각기 다른 능력을 갖춘 인공지능 에이전트들끼리 서로 회의하도록 만들면 복잡한 작업도 쉽게 해낼 수 있습니다. 
# 책의 목차를 작성하는 멀티에이전트를 만드는 방법을 다루겠습니다.

# 사용자와 함께 목차를 작성하는 에이전트

# '천 리 길도 한 걸음부터'라는 말이 있습니다. 
# 먼저 이 장에서 만들 멀티에이전트에 대해 알아보고 사용자의 질문에 대답하는 챗봇을 만든 다음, 사용자와 함께 목차를 만드는 챗봇을 완성해 보겠습니다.

# 이 장에서 만드는 멀티에이전트 이어지는 실습에서는 작은 단위의 인공지능 프로그램들이 서로 협업하여 작업을 수행하는 멀티에이전트로 발전하는 과정을 살펴보겠습니다. 
# 이를 통해 멀티에이전트가 단순한 요구사항부터 매우 복잡한 업무까지 어떻게 수행해 내는지 알아보겠습니다.
# 다음 그림은 'OOO에 관한 보고서를 써야 하니까 인터넷에서 자료를 조사하고 목차를 작성해 와.'라는 한 문장을 수행하기 위해서 조직된 AI 멀티에이전트의 시스템입니다. 
# 각각의 노드(에이전트)는 고유한 임무를 수행하고 자신의 작업 결과를 다른 AI 에이전트들과 공유하면서 작업을 완수합니다.

# 이 장에서는 AI 에이전트들이 개별 역할을 각각 수행하면서 조화를 이루어 복잡한 업무를 처리하는 과정을 직접 구현해 볼 수 있습니다. 
# 이를 통해 인공지능을 활용한 과업 수행의 기본 원리를 이해하고, 더 나아가 여러분의 프로젝트에 실제로 적용할 아이디어를 얻을 수 있을 것 입니다.

# 사용자와 의사소통하는 커뮤니케이터 에이전트 사용자와 단순한 대화를 할 수 있는 커뮤니케이터 에이전트 communicator를 만들겠습니다. 
# 이어지는 실습에서 기능을 계속 추가할 예정입니다. 사용자가 메시지를 입력하면 커뮤니 케이터 에이전트가 답변을 생성하고 결과를 내놓는 단순한 구조입니다.

# 커뮤니케이터 에이전트 communicator 만들기


# 새로운 파이썬 파일인 book_writer.py을 만들고 다음과 같이 코드를 작성합니다. 대부분 은 12-1절에서 챗봇을 만들 때 작성한 코드를 가져온 것입니다. 
# 새로 등장한 코드 위주로 살펴보겠습니다.
from dotenv import load_dotenv # pip install dotenv
import os

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict
from typing import List

from utils import save_state  # 같은 모듈에 파일 셍성 utils.py

from datetime import datetime
import os 

# 현재 폴더 경로 찾기
# 랭그래프 이미지로 저장 및 추후 작업 결과 파일 저장 경로로 활용
filename = os.path.basename(__file__) # 현재 파일명 반환
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로 

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o") 

# 상태 정의
# 상태를 정의하겠습니다. 앞으로 더 확장해 나가겠지만 현재는 대화 기록을 담아 둘 수 있도록 messages라는 변수에 리스트 자료형을 사용합니다. 
# 이 리스트에 들어갈 수 있는 자료형은 AnyMessage 혹은 문자열(str)입니다. 
# 꼭 랭체인 메시지가 아니라 문자열로 들어오더라도 GPT에서 처리할 수 있으므로 AnyMessage뿐만 아니라 str로도 담을 수 있게 했습니다.

class State(TypedDict):
    messages: List[AnyMessage | str]

# 랭체인에서 AnyMessage는 여러 종류의 메시지 타입을 하나로 통합하여 표현하기 위해 사용되는 타입 별칭type alias입니다. 
# 즉, AnyMessage는 다음과 같은 다양한 메시지 클래스를 포함하는 유니온union 타입으로 정의되어 있습니다. 
# AnyMessage를 사용하면 다양한 메시지 타입을 모두 받아들일 수 있어서 코드의 유연성과 가독성을 높일 수 있습니다. 
# 예를 들어 채팅 기록이나 메시지 리스트를 처리할 때 AnyMessage 타입으로 선언하면 앞에 언급한 모든 메시지 객체를 하나의 리스트로 다룰 수 있습니다

# AlMessage : AI(모델)가 생성한 응답 메시지
# HumanMessage : 사용자(인간)가 입력한 메시지 
# SystemMessage : 시스템에서 모델의 행동이나 대화의 맥락을 지정하는 메시지
# ToolMessage : 도구(tool)의 호출 결과를 나타내는 메시지




# 사용자와 대화하는 에이전트 communicator를 만들겠습니다. 
# 이 AI 에이전트는 목차를 작성하는 AI 팀의 일원으로 기존 대화 내용을 바탕으로 사용자와 상호 작용하며 대화하는 임무를 수행합니다.
def communicator(state: State):
    print("\n\n============ COMMUNICATOR ============")

    # 시스템 프롬프트 정의
    # 임무 내용을 communicator_system_prompt에 PromptTemplate를 이용해 프롬프트로 정의합니다. 이 프롬프트는 시스템 프롬프트 역할을 합니다
    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서, 
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다. 

        messages: {messages}
        """
    )

    # 시스템 프롬프트와 모델을 연결
    # 이 시스템 프롬프트는 11m에 체인으로 연결됩니다. 프롬프트가 기존 대화 내용을 계속 업데이트하고 답변할수 있도록 state에서 messages를 받을 수 있게 합니다.
    system_chain = communicator_system_prompt | llm

    # 상태에서 메시지를 가져옴
    messages = state["messages"]

    # 입력값 정의
    inputs = {"messages": messages}

    # 스트림되는 메시지를 출력하면서, gathered에 모으기
    # system_chain.stream(inputs)로 스트림 출력을 하기 위해 빈 변수 gathered를 만들고 언어 모델에서 스트림되는 내용을 터미널 창에 출력하면서 gathered에 점차 덧붙입니다. 
    # 그리고 gathered의 값을 messages 리스트에 추가하고 messages를 딕셔너리 형태로 반환하여 상태(State)를 업데이트합니다.
    gathered = None

    print('\nAI\t: ', end='')
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    messages.append(gathered)

    return {"messages": messages}

# 상태 그래프 정의
# 이제 StateGraph로 그래프를 만들 차례입니다. 이 그래프는 Start → communicator → END의 단순한 구조로 되어 있습니다. 
# 이번에 만든 communicator 노드만 graph_builder에 추가하고, 이 노드를 START와 END에 연결한 뒤 그래프를 컴파일합니다.
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("communicator", communicator)

# Edges
graph_builder.add_edge(START, "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# 랭그래프를 도식화하는 코드입니다. 12-1절에서 작성한 주피터 노트북 코드(langgraph_simple_chatbot.ipynb)와 거의 동일하지만 
# 도식화된 결과를 PNG 파일로 저장하도록 수정했습니다. 
# 여기에서는 앞서 작성한 absolute_path를 사용하며 현재 작업하고 있는 파이썬 파일의 이름과 동일한 PNG 파일을 만듭니다.
graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))

# 상태 초기화
# 전체 워크플로의 시스템 메시지를 작성합니다. 
# 아직은 AI 에이전트가 communicator 하나뿐이지만 나중에 여러 AI 에이전트가 추가될 것을 고려하여 설계합니다. 
# 그리고 프롬프트에 현재 시각 정보를 포함하도록 하여 향후 책이나 보고서를 만들 때 언어 모델이 만들어진 시점(gpt-4o의 경우 2023년)을 기준으로 판단하는 오류를 방지할 수 있게 합니다.
state = State(
    messages = [
        SystemMessage(
                f"""
            너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가팀이다.
            사용자가 사용하는 언어로 대화하라.

            현재시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.

            """
        )
    ],
)

# 터미널 창에서 사용자의 입력을 받고 graph를 실행(invoke)하는 코드입니다. 
# 사용자가 입력한 값을 user_input 변수에 담아 워크플로를 실행합니다. 
# 반복문은 사용자가 입력한 값이 'exit',  quit', 'q'가 아니라면 계속됩니다. 
# 실제로 AI가 생성한 답변은 communicator 에이전트(노드)에서 print로 출력되며 그 후 현재 메시지 수를 파악하기 위해 print 문을 추가합니다. 
# 그리고 현재 상태를 save_state 함수로 저장합니다. 이 함수는 utils.py 파일에 따로 만들겠습니다. 
while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break
    
    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n------------------------------------ MESSAGE COUNT\t', len(state["messages"]))

    save_state(current_path, state) # 현재 state 내용 저장

# utils.py 생성하기

