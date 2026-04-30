# 책의 목차를 작성하는 콘텐츠 전략가 에이전트
# 보고서나 책을 쓰는 방법은 여러 가지 있겠지만 저는 목차를 먼저 만들고 나서 파트별로 어떤 내용을 넣을지 구상합니다. 
# AI 에이전트도 이런 방식으로 작업을 진행하도록 구현해 보겠습니다. 
# 물론 앞에서 만든 커뮤니케이터 에이전트에게 목차를 작성하라고 요청할 수도 있지만 앞으로 특정 기능을 독립적으로 수행하는 여러 AI 에이전트를 만들 계획입니다. 
# 이 실습에서는 목차 작성을 전문으로 하는 콘텐츠 전략가 에이전트 content_strategist를 만들어 보겠습니다.

# 하나의 에이전트에게 하나의 임무만 맡기면 엉뚱하게 답변할 확률을 줄일 수 있습니다. 
# 작업결과가 마음에 들지 않을 때 어떤 에이전트를 개선해야 할지 파악하기도 쉽고요. 
# 다음 그림처럼 그래프를 만들면 사용자가 입력한 메시지를 콘텐츠 전략가 에이전트 content_strategist가 우선 받고, 
# 생성한 결과를 커뮤니케이터 에이전트 communicator가 전달받아서 사용자에게 보고하게 됩니다.

from dotenv import load_dotenv # pip install dotenv
import os

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
# from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage # p389 추가
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser # p389 추가
from typing_extensions import TypedDict
from typing import List

# from utils import save_state  # 같은 모듈에 파일 셍성 utils.py
from utils import save_state, get_outline, save_outline  # p389 추가

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

# p389 추가
# 목차를 작성하는 노드(agent)
# content_strategist 에이전트는 목차를 생성하고 save_outline() 함수를 이용해 생성한 목차를
# 저장하는 역할을 합니다. State에서 관리해도 되지만 파일로 저장하면 작업 과정을 파악하기 좋고 파일을 직
# 접 수정할 수도 있습니다.
def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")

    # 시스템 프롬프트 정의
    # content_strategist 에이전트를 위한 시스템 프롬프트를 정의합니다. 지난 목차(outline)와 이전 대화 내
    # 용(messages)이 주어지면 이전 대화 내용을 바탕으로 새로운 목차를 생성하라는 문구를 추가합니다. 실행 결
    # 과를 보면서 더 적절한 프롬프트로 만들어 나가겠습니다.
    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.

        --------------------------------
        - 지난 목차: {outline}
        --------------------------------
        - 이전 대화 내용: {messages}
        """
    )

    # 시스템 프롬프트와 모델을 연결
    # 프롬프트와 언어 모델, 그리고 StrOutputParser를 연결한 체인을 만듭니다. 나중에 이 체인에 outline과 messages를 담아서 넣으면 되겠죠.
    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]        # 상태에서 메시지를 가져옴

    # get_outline 함수를 사용합니다. 이 함수는 utils.py파일에 구현할 예정입니다. 
    # content_strategist는 content_strategist_chain을 이용해 만든 목차 내용을 save_outline 함수를 이용해 저장합니다. 
    # 만약 저장되어 있던 목차(outline)가 이미 있다면 get_outline 함수로 읽어 올 수 있습니다.
    outline = get_outline(current_path) # 저장된 목차를 가져옴

    # 입력값 정의
    # content_strategist_chain에 필요한 messages와 outline을 inputs로 만듭니다
    inputs = {
        "messages": messages,
        "outline": outline
    }

    # 목차 작성
    # 스트림 방식으로 새로운 목차(out line)를 생성해서 gathered에 담습니다.
    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end='')

    print()

    # 완료된 이후에는 gathered에 담긴 목차를 save_outline 함수를 이용해 저장합니다. 
    # 이 save_outline 함수는 utils.py 파일에 곧 구현할 예정입니다.
    save_outline(current_path, gathered) # 목차 저장

    # 메시지 추가    
    # 현재 작업이 어떻게 끝났는지를 다음 노드인 communicator에 전달하기 위해 작업 결과를 AIMessage 인스턴스로 만들어 messages에 추가합니다.
    content_strategist_message = f"[Content Strategist] 목차 작성 완료 : {gathered}"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    # 대화 히스토리인 messages에 결과가 있으므로 communicator는 업데이트된 messages로 임무를 수행하게됩니다. 따라서 현재의 진행 상황을 사용자에게 제대로 보고할 수 있습니다.
    return {"messages": messages} # 메시지 업데이트

# p390추가 끝


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
# 이제 새로 만든 노드를 그래프에 등록하고 노드간 연결 관계도 앞서 본 그래프와 같이 커뮤니케이터 에이전트인 communicator 앞에 연결되도록 수정합니다.
graph_builder.add_node("content_strategist", content_strategist) # p391 추가

# Edges
# graph_builder.add_edge(START, "communicator") 
graph_builder.add_edge(START, "content_strategist") # edge 변경 및 추가  p391 추가
graph_builder.add_edge("content_strategist", "communicator")  # edge변경 및 추가  p391 추가
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