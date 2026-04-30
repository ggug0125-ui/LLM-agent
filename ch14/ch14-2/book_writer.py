# 조장 역할을 하는 슈퍼바이저 에이전트

# 여러 사람이 협업할 때 조직 구조를 고려하듯이, 각각 전문 역할을 하는 AI 에이전트가 서로 유기적으로 협력하도록 만들 때에도 여러 전략을 구상할 수 있습니다. 
# 가장 흔한 방법이 의사결정을 하는 '장'을 뽑는 것입니다. 
# 대학 때 조별 과제를 해도 조장을 뽑고 회사에서도 부서장을 세워서 누가 어떤 일을 어떻게 할지 판단하는 역할을 합니다. 
# AI 에이전트들끼리 협력하도록 만들 때에도 이런 전략을 활용할 수 있습니다.

# 조장이 필요하다! - 슈퍼바이저 에이전트
# 14-1절에서 만든 프로그램에 보고서를 만들어 달라고 요청하지 않고 처음부터 '안녕?'이라고 입력하면 엉뚱하게 인사에 대한 목차를 생성합니다. 
# 현재 그래프 구조는 사용자의 입력에 맞춰 무조건 목차를 생성하고 대화를 이어 나가게 되어 있기 때문입니다.

# 따라서 사용자가 입력한 내용을 분석해 목차를 작성하는 콘텐츠 전략가 에이전트 content_strategist가 필요할지, 
# 아니면 사용자와 소통하는 커뮤니케이터 에이전트 communicator가 필요할지 판단하는 조장이 필요합니다.

# 사용자의 요구 사항에 따라 어떤 AI 에이전트에게 일을 시킬지 판단하는 슈퍼바이저 에이전트 supervisor를 만들고, 
# 이에 맞춰 그래프 구조를 바꾸겠습니다. 
# 사용자가 메시지를 입력하면 supervisor 노드에서 목차를 쓰는 content_strategist에게 보낼지, 
# 아니면 대화를 담당하는 communicator에게 바로 보낼지를 판단합니다. 
# 목차를 작성하는 content_strategist로 가서 목차를 작성한 뒤에는 communicator로 넘어가서 결과를 사용자에게 보고하게 됩니다.
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
    task: str  # p398 추가
    # State에 task를 문자열(str) 형식으로 추가해 상태를 다른 AI 에이전트들과 공유할 수 있게 합니다. 
    # task의 자료형은 나중에 수정할 예정입니다. 이렇게 설정했을 때의 불확실성을 살펴보기 위해 우선 문자열로 정의했습니다. 
    # 아래쪽에 state를 초기화하는 부분도 task=""로 초기화해 줍니다.
 
def supervisor(state: State): # supervisor 에이전트 추가 p398 추가
# supervisor를 새로 만듭니다. 이 에이전트는 다음에 할 일이 무엇인지 판단하는 역할을 합니다.
    print("\n\n============ SUPERVISOR ============")

    # 시스템 프롬프트 정의
    # supervisor의 시스템 프롬프트입니다. 이 프롬프트에서는 content_strategist와 communicator를 선택하는 방법을 설명합니다. 
    # 입력값으로 받을 항목은 기존 목차를 의미하는 outline과 기존 대화 내용을 의미하는 messages입니다.
    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고, 
        사용자의 요구를 달성하기 위해 현재 해야할 일이 무엇인지 결정한다.

        supervisor가 활용할 수 있는 agent는 다음과 같다.     
        - content_strategist: 사용자의 요구사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다. 
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행상황을 사용자에게 보고하고, 다음 지시를 물어본다. 

        아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.

        ------------------------------------------
        previous_outline: {outline}
        ------------------------------------------
        messages:
        {messages}
        """
    )

    # 체인 연결
    # 체인을 연결하는 부분입니다. 
    # StrOutputParser로 content_strategist와 communicator 중에 하나가 나오면 supervisor_route에서 conditional_edge로 처리할 예정입니다.
    supervisor_chain = supervisor_system_prompt | llm | StrOutputParser()	

    # 메시지 가져오기
    # state의 메시지를 가져오도록 설정합니다. 이렇게 설정하면 state에 'messages'가 있으면 그 값을 반환하고, 
    # 없으면 빈 리스트 []를 반환합니다. state['messages']로 설정해도 동일하지만 다양한 방식을 소개하기 위해 사용했습니다.
    messages = state.get("messages", [])

    # inputs 설정
    # supervisor_chain에 넣을 inputs을 딕셔너리로 설정합니다.
    inputs = {
        "messages": messages,
        "outline": get_outline(current_path)
    }

    # task 문자열로 생성
    # supervisor_chain은 다음에 사용할 노드를 문자열 형태로 반환하므로 task 변수에 담습니다.
    
    task = supervisor_chain.invoke(inputs) 
   
    # 메시지 추가
    # 이 task를 AIMessage 형태로 기존 messages에 추가하고 터미널 창에도 출력합니다.
    
    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)

    # state 업데이트
    # 앞서 만든 supervisor의 결과를 이용해 노드 연결을 conditional_edge로 처리합니다. 현재 state에서 task의 값을 가져와 반환합니다.
    return {
        "messages": messages, 
        "task": task
    }

# supervisor's route
# 체인을 연결하는 부분입니다. StrOutputParser로 content_strategist와 communicator 중에 하나가 나
# 오면 supervisor_route에서 conditional_edge로 처리할 예정입니다.
def supervisor_router(state: State):
    task = state['task']
    return task

# p398 추가 끝

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
    # content_strategist_message = f"[Content Strategist] 목차 작성 완료 : {gathered}"  p394 수정
    content_strategist_message = f"[Content Strategist] 목차 작성 완료"
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

        사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다. # p394 추가

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
# 새로 만든 supervisor 노드를 그래프에 추가합니다.
graph_builder.add_node("supervisor", supervisor) # p399추가
graph_builder.add_node("communicator", communicator)
# 이제 새로 만든 노드를 그래프에 등록하고 노드간 연결 관계도 앞서 본 그래프와 같이 커뮤니케이터 에이전트인 communicator 앞에 연결되도록 수정합니다.
graph_builder.add_node("content_strategist", content_strategist) # p391 추가

# Edges
# graph_builder.add_edge(START, "communicator") 
# p399 제거 graph_builder.add_edge(START, "content_strategist") # edge 변경 및 추가  p391 추가
# 새로 추가한 supervisor 노드를 이용해 앞에서 살펴본 그래프와 같은 형태로 연결하기 위해 코드를 수정합니다.
graph_builder.add_edge(START, "supervisor") # p399 추가
graph_builder.add_conditional_edges(
    "supervisor", 
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator"
    }
) # p399 추가 끝
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
    task="" # p400 추가 
    # State에 task를 문자열(str) 형식으로 추가해 상태를 다른 AI 에이전트들과 공유할 수 있게 합니다. task의
    # 자료형은 나중에 수정할 예정입니다. 이렇게 설정했을 때의 불확실성을 살펴보기 위해 우선 문자열로 정의했
    # 습니다. 아래쪽에 state를 초기화하는 부분도 task=""로 초기화해 줍니다.
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

# (ch13_env) PS C:\Aiprojects\ch14\ch14-1-2> cd ..
# (ch13_env) PS C:\Aiprojects\ch14> cd ch14-2
# (ch13_env) PS C:\Aiprojects\ch14\ch14-2> python .\book_writer.py

# User    : 안녕?


# ============ SUPERVISOR ============
# [Supervisor] communicator


# ============ COMMUNICATOR ============

# AI      : 안녕하세요! AI팀의 진행상황을 사용자께 보고드리겠습니다. 현재 우리는 책의 각 장에 대한 세부 내용을 작성하고 있으며, 사용자께서 이미 목차를 확인하셨으니 그 부분은 생략하겠습니다. 

# 혹시 현재까지 작업에 대해 궁금하신 점이나 추가적으로 의견 주실 부분이 있을까요? 사용자님의 피드백은 저희에게 큰 도움이 됩니다.
# ------------------------------------ MESSAGE COUNT       4

# User    : q
# Goodbye!