# 챗봇이 이전 대화 내용을 기억하게 하려면 매번 수동으로 업데이트해야 했습니다. 
# 이런 방식이 필요한 경우도 있지만 특수한 상황이 아니라면 랭그래프에서 제공하는 메모리Memory를 활용해 대화 내용을 간편하게 저장할 수 있습니다.

# 랭그래프의 메모리를 다루기 전에 기본 챗봇을 만들겠습니다. langgraph_memory.py파일을 새로 만들고 다음처럼 코드를 입력합니다. 
# 코드 중 일부는 12-1절에서 만든 챗봇 코드를 그대로 가져왔습니다.

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv # pip install dotenv
import os

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

from typing import Annotated # annotated는 타입 힌트를 사용할 때 사용하는 함수
from typing_extensions import TypedDict # TypedDict는 딕셔너리 타입을 정의할 때 사용하는 함수

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    State 클래스는 TypedDict를 상속받습니다.

    속성:
        messages (Annotated[list[str], add_messages]): 메시지들은 "list" 타입을 가집니다.
       'add_messages' 함수는 이 상태 키가 어떻게 업데이트되어야 하는지를 정의합니다.
        (이 경우, 메시지를 덮어쓰는 대신 리스트에 추가합니다)
    """
    messages: Annotated[list[str], add_messages]

# StateGraph 클래스를 사용하여 State 타입의 그래프를 생성합니다.
graph_builder = StateGraph(State)


def generate(state: State):
    """
    주어진 상태를 기반으로 챗봇의 응답 메시지를 생성합니다.

    매개변수:
    state (State): 현재 대화 상태를 나타내는 객체로, 이전 메시지들이 포함되어 있습니다.
		
    반환값:
    dict: 모델이 생성한 응답 메시지를 포함하는 딕셔너리. 
          형식은 {"messages": [응답 메시지]}입니다.
    """ 
    return {"messages": [model.invoke(state["messages"])]}

graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)    

graph = graph_builder.compile()

#------------ 여기서부터 달라진 코드가 있습니다.   
# 구형버전 오류 발생 from langchain.schema import HumanMessage
from langchain_core.messages import HumanMessage

# while 반복문을 살펴보겠습니다. 파이썬의 input 함수를 사용하여 터미널 창에서 사용자가 입력한 내용을 문자열로 받아 user_input 변수에 저장합니다. 
# 이때 받은 값이 exit, quit, q 중 하나라면 대화를 종료합니다.
while True:
    user_input = input("You\t:")
    
    if user_input in ["exit", "quit", "q"]:
        break

    # user_input이 유효한 값이라면 graph.stream의 messages에 HumanMessage 클래스를 이용해 리스트 형태로 담아 실행합니다. 
    # 이때 stream_mode를 messages가 아니라 values로 설정하여 각 단계의 상태 변화를 스트림 방식으로 가져옵니다. 
    # 스트림 방식이므로 for 문을 이용해 메시지를 하나씩 가져올 수 있습니다.
    for event in graph.stream({"messages": [HumanMessage(user_input)]}, stream_mode="values"):
        event["messages"][-1].pretty_print()
        # event["messages"]에는 현재 상태의 메시지들이 저장되어 있으므로 가장 마지막 메시지를 .pretty_ print()로 터미널 창에 출력합니다. 
        # .pretty_print()는 객체의 내용을 보기 좋게 출력할 때 사용하며, 출력하는 메시지가 AIMessage인지 HumanMessage인지에 따라 터미널 창에 자동으로 구분하여 표시해 줍니다.

    print(f'\n현재 메시지 갯수: {len(event["messages"])}\n-------------------\n')
    # 마지막으로 현재 event["messages"]에 저장된 메시지의 개수를 출력하고 while 반복문은 다시 처음으로 돌아갑니다.

# 결과 확인
# (ch12_env) PS C:\Aiprojects\ch12\ch12-2> python .\1.langgraph_memory.py
# You     :난 김기원이야
# ================================ Human Message =================================

# 난 김기원이야
# ================================== Ai Message ==================================

# 안녕하세요, 김기원님! 어떻게 도와드릴까요?

# 현재 메시지 갯수: 2
# -------------------

# You     :내이름 알아?
# ================================ Human Message =================================

# 내이름 알아?
# ================================== Ai Message ==================================

# 죄송하지만, 당신의 이름을 알 수 없습니다. 하지만 당신과 대화할 준비가 되어 있습니다! 무엇을 도와드릴까요?

# 현재 메시지 갯수: 2
# -------------------

# You     :아까 김기원이라고 했는데
# ================================ Human Message =================================

# 아까 김기원이라고 했는데
# ================================== Ai Message ==================================

# 김기원이라는 이름은 한국에서 흔한 이름 중 하나입니다. 어떤 특정한 김기원에 대해 말씀하시는 건가요? 더 많은 정보를 주시면 도움이 될 것 같습니다.

# 현재 메시지 갯수: 2
# -------------------

# You     : q


# 결론
# 코드를 실행하고 과거 대화에 기반해서 대화를 이어 갈 수 있는지 터미널 창에서 테스트해 봅시다. 
# 메시지 개수가 2개에서 더 이상 늘어나지 않고, 매번 새로운 대화로 인식해서 바로 앞에서 말해 준 제 이름도 기억하지 못합니다.