# 기존 코드에서 단 몇 줄만 바꿔 메모리를 추가하면 이전 대화 내용을 기억한 상태로 대화률 이어 나갈 수 있도록 설정할 수 있습니다. 코드를 다음처럼 수정합니다.

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

# p399 추가
# 랭그래프에서 제공하는 MemorySaver를 임포트하고 memory라는 이름의 객체로 만듭니다.
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# config에서 thread_id를 abcd로 설정합니다. 이때 thread_id는 임의의 문자열로 설정합니다. 
# config의 thread_id는 일종의 대화방 ID라고 생각하면 됩니다. 
# while 문으로 메시지를 여러 번 반복해서 입력하더라도 thread_id가 유지된다면 기존 대화 내용을 계속 보존한 상태로 진행할 수 있습니다.
config = {"configurable": {"thread_id": "abcd"}}

# graph = graph_builder.compile()
# memory 객체는 graph_builder.compile(checkpointer=memory)에서 설정되어 대화 내용을 계속 쌓아갈 수 있도록 만듭니다.
graph = graph_builder.compile(checkpointer=memory)



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
    # for event in graph.stream({"messages": [HumanMessage(user_input)]}, stream_mode="values"):
    for event in graph.stream({"messages": [HumanMessage(user_input)]}, config, stream_mode="values"):
        #                                                               config 추가
        event["messages"][-1].pretty_print()
        # event["messages"]에는 현재 상태의 메시지들이 저장되어 있으므로 가장 마지막 메시지를 .pretty_ print()로 터미널 창에 출력합니다. 
        # .pretty_print()는 객체의 내용을 보기 좋게 출력할 때 사용하며, 출력하는 메시지가 AIMessage인지 HumanMessage인지에 따라 터미널 창에 자동으로 구분하여 표시해 줍니다.

    print(f'\n현재 메시지 갯수: {len(event["messages"])}\n-------------------\n')
    # 마지막으로 현재 event["messages"]에 저장된 메시지의 개수를 출력하고 while 반복문은 다시 처음으로 돌아갑니다.

# 결과 확인

# (ch12_env) PS C:\Aiprojects\ch12\ch12-2> python .\2.langgraph_memory_add_MemorySaver.py
# You     :안녕 난 김기원이라고 해
# ================================ Human Message =================================
# 안녕 난 김기원이라고 해
# ================================== Ai Message ==================================
# 안녕하세요, 김기원님! 만나서 반갑습니다. 어떻게 도와드릴까요?
# 현재 메시지 갯수: 2
# -------------------
# You     :내이름 기억해? 
# ================================ Human Message =================================
# 내이름 기억해?
# ================================== Ai Message ==================================
# 네, 김기원님! 제가 대화를 지속하면서 기억할 수는 없지만, 지금 대화 중에는 김기원님이라고 말씀해주신 것을 기억하고 있습니다. 다른 질문이나 이야기하고 싶은 것이 있으면 언제든지 말씀해 주세요!
# 현재 메시지 갯수: 4
# -------------------
# You     :오늘 날씨는?
# ================================ Human Message =================================
# 오늘 날씨는?
# ================================== Ai Message ==================================
# 죄송하지만, 현재 실시간 날씨 정보를 제공할 수는 없습니다. 하지만 날씨 정보를 확인하고 싶으시다면, 날씨 앱이나 웹사이트를 이용하시면 좋을 것 같습니다. 다른 질문이나 궁금한 점이 있으시면 말씀해 주세요!
# 현재 메시지 갯수: 6
# -------------------
# You     : q



# 이 코드를 실행해 보면 기존 대화 내용을 계속 쌓아 나가는 것을 알 수 있습니다. 새로운 메시지를 입력해도 지난 대화 내용에 기반해서 답변을 잘 생성합니다.