# # 스스로 판단하고 작업하는 멀티에이전트 만들기

# 지금까지 만든 멀티에이전트는 작업 하나가 끝날 때마다 이어서 무슨 작업을 해야 할지 사용자에게 물어봐야 했습니다. 
# 이 장에서는 에이전트들이 공유할 수 있는 공동 목표를 설정 하고 에이전트마다 그 목표를 달성하고 있는지 스스로 평가해서 다음 작업을 진행하도록 개선하겠습니다. 
# 목표를 달성했거나 에이전트가 스스로 판단하기 어려운 상황일 때는 사용자에게 질문하는 방식으로 프로그램을 발전시키겠습니다.

# 에이전트의 공동 목표 만들기

# 목표를 점검하는 비즈니스 분석가 에이전트 business_analyst를 만들고, 사용자의 의도를 파악해 에이전트의 공동 목표를 설정해 보겠습니다.

# 목표를 점검하는 비즈니스 분석가 에이전트
# 지금까지 각 AI 에이전트는 작업을 완료한 후 자신의 작업 내용을 다른 AI 에이전트들에게 공유했지만 
# 현재 멀티에이전트 시스템은 공동 목표가 모호하고 다음 작업을 위한 판단 기준도 애매합니다. 
# 이런 상황은 여러 사람이 함께 일을 할 때 종종 일어납니다. 
# 각자 맡은 일이 왜 중요한지, 잘 진행되고 있는지, 수정이 필요한지 등의 목표를 점검하는 기준이 없다면 조직의 힘은 분산되고 시너지가 나지 않습니다. 
# 책의 목차를 만드는 프로그램에서도 이와 같은 문제가 있습니다. 
# 이번 실습에서는 이 문제를 해결하기 위해 supervisor가 일을 분배하기 전에 목표를 세우는 비즈니스 분석가 에이전트 business_analyst를 만들겠습니다. 
# business_analyst는 사용자의 요구 사항과 작업의 진행 상황을 분석하여 현재 작업의 목표와 방법을 제시하는 역할을 합니다.

from dotenv import load_dotenv # pip install dotenv
import os

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List

from utils import save_state, get_outline, save_outline 
from models import Task
from tools import retrieve, web_search, add_web_pages_json_to_chroma 
from datetime import datetime

# 현재 폴더 경로 찾기
# 랭그래프 이미지로 저장 및 추후 작업 결과 파일 저장 경로로 활용
filename = os.path.basename(__file__) # 현재 파일명 반환
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로 

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o") 

# 상태 정의
class State(TypedDict):
    messages: List[AnyMessage | str]
    task_history: List[Task]    
    references: dict
    user_request: str # 사용자의 요구사항을 저장하는 변수 p456 추가

# 비즈니스 분석가 에이전트 business_analyst에게 사용자가 입력한 내용을 바탕으로 사용자의 의도를 1차로 분석하는 역할을 맡기겠습니다. 
# 이전까지는 슈퍼바이저 에이전트 supervisor가이 역할까지 수행했지만 이제 supervisor는 일 분배에 집중하고 business_analysist가 현재사용자의 요구 사황을 파악하는 데 집중하도록 만들겠습니다.

# State에 사용자의 요구 사항이 무엇인지 담아 둘 user_request를 마련합니다. 
# 이 값은 business_anaylist가 사용자와 대화한 내용과 현재의 진행 상황(목차, 참고 자료)을 바탕으로 분석하여 채울 것입니다.

def business_analyst(state: State): # business_anaylist라는 노드를 추가하기 위해 함수를 만듭니다.
    print("\n\n============ BUSINESS ANALYST ============")

    # 시스템 프롬프트를 작성합니다. 
    # business_anaylist는 목표를 위해 현 시점에서 어떤 작업을 해야 하는지 판단하는 역할입니다. 
    # 이 역할에 대해 자세히 설명하고 어떤 방식으로 답변을 생성할지 구체적으로 표현합니다. 
    # 이 판단에 사용할 자료는 지난 요구 사항, 사용자의 최근 발언, 참고 자료(벡터 검색한 결과), 현재 목차, 대화 기록입니다.
    business_analyst_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 비즈니스 애널리스트로서, 
        AI팀의 진행상황과 "사용자 요구사항"을 토대로,
        현 시점에서 '지난 요청사항 (previous_user_request)'과 최근 사용자의 발언을 바탕으로 요구사항이 무엇인지 판단한다.
        지난 요청사항이 달성되었는지 판단하고, 현 시점에서 어떤 작업을 해야 하는지 결정한다.

        다음과 같은 템플릿 형태로 반환한다. 
        ```
        - 목표: OOOO \n 방법: OOOO
        ```

        ------------------------------------
        *지난 요청사항(previous_user_request)* : {previous_user_request}
        ------------------------------------
        사용자 최근 발언: {user_last_comment}
        ------------------------------------
        참고자료: {references}
        ------------------------------------
        목차 (outline): {outline}
        ------------------------------------
        "messages": {messages}
        """
    )
    
    # 프롬프트를 언어 모델과 연결하고 문자열로 최종 출력되도록 StrOutputParser를 사용해서 ba_chain을 만듭니다.
    ba_chain = business_analyst_system_prompt | llm | StrOutputParser()

    # 이 체인에 입력할 인풋값을 설정합니다. 사용자의 마지막 발언을 가져오기 위해 기존 대화 내역(messages) 중에서 가장 뒤에 있는 HumanMessage을 user_last_comment로 정합니다.
    messages = state["messages"]

    # (3) 사용자의 마지막 발언을 가져옴
    user_last_comment = None
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_last_comment = m.content
            break

    # (3) 입력 자료 준비
    inputs = {
        "previous_user_request": state.get("user_request", None),
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline": get_outline(current_path),
        "messages": messages,
        "user_last_comment": user_last_comment
    }

    # ba_chain을 이용해 목표와 방법을 프롬프트 템플릿대로 생성하는 부분입니다. 이를 통해 사용자의 요청이 무엇인지 파악해 user_request 변수에 담습니다.
    user_request = ba_chain.invoke(inputs)

    # business_analysis가 분석해서 도출한 user_request를 AlMessage로 만들어 기존 대화 내용을 담고 있는 messages에 추가합니다.
    business_analyst_message = f"[Business Analyst] {user_request}"
    print(business_analyst_message)
    messages.append(AIMessage(business_analyst_message))

    # 현재 state를 저장합니다. 
    # 원래는 사용자가 메시지를 입력한 직후에 저장했지만 앞으로는 사용자가 입력하지 않아도 알아서 루프를 여러 번 돌 수 있으므로 여기에서도 state를 저장하도록 합니다.
    save_state(current_path, state) 

    return {
        "messages": messages,
        "user_request": user_request
    }


def supervisor(state: State): # supervisor 에이전트 추가
    print("\n\n============ SUPERVISOR ============")

    # 시스템 프롬프트 정의
    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고, 
        사용자의 요구를 달성하기 위해 현재 해야할 일이 무엇인지 결정한다.

        supervisor가 활용할 수 있는 agent는 다음과 같다.     
        - content_strategist: 사용자의 요구사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다. 
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행상황을 사용자에게 보고하고, 다음 지시를 물어본다. 
        - web_search_agent: 웹 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.
        - vector_search_agent: 벡터 DB 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.

        아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.

        ------------------------------------------
        previous_outline: {outline}
        ------------------------------------------
        messages:
        {messages}
        """
    )

    # 체인 연결
    supervisor_chain = supervisor_system_prompt | llm. with_structured_output(Task)    

    # 메시지 가져오기
    messages = state.get("messages", [])		#⑤

    # inputs 설정
    inputs = {
        "messages": messages,
        "outline": get_outline(current_path)
    }

    # task 문자열로 생성
    task = supervisor_chain.invoke(inputs) 	#⑦
    task_history = state.get("task_history", [])    # 작업 이력 가져오기
    task_history.append(task)                    	# 작업 이력에 추가

   
    # 메시지 추가
    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)

    # state 업데이트
    return {
        "messages": messages, 
        "task_history": task_history
    }

# supervisor's route
def supervisor_router(state: State):
    task = state['task_history'][-1]
    return task.agent			

def vector_search_agent(state: State):
    print("\n\n============ VECTOR SEARCH AGENT ============")
    
    tasks = state.get("task_history", [])
    task = tasks[-1]
    if task.agent != "vector_search_agent":
        raise ValueError(f"Vector Search Agent가 아닌 agent가 Vector Search Agent를 시도하고 있습니다.\n {task}")

    vector_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로, 
        목차(outline) 작성에 필요한 정보를 벡터 검색을 통해 찾아내는 Agent이다.

        현재 목차(outline)을 작성하는데 필요한 정보를 확보하기 위해, 
        다음 내용을 활용해 적절한 벡터 검색을 수행하라. 

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목차(outline): {outline}
        """
    )

    # inputs 설정
    mission = task.description
    references = state.get("references", {"queries": [], "docs": []})
    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {
        "mission": mission,
        "references": references,
        "messages": messages,
        "outline": outline
    }

    # LLM과 벡터 검색 모델 연결
    llm_with_retriever = llm.bind_tools([retrieve]) 
    vector_search_chain = vector_search_system_prompt | llm_with_retriever

    # LLM과 벡터 검색 모델 연결
    search_plans = vector_search_chain.invoke(inputs)
    # 검색할 내용 출력
    for tool_call in search_plans.tool_calls:
        print('-----------------------------------', tool_call)
        args = tool_call["args"]
       
        query = args["query"] 
        retrieved_docs = retrieve(args)
		#① (1) 결과 담아 두기
        references["queries"].append(query) 
        references["docs"] += retrieved_docs
    
    unique_docs = []
    unique_page_contents = set()

    for doc in references["docs"]:
        if doc.page_content not in unique_page_contents:
            unique_docs.append(doc)
            unique_page_contents.add(doc.page_content)
    references["docs"] = unique_docs

    # 검색 결과 출력 – 쿼리 출력
    print('Queries:--------------------------')
    queries = references["queries"]
    for query in queries:
        print(query)
    
    # 검색 결과 출력 – 문서 청크 출력
    print('References:--------------------------')
    for doc in references["docs"]:
        print(doc.page_content[:100])
        print('--------------------------')

    # task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 새로운 task 추가
    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at=""
    )
    tasks.append(new_task)

    # vector search agent의 작업후기를 메시지로 생성
    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    message = AIMessage(msg_str)
    print(msg_str)

    messages.append(message)
    # state 업데이트
    return {
        "messages": messages,
        "task_history": tasks,
        "references": references
    }


# 목차를 작성하는 노드(agent)
def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")

    # 시스템 프롬프트 정의
    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
        목차를 작성하는데 필요한 정보는 "참고 자료"에 있으므로 활용한다. 

        --------------------------------
        - 지난 목차: {outline}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 참고 자료: {references}
        """
    )

    # 시스템 프롬프트와 모델을 연결
    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]        # 상태에서 메시지를 가져옴
    outline = get_outline(current_path) # 저장된 목차를 가져옴

    # 입력값 정의
    inputs = {
        "messages": messages,
        "outline": outline, 
        "references": state.get("references", {"queries": [], "docs": []})
    }

    # 목차 작성
    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end='')

    print()

    save_outline(current_path, gathered) # 목차 저장

    # 메시지 추가    
    content_strategist_message = f"[Content Strategist] 목차 작성 완료"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    task_history = state.get("task_history", []) # task_history 가져오기
    # 최근 task 작업완료(done) 처리하기
    if task_history[-1].agent != "content_strategist": 
        raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")
    
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 다음 작업이 communicator로 사용자와 대화하는 것이므로 새 작업 추가 
    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at=""
    )
    task_history.append(new_task)

    print(new_task)

    # 현재 state를 업데이트한다. 
    return {
        "messages": messages,
        "task_history": task_history
    }


def web_search_agent(state: State): #① (0)
    print("\n\n============ WEB SEARCH AGENT ============")

    # 작업 리스트 가져와서 web search agent 가 할 일인지 확인하기
    tasks = state.get("task_history", [])
    task = tasks[-1]

    if task.agent != "web_search_agent":
        raise ValueError(f"Web Search Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}")
    
    #③ 시스템 프롬프트 정의
    web_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로, 
        목차(outline) 작성에 필요한 정보를 웹 검색을 통해 찾아내는 Web Search Agent이다.

        현재 부족한 정보를 검색하고, 복합적인 질문은 나눠서 검색하라.

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목차(outline): {outline}
        --------------------------------
        - 현재 시각 : {current_time}
        """
    )
    
    #④ 기존 대화 내용 가져오기
    messages = state.get("messages", [])

    #⑤ 인풋 자료 준비하기
    inputs = {
        "mission": task.description,
        "references": state.get("references", {"queries": [], "docs": []}),
        "messages": messages,
        "outline": get_outline(current_path),
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    #⑥ LLM과 웹 검색 모델 연결
    llm_with_web_search = llm.bind_tools([web_search])

    #⑦ 시스템 프롬프트와 모델을 연결
    web_search_chain = web_search_system_prompt | llm_with_web_search

    #⑧ 웹 검색 tool_calls 가져오기
    search_plans = web_search_chain.invoke(inputs)

    #⑨ 어떤 내용을 검색했는지 담아두기
    queries = []

    #⑩ 검색 계획(tool_calls)에 따라 검색하기
    for tool_call in search_plans.tool_calls:
        print('-------- web search --------', tool_call)
        args = tool_call["args"]
        
        queries.append(args["query"])

        # (10)  검색 결과를 chroma에 추가
        _, json_path = web_search.invoke(args)
        print('json_path:', json_path)

        # (10)  JSON 파일을 chroma에 추가
        add_web_pages_json_to_chroma(json_path)

    #⑪ (11) task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    #⑪ (11) 새로운 task 추가
    task_desc = "AI팀이 쓸 책의 세부 목차를 결정하기 위한 정보를 벡터 검색을 통해 찾아낸다."
    task_desc += f" 다음 항목이 새로 추가되었다\n: {queries}"
    
    new_task = Task(
        agent="vector_search_agent",
        done=False,
        description=task_desc,
        done_at=""
    )

    tasks.append(new_task)

    #⑫ (12) 작업 후기 메시지
    msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    messages.append(AIMessage(msg_str))

    #⑬ (13) state 업데이트
    return {
        "messages": messages,
        "task_history": tasks
    }


# 사용자와 대화할 노드(agent): communicator
def communicator(state: State):
    print("\n\n============ COMMUNICATOR ============")

    # 시스템 프롬프트 정의
    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서, 
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다. 

        사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.
        outline: {outline} 
        --------------------------------
        messages: {messages}
        """
    )

    #② 시스템 프롬프트와 모델을 연결
    system_chain = communicator_system_prompt | llm

    # 상태에서 메시지를 가져옴
    messages = state["messages"]

    # 입력값 정의
    inputs = {
        "messages": messages,
        "outline": get_outline(current_path)
    }

    # 스트림되는 메시지를 출력하면서, gathered에 모으기
    gathered = None

    print('\nAI\t: ', end='')
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    messages.append(gathered)

    task_history = state.get("task_history", []) 
    if task_history[-1].agent != "communicator":
        raise ValueError(f"Communicator가 아닌 agent가 대화를 시도하고 있습니다.\n {task_history[-1]}")
    
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        "messages": messages,
        "task_history": task_history
    }


# 상태 그래프 정의
graph_builder = StateGraph(State)

# Nodes
# 그래프에 business_analyst 노드와 엣지를 추가합니다. 
# 이전에는 사용자의 입력 내용이 supervisor에서 처음으로 처리되었지만, 이제는 business_analyst가 먼저 사용자의 입력을 받아서 목표가 무엇인지 판단하는 구조로 변경했습니다. 
# 초기 state 설정도 user_request가 포함되도록 수정합니다
graph_builder.add_node("business_analyst", business_analyst) # p458 추가
graph_builder.add_node("supervisor", supervisor)     
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)

# Edges
graph_builder.add_edge(START, "business_analyst") # p458 수정
graph_builder.add_edge("business_analyst", "supervisor") # p458 추가
graph_builder.add_conditional_edges(
    "supervisor", 
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator",
        "vector_search_agent": "vector_search_agent", 
        "web_search_agent": "web_search_agent"
    }
)
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("web_search_agent", "vector_search_agent") #③
graph_builder.add_edge("vector_search_agent", "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))

# 상태 초기화
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
    task_history=[], 
    references={"queries": [], "docs": []}, # p458 추가
    user_request="" # p458 추가
)

while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break
    
    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n------------------------------------ MESSAGE COUNT\t', len(state["messages"]))

    save_state(current_path, state) # 현재 state 내용 저장


# (ch13_env) PS C:\Aiprojects\ch14\ch14-4> 
# (ch13_env) PS C:\Aiprojects\ch14\ch14-4> cd ..
# (ch13_env) PS C:\Aiprojects\ch14> cd ..
# (ch13_env) PS C:\Aiprojects> cd ch15
# (ch13_env) PS C:\Aiprojects\ch15> cd ch15-1
# (ch13_env) PS C:\Aiprojects\ch15\ch15-1> python .\book_writer.py 
# USER_AGENT environment variable not set, consider setting it to identify your requests.
# Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
# Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given

# User    : HYBE와 JYP를 비교하는 책 쓰자. CEO와 경영전략에 대해 써줘. 


# ============ BUSINESS ANALYST ============
# [Business Analyst] ```
# - 목표: HYBE와 JYP를 비교하는 책 작성
#  방법: 각 회사의 CEO와 경영전략에 대한 정보를 수집하고 분석하여 비교하는 내용을 작성한다. 
#         - HYBE와 JYP의 CEO들의 리더십 스타일과 배경, 경영철학 등을 조사한다.
#         - 두 회사의 경영전략, 사업 확장, 시장 접근 방식 등을 분석한다.
#         - 비교 분석을 위한 자료를 정리하여 목차를 작성하고, 이를 바탕으로 챕터별로 내용을 구성한다.
# ```


# ============ SUPERVISOR ============
# [Supervisor] agent='content_strategist' done=False description='사용자가 HYBE와 JYP를 비교하는 책을 원하며, CEO와 경영전략에 대해 작성하도록 요청하였습니다. 이를 바탕으로 책의 목차를 작성합니다.' done_at='2026-04-27T14:46:25'


# ============ CONTENT STRATEGIST ============
# HYBE와 JYP를 비교하는 책의 세부 목차를 제안드립니다. 이 목차는 이전 대화에서 요청하신 CEO와 경영전략에 대한 내용을 중점으로 구성되었습니다.

# ### 목차 제안: HYBE와 JYP 비교 분석

# 1. **서문**
#    - 책의 목적과 구성 소개
#    - HYBE와 JYP의 중요성 및 선정 이유
   
# 2. **HYBE와 JYP 개요**
#    - 두 회사의 역사와 배경
#    - 주요 성과 및 업적

# 3. **CEO 분석**
#    - HYBE의 CEO
#      - 리더십 스타일
#      - 경영 철학과 비전
#    - JYP의 CEO
#      - 리더십 스타일
#      - 경영 철학과 비전
#    - 두 CEO의 비교

# 4. **경영전략 비교**
#    - HYBE의 경영전략
#      - 사업 확장 및 혁신 사례
#      - 시장 접근 방식
#    - JYP의 경영전략
#      - 사업 확장 및 혁신 사례
#      - 시장 접근 방식
#    - 전략적 차이점과 공통점

# 5. **사업 확장 및 시장 접근**
#    - 글로벌 시장에서의 활동
#    - 팬덤 관리 전략
#    - 멀티미디어와 기술 활용

# 6. **재무 성과 분석**
#    - 최근 5년간의 재무 성과 비교
#    - 투자 및 수익 구조
  
# 7. **미래 전망**
#    - 엔터테인먼트 산업의 트렌드
#    - HYBE와 JYP의 미래 전략 예측

# 8. **결론**
#    - 비교 분석 결과 요약
#    - HYBE와 JYP의 향후 과제와 기회

# 9. **부록**
#    - 추가 자료 및 참고 문헌
#    - 인터뷰 및 참고한 자료 목록

# 이 목차는 HYBE와 JYP의 CEO와 경영전략에 중점을 두어 두 회사의 본질적 차이와 유사성을 심층적으로 탐구하고자 합니다. 추가하고 싶은 내용이나 수정할 부분이 있다면 말씀해 주세요.
# [Content Strategist] 목차 작성 완료
# agent='communicator' done=False description='AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다' done_at=''


# ============ COMMUNICATOR ============

# AI      : 안녕하세요! 현재 AI팀은 HYBE와 JYP를 비교하는 책의 목차를 구성하는 데 성공했습니다. 이 책은 두 회사의 CEO와 경영전략에 대한 깊이 있는 분석을 통해 그들의 본질적 차이와 유사성을 탐구하는 것을 목표로 하고 있습니다. 목차는 이미 보셨다고 하니, 진행 상황에 대해 말씀드리겠습니다.

# 현재 우리는 각 챕터에 필요한 데이터를 수집하고 분석하는 단계에 있습니다. HYBE와 JYP의 CEO들의 리더십 스타일, 경영 철학, 그리고 두 회사의 경영전략 및 사업 확장 방식을 자세히 조사하고 있습니다. 이를 통해 두 회사의 전략적 차이점과 공통점을 명확히 할 수 있을 것입니다.

# 혹시 추가하고 싶은 내용이나 수정할 부분이 있으신가요? 어떤 의견이든 자유롭게 말씀해 주시면, 책의 내용에 반영하는 데 큰 도움이 될 것입니다.
# ------------------------------------ MESSAGE COUNT       6
