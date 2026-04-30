# 스스로 리뷰하고 수정하는 에이전트로 발전시키기

# 이제 AI 에이전트들이 회의하여 작성한 목차를 보면서 사람이 검토하고 수정 방향을 알려 줄수 있습니다. 
# 그런데 인공지능이 스스로 내용을 검토하고 수정할 수 있다면 더욱 편리하겠죠? 
# 작업한 결과를 확인하는 목차 리뷰 에이전트를 만들어 사람이 확인하지 않아도 
# AI 에이전트가 파악할 수 있는 문제점은 스스로 해결하고 더 발전된 결과를 제공할 수 있도록 만들어 보겠습니다.

# 목차 리뷰 에이전트
# 콘텐츠 전략가 에이전트 content_strategist가 목차를 잘 작성했는지 판단하는 역할을 하는 목차 검토 에이전트 outline_reviewer를 만들겠습니다. 
# outline_reviewer가 목차를 보고 '이런 부분은 이렇게 수정하면 좋겠다'고 조언하면 business_analyst가 어떤 작업을 할지 판단해서 다시 작업을 진행하는 방식입니다.

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
from dotenv import load_dotenv # pip install dotenv
import os

load_dotenv()

# 현재 폴더 경로 찾기
# 랭그래프 이미지로 저장 및 추후 작업 결과 파일 저장 경로로 활용
filename = os.path.basename(__file__) # 현재 파일명 반환
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로 

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o") 

# 상태 정의
# 목차 조언 항목 추가하고 business_analyst에 반영하기

# 목차 검토 에이전트 outline_reviewer가 목차를 분석하고 조언하는 내용을 상태에 담아 비즈니스 분석가 에이전트 business_analyst가 활용할 수 있도록 만들어 보겠습니다. 
# 지금까지 계속 작업했던 book_writer.py 파일을 수정하면 됩니다.

# 목차를 리뷰하는 에이전트를 만들기 전에, State에 이 에이전트가 목차를 분석해 조언한 내용을 담아 둘 ai_recommendation 변수를 새로 만듭니다.
class State(TypedDict):
    messages: List[AnyMessage | str]
    task_history: List[Task]    
    references: dict
    user_request: str 
    ai_recommendation: str # AI의 추천을 저장하는 변수 p474 추가

    # 무한루프 p482
    # 현재 구조에서는 AI 종종 에이전트끼리 계속 작업을 주고받으면서 무한히 개선하려 하는 경우가 있습니다. 
    # 이런 경우 크게 개선되는 요소는 없이 GPT 토큰과 타빌리 검색 쿼리만 소모하게 됩니다. 
    # 시간도 오래 걸리고 사용자의 의견을 추가로 반영할 시간도 주지 않는 문제가 있습니다.
    # 결국에는 재귀 한계recursion limit에 걸려 다음과 같이 오류를 뱉습니다. 
    # 이 오류는 토큰을 무한히 소모하는 사람들을 위해 랭그래프에서 만들어 둔 안전 장치입니다. 
    # 최대 25바퀴 이상 돌지 못하게 기본으로 설정되어 있고, 필요하다면 시도 횟수를 더 늘릴 수 있는 옵션도 제공합니다.

    supervisor_call_count: int # supervisor 호출 횟수를 저장하는 변수 p482 추가
    # supervisor가 2회 이하로 호출되었다면 원래대로 다음 작업을 적절하게 선택하도록 하고 2회를 초과할 경우 사용자와 대화하는 communicator가 다음 작업이 되도록 지정합니다.
    # 그리고 state 값을 반환할 때는 supervisor_call_count에 1을 더한 상태로 업데이트하여 반복 호출을 추적할 수 있도록 합니다.
    
def business_analyst(state: State): #
    print("\n\n============ BUSINESS ANALYST ============")

    #② (1) 시스템 프롬프트 정의
    business_analyst_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 비즈니스 애널리스트로서, 
        AI팀의 진행상황과 "사용자 요구사항"을 토대로,
        현 시점에서 'ai_recommendation'과 최근 사용자의 발언을 바탕으로 요구사항이 무엇인지 판단한다.
        지난 요청사항이 달성되었는지 판단하고, 현 시점에서 어떤 작업을 해야 하는지 결정한다.

        다음과 같은 템플릿 형태로 반환한다. 
        ```
        - 목표: OOOO \n 방법: OOOO
        ```

        ------------------------------------
        *AI 추천(ai_recommendation)* : {ai_recommendation}  
        # p475추가 '사용자의 이전 요구 사항'을 이용해서 판단하라고 되어 있던 시스템 프롬프트를 ai_recommendation을 바탕으로 판단하도록 수정합니다.
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
    
    #③ (2) 시스템 프롬프트와 모델을 연결
    ba_chain = business_analyst_system_prompt | llm | StrOutputParser()

    #(3) 상태에서 메시지를 가져옴
    messages = state["messages"]

    # (3) 사용자의 마지막 발언을 가져옴
    user_last_comment = None
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_last_comment = m.content
            break

    # (3) 입력 자료 준비
    inputs = {
        "ai_recommendation": state.get("ai_recommendation", None), # p475 추가 
        # 프롬프트가 수정되었으므로 이에 맞춰 inputs도 수정합니다. State에 ai_recommendation을 추가했으므로 이 값을 이용합니다.
        "previous_user_request": state.get("user_request", None),
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline": get_outline(current_path),
        "messages": messages,
        "user_last_comment": user_last_comment
    }

    #⑤ (4) 시스템 프롬프트를 통해 사용자 요구사항을 분석
    user_request = ba_chain.invoke(inputs)

    #⑥ (5) businessage analyst의 결과를 메시지에 추가
    business_analyst_message = f"[Business Analyst] {user_request}"
    print(business_analyst_message)
    messages.append(AIMessage(business_analyst_message))

    save_state(current_path, state) #⑦ (6) 현재 state 내용 저장

    return {
        "messages": messages,
        "user_request": user_request,
        "ai_recommendation": "" # p475 추가 
        # ai_recommendation은 business_analyst가 사용자의 의도와 현재 상황을 
        # 바탕으로 무엇을 해야 할지 판단 하는 데만 필요하므로 다른 부분에 실수로 영향을 끼치지 않도록 처리합니다. 
        # state에서 ai_recommendation을 빈 값으로 만들기 위해 ""로 업데이트합니다.
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
        - web_search_agent: vector_search_agent를 시도하고, 검색 결과(references)에 필요한 정보가 부족한 경우 사용한다. 웹 검색을 통해 해당 정보를 Vector DB에 보강한다. 
        - vector_search_agent: 목차 작성을 위해 필요한 자료를 확보하기 위해 벡터 DB 검색을 한다. 

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

    supervisor_call_count = state.get("supervisor_call_count", 0) # p483 추가

    if supervisor_call_count > 2:
        print("Supervisor 호출 횟수 초과: Communicator 호출")
        task = Task(
            agent="communicator", 
            done=False, 
            description="supervisor 호출 횟수 초과했으므로, 현재까지의 진행상황을 사용자에게 보고한다. ",
            done_at="", 
        )
    else:
        task = supervisor_chain.invoke(inputs) # p483 추가 끝

    task_history = state.get("task_history", [])    # 작업 이력 가져오기
    task_history.append(task)                    	# 작업 이력에 추가
   
    # 메시지 추가
    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)

    # state 업데이트
    return {
        "messages": messages, 
        "task_history": task_history, 
        "supervisor_call_count": supervisor_call_count + 1
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

    # # communicator로 보내는 Task 생성 코드 삭제 p479
    # 벡터 검색 에이전트 vector_search_agent도 비즈니스 분석가 에이전트 business_analyzer에게 조언할 수 있게 수정해 보겠습니다. 
    # 앞에서 설계한 그래프를 떠올려 보면 벡터 DB에서 관련 문서를 검색하는 vector_search_agent도 작업이 종료되면 business_analyzer로 연결됩니다. 
    # 따라서 기존 코드에서 작업을 종료한 후 커뮤니케이터 에이전트 communicator로 연결되는 작업을 생성하는 대신 정보를 business_analyzer로 전달할 수 있도록 수정해 보겠습니다.

    # book_writer.py에서 목차 작성이 종료된 후 communicator로 연결되는 작업을 생성하는 코드를 삭제합니다. 
    # 그 대신 ai_recommendation을 작성할 수 있도록 수정합니다. 
    # 벡터 검색 결과로 자료가 충분히 모였다면 목차를 수정하거나 개선할 수 있도록 content_strategist 를 추천하는 메시지를 작성합니다. 
    # 만약 벡터 검색으로 자료를 충분히 찾지 못했다면 AI 에이전트들이 알아서 다음 업무를 선택할 것입니다.
    # new_task = Task(
    #     agent="communicator",
    #     done=False,
    #     description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
    #     done_at=""
    # )
    # tasks.append(new_task)

    # vector search agent의 작업후기를 메시지로 생성
    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    message = AIMessage(msg_str)
    print(msg_str)

    messages.append(message)
    ai_recommendation = "현재 참고자료(references)가 목차(outline)를 개선하는데 충분한지 확인하라. 충분하다면 content_strategist로 목차 작성을 하라. " # p479 추가
    
    # state 업데이트
    return {
        "messages": messages,
        "task_history": tasks,
        "references": references,
        "ai_recommendation": ai_recommendation # p479 추가
    }


# 목차를 작성하는 노드(agent)
def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")
    
    task_history = state.get("task_history", []) # 작업 이력 가져오기 (아래에 있던 코드 위로 이동)
    task = task_history[-1]
    if task.agent != "content_strategist":
        raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task}")
    
    # 시스템 프롬프트 정의
    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
        목차를 작성하는데 필요한 정보는 "참고 자료"에 있으므로 활용한다. 
        
        다음 정보를 활용하여 목차를 작성하라. 
        - 사용자 요구사항(user_request)
        - 작업(task)
        - 검색 자료 (references)
        - 기존 목차 (previous_outline)
        - 이전 대화 내용(messages)

        너의 작업 목표는 다음과 같다:
        1. 만약 "기존 목차 구조 (previous_outline)"이 존재한다면, 사용자의 요구사항을 토대로 "기존 목차 구조"에서 어떤 부분을 수정하거나 추가할지 결정한다.
        - "이번 목차 작성의 주안점"에 사용자 요구사항(user_request)을 충족시키는 것을 명시해야 한다.
        2. 책의 전반적인 구조(chapter, section)를 설계하고, 각 chpater와 section의 제목을 정한다.
        3. 책의 전반적인 세부구조(chapter, section, sub-section)를 설계하고, sub-section 하부의 주요 내용을 리스트 형태로 정리한다.
        4. 목차의 논리적인 흐름이 사용자 요구를 충족시키는지 확인한다.
        5. 참고자료 (references)를 적극 활용하여 근거에 기반한 목차를 작성한다.
        6. 참고문헌은 반드시 참고자료(references) 자료를 근거로 작성해야 하며, 최대한 풍부하게 준비한다. URL은 전체 주소를 적어야 한다.
        7. 추가 자료나 리서치가 필요한 부분을 파악하여 supervisor에게 요청한다.

        사용자 요구사항(user_request)을 최우선으로 반영하는 목차로 만들어야 한다. 

        --------------------------------
        - 사용자 요구사항(user_request): 
        {user_request}
        --------------------------------
        - 작업(task): 
        {task}
        --------------------------------
        - 참고 자료 (references)
        {references}
        --------------------------------
        - 기존 목차 (previous_outline)
        {outline}
        --------------------------------
        - 이전 대화 내용(messages)
        {messages}
        --------------------------------

        작성 형식 아래 양식을 지키되 하부 항목으로 더 세분화해도 좋다. 목차(outline) 양식의 챕터, 섹션 등 항목의 갯수는 필요한만큼 추가하라. 
        섹션 갯수는 최소 2개 이상이어야 하며, 더 많으면 좋다. 

        outline_template은 예시로 앞부분만 제시한 것이다. 각 장은 ':---CHAPTER DIVIDER---:'로 구분한다.
        outline_template:
        {outline_template}

        사용자가 추가 피드백을 제공할 수 있도록 논리적인 흐름과 주요 목차 아이디어를 제안하라.    
        """
    )

    # 시스템 프롬프트와 모델을 연결
    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    user_request = state.get("user_request", "") # 사용자 요구사항 가져오기

    messages = state["messages"]        # 상태에서 메시지를 가져옴
    outline = get_outline(current_path) # 저장된 목차를 가져옴

    with open(f"{current_path}/templates/outline_template.md", "r", encoding='utf-8') as f:
        outline_template = f.read()  

    # 입력값 정의
    inputs = {
        "user_request": user_request,  #사용자 요구사항 user_request
        "task": task,
        "messages": messages,
        "outline": outline, 
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline_template": outline_template # 템플릿 이용하기
    }

    # 목차 작성
    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end='')

    print()

    save_outline(current_path, gathered) # 목차 저장

    # 템플릿을 활용한 작업 후기 메시지 찾기
    if '-----: DONE :-----' in gathered:
        review = gathered.split('-----: DONE :-----')[1]
    else: 
        review = gathered[-200:]


    # 메시지 추가    
    content_strategist_message = f"[Content Strategist] 목차 작성 완료: outline 작성 완료\n {review}"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    # task_history = state.get("task_history", []) # task_history 가져오기
    # # 최근 task 작업완료(done) 처리하기
    # if task_history[-1].agent != "content_strategist": 
    #     raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")
    
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # # 다음 작업이 communicator로 사용자와 대화하는 것이므로 새 작업 추가 
    # new_task = Task(
    #     agent="communicator",
    #     done=False,
    #     description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
    #     done_at=""
    # )
    # task_history.append(new_task)
    # print(new_task)

    # 현재 state를 업데이트한다. 
    return {
        "messages": messages,
        "task_history": task_history
    }

# p476 추가
# 이제 목차를 리뷰하는 에이전트 outline_reviewer를 만들겠습니다. 
# outline_reviewer는 content_strategist가 목차를 생성한 뒤 자동으로 실행될 예정입니다.
def outline_reviewer(state: State): # (0)
    print("\n\n============ OUTLINE REVIEWER ============")

    # outline_reviewer의 시스템 프롬프트는 지금까지 content_strategist가 만든 목차의 문제점이 무엇인지 파악하는데 초점을 맞춰야 합니다. 
    # 목차 구성이 사용자의 요구 사항에 맞는지, 근거 없이 작성하지 않았는지, 참고 자료를 충분히 활용했는지 등을 체크리스트로 확인하도록 합니다. 
    # 예를 들어 목차를 만들 때 참고하는 웹 페이지 중 'example.com' 같은 더미 URL이나 가짜 URL 이 남아 있는지 확인합니다.
    outline_reviewer_system_prompt = PromptTemplate.from_template(
        """
        너는 AI팀의 목차 리뷰어로서, AI팀이 작성한 목차(outline)를 검토하고 문제점을 지적한다. 

        - outline이 사용자의 요구사항을 충족시키는지 여부
        - outline의 논리적인 흐름이 적절한지 여부
        - 근거에 기반하지 않은 내용이 있는지 여부
        - 주어진 참고자료(references)를 충분히 활용했는지 여부
        - 참고자료가 충분한지, 혹은 잘못된 참고자료가 있는지 여부
        - example.com 같은 더미 URL이 있는지 여부: 
        - 실제 페이지 URL이 아닌 대표 URL로 되어 있는 경우 삭제 해야함: 어떤 URL이 삭제되어야 하는지 명시하라.
        - 기타 리뷰 사항

        그 분석 결과를 설명하고, 다음 어떤 작업을 하면 좋을지 제안하라.
        
        - 분석결과: outline이 사용자의 요구사항을 충족시키는지 여부
        - 제안사항: (vector_search_agent, communicator 중 어떤 agent를 호출할지)

        ------------------------------------------
        user_request: {user_request}
        ------------------------------------------
        references: {references}
        ------------------------------------------
        outline: {outline}
        ------------------------------------------
        messages: {messages}
        """
    )
    
    # outline_reviewer의 프롬프트에 맞게 inputs를 설정합니다. content_strategist에서 생성된 목차와 관련 정보를 전달해야 합니다
    user_request = state.get("user_request", None)
    outline = get_outline(current_path)
    references = state.get("references", {"queries": [], "docs": []})
    messages = state.get("messages", [])

    inputs = {
        "user_request": user_request,
        "outline": outline,
        "references": references,
        "messages": messages
    }

    # 시스템 프롬프트와 모델을 연결
    outline_reviewer_chain = outline_reviewer_system_prompt | llm

    # 목차 리뷰가 길어질 수 있으므로 출력 방식을 .invoke가 아닌 .stream 방식으로 설정하여 리뷰 결과를 터미널 창에서 스트림 방식으로 확인할 수 있게 합니다.
    review = outline_reviewer_chain.stream(inputs)

    gathered = None

    for chunk in review:
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    # outline_review 에이전트의 작업 후기를 메시지에 추가
    # 목차 리뷰가 완료되면 이 내용을 기존 대화 목록인 messages에 outline_reviewer의 작업 후기로 추가합니다.
    if '[OUTLINE REVIEW AGENT]' not in gathered.content:
        gathered.content = f"[OUTLINE REVIEW AGENT] {gathered.content}"

    print(gathered.content)
    messages.append(gathered)

    # 리뷰 결과는 ai_recommendation의 값으로도 활용합니다.
    ai_recommendation = gathered.content

    # 수정한 state를 업데이트하기 위해 이 값을 반환합니다.
    return {"messages": messages, "ai_recommendation": ai_recommendation}

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
        "task_history": task_history,
        "supervisor_call_count": 0 # p484 추가
        # 사용자에게 진행 상황을 보고하는 커뮤니티케이터 에이전트 communicator가 작동되었을 때는 supervisor_call_count를 0으로 리셋해야 합니다. 
        # 그래야 또다시 supervisor의 호출이 2회를 초과할 때까지는 AI 에이전트들끼리 알아서 작업할 수 있으니까요.`
    }


# 상태 그래프 정의
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("business_analyst", business_analyst)
graph_builder.add_node("supervisor", supervisor)     
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("outline_reviewer", outline_reviewer)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)

# Edges
graph_builder.add_edge(START, "business_analyst")
graph_builder.add_edge("business_analyst", "supervisor")
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

# 새로 만든 목차 리뷰 에이전트 outline_reviewer를 노드로 추가하고 edge를 시스템의 흐름에 맞게 설정합니다. 
# content_strategist의 작업이 종료되면 곧바로 outline_reviewer로 작업이 넘어갑니다. 
# 그리고 outline_reivewer의 작업이 종료되면 business_analyst로 전달되어 후속 작업이 진행되도록 처리합니다.
graph_builder.add_edge("content_strategist", "outline_reviewer") # p480 추가
graph_builder.add_edge("outline_reviewer", "business_analyst")  # p480 추가
graph_builder.add_edge("web_search_agent", "vector_search_agent") 
graph_builder.add_edge("vector_search_agent", "business_analyst")
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
    references={"queries": [], "docs": []}, 
    user_request=""
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


# 실행 결과
# (ch13_env) PS C:\Aiprojects\ch15\ch15-3> python .\book_writer.py
# USER_AGENT environment variable not set, consider setting it to identify your requests.
# Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
# Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given

# User    : ai 프롬프트와 ai 에이전트관련 교재를 만들어 보자.


# ============ BUSINESS ANALYST ============
# [Business Analyst] ```
# - 목표: AI 프롬프트와 AI 에이전트에 관한 교재 제작
# - 방법: 
#   1. AI 프롬프트와 AI 에이전트 관련한 최신 자료 및 연구를 조사하여 내용 구성.
#   2. 사용자 요구사항에 맞춘 목차 작성 및 세부 내용 기획.
#   3. 기존 '사용자의 이전 요구 사항' 고려하여 ai_recommendation 시스템 프롬프트 수정.
#   4. 초안 작성 후 피드백을 통해 내용 수정 및 보완.
# ```


# ============ SUPERVISOR ============
# [Supervisor] agent='content_strategist' done=False description='사용자의 요구사항에 맞춘 AI 프롬프트와 AI 에이전트 관련 교재의 목차를 작성한다.' done_at=''


# ============ CONTENT STRATEGIST ============
# # 기획의도 및 제시하고자 하는 메시지  

# ## 책의 기획 의도와 독자들에게 전달하고자 하는 메시지를 더욱 구체적으로 서술합니다.  
#   1. AI 프롬프트와 AI 에이전트의 중요성 및 활용 가능성을 소개합니다.
#   2. 최신 연구와 실무 사례를 통해 독자들이 쉽게 이해할 수 있도록 설명합니다.
#   3. AI 기술을 통해 창의적이고 효율적인 문제 해결을 돕고자 합니다.
  
# ## 이번 목차 작성의 주안점
#   1. 사용자 요구사항에 맞춘 AI 프롬프트와 AI 에이전트 관련 최신 자료와 연구 기반 내용 구성.
#   2. 실무적 관점에서 AI 프롬프트와 에이전트를 활용하는 방법론 제시.
#   3. 피드백을 통해 초안 완성도를 높일 수 있도록 유연한 구조 설계.

# ---------------------------------------------------------------------------------
# ---
# # 보고서 제목
# ## 보고서 부제목: AI 혁신을 이끄는 프롬프트와 에이전트의 세계

# # chapter 제목 및 내용 간략 소개
# ## Chapter 1: AI 프롬프트와 에이전트의 기본 개념
# - AI 프롬프트와 에이전트의 기본 개념과 역사적 배경을 이해합니다.

# ## Chapter 2: AI 프롬프트 설계와 효과적인 사용법
# - 프롬프트 설계의 원리와 실제 사례를 통해 효과적인 사용법을 제시합니다.

# ## Chapter 3: AI 에이전트의 작동 원리와 구현
# - AI 에이전트의 작동 원리와 구현 방법을 심층적으로 탐구합니다.

# ## Chapter 4: AI 프롬프트 및 에이전트의 최신 동향
# - 최신 연구 자료를 통해 AI 프롬프트와 에이전트의 발전 동향을 분석합니다.

# ## Chapter 5: 실무 적용 사례와 향후 전망
# - 다양한 산업에서의 실제 적용 사례와 AI 프롬프트 및 에이전트의 미래 전망을 탐구합니다.

# ---
# # 세부 목차
# ---------------------------------------------------------------------------------

# :---OUTLINE STARTS HERE:---:

# ## Chapter 1: AI 프롬프트와 에이전트의 기본 개념  
# - **Chapter 목적:** AI 프롬프트와 에이전트의 근본적인 이해를 돕고, 관련 기초 지식을 제공합니다.  
# - **Chapter 내용:** AI 기술의 역사적 발전 및 프롬프트와 에이전트의 정의, 역할, 필요성을 설명합니다.  

# ### Section 1.1: AI의 역사와 발전  
# - **Section 목적:** AI의 발전 과정을 이해하고, 프롬프트와 에이전트의 출현 배경을 설명합니다.  
# - **Section 내용:** AI의 주요 발전 단계, 기술적 진보, 프롬프트와 에이전트의 출현 배경.  
# - **주요 내용:**  
#   - AI의 역사적 배경.  
#   - 프롬프트와 에이전트의 초기 개발 사례.  
#   - ...  
# - **참고문헌:**  
#   1. [AI의 역사와 발전](https://example.com): AI의 발전 과정을 상세히 설명한 문헌.  
#   2. [프롬프트와 에이전트의 출현](https://example.com): 초기 개발 사례를 소개한 자료.  

# ### Section 1.2: 프롬프트와 에이전트의 정의  
# - **Section 목적:** 프롬프트와 에이전트의 정의 및 역할을 명확히 합니다.  
# - **Section 내용:** 각 용어의 정의, 역할, 중요성에 대해 설명합니다.  
# - **주요 내용:**  
#   - 프롬프트의 정의와 기능.  
#   - 에이전트의 역할과 필요성.  
#   - ...  
# - **참고문헌:**  
#   1. [프롬프트의 정의](https://example.com): 프롬프트의 개념과 기능을 설명한 자료.  
#   2. [에이전트의 역할](https://example.com): 에이전트의 필요성을 논의한 연구.  

# :---CHAPTER DIVIDER---:

# ## Chapter 2: AI 프롬프트 설계와 효과적인 사용법
# - **Chapter 목적:** 효과적인 AI 프롬프트 설계 방법 및 실제 활용 사례를 제시합니다.  
# - **Chapter 내용:** 프롬프트 설계의 원리, 실무 적용 사례, 디자인 전략 등을 다룹니다.  

# ### Section 2.1: 프롬프트 설계 원리  
# - **Section 목적:** 효과적인 프롬프트 설계의 기본 원리를 설명합니다.  
# - **Section 내용:** 프롬프트의 구성 요소, 설계 전략, 영향력 분석.  
# - **주요 내용:**  
#   - 프롬프트의 구성 요소 분석.  
#   - 설계 전략 및 사례 연구.  
#   - ...  
# - **참고문헌:**  
#   1. [프롬프트 설계의 원리](https://example.com): 프롬프트 구성과 전략을 설명한 자료.  

# ### Section 2.2: 실제 사례를 통한 프롬프트 활용  
# - **Section 목적:** 다양한 실제 사례를 통해 프롬프트 활용 방법을 제시합니다.  
# - **Section 내용:** 프롬프트 적용 사례, 성공 및 실패 사례 분석.  
# - **주요 내용:**  
#   - 성공적인 프롬프트 활용 사례.  
#   - 실패 사례 분석 및 교훈.  
#   - ...  
# - **참고문헌:**  
#   1. [프롬프트 활용 사례](https://example.com): 성공 및 실패 사례를 포함한 자료.  

# :---CHAPTER DIVIDER---:

# ## Chapter 3: AI 에이전트의 작동 원리와 구현
# - **Chapter 목적:** AI 에이전트의 작동 원리와 구현 방법을 심층적으로 분석합니다.  
# - **Chapter 내용:** 에이전트의 구조, 작동 방식, 구현 기술 등을 설명합니다.  

# ### Section 3.1: 에이전트의 구조와 작동 방식  
# - **Section 목적:** 에이전트의 구조와 작동 방식을 이해합니다.  
# - **Section 내용:** 에이전트의 내부 구조, 작동 방식, 기술적 특징.  
# - **주요 내용:**  
#   - 에이전트 구조 분석.  
#   - 작동 방식 및 기술적 특징.  
#   - ...  
# - **참고문헌:**  
#   1. [에이전트의 구조](https://example.com): 에이전트의 구조를 설명한 자료.  

# ### Section 3.2: 에이전트 구현 기술  
# - **Section 목적:** 에이전트의 구현 기술을 탐구합니다.  
# - **Section 내용:** 구현 기술, 프로그래밍 언어, 도구 및 프레임워크.  
# - **주요 내용:**  
#   - 구현 기술 개요.  
#   - 사용되는 프로그래밍 언어 및 도구.  
#   - ...  
# - **참고문헌:**  
#   1. [에이전트 구현 기술](https://example.com): 구현 기술과 사용 도구를 설명한 자료.  

# :---CHAPTER DIVIDER---:

# ## Chapter 4: AI 프롬프트 및 에이전트의 최신 동향
# - **Chapter 목적:** 최신 연구 자료를 통해 AI 프롬프트 및 에이전트의 발전 동향을 분석합니다.  
# - **Chapter 내용:** 최신 기술 트렌드, 연구 결과, 미래 전망 등을 다룹니다.  

# ### Section 4.1: 최신 기술 트렌드  
# - **Section 목적:** 현재 AI 프롬프트 및 에이전트 기술의 트렌드를 이해합니다.  
# - **Section 내용:** 최신 연구 동향, 기술 발전 사례.  
# - **주요 내용:**  
#   - 최신 기술 트렌드 분석.  
#   - 연구 및 개발 사례.  
#   - ...  
# - **참고문헌:**  
#   1. [최신 기술 트렌드](https://example.com): 최신 연구 동향을 다룬 자료.  

# ### Section 4.2: 미래 전망과 과제  
# - **Section 목적:** AI 프롬프트 및 에이전트가 직면한 과제를 이해하고 미래 전망을 제시합니다.  
# - **Section 내용:** 기술적 도전과제, 사회적 영향, 미래 가능성.  
# - **주요 내용:**  
#   - 기술적 과제 및 해결 방안.  
#   - 사회적 영향 및 윤리적 고려.  
#   - ...  
# - **참고문헌:**  
#   1. [미래 전망과 과제](https://example.com): AI의 미래 가능성을 논의한 자료.  

# :---CHAPTER DIVIDER---:

# ## Chapter 5: 실무 적용 사례와 향후 전망
# - **Chapter 목적:** 다양한 산업에서의 실제 적용 사례와 AI 프롬프트 및 에이전트의 미래 전망을 탐구합니다.  
# - **Chapter 내용:** 산업별 적용 사례, 성공 사례 분석, 향후 발전 방향.  

# ### Section 5.1: 산업별 적용 사례  
# - **Section 목적:** 다양한 산업에서 AI 프롬프트 및 에이전트의 활용 사례를 제시합니다.  
# - **Section 내용:** 산업별 적용 사례, 성공 및 실패 사례 분석.  
# - **주요 내용:**  
#   - 산업별 성공 사례.  
#   - 실패 사례 분석 및 교훈.  
#   - ...  
# - **참고문헌:**  
#   1. [산업별 적용 사례](https://example.com): 다양한 산업 분야의 사례를 다룬 자료.  

# ### Section 5.2: 향후 발전 방향과 전망  
# - **Section 목적:** AI 프롬프트 및 에이전트의 향후 발전 방향을 제시합니다.  
# - **Section 내용:** 연구 및 개발 방향, 사회적 및 경제적 영향.  
# - **주요 내용:**  
#   - 향후 기술 발전 방향.  
#   - 경제적 및 사회적 영향 분석.  
#   - ...  
# - **참고문헌:**  
#   1. [향후 발전 방향](https://example.com): 미래 기술 발전 방향을 다룬 자료.  

# -----: DONE :-----

# + 목차 작성 후기
# - 사용자의 요구사항이 적절히 반영되었으며, AI 프롬프트와 에이전트 관련 최신 자료와 연구를 기반으로 구성하였습니다.
# - 피드백 과정을 통해 초안을 수정 보완할 여지를 남겨두어 유연한 구조를 설계하였습니다. 
# - 향후 실제 자료와의 연계성을 높이기 위한 추가 리서치가 필요할 것으로 보이며, 참고문헌 부분은 실제 자료 확보 후 업데이트가 필요합니다.
# [Content Strategist] 목차 작성 완료: outline 작성 완료
 

# + 목차 작성 후기
# - 사용자의 요구사항이 적절히 반영되었으며, AI 프롬프트와 에이전트 관련 최신 자료와 연구를 기반으로 구성하였습니다.
# - 피드백 과정을 통해 초안을 수정 보완할 여지를 남겨두어 유연한 구조를 설계하였습니다. 
# - 향후 실제 자료와의 연계성을 높이기 위한 추가 리서치가 필요할 것으로 보이며, 참고문헌 부분은 실제 자료 확보 후 업데이트가 필요합니다.


# ============ OUTLINE REVIEWER ============
# ### 분석 결과:

# 1. **사용자 요구사항 충족 여부:** 
#    - 주어진 아웃라인은 사용자의 목표와 방법론에 어느 정도 부합합니다. AI 프롬프트와 에이전트에 대한 기본 개념, 설계 및 구현, 최신 동향, 실무 적용 사례 등을 포함하여 사용자가 제시한 요구사항을 충족하고 있습니다.

# 2. **논리적인 흐름의 적절성:** 
#    - 아웃라인은 개념에서 시작하여 설계, 구현, 최신 동향, 적용 사례까지 이어지는 논리적인 흐름을 가지고 있습니다. 각 챕터의 목적과 내용이 잘 구성되어 있어 독자가 점진적으로 이해할 수 있도록 돕습니다.

# 3. **근거 기반 여부:** 
#    - 참고문헌이 제시되어 있으나, 모두 placeholder URL(`https://example.com`)로 실제 근거 자료가 제공되지 않았습니다. 이는 아웃라인의 신뢰성을 저하시킬 수 있습니다.

# 4. **참고자료 활용 여부:** 
#    - 현재 참고자료가 없으며, 실제 자료 확보 후 업데이트가 필요합니다. 참고문헌 부분도 아직 업데이트되지 않은 상태입니다.

# 5. **더미 URL 존재 여부:** 
#    - 다수의 더미 URL(`https://example.com`)이 존재하며, 이는 모두 삭제되거나 실제 자료로 대체되어야 합니다.

# 6. **기타 리뷰 사항:** 
#    - '목차 작성의 주안점'과 '목차 작성 후기'에서 사용자의 요구사항 반영 여부와 피드백 과정의 유연성에 대해 언급하고 있지만, 구체적인 데이터 기반의 피드백과 수정 과정에 대한 내용이 추가되면 좋겠습니다.

# ### 제안 사항:

# - **다음 작업 제안:** 
#   - 아웃라인의 각 챕터와 섹션에 대한 구체적인 참고자료 및 신뢰할 수 있는 URL을 확보하고 추가하는 작업이 필요합니다.
#   - `vector_search_agent`를 호출하여 관련된 최신 연구 자료 및 실제 사례를 탐색하고, 이를 기반으로 아웃라인을 보강할 것을 제안합니다. 

# 이 작업을 통해 아웃라인의 신뢰성과 완성도를 높이고, 실제 자료와의 연계성을 강화할 수 있을 것입니다.[OUTLINE REVIEW AGENT] ### 분석 결과:

# 1. **사용자 요구사항 충족 여부:** 
#    - 주어진 아웃라인은 사용자의 목표와 방법론에 어느 정도 부합합니다. AI 프롬프트와 에이전트에 대한 기본 개념, 설계 및 구현, 최신 동향, 실무 적용 사례 등을 포함하여 사용자가 제시한 요구사항을 충족하고 있습니다.

# 2. **논리적인 흐름의 적절성:** 
#    - 아웃라인은 개념에서 시작하여 설계, 구현, 최신 동향, 적용 사례까지 이어지는 논리적인 흐름을 가지고 있습니다. 각 챕터의 목적과 내용이 잘 구성되어 있어 독자가 점진적으로 이해할 수 있도록 돕습니다.

# 3. **근거 기반 여부:** 
#    - 참고문헌이 제시되어 있으나, 모두 placeholder URL(`https://example.com`)로 실제 근거 자료가 제공되지 않았습니다. 이는 아웃라인의 신뢰성을 저하시킬 수 있습니다.

# 4. **참고자료 활용 여부:** 
#    - 현재 참고자료가 없으며, 실제 자료 확보 후 업데이트가 필요합니다. 참고문헌 부분도 아직 업데이트되지 않은 상태입니다.

# 5. **더미 URL 존재 여부:** 
#    - 다수의 더미 URL(`https://example.com`)이 존재하며, 이는 모두 삭제되거나 실제 자료로 대체되어야 합니다.

# 6. **기타 리뷰 사항:** 
#    - '목차 작성의 주안점'과 '목차 작성 후기'에서 사용자의 요구사항 반영 여부와 피드백 과정의 유연성에 대해 언급하고 있지만, 구체적인 데이터 기반의 피드백과 수정 과정에 대한 내용이 추가되면 좋겠습니다.

# ### 제안 사항:

# - **다음 작업 제안:** 
#   - 아웃라인의 각 챕터와 섹션에 대한 구체적인 참고자료 및 신뢰할 수 있는 URL을 확보하고 추가하는 작업이 필요합니다.
#   - `vector_search_agent`를 호출하여 관련된 최신 연구 자료 및 실제 사례를 탐색하고, 이를 기반으로 아웃라인을 보강할 것을 제안합니다. 

# 이 작업을 통해 아웃라인의 신뢰성과 완성도를 높이고, 실제 자료와의 연계성을 강화할 수 있을 것입니다.


# ============ BUSINESS ANALYST ============
# [Business Analyst] ```
# - 목표: AI 프롬프트와 AI 에이전트 관련 교재의 신뢰성과 완성도 향상
# - 방법: 
#   1. 아웃라인의 각 챕터와 섹션에 대한 구체적인 참고자료 및 신뢰할 수 있는 URL 확보 및 추가.
#   2. `vector_search_agent`를 활용하여 최신 연구 자료 및 실제 사례를 탐색하고 이를 아웃라인에 반영.
#   3. 더미 URL을 실제 자료로 대체하여 신뢰성을 높임.
#   4. 피드백 과정을 통해 데이터 기반의 수정 및 보완.
# ```


# ============ SUPERVISOR ============
# [Supervisor] agent='vector_search_agent' done=False description='최신 연구 자료 및 실제 사례를 탐색하여 아웃라인을 보강하고, 신뢰할 수 있는 참고자료 및 URL을 확보한다.' done_at=''


# ============ VECTOR SEARCH AGENT ============
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 프롬프트와 AI 에이전트의 최신 연구 및 실무 사례'}, 'id': 'call_OFm7LTjU7nxFPtKTeO7A1sdM', 'type': 'tool_call'}
# C:\Aiprojects\ch15\ch15-3\book_writer.py:260: LangChainDeprecationWarning: The method `BaseTool.__call__` was deprecated in langchain-core 0.1.47 and will be removed in 1.0. Use invoke instead.
#   retrieved_docs = retrieve(args)
# Failed to send telemetry event CollectionQueryEvent: capture() takes 1 positional argument but 3 were given
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 프롬프트 설계 원리와 성공 사례'}, 'id': 'call_gbh9Adu3JxzjdOEr9jrnSHPH', 'type': 'tool_call'}
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 에이전트의 구조, 작동 방식 및 구현 기술'}, 'id': 'call_jSjFxP9zVxIvi1cLAvzRUioC', 'type': 'tool_call'}
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 프롬프트 및 에이전트의 최신 기술 트렌드와 미래 전망'}, 'id': 'call_5z26y04plMokSHf2LgvWMJws', 'type': 'tool_call'}
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 프롬프트 및 에이전트의 다양한 산업별 적용 사례'}, 'id': 'call_EH8lJQFUbm3iPnd67OOHNFA5', 'type': 'tool_call'}
# Queries:--------------------------
# AI 프롬프트와 AI 에이전트의 최신 연구 및 실무 사례
# AI 프롬프트 설계 원리와 성공 사례
# AI 에이전트의 구조, 작동 방식 및 구현 기술
# AI 프롬프트 및 에이전트의 최신 기술 트렌드와 미래 전망
# AI 프롬프트 및 에이전트의 다양한 산업별 적용 사례
# References:--------------------------
# [VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: ['AI 프롬프트와 AI 에이전트의 최신 연구 및 실무 사례', 'AI 프롬프트 설계 원리와 성공 사례', 'AI 에이전트의 구조, 작동 방식 및 구현 기술', 'AI 프롬프트 및 에이전트의 최신 기술 트렌드와 미래 전망', 'AI 프롬프트 및 에이전트의 다양한 산업별 적용 사례']


# ============ BUSINESS ANALYST ============
# [Business Analyst] ```
# - 목표: AI 프롬프트와 AI 에이전트 관련 교재의 신뢰성과 완성도 향상
# - 방법: 
#   1. 아웃라인의 각 챕터와 섹션에 대한 구체적인 참고자료 및 신뢰할 수 있는 URL 확보 및 추가.
#   2. `vector_search_agent`를 활용하여 최신 연구 자료 및 실제 사례를 탐색하고 이를 아웃라인에 반영.
#   3. 더미 URL을 실제 자료로 대체하여 신뢰성을 높임.
#   4. 피드백 과정을 통해 데이터 기반의 수정 및 보완.
# ```


# ============ SUPERVISOR ============
# [Supervisor] agent='vector_search_agent' done=False description='최신 연구 자료 및 실제 사례를 탐색하여 아웃라인을 보강하고, 신뢰할 수 있는 참고자료 및 URL을 확보한다.' done_at=''


# ============ VECTOR SEARCH AGENT ============
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 프롬프트와 AI 에이전트의 최신 연구 및 실무 사례'}, 'id': 'call_72NOij8jBMBqwbW0LiOXq2j8', 'type': 'tool_call'}
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 프롬프트 설계 원리와 성공 사례'}, 'id': 'call_YWG0eslBl97MD1UJyuDKSoV3', 'type': 'tool_call'}
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 에이전트의 구조, 작동 방식 및 구현 기술'}, 'id': 'call_mBDnqbiFJ6DKRG6XuryGflkm', 'type': 'tool_call'}
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 프롬프트 및 에이전트의 최신 기술 트렌드와 미래 전망'}, 'id': 'call_LFjL0FGGImNNCvdYBfHzOoTc', 'type': 'tool_call'}
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'AI 프롬프트 및 에이전트의 다양한 산업별 적용 사례'}, 'id': 'call_995o7DYOoKB1LCLaXTs8AUki', 'type': 'tool_call'}
# Queries:--------------------------
# AI 프롬프트와 AI 에이전트의 최신 연구 및 실무 사례
# AI 프롬프트 설계 원리와 성공 사례
# AI 에이전트의 구조, 작동 방식 및 구현 기술
# AI 프롬프트 및 에이전트의 최신 기술 트렌드와 미래 전망
# AI 프롬프트 및 에이전트의 다양한 산업별 적용 사례
# AI 프롬프트와 AI 에이전트의 최신 연구 및 실무 사례
# AI 프롬프트 설계 원리와 성공 사례
# AI 에이전트의 구조, 작동 방식 및 구현 기술
# AI 프롬프트 및 에이전트의 최신 기술 트렌드와 미래 전망
# AI 프롬프트 및 에이전트의 다양한 산업별 적용 사례
# References:--------------------------
# [VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: ['AI 프롬프트와 AI 에이전트의 최신 연구 및 실무 사례', 'AI 프롬프트 설계 원리와 성공 사례', 'AI 에이전트의 구조, 작동 방식 및 구현 기술', 'AI 프롬프트 및 에이전트의 최신 기술 트렌드와 미래 전망', 'AI 프롬프트 및 에이전트의 다양한 산업별 적용 사례', 'AI 프롬프트와 AI 에이전트의 최신 연구 및 실무 사례', 'AI 프롬프트 설계 원리와 성공 사례', 'AI 에이전트의 구조, 작동 방식 및 구현 기술', 'AI 프롬프트 및 에이전트의 최신 기술 트렌드와 미래 전망', 'AI 프롬프트 및 에이전트의 다양한 산업별 적용 사례']


# ============ BUSINESS ANALYST ============
# [Business Analyst] ```
# - 목표: AI 프롬프트와 AI 에이전트 관련 교재의 신뢰성과 완성도 향상
# - 방법: 
#   1. 아웃라인의 각 챕터와 섹션에 대한 구체적인 참고자료 및 신뢰할 수 있는 URL 확보 및 추가.
#   2. `vector_search_agent`를 활용하여 최신 연구 자료 및 실제 사례를 탐색하고 이를 아웃라인에 반영.
#   3. 더미 URL을 실제 자료로 대체하여 신뢰성을 높임.
#   4. 피드백 과정을 통해 데이터 기반의 수정 및 보완.
# ```


# ============ SUPERVISOR ============
# Supervisor 호출 횟수 초과: Communicator 호출
# [Supervisor] agent='communicator' done=False description='supervisor 호출 횟수 초과했으므로, 현재까지의 진행상황을 사용자에게 보고한다. ' done_at=''


# ============ COMMUNICATOR ============

# AI      : 안녕하세요! AI 팀의 커뮤니케이터입니다. 현재 AI 프롬프트와 AI 에이전트 관련 교재 제작 프로젝트의 진행 상황을 안내드리겠습니다.

# ### 진행 상황
# 1. **목차 작성 완료**: 사용자의 요구사항을 반영하여 AI 프롬프트와 에이전트 관련 교재의 목차를 작성했습니다. 책의 기획 의도와 전달하고자 하는 메시지를 명확히 하여 독자들이 이해할 수 있도록 구체적으로 구성했습니다.

# 2. **아웃라인 리뷰**: 아웃라인의 각 챕터는 개념에서 시작하여 설계, 구현, 최신 동향, 실무 적용 사례까지 다루고 있습니다. 논리적인 흐름이 잘 구성되어 있으며, 독자가 쉽게 이해할 수 있도록 설계되었습니다.

# 3. **신뢰성 강화 작업**: 현재 아웃라인의 신뢰성을 높이기 위해 구체적인 참고자료와 신뢰할 수 있는 URL을 확보하고 있습니다. `vector_search_agent`를 활용하여 최신 연구 자료 및 실무 사례를 탐색 중이며, 이를 아웃라인에 반영할 계획입니다.

# 4. **피드백 및 수정**: 피드백 과정을 통해 데이터 기반의 수정 및 보완을 진행하고 있으며, 더미 URL을 실제 자료로 대체하여 신뢰성을 높이고자 합니다.

# ### 다음 단계
# - 최신 연구 자료를 기반으로 아웃라인을 보강하고, 신뢰할 수 있는 참고자료 및 URL을 추가할 예정입니다.
# - 피드백을 지속적으로 반영하여 초안의 완성도를 높여 나가겠습니다.

# 사용자께서 추가로 궁금하신 점이나 의견이 있으시다면 언제든지 말씀해 주세요. 여러분의 피드백은 더욱 나은 콘텐츠 제작에 큰 도움이 됩니다!
# ------------------------------------ MESSAGE COUNT       15

# User    : q