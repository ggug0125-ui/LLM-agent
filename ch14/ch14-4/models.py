from pydantic import BaseModel, Field
from typing import Literal

class Task(BaseModel):
    agent: Literal[
        "content_strategist",
        "communicator",
        "web_search_agent",  # p449 추가
        "vector_search_agent",
    ] = Field(
        ...,
        description="""
        작업을 수행하는 agent의 종류.
        - content_strategist: 콘텐츠 전략을 수립하는 작업을 수행한다. 사용자의 요구사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다. 
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행상황을 사용자에게 보고하고, 다음 지시를 물어본다.
        - web_search_agent: 웹 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.  # p449 추가
        - vector_search_agent: 벡터 DB 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.
        """
    )
	
    done: bool = Field(..., description="종료 여부")
    description: str = Field(..., description="어떤 작업을 해야 하는지에 대한 설명")
	
    done_at: str = Field(..., description="할 일이 완료된 날짜와 시간")
	
    def to_dict(self):
        return {
            "agent": self.agent,
            "done": self.done,
            "description": self.description,
            "done_at": self.done_at
        }  
    
# p452 결과
# (ch13_env) PS C:\Aiprojects\ch14\ch14-3> 
# (ch13_env) PS C:\Aiprojects\ch14\ch14-3> cd ..
# (ch13_env) PS C:\Aiprojects\ch14> cd ch14-4
# (ch13_env) PS C:\Aiprojects\ch14\ch14-4> python .\book_writer.py
# USER_AGENT environment variable not set, consider setting it to identify your requests.
# Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
# Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given

# User    : HYPE와 JYP 비교하는 책 쓰자. 우선 검색부터 해


# ============ SUPERVISOR ============
# [Supervisor] agent='web_search_agent' done=False description='사용자가 요청한 HYPE와 JYP에 대한 정보를 웹 검색을 통해 수집한다.' done_at='2026-04-27T14:33:32'


# ============ WEB SEARCH AGENT ============
# -------- web search -------- {'name': 'web_search', 'args': {'query': 'HYPE 엔터테인먼트 회사 정보'}, 'id': 'call_YnzeopwTXHeQRZ95E4BHTW7l', 'type': 'tool_call'}
# json_path: C:\Aiprojects\ch14\ch14-4/data/resources_2026_0427_143355.json
# Documents를 Chroma DB에 저장합니다.
# {'title': 'Hybe - Wikipedia', 'source': 'https://en.wikipedia.org/wiki/Hybe'}
# {'title': 'HYBE', 'source': 'https://hybecorp.com/eng/company/business'}
# {'title': '(주)하이브 2026년 기업정보 | 직원수, 근무환경, 복리후생 등 - 사람인', 'source': 'https://www.saramin.co.kr/zf_user/company-info/view/csn/RkRqYjFMRnM0UWEwcHc4a3g3clJ3dz09/company_nm/(%EC%A3%BC)%ED%95%98%EC%9D%B4%EB%B8%8C?srsltid=AfmBOoosFgOt09_7UtSr_dF0-sLzs_NAACCaN2t2l2ux4e3N_hwWuYG6'}
# {'title': 'HYBE', 'source': 'https://hybecorp.com/eng/company/artist'}
# {'title': '회사 정보 | MÄGERLE', 'source': 'https://www.maegerle.com/ko/%ED%9A%8C%EC%82%AC-%EC%A0%95%EB%B3%B4'}
# Splitting documents...
# 5개의 문서를 1000자 크기로 중첩 100자로 분할합니다.

# 총 397개의 문서로 분할되었습니다.
# -------- web search -------- {'name': 'web_search', 'args': {'query': 'JYP 엔터테인먼트 회사 정보'}, 'id': 'call_PydDoOO5dJQSBPxynPyMAI9q', 'type': 'tool_call'}
# json_path: C:\Aiprojects\ch14\ch14-4/data/resources_2026_0427_143408.json
# Documents를 Chroma DB에 저장합니다.
# Failed to send telemetry event CollectionGetEvent: capture() takes 1 positional argument but 3 were given
# {'title': 'JYP엔터테인먼트 - 나무위키', 'source': 'https://namu.wiki/w/JYP%EC%97%94%ED%84%B0%ED%85%8C%EC%9D%B8%EB%A8%BC%ED%8A%B8'}
# {'title': 'JYP Entertainment - LEADER IN ENTERTAINMENT', 'source': 'https://kr.linkedin.com/company/jyp-entertainment'}
# {'title': 'JYP Entertainment', 'source': 'https://www.jype.com/ko/JYP'}
# {'title': 'JYP Entertainment', 'source': 'https://www.jype.com/ko/JYP/Contact'}
# {'title': 'JYP Entertainment', 'source': 'https://www.jype.com/'}
# Splitting documents...
# 5개의 문서를 1000자 크기로 중첩 100자로 분할합니다.

# 총 141개의 문서로 분할되었습니다.


# ============ VECTOR SEARCH AGENT ============
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'HYPE 엔터테인먼트 회사 정보'}, 'id': 'call_gO4QYgCA7VvmF1nJk6k6RD08', 'type': 'tool_call'}
# C:\Aiprojects\ch14\ch14-4\book_writer.py:161: LangChainDeprecationWarning: The method `BaseTool.__call__` was deprecated in langchain-core 0.1.47 and will be removed in 1.0. Use invoke instead.
#   retrieved_docs = retrieve(args)
# Failed to send telemetry event CollectionQueryEvent: capture() takes 1 positional argument but 3 were given
# ----------------------------------- {'name': 'retrieve', 'args': {'query': 'JYP 엔터테인먼트 회사 정보'}, 'id': 'call_mvFqFryKzooujWP1toaWmnDu', 'type': 'tool_call'}
# Queries:--------------------------
# HYPE 엔터테인먼트 회사 정보
# JYP 엔터테인먼트 회사 정보
# References:--------------------------
# * **업력 22년차** 

#   2005년 2월 4일 설립
# * 기업형태: 1000대기업, 대기업, 코스피, 주식회사
# * **905 명** 

#   출처: 국민연금

# 업종
# :   음악
# --------------------------
# HYBE

# ![HYBE](https://clogo.saramin.co.kr/workenv-bg/202306/14/rw7zws_t7er-yowa2y_workenv-bg.jpg/sri
# --------------------------
# ## 한눈에 보는 우리 회사, 미리 확인해 보세요!

# * #서울지하철경의중앙  경의중앙
# * #서울지하철1호선  1

# * + 6   ![HYBE](https://clogo.saram
# --------------------------
# **Hybe Co., Ltd.**[[1]](#cite_note-Company_established-1) ([Korean](/wiki/Korean_language "Korean la
# --------------------------
# ## External links

# * [Official website](https://hybecorp.com/eng/)
# * Business data for Hybe:

#   + [G
# --------------------------
# JYP ENTERTAINMENT

# * [Youtube](https://www.youtube.com/jypentertainment)
# * [Instagram](https://www.i
# --------------------------
# ### Website
# [https://www.jype.com/](https://www.jype.com/)
# ### Crunchbase
# [N/A](N/A)
# ### LinkedIn
# [h
# --------------------------
# |  | [박진영](/w/%EB%B0%95%EC%A7%84%EC%98%81 "박진영") · [홍승성](/w/%ED%99%8D%EC%8A%B9%EC%84%B1 "홍승성")[[1]](
# --------------------------
# |  |
# | --- |
# | **JYP엔터테인먼트 JYP Entertainment** |
# |  | **정식:** 주식회사 제이와이피엔터테인먼트 **영문:** JYP Entertain
# --------------------------
# [7](#s-7). JYP NATION[8](#s-8). 지배구조

# [8.1](#s-8.1). 우회상장

# [9](#s-9). [논란 및 사건 사고](/w/JYP%EC%97%94%E
# --------------------------
# [VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: ['HYPE 엔터테인먼트 회사 정보', 'JYP 엔터테인먼트 회사 정보']


# ============ COMMUNICATOR ============

# AI      : 안녕하세요! 현재 저희 AI팀은 HYPE와 JYP 엔터테인먼트에 대한 정보를 수집하는 단계를 완료했습니다. 웹 검색과 벡터 검색을 통해 두 회사의 다양한 정보를 확보한 상태입니다. 다음 단계로는 이 정보를 바탕으로 목차를 구성하고, 본격적인 내용 작성을 진행할 계획입니다. 

# 혹시 이 책에서 특별히 다뤘으면 하는 주제가 있거나, 추가로 궁금한 점이 있다면 말씀해 주세요. 이를 반영하여 더욱 유익한 내용을 제공할 수 있도록 하겠습니다.
# ------------------------------------ MESSAGE COUNT       6

# User    : q
