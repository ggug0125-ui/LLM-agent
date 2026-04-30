# 14-3 웹 검색과 RAG를 활용하는 벡터 검색 에이전트

# 14-2절에서 만든 AI 에이전트가 작성한 목차는 그럴싸하지만 확실한 근거가 없으므로 이대로 사용해도 될지 확인해야 합니다. 
# 또한 더 좋은 책을 만들려면 참고 문헌을 찾아서 목차를 보완하는 과정도 필요합니다. 
# 이때 인터넷 검색을 하거나 갖고 있는 책이나 문서를 이용할 수 있습니다. 이런 작업은 앞에서 이미 여러 차례 시켜 보았습니다. 
# 인터넷 검색은 덕덕고 DuckDuckgo와 타빌리Tavily를 활용하고, 
# 기존에 갖고 있던 문서에서 쓸 만한 자료를 찾아 반영하는 작업도 앞에서 배운 RAG로 구현할 수 있습니다. 
# 웹 검색과 RAG를 활용하는 AI 에이전트를 개발해 보겠습니다.

# 벡터 DB를 활용해 효율적으로 웹 검색하기
# 10장에서는 덕덕고와 타빌리로 웹 검색 기능을 랭체인에 추가하는 방법을 배웠습니다. 
# 질문을 받을 때마다 매번 인터넷을 검색해 답변하는 방식이었습니다. 
# 그러나 이러한 방식은 책이나 보고서의 목차를 작성하는 데 적합하지 않습니다. 
# 책이나 보고서의 목차를 작성하려면 웹 김색에서 찾은 문서의 전문을 읽고 내용을 파악해야 하는데 
# 책을 쓸 때는 같은 문서 전체를 여러 번 검색해야 할 수 있습니다. 
# 이때 타빌리를 사용하면 비용이 계속 발생하므로 같은 문서를 여러 번 검색하지 않도록 주의해야 합니다.
# 검색을 여러 번 하지 않고 검색한 내용을 모두 대화 내용에 담아 언어 모델이 찾아서 활용하게 할 수도 있습니다. 
# 하지만 매번 대량의 텍스트률 언어 모델에 보내야 하므로 토큰을 많이 사용하게 됩니다. 
# 이 방법은 비용이 많이 들 뿐만 아니라 답변 품질도 좋지 않을 수 있습니다. 
# 사람에게 수십 페이지에 달하는 문서의 내용을 물어보면 제대로 대답하기 어려운 것과 마찬가지죠. 
# 또한 사용하는 언어 모델의 컨텍스트 윈도우가 금방 한계에 도달해서 오류가 발생할 수 도 있습니다.
# 이러한 문제를 해결하기 위해 웹에서 검색된 문서 전체를 벡터 DB에 저장하여 활용할 수 있습니다. 
# 책 내용을 구성하는 데 필요한 질문이 있을 때 벡터 DB에서 RAG를 통해 검색하여 활용해 보겠습니다.

# 웹 검색 기능 만들기

# 먼저 웹 검색 기능을 구현하고 WebBaseLoader를 활용해 웹 페이지의 전문을 읽어오는 방법을 알아보겠습니다. 
# 그리고 검색 결과률 JSON 파일로 저장해 보겠습니다.

# 웹 검색을 위한 타빌리 검색 함수 만들기 
# 타빌리 검색 기능을 활용해서 함수를 만들고 랭체인 도구로 사용하겠습니다.

# 현재 작업 중인 폴더에 새로운 파이썬 파일 tools.py를 생성하고 그 안에 다음과 같이 web_search 함수를 도구로 작성합니다.

# deactivate
# PS C:\Aiprojects> .\ch13_env\Scripts\activate
# (ch13_env) PS C:\Aiprojects> 

from dotenv import load_dotenv # pip install dotenv
import os

load_dotenv()

from tavily import TavilyClient # pip install tavily-python
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from datetime import datetime

import json
import os
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로

# @tool 데코레이터는 함수를 랭체인이나 랭그래프에서 도구로 등록할 때 사용합니다. 이렇게 하면 해당 함수는 . invoke()를 통해 호출할 수 있게 됩니다.
@tool 
def web_search(query: str):

    # 이때 함수에 대한 설명을 작성해야 오류가 발생하지 않습니다. 여기에서는 '주어진 query에 대해 웹 검색을
    # 하고 결과를 반환한다.'라고 간단한 설명을 추가합니다.
    """
    주어진 query에 대해 웹검색을 하고, 결과를 반환한다.

    Args:
        query (str): 검색어

    Returns:
        dict: 검색 결과
    """
    client = TavilyClient()

    content = client.search(
        query, 
        search_depth="advanced",
        include_raw_content=True,
        # include_raw_content를 True로 설정하여 검색된 페이지의 전문을 가져오도록 만듭니다.
    )

    # results = content
    results = content["results"]   # p417 수정
    # 이제 web_search 함수를 수정할 차례입니다. 타빌라 검색 결과로 받은 content는 딕셔너리 형태입니다. 그
    # 중에 results 요소만 선택합니다. results에 들어 있는 항목 중에서 raw_content가 None인 경우에는
    # Load_web_page 함수를 이용해 페이지 내용을 가져와 raw_content에 담습니다. 이때 일부 웹 사이트에서
    # Load_web_page가 실패할 수 있으므로 try ~ except 문으로 오류를 처리합니다. 만약 load_web_page가
    # 실패하면 content의 원래 값을 그대로 raw_content에 넣도록 처리합니다.
    
    for result in results:
        if result["raw_content"] is None:
            try:
                result["raw_content"] = load_web_page(result["url"])
            except Exception as e:
                print(f"Error loading page: {result['url']}")
                print(e)
                result["raw_content"] = result["content"]

    
    # p417 추가 끝
    # p419 수정 return results

    # p419 추가 # 현 디릭토리에 /data/ 폴더 생성
    resources_json_path = f'{current_path}/data/resources_{datetime.now().strftime('%Y_%m%d_%H%M%S')}.json'
    with open(resources_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
   
    return results, resources_json_path  # (3) 검색 결과와 JSON 파일 경로 반환



# p416 추가 pip install beautifulsoup4
def load_web_page(url: str):
    loader = WebBaseLoader(url, verify_ssl=False)

    content = loader.load()
    # return content
    raw_content = content[0].page_content.strip()   
    # 먼저 페이지 내용만 반환하도록 Load_web_page 함수를 수정합니다. WebBaseLoader는 여러 URL을 한 번
    # 에 처리하여 리스트로 반환하는데, 여기서는 URL을 하나씩 처리하므로 첫 번째 요소만 결과에서 꺼내 사용
    # 합니다. 이때 page_content에는 WebBaseLoader로 가져온 내용이 있으므로 page_content만 추출하고 앞
    # 뒤 공백을 제거하기 위해 strip()을 사용합니다. 또한 가져온 페이지 내용에 탭이나 줄 바꿈이 과도하게 포
    # 함될 수 있으므로 줄 바꿈이나 탭이 연속으로 3개 이상 있으면 whi le 문을 사용해 정리합니다. 이렇게 수정한
    # load_web_page 함수는 매개변수로 받은 URL의 페이지 내용을 문자열로 반환하게 됩니다.

    while '\n\n\n' in raw_content or '\t\t\t' in raw_content:
        raw_content = raw_content.replace('\n\n\n', '\n\n')
        raw_content = raw_content.replace('\t\t\t', '\t\t')
        
    return raw_content

# 코드 하단에는 이 파일만 실행해도 테스트할 수 있도록 __ name __ 으로 main 영역을 만들어 두었습나다. 
# web_search.invoke() 안에 검색하고 싶은 내용을 입력하고 실행하면 됩니다.
if __name__ == "__main__":
    # results = web_search.invoke("2025년 한국 경제 전망")
    # 
    # print(results)

    # result = load_web_page("https://eiec.kdi.re.kr/publish/columnView.do?cidx=15029&ccode=&pp=20&pg=&sel_year=2025&sel_month=01")
    # print(result)

    #p418 수정
    # p419 수정 results = web_search.invoke("2025년 한국 경제 전망")
    # p419 수정  print(results[0])
    results, resources_json_path = web_search.invoke("2025년 한국 경제 전망")
    print(results)


# 작업하는 폴더에 /data 폴더를 만들고 코드에 '2025년 한국 경제 전망'을 입력한 후 실행하면 리스트 안에 딕셔너리가 담겨 있는 JSON 파일로 잘 저장됩니다.