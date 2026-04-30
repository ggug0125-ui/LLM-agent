# 벡터 DB 만들기

# 이제 웹 검색을 하면 검색 결과가 JSON 파일로 저장됩니다. 
# 앞서 설명했듯이 JSON 파일의 내용이 많아질수록 언어 모델을 활용해 원하는 답변을 받기 어려워지므로 검색 결과를 벡터 DB에 저장해서 활용하겠습니다.

# 웹 검색 결과를 랭체인 Document 객체로 변환하기
# 09장과 13장에서 RAG를 만들 때 PDF를 읽어 랭체인 Document 객체로 변환한 뒤, text splitter를 이용해 청크 단위로 나누고 이 청크들을 임베딩하여 벡터 DB에 저장했습니다. 
# 이번에도 동일한 방법을 사용합니다. 
# 다른 점은 이번에는 PDF가 아니라 JSON 형태로 저장된 웹 페이지 정보를 랭체인 Document 객체로 변환한다는 점입니다. 이를 위해 함수 2개를 사용할 것입니다.



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

# p421추가
# web_page_to_document 함수를 만듭니다. 이 함수는 하나의 웹 페이지 정보가 들어 오면 이를 랭체인의 Document 객체로 변환하는 기능을 합니다.
from langchain_core.documents import Document  
from langchain_text_splitters import RecursiveCharacterTextSplitter

import json
import os
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로



# p425 추가 RAG를 위한 설정
# 임베딩은 OpenAIEmbeddings의 text-embedding-3-large를 선택하고 벡터 DB는 크로마 DB를 선택합니다. 
# 크로마 DB가 저장될 폴더 위치를 설정하고 vectorstore 변수에 Chroma를 선언해 담습니다.
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 오픈AI Embedding 설정
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# 크로마 DB 저장 경로 설정
persist_directory = f"{current_path}/data/chroma_store"

# Chroma 객체 생성
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
# p425 추가 끝

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

# p422 추가 
def web_page_to_document(web_page):
    # raw_content와 content 중 정보가 많은 것을 page_content로 한다.
    # 타빌리 검색에서 페이지를 열기 전에 얻을 수 있는 페이지 정보는 content에 기록되고, 실제로 페이지를 열
    # 었을 때의 전문이 raw_content에 기록됩니다. 하지만 raw_content에 정보가 부족한 경우가 있습니다. 이는
    # 최근에 웹 페이지 내용이 삭제되었거나 수정된 경우에 해당하는데, 이런 경우 raw_content와 content 중에
    # 길이가 더 긴 정보를 Document의 page_content로 사용합니다.
    if len(web_page['raw_content']) > len(web_page['content']):
        page_content = web_page['raw_content']
    else:
        page_content = web_page['content']

    # 랭체인 Document로 변환
    # 웹 페이지의 정보를 랭체인의 Document로 변환하는 코드입니다. 랭체인에서 제공하는 Document 클래스는
    # 기본적으로 벡터 검색에 활용할 수 있는 문서의 실제 내용인 page_content와 문서에 대한 추가 정보인
    # metadata를 갖고 있습니다. 앞선 실습의 결과를 보면 리스트에 각 페이지 정보가 있고 그 안에 title, url,
    # content, score, raw_content 키에 해당하는 값이 포함되어 있습니다. page_content에 1에서 정의한
    # page_content를, metadata에 웹 페이지의 title과 url을 대입하여 Document를 생성합니다. 그리고 docu
    # ment를 반환합니다.
    document = Document(
        page_content=page_content,
        metadata={
            'title': web_page['title'],
            'source': web_page['url']
        }
    )

    return document


# p422 추가
# 다음으로 web_page_json_to_documents 함수를 만듭니다. 
# 이 함수는 json_file 경로를 입력받아 파일을 읽고 각 웹 페이지 정보를 web_page_to_document 함수에 하나씩 전달하여 Document 객체로 변환된 값을 받습니다. 
# 변환된 Document들은 리스트에 차례대로 추가하여 반환합니다. 
# 테스트를 위해 이 파일의 메인 영역에는 앞서 생성된 JSON 파일 경로를 web_page_json_to_documents에 추가합니다. 그리고 documents의 마지막 요소를 출력합니다.
def web_page_json_to_documents(json_file):
    with open(json_file, "r", encoding='utf-8') as f:
        resources = json.load(f)

    documents = []

    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)

    return documents

# p424 추가
# Document 객체들을 청크 단위로 자르기
# 웹 페이지의 정보를 RAG 방식으로 활용하려면 청크 단위로 나누어야 합니다. 웹 페이지 정보를 랭체인의 Document 객체로 변환하는 함수를 만들었으므로 
# 09-2절에서 배운 Recursive CharacterTextSplitter를 활용해서 청크 단위로 나눌 수 있습니다.

# split_documents 함수는 여러 개의 documents를 매개변수로 받아 chunk_size와 chunk_overlap에 맞춰 청크 단위로 나누는 역할을 합니다. 
# 09-2절에서 RAG를 사용한 코드와 거의 같으며 함수가 잘 작동하는지 확인하기 위해 print 문을 추가했습니다. 
# 이 코드를 테스트하기 위해 main 부분에서 split_documents 함수를 호출합니다.
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print('Splitting documents...')
    print(f"{len(documents)}개의 문서를 {chunk_size}자 크기로 중첩 {chunk_overlap}자로 분할합니다.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(documents)

    print(f"총 {len(splits)}개의 문서로 분할되었습니다.")
    return splits



# p427 추가 documents를 chroma DB에 저장하는 함수
def documents_to_chroma(documents, chunk_size=1000, chunk_overlap=100):
    print("Documents를 Chroma DB에 저장합니다.")

    # documents의 url 가져오기
    # 매개변수로 받은 documents의 metadata에 있는 source를 이용해 URL을 가져옵니다.
    urls = [document.metadata['source'] for document in documents]

    # 이미 vectorstore에 저장된 urls 가져오기
    # 이미 벡터 DB에 저장된 문서들의 metadata에서 URL을 가져옵니다.
    stored_metadatas = vectorstore._collection.get()['metadatas'] 
    stored_web_urls = [metadata['source'] for metadata in stored_metadatas] 

    # 새로운 urls만 남기기
    # 저장할 문서들과 이미 저장되어 있던 문서들의 URL은 set을 사용해 중복을 없애서 아직 벡터 DB에 저장되어 있지 않은 URL들을 골라냅니다.
    new_urls = set(urls) - set(stored_web_urls)

    # 새로운 urls에 대한 documents만 남기기
    # 새로운 URL에 해당하는 document들만 골라서 담습니다.
    new_documents = []

    for document in documents:
        if document.metadata['source'] in new_urls:
            new_documents.append(document)
            print(document.metadata)

    # 새로운 documents를 Chroma DB에 저장
    # 이 문서들을 split_documents 함수를 이용해 청크 단위로 자릅니다.
    splits = split_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 크로마 DB에 저장 : 만약 새로 저장할 문서가 있다면 크로마 DB에 청크들을 저장합니다.
    if splits:
        vectorstore.add_documents(splits)
    else:
        print("No new urls to process")

# json 파일에서 documents를 만들고, 그 documents들을 Chroma DB에 저장
# JSON 파일에서 documents를 만들고 그 documents들을 크로마 DB에 저장합니다. 그리고 documents_to_chroma 함수를 실행시킵니다.
def add_web_pages_json_to_chroma(json_file, chunk_size=1000, chunk_overlap=100):
    documents = web_page_json_to_documents(json_file)
    documents_to_chroma(
        documents, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
# p427 추가 끝

@tool # p428 추가
# 리트리버 만들기
# 아직 벡터 DB에 몇 개의 문서만 저장되어 있지만 벡터 검색이 잘 작동하는지 확인해 보겠습니다.

# 벡터 검색을 확인하는 retrieve 함수를 만듭니다. 이 함수는 query와 top_k를 매개변수를 받습니다. 
# retrieve 함수는 13장에서 만든 rag_with_langgraph.ipynb를 그대로 가져온것이며, 
# 사용자의 질문에 응답할 수 있도록 랭그래프에 연결하기 위해 @tool 데코레이터를 붙였습니다. 
# 메인 부분에는 리트리버를 테스트하는 코드를 추가합니다.
def retrieve(query: str, top_k: int=5):
    """
    주어진 query에 대해 벡터 검색을 수행하고, 결과를 반환한다.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs

# p428 추가 끝

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

    # p418 수정
    # p419 수정 results = web_search.invoke("2025년 한국 경제 전망")
    # p419 수정 print(results[0])
    # p423 수정 results, resources_json_path = web_search.invoke("2025년 한국 경제 전망")
    # p423 수정 print(results)

    # p427 제거 documents = web_page_json_to_documents(f'{current_path}/data/resources_2026_0427_121737.json')   # p423 파일명 확인 후 실행
    # p424 제거 print(documents[-1]) # p423 수정
    
    # p427 제거 splits = split_documents(documents)
    # p427 제거 print(splits)
    # 결과 확인 
    # Splitting documents...
    # 5개의 문서를 1000자 크기로 중첩 100자로 분할합니다.
    # 총 40개의 문서로 분할되었습니다.

    # 새로 만든 기능을 테스트하기 위해 이 코드의 메인 부분에서 기존에 만들었던 JSON 파일을 이용해 실행합나다.
    # p429 제거 add_web_pages_json_to_chroma(f'{current_path}/data/resources_2026_0427_121737.json') # p427 수정 (파일명확인)
    # 결과 크로마 db가 생성됨 chroma_store

    retrieved_docs = retrieve.invoke({"query": "한국 경제 위험 요소 "})
    print(retrieved_docs)
    # 이 코드를 실행한 결과 다음과 같이 리트리버가 잘 작동하는 것을 확인할 수 있습니다. 
    # 아직 벡터 DB에 저장된 문서가 많지 않아 입력한 키워드와 연관성은 그다지 높지 않을 수 있지만, 
    # 웹 검색 후 벡터 DB에 저장하는 기능을 만들어 두었으니 점점 더 많은 자료를 기반으로 유용한 정보를 반환할 겁니다.

    # [Document(metadata={'source': 'https://zerlho.com/entry/한국-경제지표-최신-동향-2025년-경제-성장률과-경기-전망-분석', 
    # 'title': '[한국 경제지표 최신 동향] 2025년 경제 성장률과 경기 전망 분석'}, page_content='📉 **한국 경제의 발목을 잡는 요인**  \n
    # ❌ **중국 경제 둔화**: 한국 수출의 20% 이상을 차지하는 중국 경제가 둔화되면서 수출 기업들의 부담이 가중되고 있습니다.  \n
    # ❌ **고물가 지속**: 소비자물가지수(CPI)가 여전히 높은 수준을 유지하면서 실질 소득이 감소하고 있습니다.  \n
    # ❌ **부동산 시장 불안**: 대출 금리 상승과 함께 부동산 시장이 위축되면서 건설·부동산 업종이 침체된 모습입니다.\n\n---\n\n## 
    # **2. 경기 침체 가능성 및 대응 전략**\n\n### 💡 **2025년 경기 침체 신호는?**\n\n경기 침체의 가능성을 판단하는 핵심 지표들을 살펴보면, 
    # 아직 확실한 침체라고 보기는 어렵지만 **일부 우려 요소**가 존재합니다.\n\n📉 **경기 침체를 시사하는 주요 지표들**\n\n* 
    # **제조업 PMI(구매관리자지수) 49.2** 📉 (50 이하일 경우 경기 위축 신호)\n* **가계부채 1,900조 원 돌파** 📉 
    # (사상 최고 수준)\n* **소비자 심리지수(CSI) 하락** 📉 (소비 위축 가능성)\n\n![](https://blog


    # 관련 높은 청크 찾는 벡터 검색 에이전트
    # 지금까지 인터넷 검색 후 검색 결과를 벡터 DB에 저장하는 기능과 RAG를 구현하는 기능을 개발했지만 아직 랭그래프에 연결하여 사용하지는 않았습니다. 
    # 벡터 DB에 저장한 내용에서 검색 내용과 관련이 높은 청크 문서를 가져오는 벡터 검색 에이전트 vector_search_agent를 추가하고 어떤 식으로 동작하는지 확인해 봅시다
    # 전에 만든 book_writer.py, models.py, utils.py를 활용한다 복사 붙이기