import os
import json

def save_state(current_path, state):
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")
    
    state_dict = {}

    messages = [(m.__class__.__name__, m.content) for m in state["messages"]]
    state_dict["messages"] = messages
    
    with open(f"{current_path}/data/state.json", "w", encoding='utf-8') as f:
        json.dump(state_dict, f, indent=4, ensure_ascii=False)

# p392 추가
# get_outline 함수의 매개변수는 current_path뿐입니다. 
# 만약 current_path/data 폴더에 outline.md 파일이 있으면 그 파일 내용을 읽어서 반환합니다. 
# 없는 경우에 대비해서 outline 변수에 '아직 작성된 목차가 없습니다.'라는 기본 문구를 설정해 두었으므로 파일이 없다면 이 문구가 반환됩니다.
def get_outline(current_path):
    outline = '아직 작성된 목차가 없습니다.'

    if os.path.exists(f"{current_path}/data/outline.md"):
        with open(f"{current_path}/data/outline.md", "r", encoding='utf-8') as f:
            outline = f.read()  
    return outline

# save_outline 함수는 current_path와 outline을 매개변수로 받습니다. 
# 매개변수로 받은 outline의 값울 current_path/data 폴더에 outline.md라는 파일명으로 저장합니다.
def save_outline(current_path, outline):
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")
    
    with open(f"{current_path}/data/outline.md", "w", encoding='utf-8') as f:
        f.write(outline)
    return outline

# 이제 대화를 통해 목차를 작성할 수 있습니다. 
# 이 코드를 실행해 보면 목차를 작성하고 어떻게 수정하면 좋을지 물어봅니다. 
# 목차를 작성하기 위해 현재 2개의 에이전트가 일을 하고 있습니다. 
# 사용자가 원하는 내용을 입력하면 콘텐츠 전략가 에이전트 content_strategist가 목차를 작성하고
#  커뮤니케이터 에이전트 communicator가 진행 상황을 사용자에게 보고합니다.

# book_writer.py에 가서 실행한다.
# (ch13_env) PS C:\Aiprojects\ch14\ch14-1> cd ..            
# (ch13_env) PS C:\Aiprojects\ch14> cd ch14-1-1

# (ch13_env) PS C:\Aiprojects\ch14\ch14-1-1> python .\book_writer.py

# User    : JYP와 HYBE의 경영전략과 기업문화를 비교하는 책을 쓰고 싶어


# ============ CONTENT STRATEGIST ============
# 사용자의 요청에 따라 JYP와 HYBE의 경영 전략과 기업 문화를 비교하는 책을 쓰는 것이 목표입니다. 지난 목차가 이미 사용자의 요구에 부합하고 체계적으로 구성되어 있으므로 큰 변동 없이 유지하되, 추가적으로 사용자가 궁금해할 수 있는 부분을 보강하거나 최신 트렌드를 반영할 수 있습니다. 다음은 수정 및 보완된 목차 제안입니다:

# ### 목차 제안: JYP와 HYBE의 경영 전략과 기업 문화 비교 (수정 및 보완)

# 1. **서문**
#    - 책의 목적과 구성
#    - K-POP 산업의 배경과 중요성
#    - 이 책을 통해 얻을 수 있는 인사이트

# 2. **JYP와 HYBE 소개**
#    - JYP 엔터테인먼트 개요
#    - HYBE(구 빅히트 엔터테인먼트) 개요
#    - 두 기업의 설립자와 초기 역사
#    - 현재의 위치와 업적

# 3. **경영 전략 비교**
#    - 아티스트 발굴과 육성 전략
#      - JYP의 연습생 시스템
#      - HYBE의 글로벌 오디션 프로그램
#      - 최근 트렌드와 시스템의 진화
#    - 음악 제작과 콘텐츠 기획
#      - JYP의 음악 프로듀싱 철학
#      - HYBE의 스토리텔링과 세계관 구축
#      - AI 및 기술의 활용

# 4. **기업 문화 비교**
#    - 조직 구조와 운영 방식
#      - JYP의 수평적 조직 문화
#      - HYBE의 혁신적 운영 방식
#      - 두 기업의 조직 문화 변화와 대응
#    - 직원 복지와 인재 관리
#      - JYP의 인재 육성 프로그램
#      - HYBE의 글로벌 인재 확보 전략
#      - 지속 가능한 경영을 위한 노력

# 5. **브랜드 전략과 글로벌 진출**
#    - JYP의 해외 시장 개척 사례
#    - HYBE의 글로벌 팬덤 전략
#    - 두 기업의 협업과 파트너십 사례
#    - 글로벌 시장에서의 성공 전략

# 6. **성공 요인 및 과제**
#    - JYP와 HYBE의 성공 요인 분석
#    - 두 기업이 직면한 도전과 미래 과제
#    - K-POP 산업의 변화와 두 기업의 적응 전략

# 7. **결론**
#    - 두 기업의 경영 전략과 문화의 시사점
#    - K-POP 산업의 미래 전망
#    - 독자에게 전하는 메시지

# 8. **부록**
#    - 인터뷰 및 참고 자료
#    - 용어 설명 및 추가 자료
#    - 최근 데이터 및 통계

# 이 목차는 사용자 요구에 따라 JYP와 HYBE의 차별화된 경영 전략과 기업 문화를 심층적으로 분석하며, 독자들이 K-POP 산업의 현재와 미래를 이해하는 데 도움을 주고자 합니다. 추가로 원하는 내용이 있으면 언제든지 말씀해 주세요.
# [Content Strategist] 목차 작성 완료


# ============ COMMUNICATOR ============

# AI      : 안녕하세요! AI팀의 커뮤니케이터입니다. 현재 JYP와 HYBE의 경영전략과 기업문화를 비교하는 책의 목차 작성을 완료했습니다. 다음 단계로는 각 장의 세부 내용을 작성하는 것입니다. 혹시 추가로 고려해야 할 부분이나 특별히 다루고 싶은 주제가 있으신가요? 여러분의 의견을 반영할 수 있도록 도와드리겠습니다.
# ------------------------------------ MESSAGE COUNT       4

# User    :  q
# Goodbye!