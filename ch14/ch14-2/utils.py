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
# 새로운 책의 주제는 JYP와 HYBE의 경영 전략과 기업 문화를 비교하는 것입니다. 이에 맞춰 목차를 제안해 보겠습니다.

# ### 목차 제안: JYP와 HYBE의 경영 전략과 기업 문화 비교

# 1. **서문**
#    - 책의 목적과 구성
#    - K-POP 산업의 배경과 중요성

# 2. **JYP와 HYBE 소개**
#    - JYP 엔터테인먼트 개요
#    - HYBE(구 빅히트 엔터테인먼트) 개요
#    - 두 기업의 설립자와 초기 역사

# 3. **경영 전략 비교**
#    - 아티스트 발굴과 육성 전략
#      - JYP의 연습생 시스템
#      - HYBE의 글로벌 오디션 프로그램
#    - 음악 제작과 콘텐츠 기획
#      - JYP의 음악 프로듀싱 철학
#      - HYBE의 스토리텔링과 세계관 구축

# 4. **기업 문화 비교**
#    - 조직 구조와 운영 방식
#      - JYP의 수평적 조직 문화
#      - HYBE의 혁신적 운영 방식
#    - 직원 복지와 인재 관리
#      - JYP의 인재 육성 프로그램
#      - HYBE의 글로벌 인재 확보 전략

# 5. **브랜드 전략과 글로벌 진출**
#    - JYP의 해외 시장 개척 사례
#    - HYBE의 글로벌 팬덤 전략
#    - 두 기업의 협업과 파트너십 사례

# 6. **성공 요인 및 과제**
#    - JYP와 HYBE의 성공 요인 분석
#    - 두 기업이 직면한 도전과 미래 과제

# 7. **결론**
#    - 두 기업의 경영 전략과 문화의 시사점
#    - K-POP 산업의 미래 전망

# 8. **부록**
#    - 인터뷰 및 참고 자료
#    - 용어 설명 및 추가 자료

# 이 목차는 JYP와 HYBE의 경영 전략과 기업 문화를 체계적으로 비교 분석하여 독자들에게 깊은 인사이트를 제공하는 것을 목표로 합니다. 추가적인 요구사항이나 변경사항이 있다면 말씀해 주세요.
# [Content Strategist] 목차 작성 완료


# ============ COMMUNICATOR ============

# AI      : 안녕하세요! AI팀의 커뮤니케이터입니다. 현재 AI팀은 "JYP와 HYBE의 경영전략과 기업문화를 비교하는 책"의 목차 작성을 완료한 상태입니다. 다음 단계는 각 목차에 대한 세부 내용 작성입니다. 사용자님의 의견이나 추가적으로 원하는 내용이 있으신가요? 반영할 수 있도록 도와드리겠습니다.
# ------------------------------------ MESSAGE COUNT       4

# User    : Q
# Goodbye!