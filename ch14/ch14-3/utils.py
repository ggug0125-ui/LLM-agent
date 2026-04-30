import os
import json

def save_state(current_path, state):
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")
    
    state_dict = {}

    messages = [(m.__class__.__name__, m.content) for m in state["messages"]]
    state_dict["messages"] = messages

    # utils.py 파일의 save_state 함수에 task_history도 저장되도록 코드를 한 줄 추가하겠습니다. 
    # 매개변수 state를 통해 받은 값을 .to_dict() 메서드를 이용해 딕셔너리로 만드는 내용을 추가합니다. 
    # 이렇게 하면 ./data/state.json 파일에 작업 이력인 task_history까지 깔끔하게 저장됩니다.   
    state_dict["task_history"] = [task.to_dict() for task in state.get("task_history", [])] # p409추가
    

    # references p441 추가
    # 마지막으로 벡터 검색 결과로 가져온 참고 자료를 state.json 파일에 저장하여 현재 상태를 확인할 수 있도록 하겠습니다.

    # references를 저장하는 코드를 작성합니다. 
    # 특히 state의 references 항목에 담긴 문서 청크는 랭체인의 Document 객체로 되어 있어서 이를 바로 딕셔너리로 만들 수 없습니다. 
    # 그래서 문서의 metadata를 추출하여 저장합니다.
    references = state.get("references", {"queries": [], "docs": []})
    state_dict["references"] = {
        "queries": references["queries"], 
        "docs": [doc.metadata for doc in references["docs"]]
    }

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

# (ch13_env) PS C:\Aiprojects\ch14\ch14-2> cd ..
# (ch13_env) PS C:\Aiprojects\ch14> cd ch14-2-1
# (ch13_env) PS C:\Aiprojects\ch14\ch14-2-1> python .\book_writer.py

# User    : 안녕?


# ============ SUPERVISOR ============
# [Supervisor] agent='communicator' done=False description='목차가 아직 작성되지 않았으므로, 사용자에게 현재 진행 상황을 보고하고 다음 지시를 물어본다.' done_at='2026-04-24 16:24:11'


# ============ COMMUNICATOR ============

# AI      : 안녕하세요! 현재 저희 AI 팀은 책을 쓰기 위한 준비 단계에 있으며, 아직 목차는 작성되지 않았습니다. 사용자의 의견과 요구를 반영하여 진행하려고 하니, 원하는 방향이나 추가적인 의견이 있다면 말씀해 주세요. 여러분의 피드백은 저희에게 큰 도움이 됩니다. 감사합니다!
# ------------------------------------ MESSAGE COUNT       4

# User    : AI 에이전트와 에이전스의 차이를 비교하는 책을 써줘


# ============ SUPERVISOR ============
# [Supervisor] agent='content_strategist' done=False description="사용자가 요청한 'AI 에이전트와 에이전스의 차이를 비교하는 책'의 콘텐츠 전략을 수립하고, 전체 책의 목차를 작성한다." done_at='2026-04-24 16:24:11'


# ============ CONTENT STRATEGIST ============
# 사용자의 요구에 따라 "AI 에이전트와 에이전스의 차이를 비교"하는 책의 목차를 제안합니다. 이 책은 AI 에이전트와 에이전스의 개념, 역사, 응용, 그리고 앞으로의 전망을 다루는 내용을 포함할 것입니다.

# ### 목차 제안

# 1. **서문**
#    - 책의 목적과 주요 주제 소개
#    - 독자에게 주는 가치 제언

# 2. **AI 에이전트와 에이전스: 기본 개념**
#    - AI 에이전트란 무엇인가
#    - 에이전스란 무엇인가
#    - 에이전트와 에이전스의 차이점

# 3. **AI 에이전트의 역사와 발전**
#    - 초기 AI 에이전트 개념
#    - 현대 AI 에이전트의 발전
#    - AI 에이전트의 주요 기술

# 4. **에이전스의 철학적 배경**
#    - 에이전스의 철학적 기원
#    - 자율성과 책임 문제
#    - 윤리적 고려사항

# 5. **AI 에이전트의 실제 응용 사례**
#    - 산업별 AI 에이전트 활용 사례
#    - AI 에이전트의 장점과 한계

# 6. **에이전스와 사회적 영향**
#    - 에이전스가 미치는 사회적 영향
#    - 에이전스와 인간 상호작용

# 7. **미래 전망**
#    - AI 에이전트와 에이전스의 미래
#    - 기술 발전이 가져올 변화 예측
#    - 인류와 AI의 공존 방향

# 8. **결론**
#    - 주요 논점 요약
#    - 독자에게 전하는 메시지

# 9. **부록**
#    - 용어 사전
#    - 추가 자료 및 추천 도서

# 10. **참고문헌**

# 이 목차는 사용자의 피드백을 바탕으로 수정될 수 있으며, 추가적인 의견이나 주제가 있다면 언제든지 말씀해 주세요.
# [Content Strategist] 목차 작성 완료
# agent='communicator' done=False description='AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다' done_at=''


# ============ COMMUNICATOR ============

# AI      : 안녕하세요! 저희 AI 팀은 사용자님의 요청에 따라 "AI 에이전트와 에이전스의 차이를 비교"하는 책의 목차를 작성했습니다. 이 책은 AI 에이전트와 에이전스의 개념, 역사, 응용, 그리고 앞으로의 전망을 다루는 내용을 포함하고 있습니다.

# 현재 목차는 완성된 상태이며, 사용자님의 의견을 반영하여 수정할 수 있습니다. 책의 내용이나 특정 주제에 대한 추가적인 의견이나 질문이 있으시면 언제든지 말씀해 주세요. 여러분의 피드백은 저희에게 큰 도움이 됩니다. 감사합니다!
# ------------------------------------ MESSAGE COUNT       8