# save_state 함수를 만들겠습니다. book_writer.py 파일에 작성할 수도 있지만 
# 파일이 너무 커지는 것을 방지하기 위해 새로운 파일인 utils.py을 같은 폴더 내에 만들어 작성하겠습니다. 
# save_state 함수는 current_path와 state를 매개변수로 받아 current_path/data 폴더에 state를 JSON 파일로 저장합니다. 
# 현재 state에는 사용자와 AI 간의 대화 내용이 담긴 messages만 존재하므로 [(m .__ class __·__ name __ , m.content) 형태로 자료형과 대화
# 내용을 튜플로 변환하여 저장합니다. 이때 ensure_ascii=False와 encoding='utf-8'을 지정하여 저장된 JSON 파일의 한글이 깨지지 않도록 합니다.

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

# book_writer.py에 가서 실행한다.
# (ch13_env) PS C:\Aiprojects> cd ch14/ch14-1
# (ch13_env) PS C:\Aiprojects\ch14\ch14-1> python .\book_writer.py

# User    : HYBE와 JYP의 경영 전략과 기업 문화에 대한 책을 써줘


# ============ COMMUNICATOR ============

# AI      : 안녕하세요! 저희 AI팀은 HYBE와 JYP의 경영 전략과 기업 문화에 대한 책을 작성하는 작업을 진행하고 있습니다. 현재까지의 진행상황을 간단히 말씀드리면, 두 기업의 역사와 성장 과정, 그리고 각 회사의 경영 전략 및 기업 문화를 분석하고 있습니다. 또한, 두 기업의 차별점과 유사점에 대한 심층적인 연구도 함께 진행 중입니다.

# 이 책에서 특별히 다루었으면 하는 내용이나 궁금한 점이 있으신가요? 여러분의 의견을 듣고 싶습니다.
# ------------------------------------ MESSAGE COUNT       3

# User    : Q
# Goodbye!

# 결론
# 이 실행 결과는 data/state.json 파일에 저장됩니다. 
# 시스템이 알아서 작업해 주면 좋겠지만 아직은 그렇지 못한 상태입니다. 
# 이러한 결과물을 보고 언어 모델을 GPT-40로 선택했는데도 그다지 쓸 만하지 않다고 생각해 버리는 사람이 많습니다. 
# 하지만 이는 아직 우리가 적절한 가이드를 제공하지 못했기 때문일 수 있습니다.