# 영어 문제만 출력 json으로 출력
from glob import glob
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기
client = OpenAI(api_key=api_key)  # OpenAI 클라이언트의 인스턴스 생성

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def image_quiz(image_path, n_trial=0, max_trial=3):
    if n_trial >= max_trial: # 최대 시도 회수에 도달하면 포기
        raise Exception("Failed to generate a quiz.")
    
    base64_image = encode_image(image_path) # 이미지를 base64로 인코딩

    quiz_prompt = """
    제공된 이미지를 바탕으로, 다음과 같은 양식으로 퀴즈를 만들어주세요. 
    정답은 1~4 중 하나만 해당하도록 출제하세요.
    토익 리스닝 문제 스타일로 문제를 만들어주세요.
    아래는 예시입니다. 
    ----- 예시 -----

    Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
    - (1) 베이커리에서 사람들이 빵을 사고 있는 모습이 담겨 있습니다.
    - (2) 맨 앞에 서 있는 사람은 빨간색 셔츠를 입고 있습니다.
    - (3) 기차를 타기 위해 줄을 서 있는 사람들이 있습니다.
    - (4) 점원은 노란색 티셔츠를 입고 있습니다.

    Listening: Which of the following descriptions of the image is incorrect?
    - (1) It shows people buying bread at a bakery.
    - (2) The person standing at the front is wearing a red shirt.
    - (3) There are people lining up to take a train.
    - (4) The clerk is wearing a yellow T-shirt.
        
    정답: (4) 점원은 노란색 티셔츠가 아닌 파란색 티셔츠를 입고 있습니다.
    (주의: 정답은 1~4 중 하나만 선택되도록 출제하세요.)
    ======
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": quiz_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ]

    try: 
        response = client.chat.completions.create(
            model="gpt-4o",  # 응답 생성에 사용할 모델 지정
            messages=messages # 대화 기록을 입력으로 전달
        )
    except Exception as e:
        print("failed\n" + e)
        return image_quiz(image_path, n_trial+1)
    
    content = response.choices[0].message.content

    if "Listening:" in content:
        return content, True
    else:
        return image_quiz(image_path, n_trial+1)


# q = image_quiz("./ch06/data/images/busan_dive.jpg")
# print(q)



txt = '' # 문제들을 계속 붙여 나가기 위해 빈 문자열 선언
eng_dict = []
no = 1 # 문제 번호를 위해 선언
for g in glob('./ch06/data/images/*.jpg'):  # ②
    q, is_suceed = image_quiz(g)

    if not is_suceed:
        continue


    divider = f'## 문제 {no}\n\n'
    print(divider)
    
    txt += divider
    # 파일명 추출해 이미지 링크 만들기
    filename = os.path.basename(g) # 마크다운에 표시할 이미지 파일 경로 설정   
    txt += f'![image]({filename})\n\n' 

    # 문제 추가
    print(q)
    txt += q + '\n\n---------------------\n\n'
    # 마크다운 파일로 저장
    with open('./ch06/data/images/image_quiz_eng.md', 'w', encoding='utf-8') as f:
        f.write(txt)

    # 영어 문제만 추출
    eng = q.split('Listening: ')[1].split('정답:')[0].strip()

    eng_dict.append({
        'no': no,
        'eng': eng,
        'img': filename
    })

    # json 파일로 저장
    with open('./ch06/data/images/image_quiz_eng.json', 'w', encoding='utf-8') as f:
        json.dump(eng_dict, f, ensure_ascii=False, indent=4)
    
    
    no += 1 # 문제 번호 증가

# v) PS C:\Aiprojects> & c:\Aiprojects\venv\Scripts\python.exe c:/Aiprojects/ch06/ch06-2/3.image_quiz.py
## 문제 1

# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 많은 사람들이 회의 공간에서 노트북을 사용하고 있습니다.
# - (2) 천장에는 여러 개의 조명이 설치되어 있습니다.
# - (3) 사람들이 자연 속에서 미팅을 하고 있습니다.
# - (4) 회의장의 벽면에 큰 전광판이 보입니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) Many people are using laptops in a conference space.
# - (2) There are several lights installed on the ceiling.
# - (3) People are meeting in a natural outdoor setting.
# - (4) A large electronic display is visible on the wall of the conference hall.

# 정답: (3) 사람들이 실내에서 미팅을 하고 있습니다.
# ## 문제 2


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 이미지에는 대형 노란색 조각이 있습니다.
# - (2) 건물의 이름은 'Local Stitch'입니다.
# - (3) 건물 외벽은 파란색으로 칠해져 있습니다.
# - (4) 건물 창문이 여러 개 보입니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) There is a large yellow sculpture in the image.
# - (2) The building is named 'Local Stitch'.
# - (3) The exterior walls of the building are painted blue.
# - (4) There are several windows visible on the building.

# 정답: (3) 건물 외벽은 파란색이 아닌 벽돌색으로 칠해져 있습니다.
# ## 문제 3


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 카페에는 노란색 벽이 있습니다.
# - (2) 손님들이 카운터에서 주문하고 있습니다.
# - (3) 창밖으로 나무가 보입니다.
# - (4) 모든 테이블에 사람들이 앉아 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) The cafe has a yellow wall.
# - (2) Customers are ordering at the counter.
# - (3) Trees are visible outside the windows.
# - (4) All tables have people sitting at them.

# 정답: (4) 모든 테이블에 사람들이 앉아 있는 것이 아니라 일부 테이블은 비어 있습니다.
# ## 문제 4


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 건물의 1층에 커피숍이 위치해 있습니다.
# - (2) 건물은 붉은 벽돌로 지어져 있습니다.
# - (3) 교차로의 신호등이 녹색으로 켜져 있습니다.
# - (4) 건물 앞 도로에 주황색 원뿔이 놓여 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) A coffee shop is located on the first floor of the building.
# - (2) The building is made of red bricks.
# - (3) The traffic light at the intersection shows green.
# - (4) There is an orange cone placed on the road in front of the building.

# 정답: (3) 교차로의 신호등은 보이지 않습니다.
# ## 문제 5


# -----
# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 다양한 종류의 빵이 진열되어 있습니다.
# - (2) 두 명의 점원이 모자를 쓰고 있는 모습이 보입니다.
# - (3) 빵 가격이 모두 동일합니다.
# - (4) 진열대는 두 층으로 나누어져 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) Various types of bread are displayed.
# - (2) Two clerks wearing caps are visible.
# - (3) All the bread prices are the same.
# - (4) The display is divided into two levels.

# 정답: (3) 빵 가격이 모두 동일하지 않습니다. 가격이 다양합니다.
# -----
# ## 문제 6


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 사람들이 건물 외부에서 작업을 하고 있습니다.
# - (2) 중간에 빨간색 의자들이 쌓여 있습니다.
# - (3) 작업자 중 한 명이 흰색 헬멧을 착용하고 있습니다.
# - (4) 오른쪽에는 "뉴트리코어"라는 상호가 보입니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) People are working outside the building.
# - (2) There are red chairs stacked in the middle.
# - (3) One of the workers is wearing a white helmet.
# - (4) On the right, you can see a sign with "뉴트리코어" (Nutricore).

# 정답: (3) 작업자 중 헬멧을 착용한 사람이 보이지 않습니다.
# ## 문제 7


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 넓은 창문을 통해 자연광이 들어오고 있습니다.
# - (2) 사람들이 책상에서 각자 공부를 하고 있는 모습입니다.
# - (3) 식사 공간으로 보이는 장소에 여러 개의 의자가 배치되어 있습니다.
# - (4) 바닥은 짙은 색의 나무로 마감되어 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) Natural light is coming through large windows.
# - (2) People are studying individually at tables.
# - (3) There are several chairs arranged in what appears to be a dining area.
# - (4) The floor is finished with dark wood.

# 정답: (2) 사람들이 공부하는 모습이 아닌, 대화하는 모습이 담겨 있습니다.
# ## 문제 8


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 사람들이 버스에 탑승하기 위해 줄을 서 있습니다.
# - (2) 커피숍 창문에는 "COFFEE & BAKERY"라는 글자가 보입니다.
# - (3) 정류장 옆에 나무들이 줄지어 서 있습니다.
# - (4) 사람이 모두 검은색 옷을 입고 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) People are queuing to board a bus.
# - (2) The window of the café has the words "COFFEE & BAKERY" on it.
# - (3) Trees are lined up next to the station.
# - (4) Everyone is wearing black clothes.

# 정답: (4) 옷의 색깔이 모두 검은색인 것은 아닙니다.