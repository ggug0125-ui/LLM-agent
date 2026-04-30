from glob import glob # 추후 for문으로 여러 파일의 경로를 가져오기 위해 선언
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


q = image_quiz("./ch06/data/images/busan_dive.jpg")
print(q)



txt = '' # ①  문제들을 계속 붙여 나가기 위해 빈 문자열 선언
no = 1 # 문제 번호를 위해 선언
for g in glob('./ch06/data/images/*.jpg'):  # ②
    q, is_suceed = image_quiz(g)

    if not is_suceed:
        continue


    divider = f'## 문제 {no}\n\n'
    print(divider)
    
    txt += divider  # ③
    # 파일명 추출해 이미지 링크 만들기
    filename = os.path.basename(g) # ③ 마크다운에 표시할 이미지 파일 경로 설정   
    txt += f'![image]({filename})\n\n' # ③

    # 문제 추가
    print(q)
    txt += q + '\n\n---------------------\n\n'
    # ④ 마크다운 파일로 저장
    with open('./ch06/data/images/image_quiz_eng.md', 'w', encoding='utf-8') as f:
        f.write(txt)
    
    no += 1 # 문제 번호 증가


# 한글과 영어평가 가능
# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 많은 사람들이 테이블에 앉아 있습니다.
# - (2) 벽에 "DIVE 2024 IN BUSAN"이라는 문구가 보입니다.
# - (3) 사람들이 정장을 입고 회의에 참석하고 있습니다.
# - (4) 천장에는 조명이 많이 설치되어 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) Many people are seated at tables.
# - (2) There is a sign on the wall saying "DIVE 2024 IN BUSAN."
# - (3) People are attending a meeting in formal suits.
# - (4) There are many lights installed on the ceiling.

# 정답: (3) 사람들이 정장을 입고 있지는 않고, 캐주얼한 복장을 하고 있습니다.
# ## 문제 2


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 노란색 대형 조형물이 중앙에 위치해 있습니다.
# - (2) 건물 벽면에 "Local Stitch"라는 문구가 보입니다.
# - (3) 건물 사이에 작은 정원이 가꾸어져 있습니다.
# - (4) 사람들은 벤치에 앉아 휴식을 취하고 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) A large yellow sculpture is located in the center.
# - (2) The phrase "Local Stitch" is visible on the building wall.
# - (3) There is a small garden cultivated between the buildings.
# - (4) People are sitting on benches taking a rest.

# 정답: (4) 사람들은 벤치에 앉아 있지 않습니다.
# ## 문제 3


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?

# - (1) 카페에는 노란색 벽이 있습니다.
# - (2) 여러 개의 빈 의자와 테이블이 있습니다.
# - (3) 커피를 만드는 직원이 두 명 보입니다.
# - (4) 바리스타들이 앞치마를 착용하고 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?

# - (1) The cafe has a yellow wall.
# - (2) There are multiple empty chairs and tables.
# - (3) Two employees making coffee are visible.
# - (4) The baristas are wearing aprons.

# 정답: (3) 커피를 만드는 직원은 한 명만 보입니다.
# ## 문제 4


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 카페 창문에 "PERSONAL COFFEE"라는 글자가 보입니다.
# - (2) 건물은 벽돌로 지어져 있습니다.
# - (3) 1층에 자동차가 주차되어 있습니다.
# - (4) 인도 위에 오렌지색 원뿔이 놓여 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) The cafe window has the words "PERSONAL COFFEE."
# - (2) The building is made of bricks.
# - (3) There is a car parked on the ground floor.
# - (4) An orange cone is placed on the sidewalk.

# 정답: (3) 1층에는 자동차가 주차되어 있지 않습니다.
# ## 문제 5


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 다양한 종류의 빵이 진열되어 있습니다.
# - (2) 유리 진열장 뒤에서 파티셰가 빵을 포장하고 있습니다.
# - (3) 빵의 가격이 라벨에 적혀 있습니다.
# - (4) 진열장은 나무로 만들어져 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) Various types of bread are displayed.
# - (2) A pastry chef is wrapping up bread behind the glass showcase.
# - (3) The price of the bread is written on labels.
# - (4) The showcase is made of wood.

# 정답: (2) 파티셰가 빵을 포장하는 모습은 보이지 않습니다.
# ## 문제 6


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 사람들이 건물 앞에서 작업을 하고 있습니다.
# - (2) "베이커리 카페"라는 문구가 보이는 간판이 있습니다.
# - (3) 작업자들 중 한 명이 노란색 모자를 착용하고 있습니다.
# - (4) 나무가 건물 바로 앞에 심어져 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) People are working in front of a building.
# - (2) There is a sign with the words "bakery cafe."
# - (3) One of the workers is wearing a yellow hat.
# - (4) A tree is planted right in front of the building.

# 정답: (3) 작업자들 중 한 명이 노란색 모자를 착용하고 있지는 않습니다.
# ## 문제 7


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 여러 사람이 창가 근처에서 대화를 나누고 있습니다.
# - (2) 바닥이 나무 마루로 되어 있습니다.
# - (3) 사람들은 전부 서서 대화를 하고 있습니다.
# - (4) 천장에는 여러 개의 조명이 설치되어 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) Several people are having conversations near the windows.
# - (2) The floor is made of wooden planks.
# - (3) All the people are standing while conversing.
# - (4) There are multiple lights installed on the ceiling.

# 정답: (3) 사람들은 전부 서서 대화를 하고 있지 않고, 앉아 있는 사람들도 있습니다.
# ## 문제 8


# Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
# - (1) 사람들이 버스에 타기 위해 줄을 서 있습니다.
# - (2) 한 사람이 전화 통화를 하고 있습니다.
# - (3) 창문에 "COFFEE & BAKERY"라는 글씨가 있습니다.
# - (4) 사람들은 버스에서 내리고 있습니다.

# Listening: Which of the following descriptions of the image is incorrect?
# - (1) People are lining up to get on the bus.
# - (2) A person is talking on the phone.
# - (3) The window has the words "COFFEE & BAKERY."
# - (4) People are getting off the bus.

# 정답: (4) 사람들은 버스에서 내리고 있는 것이 아니라 타고 있습니다.
