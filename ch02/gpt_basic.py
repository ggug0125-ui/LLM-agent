# 언어 모델이 사용자가 입력한 텍스트에 대화 형식으로 응답을 생성하는 Chat Completion을 사용
# 2022년 월드컵 우승팀을 물어보고 답변을 받아오는 기능

from openai import OpenAI
from dotenv import load_dotenv
import os
# api키 입력
# api_key = '키변경.env'	
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# client.chat.completions.create() 포함된 값 중 model은 어떤 언어 모델을 사용할 지 정하는 부분
# 1. 플래그십 및 추론형 모델 (고성능)가장 복잡한 논리 구조나 코딩, 수학적 추론이 필요할 때 사용합니다.
# o1 시리즈 (o1-preview, o1-mini): 복잡한 추론에 특화된 모델입니다. 
# 답변을 내놓기 전 스스로 생각하는 과정을 거치며, chat.completions에서 높은 논리력을 발휘합니다.

# GPT-4o (gpt-4o): 현재 가장 범용적으로 쓰이는 옴니(Omni) 모델입니다. 
# 텍스트, 이미지, 음성을 동시에 처리하며 속도와 지능의 균형이 가장 좋습니다.

# 2. 효율성 및 비용 최적화 모델빠른 응답 속도와 저렴한 비용이 중요할 때 적합합니다.
# GPT-4o mini (gpt-4o-mini): 기존의 gpt-3.5-turbo를 대체하는 모델로, 
# 매우 저렴하면서도 이전의 4세대급 성능을 보여줍니다. 가벼운 텍스트 요약이나 단순 챗봇에 최적입니다.

# 3. 특정 기능 특화 모델GPT-4 Turbo (gpt-4-turbo): GPT-4o 출시 이전의 고성능 모델로, 
# 특정 과거 버전의 일관성이 필요할 때 여전히 사용됩니다.

# 모델 선택 가이드 테이블
# 모델명	     주요 특징	                         추천 용도
# o1-preview	깊은 추론, 복잡한 과학/수학 문제	     고난도 알고리즘 설계, 논문 분석
# gpt-4o	    높은 지능, 멀티모달(이미지 인식 등)	   대부분의 복잡한 비즈니스 로직, 코딩
# gpt-4o-mini 매우 빠름, 압도적 가성비	            단순 고객 응대, 대량의 데이터 분류

response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.1,  
  # n=3, Choice(...), Choice(...), Choice(...) 으로 답변이 3개 만들어 짐!
  # 문장을 생성할 때 무작위성을 조절함
  # 0에 가까울수록 안정적 / 1에 가까울 수록 창의적(일관성없음)

  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "2022년 월드컵 우승팀은 어디야?"},
  ]
  # messages는 과거의 대화를 기반으로 적절한 응답을 생성하는데 필요한 매개변수
  # 맥락에 맞게 응답을 생성하로고 역할(role), 내용(content) dict 타입으로 만들어 리스트로 쌓아 api로 보냄
  # role은 system(GPT의 역할), user(언어모델과 대화를 나누는 사용자), assistant(언어모델의 답변)
  # content()는 사용자의 질문을 담고 있다.
)

print(response)

print('----')	
print(response.choices[0].message.content) 
# response
# client.chat.completions.create() 호출 결과 전체
# 메타데이터 + 여러 후보 답변(choices) 포함된 객체

# response.choices
# 모델이 생성한 답변 후보 리스트
# 보통 1개만 생성되지만, 여러 개 요청하면 여러 개 들어있음
# response.choices
# [Choice(...), Choice(...), ...]

# response.choices[0]
# 첫 번째 답변 선택, 대부분 여기만 쓰면 됨

# .message
# 해당 답변의 메시지 객체
# 구조:
# {
#   "role": "assistant",
#   "content": "실제 답변 내용"
# }
# .content
# 진짜 우리가 보고 싶은 답변 텍스트

# print(response) 결과 assistant는 GPT가 제공하는 답변
# (venv) PS C:\Aiprojects> cd .\ch02\
# (venv) PS C:\Aiprojects\ch02> py.exe .\gpt_basic.py
# ChatCompletion(id='chatcmpl-DQ3HB3yt3ljUz7RJWAPoMp0vmpWA4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='2022년 FIFA 월
# 드컵에서는 아르헨티나가 우승을 차지했습니다. 아르헨티나는 결승전에서 프랑스를 상대로 승리하여 월드컵 트로피를 들어올렸습니다.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None, annotations=[]))], created=1775101493, model='gpt-4o-2024-08-06', object='chat.completion', service_tier='default', system_fingerprint='fp_c8b70290c4', usage=CompletionUsage(completion_tokens=52, prompt_tokens=30, total_tokens=82, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
# ----
# print(response.choices[0].message.content) 결과 답변 내용만 출력
# 2022년 FIFA 월드컵에서는 아르헨티나가 우승을 차지했습니다. 아르헨티나는 결승전에서 프랑스를 상대로 승리하여 월드컵 트로피를 들어올렸습니다.
