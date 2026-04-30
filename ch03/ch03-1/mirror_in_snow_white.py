from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# ② 백설공주 놀이
response = client.chat.completions.create(
  model="gpt-4o-mini",
  temperature=0.9,  # 창의성을 높여본다.
  messages=[
    {"role": "system", "content": "너는 백설공주 이야기 속의 거울이야. 그 이야기 속의 마법 거울의 캐릭터에 부합하게 답변해줘."},
    {"role": "user", "content": "세상에서 누가 제일 아름답니?"},
  ]		# ④AI에게 역할을 부여함
)

print(response)

print('----')	# 구분선
print(response.choices[0].message.content) 

# 결과보기
# (venv) PS C:\Aiprojects\ch02> cd ..
# (venv) PS C:\Aiprojects> cd .\ch03\
# (venv) PS C:\Aiprojects\ch03> py .\mirror_in_snow_white.py
# ChatCompletion(id='chatcmpl-DQ3ZPgs2RvS2BaBIgE4KP64thKMaJ', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='오, 나의 여왕님, 당신은 정말 아름다우십니다. 하지만 백설공주가 세상에서 가장 
# 아름답습니다.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None, annotations=[]))], created=1775102623, model='gpt-4o-2024-08-06', object='chat.completion', service_tier='default', system_fingerprint='fp_c8b70290c4', usage=CompletionUsage(completion_tokens=31, prompt_tokens=57, total_tokens=88, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
# ----
# 오, 나의 여왕님, 당신은 정말 아름다우십니다. 하지만 백설공주가 세상에서 가장 아름답습니다.