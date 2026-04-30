from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# ② 퓨샷프롬프팅하면 여러가지의 힌트를 제공해줄 수 있다.
response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.9,  # ③
  messages=[
    {"role": "system", "content": "너는 유치원 학생이야. 유치원생처럼 답변해줘."},
    {"role": "user", "content": "참새"},
    {"role": "assistant", "content": "짹짹"}, # 힌트1
    {"role": "user", "content": "말"},
    {"role": "assistant", "content": "히이잉"}, # 힌트2
    {"role": "user", "content": "개구리"},
    {"role": "assistant", "content": "개굴개굴"}, # 힌트3
    {"role": "user", "content": "뱀"},
  ]		# ④ 퓨샷프롬프트
)

print(response)

print('----')	# ⑤ 구분선
print(response.choices[0].message.content) 


# 결과보기 -> 뱀 (스스슥!!!)
# (venv) PS C:\Aiprojects\ch03> py .\few_shot.py
# ChatCompletion(id='chatcmpl-DQ3kV7CQ7fbqIdoHj6XBKQUpEtg0m', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='스스슥!', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None, annotations=[]))], created=1775103311, model='gpt-4o-2024-08-06', object='chat.completion', service_tier='default', system_fingerprint='fp_c8b70290c4', usage=CompletionUsage(completion_tokens=5, prompt_tokens=75, total_tokens=80, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
# ----
# 스스슥!
# (venv) PS C:\Aiprojects\ch03> 