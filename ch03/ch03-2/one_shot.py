from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# ② 질문에 대한 답변 힌트를 제공함!! assistant
response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.9,  # ③
  messages=[
    {"role": "system", "content": "너는 유치원 학생이야. 유치원생처럼 답변해줘."},
    {"role": "user", "content": "참새"},
    {"role": "assistant", "content": "짹짹"}, # 정답을 유도하는 원샷프롬프트!!!!
    {"role": "user", "content": "오리"},
  ]		# ④ 오리~ 꽥꽥???
)

print(response)

print('----')	# ⑤
print(response.choices[0].message.content) 

# (venv) PS C:\Aiprojects\ch03> py .\one_shot.py
# ChatCompletion(id='chatcmpl-DQ3iVzbY20cTjWGMHz8kOkcA8nlkh', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='꽥꽥!', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None, annotations=[]))], created=1775103187, model='gpt-4o-2024-08-06', object='chat.completion', service_tier='default', system_fingerprint='fp_c8b70290c4', usage=CompletionUsage(completion_tokens=7, prompt_tokens=46, total_tokens=53, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
# ----
# 꽥꽥!
# (venv) PS C:\Aiprojects\ch03> 