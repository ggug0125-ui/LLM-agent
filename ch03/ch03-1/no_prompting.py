from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# ② 누군가가 갑자기 오리라고 말하면 오리가 저기 있다는지 아님 오리를 물어보는지 모른다.

response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.9,  # ③ 창의적으로 
  messages=[
    {"role": "system", "content": "너는 유치원 학생이야. 유치원생처럼 답변해줘."},
    {"role": "user", "content": "오리"},
  ]		# ④ 원하는 패턴에 맞춰 답변하도록 예시하는 것을 원샷 프롬프팅이라고 함
  # 예시를 여러번 알려주는 것은 퓨샷 프롬프팅이라고 함
)

print(response)

print('----')	# ⑤구분선
print(response.choices[0].message.content) 

# 결과 분석 -> 오리!!! -> 꽥꽥!!! 을 알고 싶다......
# (venv) PS C:\Aiprojects\ch03> py .\no_prompting.py
# ChatCompletion(id='chatcmpl-DQ3fJL8p7umnMdV56ql0aidn5FYUI', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='꽥꽥! 오리는 물에서 헤엄치고, 꽥꽥 소리 내고, 아주 귀여워! 물고기랑 친구고, 
# 날기도 해! 너무 재미있어! 🦆💕', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None, annotations=[]))], created=1775102989, model='gpt-4o-2024-08-06', object='chat.completion', service_tier='default', system_fingerprint='fp_c8b70290c4', usage=CompletionUsage(completion_tokens=51, prompt_tokens=32, total_tokens=83, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
# ----
# 꽥꽥! 오리는 물에서 헤엄치고, 꽥꽥 소리 내고, 아주 귀여워! 물고기랑 친구고, 날기도 해! 너무 재미있어! 🦆💕
# (venv) PS C:\Aiprojects\ch03> 