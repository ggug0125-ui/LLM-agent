# from langchain_openai import ChatOpenAI
# pip uninstall langchain langchain-community langchain-core langsmith -y
# pip install langchain langchain-community langchain-ollama ollama youtube-transcript-api youtube-search

from langchain_ollama import ChatOllama # pip install langchain-ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOllama(model="deepseek-r1:latest")  # 다운받은 버전 일치
messages = [
    SystemMessage("너는 사용자를 도와주는 상담사야."),
]

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break
    
    messages.append( 
        HumanMessage(user_input)
    )  
    
    # 스트림 방식으로 수정 p318
    response = llm.stream(messages)
    #②
    ai_message = None
    for chunk in response:
        print(chunk.content, end="")
        if ai_message is None:
            ai_message = chunk
        else:
            ai_message += chunk
    print('')
	#③ think eepseek-r1 특징:
        # 어떤 경우 → <think> ... </think> 포함
        # 어떤 경우 → 그냥 답변만 줌
    if "</think>" in ai_message.content:
        message_only = ai_message.content.split("</think>", 1)[1].strip()
    else:
        message_only = ai_message.content.strip()

    # print("AI: " + response.content)
