import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


llm = ChatOllama(model=os.getenv("OLLAMA_CHAT_MODEL", "deepseek-r1:7b"))
messages = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    response = llm.stream(messages)

    ai_message = None
    for chunk in response:
        print(chunk.content, end="")
        if ai_message is None:
            ai_message = chunk
        else:
            ai_message += chunk
    print("")

    content = ai_message.content.strip() if ai_message else ""
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()

    messages.append(AIMessage(content=content))
