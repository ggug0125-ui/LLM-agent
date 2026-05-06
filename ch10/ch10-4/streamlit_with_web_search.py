import os
from datetime import datetime

import pytz
import streamlit as st
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY를 찾지 못했습니다. Aiprojects/.env 파일을 확인하세요.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)


@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환한다. timezone은 예: Asia/Seoul 형식이다."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timezone} ({location}) 현재시각 {now}"
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"


@tool
def get_web_search(query: str, search_period: str = "m") -> str:
    """
    인터넷 검색을 수행한다.

    Args:
        query: 검색어
        search_period: 검색 기간. d=일, w=주, m=월, y=년
    """
    period_map = {
        "d": "d",
        "w": "w",
        "m": "m",
        "y": "y",
        "day": "d",
        "week": "w",
        "month": "m",
        "year": "y",
    }
    timelimit = period_map.get(search_period, "m")

    with DDGS() as ddgs:
        results = list(
            ddgs.text(
                query,
                region="kr-kr",
                timelimit=timelimit,
                max_results=5,
            )
        )

    if not results:
        return "검색 결과를 찾지 못했습니다."

    lines = []
    for index, item in enumerate(results, start=1):
        title = item.get("title", "")
        href = item.get("href", "")
        body = item.get("body", "")
        lines.append(f"{index}. {title}\n{body}\nURL: {href}")

    return "\n\n".join(lines)


tools = [get_current_time, get_web_search]
tool_dict = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


def get_ai_response(messages):
    gathered = None
    for chunk in llm_with_tools.stream(messages):
        yield chunk
        gathered = chunk if gathered is None else gathered + chunk

    if not gathered or not gathered.tool_calls:
        return

    st.session_state.messages.append(gathered)

    for tool_call in gathered.tool_calls:
        selected_tool = tool_dict[tool_call["name"]]
        if tool_call["name"] == "get_web_search":
            query = tool_call.get("args", {}).get("query", "")
            st.info(f"인터넷 검색 중입니다: {query}")
        elif tool_call["name"] == "get_current_time":
            st.info("현재 시간을 확인 중입니다...")

        tool_msg = selected_tool.invoke(tool_call)
        st.session_state.messages.append(tool_msg)

        if tool_call["name"] == "get_web_search":
            st.success("인터넷 검색 결과를 확인했습니다. 답변을 정리하는 중입니다.")

    for chunk in get_ai_response(st.session_state.messages):
        yield chunk


st.title("인터넷 검색 챗봇")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(
            content=(
                "너는 인터넷 검색 도구를 사용해서 최신 정보를 확인한 뒤 한국어로 답변하는 "
                "AI 검색 어시스턴트다. 최신 이슈나 반응을 묻는 질문에는 반드시 검색 도구를 사용해라."
            )
        ),
        AIMessage(content="궁금한 최신 이슈를 물어보세요."),
    ]

for msg in st.session_state.messages:
    if not msg.content:
        continue

    if isinstance(msg, SystemMessage):
        st.chat_message("system").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.chat_message("tool"):
            st.write(msg.content)

if prompt := st.chat_input("질문을 입력하세요"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.spinner("인터넷 검색 여부를 판단하고 있습니다..."):
        response = get_ai_response(st.session_state.messages)
        result = st.chat_message("assistant").write_stream(response)

    st.session_state.messages.append(AIMessage(content=result))
