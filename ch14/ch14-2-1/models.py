from pydantic import BaseModel, Field
from typing import Literal

# Task 클래스는 agent, done, description, done_at의 4개 필드가 있습니다. 
# 이전 실습에서 supervisor_chain을 StrOutputParser로 지정했을 때는 원하는 문자만 출력하도록 강제하지 못했습니다. 
# 우리가 원했던 결과는 content_strategist 또는 communicator 중에서 하나의 단어만 출력하는 것입니다. 
# 이런 경우 agent 필드처럼 Literal로 지정하고 원하는 문구를 써주면 됩니다. 그리고 이에 대한 설명을 description 에 작성합니다.
class Task(BaseModel):
    agent: Literal[
        "content_strategist",
        "communicator",
    ] = Field(
        ...,
        description="""
        작업을 수행하는 agent의 종류.
        - content_strategist: 콘텐츠 전략을 수립하는 작업을 수행한다. 사용자의 요구사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다. 
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행상황을 사용자에게 보고하고, 다음 지시를 물어본다.
        """
    )
	
    # done은 이 일을 끝냈는지 여부를 bool로 기록하는 필드입니다. 
    # 그리고 description 필드는 Task가 어떤 종류의 일인지 설명합니다. 
    # description 필드는 뒤에서 보겠지만 다음 노드, 즉 에이전트가 어떤 방식으로 일을 해야 할지 기록해 두므로 전체 흐름에서 중요한 역할을 합니다.
    done: bool = Field(..., description="종료 여부")
    description: str = Field(..., description="어떤 작업을 해야 하는지에 대한 설명")
	
    # done_at은 작업이 종료된 시각을 저장할 문자열 필드입니다.
    done_at: str = Field(..., description="할 일이 완료된 날짜와 시간")
	
    # to_dict는 이 Task를 파일에 저장할 때 사용하기 위해 작성한 함수입니다.
    def to_dict(self):
        return {
            "agent": self.agent,
            "done": self.done,
            "description": self.description,
            "done_at": self.done_at
        }  