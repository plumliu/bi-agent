from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.state import AgentState
from app.core.config import settings  # 假设你会在 config 中存放 API_KEY 和 Model Name
from app.prompts.router_prompt import ROUTER_SYSTEM_TEMPLATE

# 1. 定义路由的输出结构
# 使用 Literal 严格限制 LLM 只能从这几个选项里选，防止幻觉
ScenarioType = Literal[
    "clustering",  # 聚类
    "anomaly",  # 异常分析
    "decomposition",  # 趋势分解

    "association",  # 关联分析
    "forecast",  # 时序预测
    "classification",  # 分类

    "custom", # 复杂分析或者前几项标准分析之外的情景

    "unknown"  # 如果用户在闲聊或无法识别
]


class RouterOutput(BaseModel):
    """路由器的决策结果结构"""
    scenario: ScenarioType = Field(
        ...,
        description="最适合用户需求的算法场景分类"
    )
    reasoning: str = Field(
        ...,
        description="选择该场景的理由，简短的一句话即可，用于调试或告知用户"
    )


# 2. 初始化 LLM
llm = ChatOpenAI(
    model=settings.LLM_FLASH_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

# 绑定结构化输出，强制 LLM 返回 JSON 格式
structured_llm = llm.with_structured_output(RouterOutput)


# 3. Router 节点函数
def router_node(state: AgentState) -> dict:
    """
    路由节点：分析用户输入和数据Schema，决定算法场景。
    """
    print("--- [Router] 分析用户的意图中... ---")

    user_input = state["user_input"]
    data_schema = state.get("data_schema")

    system_prompt = ROUTER_SYSTEM_TEMPLATE.format(data_schema=data_schema)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    # 调用 LLM 获取结构化结果
    try:
        result: RouterOutput = structured_llm.invoke(messages)

        # 打印日志方便观察
        print(f"算法场景: {result.scenario} | 理由: {result.reasoning}")

        return {
            "scenario": result.scenario,
            "messages": [SystemMessage(content=result.reasoning)],
            "modeling_insight": f"{result.reasoning}"
        }

    except Exception as e:
        print(f"路由失败: {e}")
        return {"scenario": "unknown", "modeling_insight": f"路由失败，{e}"}