from typing import Literal
from langchain_core.messages import SystemMessage, AIMessage # 引入 AIMessage
from langchain_openai import ChatOpenAI, OpenAI
from pydantic import BaseModel, Field

from app.core.state import AgentState
from app.core.config import settings
from app.prompts.router_prompt import ROUTER_SYSTEM_TEMPLATE

# 1. 定义路由的输出结构 (保持不变)
ScenarioType = Literal[
    "clustering", "anomaly", "decomposition", "association", "forecast", "classification", # sop 场景
    "custom",  # custom 场景
    "unknown" # 拒绝
]

class RouterOutput(BaseModel):
    """路由器的决策结果结构"""
    scenario: ScenarioType = Field(..., description="最适合用户需求的算法场景分类")
    reasoning: str = Field(..., description="选择该场景的理由，简短的一句话即可，用于调试或告知用户")

# 2. 初始化 LLM (保持不变)
llm = ChatOpenAI(
    model=settings.LLM_FLASH_MODEL_NAME,
    temperature=0.1,
    api_key=settings.OPENAI_API_KEY_FLASH,
    use_responses_api=settings.USE_RESPONSES_API_FLASH,
    base_url=settings.OPENAI_API_BASE_FLASH,
)
structured_llm = llm.with_structured_output(RouterOutput)

# 3. Router 节点函数
def router_node(state: AgentState) -> dict:
    """
    路由节点：分析用户输入和数据Schema，决定算法场景。
    """
    print("--- [Router] 分析用户的意图中... ---")
    data_schema = state.get("data_schema", "")

    # 1. 构建纯净的系统规则 (完美命中 Cache)
    system_prompt = ROUTER_SYSTEM_TEMPLATE.format(data_schema=data_schema)

    # 2. 组装 Messages：系统规则 + 之前积攒的所有对话历史 (包含用户的真实提问)
    messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])

    # 调用 LLM 获取结构化结果
    try:
        result: RouterOutput = structured_llm.invoke(messages)

        print(f"算法场景: {result.scenario} | 理由: {result.reasoning}")

        return {
            "scenario": result.scenario,
            "messages": [AIMessage(content=f"[路由决策] Router 节点将本任务路由到了 {result.scenario} 场景")],
            "reasoning": result.reasoning
        }

    except Exception as e:
        print(f"路由失败: {e}")
        return {"scenario": "unknown", "reasoning": f"路由失败，{e}"}