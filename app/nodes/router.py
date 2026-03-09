from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAI
from openai import APIError
from pydantic import BaseModel, Field
import json
import re

from app.core.state import WorkflowState
from app.core.config import settings
from app.prompts.router_prompt import ROUTER_SYSTEM_TEMPLATE, ROUTER_CONTEXT_TEMPLATE

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

# 2. 初始化 LLM
llm = ChatOpenAI(
    model=settings.LLM_FLASH_MODEL_NAME,
    temperature=0.1,
    api_key=settings.OPENAI_API_KEY_FLASH,
    use_responses_api=settings.USE_RESPONSES_API_FLASH,
    base_url=settings.OPENAI_API_BASE_FLASH,
    max_retries=5,
    timeout=120,
)

# 添加智能重试机制
llm = llm.with_retry(
    stop_after_attempt=5,
    retry_if_exception_type=(APIError,),
    wait_exponential_jitter=True,
)

# 3. Router 节点函数
def router_node(state: WorkflowState) -> dict:
    """
    路由节点：分析用户输入和数据Schema，决定算法场景。
    """
    print("--- [Router] 分析用户的意图中... ---")
    data_schema = state.get("data_schema", "")
    user_input = state.get("user_input", "")

    # 1. 构建静态 System Message
    system_message = SystemMessage(content=ROUTER_SYSTEM_TEMPLATE)

    # 2. HumanMessage 包含动态上下文（包括用户输入）
    context_content = ROUTER_CONTEXT_TEMPLATE.format(
        data_schema=data_schema,
        user_input=user_input
    )
    context_message = HumanMessage(content=context_content)

    # 3. 组装 Messages：静态规则 + 动态上下文
    messages = [system_message, context_message]

    # 调用 LLM 获取 JSON 结果
    try:
        response = llm.invoke(messages)

        # 手动解析 JSON 输出
        content = response.content

        # 尝试提取 JSON（支持 Markdown 代码块）
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            json_str = match.group(1).strip()
        else:
            # 如果没有代码块，尝试直接解析
            json_str = content.strip()

        # 解析 JSON
        parsed = json.loads(json_str)

        # 验证并创建 RouterOutput 对象
        result = RouterOutput(**parsed)

        print(f"算法场景: {result.scenario} | 理由: {result.reasoning}")

        return {
            "scenario": result.scenario,
            "reasoning": result.reasoning
        }

    except json.JSONDecodeError as e:
        print(f"路由失败 - JSON 解析错误: {e}")
        print(f"LLM 输出: {response.content}")
        return {"scenario": "unknown", "reasoning": f"路由失败，JSON 解析错误: {e}"}
    except Exception as e:
        print(f"路由失败: {e}")
        return {"scenario": "unknown", "reasoning": f"路由失败，{e}"}