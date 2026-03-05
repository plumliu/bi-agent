from pydantic import BaseModel, Field
from typing import Literal, Optional, List


# --- Planner 输出结构 ---
class PlanItem(BaseModel):
    description: str = Field(..., description="任务的具体描述，无需包含 ID")


class PlannerOutput(BaseModel):
    """Planner 节点的输出结构"""
    tasks: List[PlanItem] = Field(..., description="初始生成的任务列表")
