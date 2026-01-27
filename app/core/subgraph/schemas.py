from pydantic import BaseModel, Field
from typing import Literal, Optional, List


# --- Planner 输出结构 ---
class PlanItem(BaseModel):
    description: str = Field(..., description="任务的具体描述，无需包含 ID")


class PlannerOutput(BaseModel):
    """Planner 节点的输出结构"""
    tasks: List[PlanItem] = Field(..., description="初始生成的任务列表")


# --- Reflector 输出结构 (扁平化超集) ---
class ReflectorOutput(BaseModel):
    """Reflector 节点的决策输出结构"""

    # 核心决策
    action: Literal["finish_task", "retry_task", "insert_task"] = Field(
        ...,
        description="根据执行结果决定的下一步动作"
    )

    # === 场景 A: finish_task ===
    new_scratchpad: Optional[str] = Field(
        None,
        description="[Action=finish_task] 任务完成后的关键发现摘要，用于更新上下文"
    )

    # === 场景 B: retry_task ===
    retry_reason: Optional[str] = Field(
        None,
        description="[Action=retry_task] 判定失败的原因"
    )
    retry_suggestion: Optional[str] = Field(
        None,
        description="[Action=retry_task] 给 Executor 的代码修改建议"
    )

    # === 场景 C: insert_task ===
    # 注意：不需要 target_id，系统默认插入在当前任务之后
    new_task_description: Optional[str] = Field(
        None,
        description="[Action=insert_task] 新插入任务的描述"
    )
    insert_reason: Optional[str] = Field(
        None,
        description="[Action=insert_task] 为什么要插入新任务"
    )