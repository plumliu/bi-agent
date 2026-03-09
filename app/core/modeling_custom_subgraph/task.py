from pydantic import BaseModel, Field
from typing import Optional, Literal


class Task(BaseModel):
    """
    通用建模流程中的单个任务单元。
    """
    id: int = Field(..., description="任务的唯一标识符，从 1 开始")
    description: str = Field(..., description="任务的具体描述，例如'清洗缺失值'或'按日期分组求和'")
