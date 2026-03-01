from pydantic import BaseModel, Field
from typing import Optional, Literal


class Task(BaseModel):
    """
    通用建模流程中的单个任务单元。
    """
    id: int = Field(..., description="任务的唯一标识符，从 1 开始")
    description: str = Field(..., description="任务的具体描述，例如'清洗缺失值'或'按日期分组求和'")

    # 状态机: pending(待处理) -> running(执行中) -> completed(完成) | failed(失败)
    status: Literal["pending", "running", "completed", "failed"] = "pending"

    # 执行细节
    code: Optional[str] = Field(None, description="该任务生成的 Python 代码")
    result: Optional[str] = Field(None, description="代码执行后的标准输出或摘要")
    error: Optional[str] = Field(None, description="如果执行失败，记录错误信息")

    def mark_running(self):
        self.status = "running"

    def mark_completed(self, result: str):
        self.status = "completed"
        self.result = result

    def mark_failed(self, error: str):
        self.status = "failed"
        self.error = error