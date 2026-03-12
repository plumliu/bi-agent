from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any


class DataRequirement(BaseModel):
    """单个文件的数据需求"""
    file_name: str = Field(..., description="文件名")
    required_columns: Optional[List[str]] = Field(
        None,
        description="对于 feather 文件：需要的列名列表；对于 JSON 文件：None 表示使用全部数据"
    )


class VizTaskSimple(BaseModel):
    """Planner 输出的简化任务结构（由代码组装成完整的 VizTask）"""
    chart_type: str = Field(
        ..., description="图表类型"
    )
    title: str = Field(..., description="图表标题")
    description: str = Field(..., description="任务描述，说明需要生成什么数据")
    data_requirements: List[DataRequirement] = Field(
        ...,
        description="数据需求列表，指定每个文件需要哪些列（feather）或全部数据（JSON）"
    )


class VizPlannerOutput(BaseModel):
    """Planner 节点的输出结构"""
    tasks: List[VizTaskSimple] = Field(..., description="viz 任务列表")


class VizTask(BaseModel):
    """完整的可视化任务（由代码组装）"""
    task_id: str = Field(..., description="任务唯一标识符，如 'viz_task_1'")
    chart_type: str = Field(..., description="图表类型")
    title: str = Field(..., description="图表标题")
    description: str = Field(..., description="任务描述，说明需要生成什么数据")
    data_requirements: List[DataRequirement] = Field(..., description="数据需求列表")
