from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any


class VizTaskSimple(BaseModel):
    """Planner 输出的简化任务结构（由代码组装成完整的 VizTask）"""
    chart_type: Literal["scatter", "line", "bar", "pie", "radar", "heatmap", "boxplot", "table"] = Field(
        ..., description="图表类型"
    )
    title: str = Field(..., description="图表标题")
    description: str = Field(..., description="任务描述，说明需要生成什么数据")
    source_files: List[str] = Field(..., description="源文件名列表（不含路径）")


class VizPlannerOutput(BaseModel):
    """Planner 节点的输出结构"""
    tasks: List[VizTaskSimple] = Field(..., description="viz 任务列表")


class VizTask(BaseModel):
    """完整的可视化任务（由代码组装，包含文件元信息）"""
    task_id: str = Field(..., description="任务唯一标识符，如 'viz_task_1'")
    chart_type: Literal["scatter", "line", "bar", "pie", "radar", "heatmap", "boxplot", "table"] = Field(
        ..., description="图表类型"
    )
    title: str = Field(..., description="图表标题")
    description: str = Field(..., description="任务描述，说明需要生成什么数据")
    source_files: List[str] = Field(..., description="源文件名列表")

    # 透传的文件元信息（从 file_metadata 中提取）
    file_columns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="文件的列信息，格式: [{'file_name': '...', 'columns': [...]}]"
    )
