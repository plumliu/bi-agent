import operator
from typing import TypedDict, Annotated, List, Optional, Any, Dict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    LangGraph 的状态对象。
    这个字典会在图中的所有节点(Agent)之间传递。
    """

    # --- 基础对话上下文 ---
    # messages: 保存对话历史 (追加模式)
    messages: Annotated[List[BaseMessage], operator.add]

    # 原始用户输入
    user_input: str

    # --- 数据上下文 (Data Context) ---
    # [修改] 升级为结构化字典，包含 'columns', 'dtypes', 'summary'
    # 这样 Fetch_Artifacts 节点就可以只更新里面的 'columns' 列表，而不破坏其他信息
    data_schema: Dict[str, Any]

    # E2B 沙箱中的文件路径 (例如 "/home/user/data.csv")
    remote_file_path: str

    # --- 阶段 0: 路由结果 ---
    scenario: Optional[str]
    modeling_insight: Optional[str]

    # --- 建模产物 (Modeling Artifacts) ---
    # 例如: {"k_value": 3, "centroids": {...}, "silhouette_score": 0.5}
    modeling_artifacts: Optional[Dict[str, Any]]

    # 建模工作摘要
    modeling_summary: Optional[str]

    # --- 可视化配置 (Viz Node) ---
    # 这是一个 JSON 对象，对应 ECharts 配置
    viz_config: Optional[Dict[str, Any]]

    # Viz 执行状态标志 (用于控制 Viz 循环)
    viz_success: Optional[bool]

    # --- 总结 (Summary Node) ---
    final_summary: Optional[str]

    # --- 控制流标志 ---
    error_count: int