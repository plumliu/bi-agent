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

    # 原始碎片文件在沙盒中的物理路径列表 (Auto-ETL 的输入)
    # 例如: ["/home/user/raw_0.csv", "/home/user/raw_1.csv"]
    raw_file_paths: List[str]

    # 原始文件名列表 (给 LLM 参考，用于判断合并逻辑)
    # 例如: ["销售表.xlsx (Sheet: 1月)", "销售表.xlsx (Sheet: 2月)"]
    # 列表顺序必须与 raw_file_paths 一一对应
    original_filenames: List[str]

    # 结构化字典，包含 'columns', 'dtypes', 'summary'
    # 注意：初始化时为空字典 {}，由 Auto-ETL 节点运行后填充
    data_schema: Dict[str, Any]

    # PPIO 沙盒中的最终合并文件路径
    # 改为 Optional。因为图启动时该文件尚未生成。
    # Auto-ETL 成功后，会被设置为 "/home/user/data.csv"
    remote_file_path: Optional[str]

    # --- 路由结果 (Router Node) ---
    scenario: Optional[str]
    reasoning: Optional[str]

    # --- 建模产物 (Modeling Artifacts) ---
    # 例如: {"k_value": 3, "centroids": {...}, "silhouette_score": 0.5}
    modeling_artifacts: Optional[Dict[str, Any]]

    # 建模工作摘要
    modeling_summary: Optional[str]

    # 通用场景下的建模产出的所有 feather 文件的文件名
    generated_data_files: Optional[List[str]]

    # 建模产物的结构化元数据（包含文件名、列名、列描述）
    # 格式: [{"file_name": "...", "columns": [{"name": "...", "description": "..."}]}]
    file_metadata: Optional[List[Dict[str, Any]]]

    # --- 可视化配置 (Viz Node) ---
    # 这是一个 JSON 对象，对应 ECharts 配置
    viz_config: Optional[Dict[str, Any]]

    # Viz 执行状态标志 (用于控制 Viz 循环)
    viz_success: Optional[bool]

    # viz_custom 子图生成的可视化数据（格式：{"echarts": [...]}}）
    viz_data: Optional[Dict[str, Any]]

    # --- 总结 (Summary Node) ---
    final_summary: Optional[str]

    # --- 控制流标志 ---
    error_count: int