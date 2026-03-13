from typing import TypedDict, List, Optional, Any, Dict


class WorkflowState(TypedDict):
    """
    主工作流状态对象（无 messages）。
    这个字典会在图中的所有节点之间传递。
    """

    # 原始用户输入
    user_input: str

    # 原始碎片文件在沙盒中的物理路径列表 (Profiler 的输入)
    # 例如: ["/home/user/raw_0.csv", "/home/user/raw_1.csv"]
    raw_file_paths: List[str]

    # 原始文件名列表 (给 LLM 参考，用于判断合并逻辑)
    # 例如: ["销售表.xlsx (Sheet: 1月)", "销售表.xlsx (Sheet: 2月)"]
    # 列表顺序必须与 raw_file_paths 一一对应
    original_filenames: List[str]

    # 本地临时文件路径列表 (用于 Profiler 本地元信息采集)
    # 例如: ["temp/raw_session123_0.csv", "temp/raw_session123_1.csv"]
    # 列表顺序必须与 raw_file_paths 一一对应
    local_file_paths: List[str]

    # 文件元信息列表 (Profiler 输出)
    # 单文件时长度为 1，多文件时包含所有文件的元信息
    # 格式: [{"file_index": 0, "original_filename": "...", "columns": [...], ...}]
    files_metadata: List[Dict[str, Any]]

    # 合并建议列表 (Profiler 输出，仅多文件时存在)
    # 格式: [{"strategy": "concat", "involved_files": [0,1], "confidence": "high", ...}]
    merge_recommendations: Optional[List[Dict[str, Any]]]

    # [已废弃] 结构化字典，包含 'columns', 'dtypes', 'summary'
    # 已被 files_metadata 替代，保留以向后兼容
    data_schema: Dict[str, Any]

    # [已废弃] PPIO 沙盒中的最终合并文件路径
    # 已被 files_metadata 中的 remote_path 替代，保留以向后兼容
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