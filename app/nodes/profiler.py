import json
import logging
from typing import Dict, Any, List, Union

from langchain_core.messages import SystemMessage, HumanMessage

from app.core.state import WorkflowState
from ppio_sandbox.code_interpreter import Sandbox
from app.prompts.profiler_prompt import (
    PROFILER_RECOMMENDATION_SYSTEM_TEMPLATE,
    PROFILER_RECOMMENDATION_CONTEXT_TEMPLATE
)
from app.utils.csv_reader import collect_file_metadata
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import create_llm, apply_retry

step = "profiler"
logger = logging.getLogger(__name__)

llm = apply_retry(create_llm(use_flash=False))


def _parse_stdout(stdout_content: Union[str, List[str], None]) -> str:
    """兼容处理沙盒返回的 stdout"""
    if stdout_content is None:
        return ""
    if isinstance(stdout_content, list):
        return "".join(stdout_content)
    return str(stdout_content)


def _generate_merge_recommendations(
    sandbox: Sandbox,
    files_metadata: List[Dict[str, Any]],
    original_names: List[str]
) -> List[Dict[str, Any]]:
    """生成并验证合并建议（仅多文件时调用）"""
    print("--- [Profiler] 请求 LLM 生成合并建议... ---")

    # 调用 LLM 生成建议
    system_message = SystemMessage(content=PROFILER_RECOMMENDATION_SYSTEM_TEMPLATE)
    context_content = PROFILER_RECOMMENDATION_CONTEXT_TEMPLATE.format(
        files_metadata_json=json.dumps(files_metadata, ensure_ascii=False, indent=2)
    )
    context_message = HumanMessage(content=context_content)

    print(f"[Profiler] 发送给 LLM 的上下文长度: {len(context_content)} 字符")
    print(f"[Profiler] files_metadata 包含 {len(files_metadata)} 个文件")

    response = llm.invoke([system_message, context_message])

    print(f"[Profiler] LLM 原始响应（前1000字符）:\n{response.content[:1000]}")
    print(f"[Profiler] LLM 响应总长度: {len(response.content)} 字符")

    # 解析 JSON 响应
    try:
        response_text = extract_text_from_content(response.content).strip()
        # 移除可能的 markdown 代码块标记
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        recommendations_data = json.loads(response_text)
        raw_recommendations = recommendations_data.get("recommendations", [])
        print(f"[Profiler] LLM 生成了 {len(raw_recommendations)} 个初步建议")
    except Exception as e:
        logger.error(f"解析 LLM 建议失败: {e}, 响应内容: {response.content}")
        print(f"[Profiler] JSON 解析失败: {e}")
        print(f"[Profiler] 尝试解析的文本（前2000字符）:\n{response_text[:2000] if 'response_text' in locals() else response.content[:2000]}")
        return []

    # 验证每个建议
    validated_recommendations = []
    for idx, rec in enumerate(raw_recommendations):
        print(f"\n[Profiler] 验证建议 {idx + 1}/{len(raw_recommendations)}: {rec.get('recommendation_id', 'unknown')}")
        print(f"  策略: {rec.get('strategy')}, 涉及文件: {rec.get('involved_files')}")
        validated_rec = _validate_recommendation(sandbox, rec, files_metadata)
        print(f"  验证结果: {'通过' if validated_rec.get('validation_passed') else '未通过'}")
        if not validated_rec.get('validation_passed'):
            print(f"  警告: {validated_rec.get('validation_warnings', [])}")
        if validated_rec.get("validation_passed"):
            validated_recommendations.append(validated_rec)

    return validated_recommendations


def _validate_recommendation(
    sandbox: Sandbox,
    recommendation: Dict[str, Any],
    files_metadata: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """验证单个合并建议"""
    strategy = recommendation.get("strategy")

    if strategy == "concat":
        return _validate_concat(sandbox, recommendation, files_metadata)
    elif strategy == "merge":
        return _validate_merge(sandbox, recommendation, files_metadata)
    else:
        # reject 或其他策略不需要验证
        recommendation["validation_passed"] = False
        return recommendation


def _validate_concat(
    sandbox: Sandbox,
    recommendation: Dict[str, Any],
    files_metadata: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """验证 concat 建议：检查列名重合度"""
    involved_files = recommendation.get("involved_files", [])

    if len(involved_files) != 2:
        recommendation["validation_passed"] = False
        recommendation["validation_warnings"] = ["concat 策略仅支持两个文件"]
        return recommendation

    file1_meta = files_metadata[involved_files[0]]
    file2_meta = files_metadata[involved_files[1]]

    cols1 = set(file1_meta.get("columns", []))
    cols2 = set(file2_meta.get("columns", []))

    overlap = cols1 & cols2
    union = cols1 | cols2

    overlap_ratio = len(overlap) / len(union) if union else 0

    recommendation["column_overlap_ratio"] = overlap_ratio
    recommendation["validation_passed"] = overlap_ratio >= 0.9
    recommendation["validation_warnings"] = [] if overlap_ratio >= 0.9 else [
        f"列名重合度仅 {overlap_ratio:.2%}，低于 90% 阈值"
    ]

    print(
        f"[Profiler] concat 列名重合度: {overlap_ratio:.2%} | "
        f"pass={recommendation['validation_passed']}"
    )

    return recommendation


def _validate_merge(
    sandbox: Sandbox,
    recommendation: Dict[str, Any],
    files_metadata: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """验证 merge 建议：三阶段策略"""
    left_idx = recommendation.get("left_file")
    right_idx = recommendation.get("right_file")
    left_on = recommendation.get("left_on")
    right_on = recommendation.get("right_on")

    if left_idx is None or right_idx is None or not left_on or not right_on:
        recommendation["validation_passed"] = False
        recommendation["validation_warnings"] = ["merge 策略缺少必要字段 (left_on/right_on)"]
        return recommendation

    left_meta = files_metadata[left_idx]
    right_meta = files_metadata[right_idx]
    left_path = left_meta["remote_path"]
    right_path = right_meta["remote_path"]

    # 生成验证代码
    validation_code = f"""
import pandas as pd
import json

left_df = pd.read_csv('{left_path}')
right_df = pd.read_csv('{right_path}')

left_on = {json.dumps(left_on)}
right_on = {json.dumps(right_on)}
if isinstance(left_on, str):
    left_on = [left_on]
if isinstance(right_on, str):
    right_on = [right_on]

# 阶段 1: 静态键质量检查
left_non_null = left_df[left_on].notna().all(axis=1).sum()
right_non_null = right_df[right_on].notna().all(axis=1).sum()

left_unique = left_df[left_on].drop_duplicates().shape[0]
right_unique = right_df[right_on].drop_duplicates().shape[0]

left_uniqueness = left_unique / left_non_null if left_non_null > 0 else 0
right_uniqueness = right_unique / right_non_null if right_non_null > 0 else 0

# 值域交集
left_keys = set(map(tuple, left_df[left_on].dropna().values))
right_keys = set(map(tuple, right_df[right_on].dropna().values))
intersection = left_keys & right_keys
intersection_size = len(intersection)

left_coverage = intersection_size / len(left_keys) if left_keys else 0
right_coverage = intersection_size / len(right_keys) if right_keys else 0

# 阶段 2: 试验性合并
merged_df = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how='left', indicator=True)
row_multiplier = len(merged_df) / len(left_df) if len(left_df) > 0 else 0
matched_ratio = ((merged_df['_merge'] == 'both').sum() / len(left_df)) if len(left_df) > 0 else 0

result = {{
    "left_uniqueness": left_uniqueness,
    "right_uniqueness": right_uniqueness,
    "intersection_size": intersection_size,
    "left_coverage": left_coverage,
    "right_coverage": right_coverage,
    "row_multiplier": row_multiplier,
    "matched_ratio": matched_ratio
}}

print(json.dumps(result))
"""

    exec_res = sandbox.run_code(validation_code)

    if exec_res.error or exec_res.logs.stderr:
        recommendation["validation_passed"] = False
        recommendation["validation_warnings"] = [f"验证代码执行失败: {exec_res.error or exec_res.logs.stderr}"]
        return recommendation

    try:
        stdout_str = _parse_stdout(exec_res.logs.stdout)
        metrics = json.loads(stdout_str)
    except Exception as e:
        recommendation["validation_passed"] = False
        recommendation["validation_warnings"] = [f"验证结果解析失败: {e}"]
        return recommendation

    recommendation["validation_metrics"] = metrics
    print(f"[Profiler] merge 验证指标: {metrics}")

    # 阶段 3: 硬性过滤决策
    warnings = []
    passed = True

    # ① 至少一侧唯一性 ≥ 0.95
    if metrics["left_uniqueness"] < 0.95 and metrics["right_uniqueness"] < 0.95:
        passed = False
        warnings.append("两侧唯一性均低于 0.95")

    # ② 值域存在实质性重合
    if metrics["left_coverage"] < 0.7 or metrics["right_coverage"] < 0.7:
        passed = False
        warnings.append(f"值域覆盖率不足: left={metrics['left_coverage']:.2%}, right={metrics['right_coverage']:.2%}")

    # ③ row_multiplier ≤ 1.05
    if metrics["row_multiplier"] > 1.05:
        passed = False
        warnings.append(f"row_multiplier={metrics['row_multiplier']:.2f} 超过 1.05，可能存在 many-to-many 关系")

    # ④ matched_ratio ≥ 0.6
    if metrics["matched_ratio"] < 0.6:
        passed = False
        warnings.append(f"matched_ratio={metrics['matched_ratio']:.2%} 低于 60%")

    recommendation["validation_passed"] = passed
    recommendation["validation_warnings"] = warnings

    return recommendation


def profiler_node(state: WorkflowState, sandbox: Sandbox) -> Dict[str, Any]:
    """Profiler 节点：元信息收集器 + 合并建议生成器"""
    print("--- [Profiler] 收集文件元信息... ---")

    raw_paths = state.get("raw_file_paths", [])
    orig_names = state.get("original_filenames", [])
    local_paths = state.get("local_file_paths", [])

    # 1. 本地收集所有文件的元信息
    files_metadata = []
    for idx, (remote_path, local_path, name) in enumerate(zip(raw_paths, local_paths, orig_names)):
        metadata = collect_file_metadata(
            file_path=local_path,
            original_filename=name,
            file_index=idx,
            remote_path=remote_path
        )

        if "error" in metadata:
            raise RuntimeError(f"文件元信息收集失败: {metadata['error']}")

        files_metadata.append(metadata)

    print(f"--- [Profiler] 已收集 {len(files_metadata)} 个文件的元信息 ---")

    # 2. 多文件：生成并验证合并建议
    merge_recommendations = None
    if len(raw_paths) > 1:
        merge_recommendations = _generate_merge_recommendations(
            sandbox, files_metadata, orig_names
        )
        print(f"--- [Profiler] 生成了 {len(merge_recommendations)} 个验证通过的合并建议 ---")

    # 3. 返回状态更新
    return {
        "files_metadata": files_metadata,
        "merge_recommendations": merge_recommendations
    }
