import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.state import WorkflowState
from app.prompts.profiler_prompt import (
    PROFILER_RECOMMENDATION_CONTEXT_TEMPLATE,
    PROFILER_RECOMMENDATION_SYSTEM_TEMPLATE,
)
from app.utils.csv_reader import collect_file_metadata
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import apply_retry, create_llm
from app.utils.terminal_logger import print_block, print_kv, print_list, print_subheader, preview_text

logger = logging.getLogger(__name__)
llm = apply_retry(create_llm(use_flash=False))


def _generate_merge_recommendations(
    files_metadata: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate and validate merge recommendations when multiple files are uploaded."""
    print_subheader("Profiler / Merge Recommendation")
    print_kv("input_files", len(files_metadata))

    system_message = SystemMessage(content=PROFILER_RECOMMENDATION_SYSTEM_TEMPLATE)
    context_content = PROFILER_RECOMMENDATION_CONTEXT_TEMPLATE.format(
        files_metadata_json=json.dumps(files_metadata, ensure_ascii=False, indent=2)
    )
    context_message = HumanMessage(content=context_content)

    response = llm.invoke([system_message, context_message])

    try:
        response_text = extract_text_from_content(response.content).strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        recommendations_data = json.loads(response_text)
        raw_recommendations = recommendations_data.get("recommendations", [])
        print_kv("raw_recommendations", len(raw_recommendations))
    except Exception as e:
        logger.error("解析 LLM 合并建议失败: %s", e)
        print_kv("parse_error", str(e))
        print_kv("response_preview", preview_text(response.content, max_chars=280))
        return []

    validated_recommendations: List[Dict[str, Any]] = []
    for rec in raw_recommendations:
        validated = _validate_recommendation(rec, files_metadata)
        if validated.get("validation_passed"):
            validated_recommendations.append(validated)

    print_kv("validated_recommendations", len(validated_recommendations))
    if validated_recommendations:
        print_list(
            "validated_items",
            [
                f"{r.get('recommendation_id', 'n/a')} | {r.get('strategy', 'n/a')}"
                for r in validated_recommendations
            ],
            max_items=6,
        )

    return validated_recommendations


def _validate_recommendation(
    recommendation: Dict[str, Any],
    files_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    strategy = recommendation.get("strategy")

    if strategy == "concat":
        return _validate_concat(recommendation, files_metadata)
    if strategy == "merge":
        return _validate_merge(recommendation, files_metadata)

    recommendation["validation_passed"] = False
    recommendation["validation_warnings"] = ["unsupported strategy"]
    return recommendation


def _validate_concat(
    recommendation: Dict[str, Any],
    files_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    involved_files = recommendation.get("involved_files", [])

    if len(involved_files) != 2:
        recommendation["validation_passed"] = False
        recommendation["validation_warnings"] = ["concat strategy requires exactly two files"]
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
        f"column overlap ratio {overlap_ratio:.2%} is below 90%",
    ]
    return recommendation


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _print_file_eda(metadata: Dict[str, Any]) -> None:
    print_kv("rows", metadata.get("row_count"))
    print_kv("columns", metadata.get("column_count", len(metadata.get("columns", []))))
    print_kv(
        "missing_cells",
        f"{metadata.get('missing_cell_count', 0)} ({_pct(metadata.get('missing_cell_ratio', 0.0))})",
    )
    print_kv(
        "duplicate_rows",
        f"{metadata.get('duplicate_row_count', 0)} ({_pct(metadata.get('duplicate_row_ratio', 0.0))})",
    )
    print_list("column_names", metadata.get("columns", []), max_items=10)

    column_profiles = metadata.get("column_profiles", [])
    high_missing = [
        p for p in column_profiles if float(p.get("missing_ratio", 0.0)) > 0
    ]
    high_missing.sort(key=lambda p: float(p.get("missing_ratio", 0.0)), reverse=True)
    if high_missing:
        print_list(
            "missing_columns(top)",
            [
                (
                    f"{p.get('name')}: missing={p.get('missing_count')} "
                    f"({_pct(p.get('missing_ratio', 0.0))}), unique={p.get('unique_count')}"
                )
                for p in high_missing
            ],
            max_items=8,
        )
    else:
        print_kv("missing_columns", "none")

    low_card_cols = metadata.get("low_cardinality_columns", {})
    if low_card_cols:
        lines: List[str] = []
        for col, detail in low_card_cols.items():
            top_values = detail.get("top_values", [])
            top_preview = ", ".join(
                [f"{item.get('value')}({item.get('count')})" for item in top_values[:5]]
            )
            lines.append(
                (
                    f"{col}: unique={detail.get('unique_count')}, "
                    f"missing={detail.get('missing_count')} ({_pct(detail.get('missing_ratio', 0.0))}), "
                    f"top={top_preview}"
                )
            )
        print_list("low_cardinality_columns", lines, max_items=8)
    else:
        print_kv("low_cardinality_columns", "none")

    numeric_summary = metadata.get("numeric_columns_summary", {})
    if numeric_summary:
        print_list(
            "numeric_summary(top)",
            [
                (
                    f"{col}: mean={stats.get('mean')}, p50={stats.get('p50')}, "
                    f"min={stats.get('min')}, max={stats.get('max')}"
                )
                for col, stats in numeric_summary.items()
            ],
            max_items=8,
        )
    else:
        print_kv("numeric_summary", "none")


def _validate_merge(
    recommendation: Dict[str, Any],
    files_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    left_idx = recommendation.get("left_file")
    right_idx = recommendation.get("right_file")
    left_on = recommendation.get("left_on")
    right_on = recommendation.get("right_on")

    if left_idx is None or right_idx is None or not left_on or not right_on:
        recommendation["validation_passed"] = False
        recommendation["validation_warnings"] = ["merge strategy missing left/right join keys"]
        return recommendation

    left_meta = files_metadata[left_idx]
    right_meta = files_metadata[right_idx]
    left_path = left_meta.get("data_path")
    right_path = right_meta.get("data_path")

    try:
        left_df = pd.read_csv(left_path)
        right_df = pd.read_csv(right_path)
    except Exception as e:
        recommendation["validation_passed"] = False
        recommendation["validation_warnings"] = [f"failed to read csv for validation: {e}"]
        return recommendation

    left_keys = _as_list(left_on)
    right_keys = _as_list(right_on)

    if len(left_keys) != len(right_keys):
        recommendation["validation_passed"] = False
        recommendation["validation_warnings"] = ["left_on and right_on must have equal key counts"]
        return recommendation

    try:
        left_non_null = left_df[left_keys].notna().all(axis=1).sum()
        right_non_null = right_df[right_keys].notna().all(axis=1).sum()

        left_unique = left_df[left_keys].drop_duplicates().shape[0]
        right_unique = right_df[right_keys].drop_duplicates().shape[0]

        left_uniqueness = left_unique / left_non_null if left_non_null > 0 else 0
        right_uniqueness = right_unique / right_non_null if right_non_null > 0 else 0

        left_key_set = set(map(tuple, left_df[left_keys].dropna().values.tolist()))
        right_key_set = set(map(tuple, right_df[right_keys].dropna().values.tolist()))
        intersection_size = len(left_key_set & right_key_set)

        left_coverage = intersection_size / len(left_key_set) if left_key_set else 0
        right_coverage = intersection_size / len(right_key_set) if right_key_set else 0

        merged_df = pd.merge(
            left_df,
            right_df,
            left_on=left_keys,
            right_on=right_keys,
            how="left",
            indicator=True,
        )
        row_multiplier = len(merged_df) / len(left_df) if len(left_df) > 0 else 0
        matched_ratio = ((merged_df["_merge"] == "both").sum() / len(left_df)) if len(left_df) > 0 else 0

    except Exception as e:
        recommendation["validation_passed"] = False
        recommendation["validation_warnings"] = [f"validation computation failed: {e}"]
        return recommendation

    metrics = {
        "left_uniqueness": float(left_uniqueness),
        "right_uniqueness": float(right_uniqueness),
        "intersection_size": int(intersection_size),
        "left_coverage": float(left_coverage),
        "right_coverage": float(right_coverage),
        "row_multiplier": float(row_multiplier),
        "matched_ratio": float(matched_ratio),
    }

    warnings: List[str] = []
    passed = True

    if metrics["left_uniqueness"] < 0.95 and metrics["right_uniqueness"] < 0.95:
        passed = False
        warnings.append("both sides uniqueness are below 0.95")

    if metrics["left_coverage"] < 0.7 or metrics["right_coverage"] < 0.7:
        passed = False
        warnings.append(
            f"insufficient key overlap coverage: left={metrics['left_coverage']:.2%}, right={metrics['right_coverage']:.2%}"
        )

    if metrics["row_multiplier"] > 1.05:
        passed = False
        warnings.append(f"row_multiplier={metrics['row_multiplier']:.2f} > 1.05")

    if metrics["matched_ratio"] < 0.6:
        passed = False
        warnings.append(f"matched_ratio={metrics['matched_ratio']:.2%} < 60%")

    recommendation["validation_metrics"] = metrics
    recommendation["validation_passed"] = passed
    recommendation["validation_warnings"] = warnings
    return recommendation


def profiler_node(state: WorkflowState) -> Dict[str, Any]:
    """Collect metadata and generate merge recommendations in local workspace."""
    print_block("Profiler")

    raw_paths = state.get("raw_file_paths", [])
    workspace_filenames = [os.path.basename(path) for path in raw_paths]
    print_kv("raw_file_count", len(raw_paths))
    print_list("workspace_filenames", workspace_filenames, max_items=6)

    files_metadata: List[Dict[str, Any]] = []
    for idx, data_path in enumerate(raw_paths):
        workspace_filename = os.path.basename(data_path)
        metadata = collect_file_metadata(
            file_path=data_path,
            original_filename=workspace_filename,
            file_index=idx,
            data_path=data_path,
        )
        if "error" in metadata:
            raise RuntimeError(f"文件元信息收集失败: {metadata['error']}")
        files_metadata.append(metadata)
        print_subheader(f"Profiler / File {idx}")
        print_kv("name", workspace_filename)
        print_kv("data_path", data_path)
        _print_file_eda(metadata)

    merge_recommendations = None
    if len(raw_paths) > 1:
        merge_recommendations = _generate_merge_recommendations(files_metadata)
        print_kv("validated_merge_count", len(merge_recommendations))
    else:
        print_kv("merge_recommendation", "skipped(single file)")

    print_subheader("Profiler / Output")
    print_kv("files_metadata_count", len(files_metadata))
    print_kv(
        "merge_recommendation_count",
        0 if merge_recommendations is None else len(merge_recommendations),
    )

    return {
        "files_metadata": files_metadata,
        "merge_recommendations": merge_recommendations,
    }
