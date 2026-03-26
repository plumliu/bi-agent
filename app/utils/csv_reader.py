import os
from typing import Any, Dict, List

import pandas as pd


def _to_builtin(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _to_float(value: Any) -> float | None:
    value = _to_builtin(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_file_metadata(
    file_path: str,
    original_filename: str,
    file_index: int,
    data_path: str,
    max_rows_for_sample: int = 5,
    max_string_length: int = 50,
    unique_threshold: int = 20,
    max_unique_samples: int = 10,
) -> Dict[str, Any]:
    """Collect metadata for one CSV file in local session workspace."""
    if not os.path.exists(file_path):
        return {
            "error": f"File not found at {file_path}",
            "file_index": file_index,
            "original_filename": original_filename,
            "data_path": data_path,
        }

    def _truncate_value(value: Any) -> Any:
        value = _to_builtin(value)
        if isinstance(value, str) and len(value) > max_string_length:
            return value[:max_string_length] + "[truncated]"
        return value

    try:
        df = pd.read_csv(file_path)

        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).to_dict()
        row_count = int(len(df))
        column_count = int(len(columns))
        total_cells = int(row_count * column_count)
        non_null_cells = int(df.notna().sum().sum()) if total_cells > 0 else 0
        missing_cell_count = int(total_cells - non_null_cells)
        missing_cell_ratio = float(missing_cell_count / total_cells) if total_cells > 0 else 0.0

        head_rows_raw = df.head(max_rows_for_sample).to_dict(orient="records")
        head_rows: List[Dict[str, Any]] = []
        for row in head_rows_raw:
            head_rows.append({k: _truncate_value(v) for k, v in row.items()})

        missing_counts = df.isna().sum().astype(int).to_dict()
        missing_ratios = {
            col: (float(missing_counts[col] / row_count) if row_count > 0 else 0.0)
            for col in columns
        }
        non_null_counts = (row_count - df.isna().sum()).astype(int).to_dict()
        unique_counts = df.nunique(dropna=True).astype(int).to_dict()

        column_profiles: List[Dict[str, Any]] = []
        for col in columns:
            non_null = int(non_null_counts[col])
            unique = int(unique_counts[col])
            column_profiles.append(
                {
                    "name": col,
                    "dtype": str(dtypes[col]),
                    "non_null_count": non_null,
                    "missing_count": int(missing_counts[col]),
                    "missing_ratio": float(missing_ratios[col]),
                    "unique_count": unique,
                    "unique_ratio": float(unique / non_null) if non_null > 0 else 0.0,
                }
            )

        sample_unique_values: Dict[str, List[Any]] = {}
        low_cardinality_columns: Dict[str, Dict[str, Any]] = {}
        for col, unique_count in unique_counts.items():
            if unique_count < unique_threshold:
                unique_values = [_truncate_value(v) for v in df[col].dropna().unique().tolist()]
                sample_unique_values[col] = unique_values[:max_unique_samples]

                value_counts = df[col].fillna("__MISSING__").value_counts(dropna=False).head(max_unique_samples)
                low_cardinality_columns[col] = {
                    "unique_count": int(unique_count),
                    "missing_count": int(missing_counts[col]),
                    "missing_ratio": float(missing_ratios[col]),
                    "top_values": [
                        {"value": _truncate_value(k), "count": int(v)}
                        for k, v in value_counts.items()
                    ],
                }

        numeric_columns_summary: Dict[str, Dict[str, Any]] = {}
        numeric_df = df.select_dtypes(include=["number"])
        for col in numeric_df.columns.tolist():
            series = numeric_df[col]
            numeric_columns_summary[col] = {
                "mean": _to_float(series.mean()),
                "std": _to_float(series.std()),
                "min": _to_float(series.min()),
                "p25": _to_float(series.quantile(0.25)),
                "p50": _to_float(series.quantile(0.5)),
                "p75": _to_float(series.quantile(0.75)),
                "max": _to_float(series.max()),
            }

        duplicate_row_count = int(df.duplicated().sum())
        duplicate_row_ratio = float(duplicate_row_count / row_count) if row_count > 0 else 0.0

        return {
            "file_index": file_index,
            "original_filename": original_filename,
            "data_path": data_path,
            "row_count": row_count,
            "column_count": column_count,
            "columns": columns,
            "dtypes": dtypes,
            "head_rows": head_rows,
            "total_cells": total_cells,
            "missing_cell_count": missing_cell_count,
            "missing_cell_ratio": missing_cell_ratio,
            "missing_counts": missing_counts,
            "missing_ratios": missing_ratios,
            "unique_counts": unique_counts,
            "column_profiles": column_profiles,
            "sample_unique_values": sample_unique_values,
            "low_cardinality_columns": low_cardinality_columns,
            "numeric_columns_summary": numeric_columns_summary,
            "duplicate_row_count": duplicate_row_count,
            "duplicate_row_ratio": duplicate_row_ratio,
        }

    except Exception as e:
        return {
            "error": f"Error reading CSV: {str(e)}",
            "file_index": file_index,
            "original_filename": original_filename,
            "data_path": data_path,
        }
