import os

import pandas as pd
from typing import Dict, Any, List


def collect_file_metadata(
    file_path: str,
    original_filename: str,
    file_index: int,
    remote_path: str,
    max_rows_for_sample: int = 5,
    max_string_length: int = 50,
    unique_threshold: int = 20,
    max_unique_samples: int = 10
) -> Dict[str, Any]:
    """
    收集单个文件的完整元信息（本地执行）

    Returns:
        {
        "file_index": int,
            "original_filename": str,
            "remote_path": str,
         "row_count": int,
            "columns": List[str],
            "dtypes": Dict[str, str],
            "head_rows": List[Dict],
            "missing_counts": Dict[str, int],
            "unique_counts": Dict[str, int],
          "sample_unique_values": Dict[str, List]
        }
    """
    if not os.path.exists(file_path):
        return {
            "error": f"File not found at {file_path}",
            "file_index": file_index,
          "original_filename": original_filename,
            "remote_path": remote_path
        }

    def _truncate_value(value: Any) -> Any:
        if isinstance(value, str) and len(value) > max_string_length:
            return value[:max_string_length] + "[截断]"
        return value

    try:
        df = pd.read_csv(file_path)

        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).to_dict()
        row_count = int(len(df))

        head_rows_raw = df.head(max_rows_for_sample).to_dict(orient="records")
        head_rows: List[Dict[str, Any]] = []
        for row in head_rows_raw:
            head_rows.append({k: _truncate_value(v) for k, v in row.items()})

        missing_counts = df.isna().sum().astype(int).to_dict()
        unique_counts = df.nunique(dropna=True).astype(int).to_dict()

        sample_unique_values: Dict[str, List[Any]] = {}
        for col, unique_count in unique_counts.items():
            if unique_count < unique_threshold:
                unique_values = df[col].dropna().unique().tolist()
                sample_unique_values[col] = unique_values[:max_unique_samples]

        return {
            "file_index": file_index,
            "original_filename": original_filename,
            "remote_path": remote_path,
            "row_count": row_count,
            "columns": columns,
            "dtypes": dtypes,
            "head_rows": head_rows,
            "missing_counts": missing_counts,
            "unique_counts": unique_counts,
            "sample_unique_values": sample_unique_values
        }

    except Exception as e:
        return {
            "error": f"Error reading CSV: {str(e)}",
            "file_index": file_index,
            "original_filename": original_filename,
            "remote_path": remote_path
        }

