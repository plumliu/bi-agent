import os

import pandas as pd
from typing import Dict, Any


def get_csv_schema(file_path: str) -> Dict[str, Any]:
    """
    读取 CSV 文件，返回结构化的 Data Schema。

    Returns:
        {
            "columns": List[str],        # 列名列表 (方便代码操作)
            "dtypes": Dict[str, str],    # 数据类型 (方便逻辑判断)
            "summary": str               # 格式化文本 (供 Router LLM 阅读)
        }
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found at {file_path}"}

    try:
        # 只读取前 3 行
        df = pd.read_csv(file_path, nrows=2)

        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).to_dict()
        samples = df.to_dict(orient='records')

        return {
            "columns": columns,
            "dtypes": dtypes,
            "samples": samples,
            "summary": ""
        }

    except Exception as e:
        return {"error": f"Error reading CSV: {str(e)}"}


if __name__ == "__main__":
    file_path = "../../temp/temp_data.csv"
    schema = get_csv_schema(file_path)
    print(schema)