import io
import pandas as pd
from typing import List


def parse_file_content(content: bytes, filename: str) -> List[pd.DataFrame]:
    """
    解析文件二进制内容。

    返回: List[pd.DataFrame]
    - 对于 CSV: 返回包含 1 个 DataFrame 的列表 [df]
    - 对于 Excel: 返回包含所有 Sheet 的 DataFrame 列表 [df_sheet1, df_sheet2, ...]
    """

    # 1. 魔法头检测
    file_header = content[:8]
    is_xlsx = file_header.startswith(b'PK')
    is_xls = file_header.startswith(b'\xd0\xcf\x11\xe0')

    # 准备容器
    dfs_list: List[pd.DataFrame] = []
    input_stream = io.BytesIO(content)

    print(f"[FileParser] 正在解析: {filename}, 魔法头判定: XLSX={is_xlsx}, XLS={is_xls}")

    # ==========================
    # 分支 1: 处理 Excel (支持多 Sheet)
    # ==========================
    if is_xlsx or is_xls:
        excel_data = None  # 这是一个字典 {sheet_name: df} 或 单个 df

        # 定义通用读取参数: sheet_name=None 表示读取所有 sheet
        read_params = {"sheet_name": None}

        # --- 方案 A: Calamine (最优先，速度快且容错高) ---
        try:
            # 注意: calamine 需要 pandas >= 2.2.0
            input_stream.seek(0)
            excel_data = pd.read_excel(input_stream, engine='calamine', **read_params)
            print(f"[FileParser] {filename} -> calamine 解析成功 (多Sheet模式)")
        except (ImportError, Exception) as e:
            if isinstance(e, ImportError):
                print("[FileParser] 未安装 python-calamine，跳过")
            else:
                print(f"[FileParser] calamine 失败 ({e})")

        # --- 方案 B: Openpyxl (仅 xlsx) ---
        if excel_data is None and is_xlsx:
            try:
                input_stream.seek(0)
                excel_data = pd.read_excel(input_stream, engine='openpyxl', **read_params)
                print(f"[FileParser] {filename} -> openpyxl 解析成功")
            except Exception as e:
                print(f"[FileParser] openpyxl 失败 ({e})")

        # --- 方案 C: xlrd (仅 xls) ---
        if excel_data is None and is_xls:
            try:
                input_stream.seek(0)
                excel_data = pd.read_excel(input_stream, engine='xlrd', **read_params)
                print(f"[FileParser] {filename} -> xlrd 解析成功")
            except Exception as e:
                print(f"[FileParser] xlrd 失败 ({e})")

        # --- 处理 Excel 解析结果 ---
        if excel_data is not None:
            # 当 sheet_name=None 时，Pandas 返回 Dict[str, DataFrame]
            if isinstance(excel_data, dict):
                for sheet_name, df in excel_data.items():
                    if not df.empty:
                        # 可选: 把 sheet名 作为一列加进去，防止数据混淆
                        # df["__sheet_source__"] = sheet_name
                        print(f"  -> 提取 Sheet: {sheet_name}, 行数: {len(df)}")
                        dfs_list.append(df)
            else:
                # 极其罕见的情况（通常不会走到这，除非去掉了 sheet_name=None）
                dfs_list.append(excel_data)

            return dfs_list

    # ==========================
    # 分支 2: 处理 CSV (单表)
    # ==========================
    # 如果不是 Excel，或者 Excel 解析全挂了，尝试 CSV
    print(f"[FileParser] 尝试作为 CSV 解析: {filename}")
    encodings = ['utf-8', 'gbk', 'gb18030']

    for encoding in encodings:
        try:
            input_stream.seek(0)
            df = pd.read_csv(input_stream, encoding=encoding)
            print(f"[FileParser] {filename} -> CSV 解析成功 (编码: {encoding})")
            return [df]  # 统一返回列表格式
        except Exception:
            continue

    # 4. 彻底失败
    raise ValueError(f"文件解析失败: {filename}。请确认文件未损坏。")