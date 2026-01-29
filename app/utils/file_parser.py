import io
import pandas as pd
from typing import List

import io
import pandas as pd
from typing import List, Tuple


def parse_file_content(content: bytes, filename: str) -> List[Tuple[str, pd.DataFrame]]:
    """
    解析文件二进制内容为 Pandas DataFrame 列表。

    参数:
        content: 文件的二进制内容
        filename: 文件名 (仅用于日志打印，不参与解析逻辑)

    返回:
        List[Tuple[str, pd.DataFrame]]
        列表中的每个元素是一个元组：(Sheet名称或标识, DataFrame数据)
        - 对于 Excel: [("Sheet1", df1), ("Sheet2", df2), ...]
        - 对于 CSV:   [("CSV_Content", df)]
    """

    # 1. 魔法头检测 (Magic Bytes)
    # 相比依赖 .xlsx 后缀，检测文件头前8个字节更靠谱
    file_header = content[:8]
    is_xlsx = file_header.startswith(b'PK')  # ZIP 格式 (现代 Excel .xlsx)
    is_xls = file_header.startswith(b'\xd0\xcf\x11\xe0')  # OLE2 格式 (旧版 Excel .xls)

    # 准备容器
    results: List[Tuple[str, pd.DataFrame]] = []
    input_stream = io.BytesIO(content)

    print(f"[FileParser] 正在解析: {filename}, 判定类型: XLSX={is_xlsx}, XLS={is_xls}")

    # ==========================
    # 分支 1: 处理 Excel (支持多 Sheet)
    # ==========================
    if is_xlsx or is_xls:
        excel_data = None  # 这是一个字典 {sheet_name: df} 或 单个 df

        # 定义通用读取参数: sheet_name=None 表示读取所有 sheet
        read_params = {"sheet_name": None}

        # --- 方案 A: Calamine (最优先，速度快且容错高) ---
        try:
            # seek(0) 非常重要，重置指针到文件开头
            input_stream.seek(0)
            excel_data = pd.read_excel(input_stream, engine='calamine', **read_params)
            print(f"[FileParser] {filename} -> calamine 解析成功 (多Sheet模式)")
        except (ImportError, Exception) as e:
            if isinstance(e, ImportError):
                print("[FileParser] 未安装 python-calamine，跳过强力模式")
            else:
                print(f"[FileParser] calamine 解析尝试失败: {e}")

        # --- 方案 B: Openpyxl (仅 xlsx) ---
        if excel_data is None and is_xlsx:
            try:
                input_stream.seek(0)
                excel_data = pd.read_excel(input_stream, engine='openpyxl', **read_params)
                print(f"[FileParser] {filename} -> openpyxl 解析成功")
            except Exception as e:
                print(f"[FileParser] openpyxl 解析尝试失败: {e}")

        # --- 方案 C: xlrd (仅 xls) ---
        if excel_data is None and is_xls:
            try:
                input_stream.seek(0)
                excel_data = pd.read_excel(input_stream, engine='xlrd', **read_params)
                print(f"[FileParser] {filename} -> xlrd 解析成功")
            except Exception as e:
                print(f"[FileParser] xlrd 解析尝试失败: {e}")

        # --- 处理 Excel 解析结果 ---
        if excel_data is not None:
            # 当 sheet_name=None 时，Pandas 返回 Dict[str, DataFrame]
            if isinstance(excel_data, dict):
                for sheet_name, df in excel_data.items():
                    if not df.empty:
                        # 强制转为字符串，防止纯数字 Sheet 名导致类型错误
                        s_name = str(sheet_name)
                        print(f"  -> 提取 Sheet: {s_name}, 行数: {len(df)}")
                        results.append((s_name, df))
            else:
                # 极其罕见的情况（单表返回且不是dict，可能是某些特定版本的pandas行为）
                # 为了健壮性，给它一个默认名字
                if not excel_data.empty:
                    results.append(("Sheet1", excel_data))

            if not results:
                raise ValueError(f"Excel 文件 {filename} 解析成功但内容为空或无有效 Sheet")

            return results

    # ==========================
    # 分支 2: 处理 CSV (单表)
    # ==========================
    print(f"[FileParser] 尝试作为 CSV 解析: {filename}")

    # 常见的 CSV 编码尝试顺序
    encodings = ['utf-8', 'gbk', 'gb18030']

    for encoding in encodings:
        try:
            input_stream.seek(0)
            df = pd.read_csv(input_stream, encoding=encoding)
            print(f"[FileParser] {filename} -> CSV 解析成功 (编码: {encoding})")

            return [("CSV", df)]
        except Exception:
            continue

    # 4. 彻底失败
    raise ValueError(f"文件解析失败: {filename}。请确认文件格式正确(Excel/CSV)且未损坏。")