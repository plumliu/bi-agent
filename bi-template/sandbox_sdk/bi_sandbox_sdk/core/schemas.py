from typing import Dict, Any
import pandas as pd
from pydantic import BaseModel, ConfigDict


class SDKResult(BaseModel):
    """
    所有 BI Sandbox SDK 算法的标准返回对象。

    属性:
        result_df (pd.DataFrame): 处理完毕的数据框，可直接保存为 feather。
        artifacts (Dict[str, Any]): 供前端渲染的标准化 JSON 字典。
    """
    result_df: pd.DataFrame
    artifacts: Dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)


