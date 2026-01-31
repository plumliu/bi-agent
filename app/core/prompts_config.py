import os
import yaml

def load_prompts_config(step: str, scenario: str):
    base_path = os.path.join(os.path.dirname(__file__), "../prompts/scenarios")
    path = os.path.join(base_path, f"{step}_{scenario}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{step}_{scenario} 提示词未找到: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)