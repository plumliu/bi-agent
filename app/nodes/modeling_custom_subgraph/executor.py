import json
import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from ppio_sandbox.code_interpreter import Sandbox

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.agents.modeling_custom_agent import create_modeling_custom_agent
from app.utils.extract_text_from_content import extract_text_from_content


def create_executor_node(sandbox: Sandbox):
    """创建 executor 节点（工厂函数）"""

    def executor_node(state: CustomModelingState) -> Dict[str, Any]:
        print("--- [Modeling Subgraph] Executor: 开始执行任务 ---")

        # 获取上下文
        scenario = state.get("scenario")
        remote_file_path = state.get("remote_file_path", "")
        plan = state.get("plan")

        if not plan or len(plan) == 0:
            raise RuntimeError("Executor: plan 为空")

        # 将 plan 转换为文本
        plan_str = "\n".join([f"任务 {i + 1}: {t.description}" for i, t in enumerate(plan)])

        # 创建 agent
        agent = create_modeling_custom_agent(sandbox, scenario)

        # 构建动态上下文
        config = load_prompts_config("modeling", scenario)
        context_template = config.get('executor_context_template')
        context_content = context_template.format(
            remote_file_path=remote_file_path,
            plan_str=plan_str
        )

        # 调用 agent
        agent_result = agent.invoke({
            "remote_file_path": remote_file_path,
            "plan": plan,
            "messages": [HumanMessage(content=context_content)]
        })

        # 提取结果
        print("--- [Modeling Subgraph] Agent 执行完成，提取结果 ---")

        # 从 agent_result 的 messages 中获取最后一条消息
        messages = agent_result.get("messages", [])
        if not messages:
            return {
                "modeling_summary": "执行完成",
                "generated_data_files": [],
                "file_metadata": []
            }

        last_message = messages[-1]
        answer = extract_text_from_content(last_message.content)

        # 尝试从回复中提取 JSON 格式的产物清单
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', answer, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(1)
                file_metadata = json.loads(json_str)
                print(f"--- [Modeling Subgraph] 成功解析产物清单，包含 {len(file_metadata)} 个文件 ---")

                return {
                    "modeling_summary": answer,
                    "generated_data_files": [item["file_name"] for item in file_metadata],
                    "file_metadata": file_metadata
                }
            except json.JSONDecodeError:
                print("--- [Modeling Subgraph] JSON 解析失败，回退到文件扫描 ---")

        # 回退：扫描文件系统
        try:
            ls_result = sandbox.commands.run("ls /home/user/*.feather /home/user/*.json 2>/dev/null || true")
            if ls_result.stdout:
                files = [f.split('/')[-1] for f in ls_result.stdout.strip().split('\n') if f and '/' in f]
                print(f"--- [Modeling Subgraph] 发现 {len(files)} 个产物文件 ---")
                return {
                    "modeling_summary": answer,
                    "generated_data_files": files,
                    "file_metadata": []
                }
        except Exception as e:
            print(f"--- [Modeling Subgraph] 扫描文件失败: {e} ---")

        return {
            "modeling_summary": answer,
            "generated_data_files": [],
            "file_metadata": []
        }

    return executor_node