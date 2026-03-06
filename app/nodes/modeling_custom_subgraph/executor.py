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
        user_input = state.get("user_input", "")  # 添加：获取用户原始需求

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
            plan_str=plan_str,
            user_input=user_input
        )

        # 调用 agent（agent 内部会循环执行直到完成所有任务）
        # 增加 recursion_limit 以支持多任务执行
        invoke_config = {
            "recursion_limit": 100  # 增加递归限制，支持执行多个任务
        }

        agent_result = agent.invoke(
            {
                "remote_file_path": remote_file_path,
                "plan": plan,
                "messages": [HumanMessage(content=context_content)]
            },
            config=invoke_config
        )

        # 提取结果
        print("--- [Modeling Subgraph] Agent 执行完成，提取结果 ---")

        # 从 agent_result 的 messages 中获取最后一条消息
        messages = agent_result.get("messages", [])

        # 调试：打印 messages 数量和类型
        print(f"--- [Debug] Agent 返回了 {len(messages)} 条消息 ---")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:100] if hasattr(msg, 'content') else "N/A"
            print(f"  Message {i+1}: {msg_type} - {content_preview}...")

            # 检查是否有 tool_calls
            if hasattr(msg, 'tool_calls'):
                print(f"    tool_calls: {msg.tool_calls}")
            if hasattr(msg, 'additional_kwargs'):
                print(f"    additional_kwargs: {msg.additional_kwargs}")

        # 打印最后一条消息的完整内容
        if messages:
            last_msg = messages[-1]
            print(f"\n--- [Debug] 最后一条消息完整内容 ---")
            print(f"Type: {type(last_msg).__name__}")
            print(f"Content: {last_msg.content}")
            if hasattr(last_msg, 'tool_calls'):
                print(f"Tool calls: {last_msg.tool_calls}")

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
                result = {
                    "modeling_summary": answer,
                    "generated_data_files": files,
                    "file_metadata": []
                }
                print(f"--- [Debug] Executor 返回结果（文件扫描）---")
                print(f"  modeling_summary: {result['modeling_summary'][:100]}...")
                print(f"  generated_data_files: {result['generated_data_files']}")
                print(f"  file_metadata: {result['file_metadata']}")
                return result
        except Exception as e:
            print(f"--- [Modeling Subgraph] 扫描文件失败: {e} ---")

        result = {
            "modeling_summary": answer,
            "generated_data_files": [],
            "file_metadata": []
        }
        print(f"--- [Debug] Executor 返回结果（最终兜底）---")
        print(f"  modeling_summary: {result['modeling_summary'][:100] if result['modeling_summary'] else 'EMPTY'}...")
        print(f"  generated_data_files: {result['generated_data_files']}")
        print(f"  file_metadata: {result['file_metadata']}")
        return result

    return executor_node