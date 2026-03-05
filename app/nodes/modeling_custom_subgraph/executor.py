import time
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.tools.python_interpreter import create_code_interpreter_tool
from ppio_sandbox.code_interpreter import Sandbox

from app.utils.extract_text_from_content import extract_text_from_content

# 1. 初始化基础 LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    use_responses_api=settings.USE_RESPONSES_API,
)

step = "modeling"


def executor_node(state: CustomModelingState, sandbox: Sandbox, config: RunnableConfig):
    """
    【Executor 节点】(采用 partial 依赖注入模式)
    职责：看全局计划 -> 思考 -> 决定调用工具写代码 (或给出最终结论)
    """
    print("--- [Modeling Subgraph] Executor: 正在思考与规划行动... ---")

    # 1. 动态生成并绑定真实的沙盒工具
    python_tool = create_code_interpreter_tool(sandbox)
    llm_with_tools = llm.bind_tools([python_tool])

    # 2. 初始化时间统计
    metrics = state.get("metrics") or {}
    metrics.setdefault("llm_duration", 0.0)

    # 3. 获取上下文与 Planner 制定的战略计划
    plan = state.get("plan")
    if plan is None or not isinstance(plan, list):
        raise RuntimeError("--- [Modeling Subgraph] Executor: 错误! 'plan' 不在 state 中! ")
    elif len(plan) == 0:
        raise RuntimeError("--- [Modeling Subgraph] Executor: 错误! 'plan' 是空的! ")

    remote_file_path = state.get("remote_file_path", "")

    # 将结构化的 Plan 转化为文本指南
    plan_str = "\n".join([f"任务 {i + 1}: {t.description}" for i, t in enumerate(plan)])

    # 4. 加载 Prompt 模板
    scenario = state.get("scenario")
    config = load_prompts_config(step, scenario)
    instruction_template = config.get('executor_instruction')

    # 构造核心 System Prompt
    system_content = instruction_template.format(
        remote_file_path=remote_file_path,
        plan_str=plan_str
    )

    # 5. 组装消息列表 (系统指令 + 对话历史记忆)
    messages = [SystemMessage(content=system_content)] + state.get("messages", [])

    # 6. 调用 LLM (大脑决策阶段)
    print("--- [Modeling Subgraph] Executor: 调用大模型中... ---")
    llm_start_time = time.perf_counter()

    # 让大模型自己决定是输出 tool_calls(写代码)，还是直接输出文本(宣布完工)
    response = llm_with_tools.invoke(messages, config=config)

    current_llm_time = time.perf_counter() - llm_start_time
    metrics["llm_duration"] += current_llm_time
    print(f"--- [Time] Executor LLM 决策耗时: {current_llm_time:.2f} 秒")

    updates = {
        "messages": [response],
        "metrics": metrics
    }

    if not response.tool_calls:
        print("--- [Modeling Subgraph] 建模成功... ---")
        answer = extract_text_from_content(response.content)
        print(answer)

        # 尝试从回复中提取 JSON 格式的产物清单
        import re
        import json

        # 查找 JSON 代码块
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', answer, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(1)
                file_metadata = json.loads(json_str)

                print(f"--- [Modeling Subgraph] 成功解析产物清单，包含 {len(file_metadata)} 个文件 ---")

                # 存储结构化的文件元数据
                updates["file_metadata"] = file_metadata

                # 提取文件名列表（保持向后兼容）
                updates["generated_data_files"] = [item["file_name"] for item in file_metadata]

                # 存储完整的 modeling_summary（包含业务总结和 JSON）
                updates["modeling_summary"] = answer

            except json.JSONDecodeError as e:
                print(f"--- [Modeling Subgraph] JSON 解析失败: {e}，回退到文件扫描 ---")
                # 回退：扫描文件系统
                updates["modeling_summary"] = answer
                try:
                    ls_result = sandbox.commands.run("ls /home/user/*.feather /home/user/*.json 2>/dev/null || true")
                    if ls_result.stdout:
                        files = [f.split('/')[-1] for f in ls_result.stdout.strip().split('\n') if f and '/' in f]
                        updates["generated_data_files"] = files
                    else:
                        updates["generated_data_files"] = []
                except:
                    updates["generated_data_files"] = []
        else:
            print("--- [Modeling Subgraph] 未找到 JSON 清单，回退到文件扫描 ---")
            # 回退：扫描文件系统
            updates["modeling_summary"] = answer
            try:
                ls_result = sandbox.commands.run("ls /home/user/*.feather /home/user/*.json 2>/dev/null || true")
                if ls_result.stdout:
                    files = [f.split('/')[-1] for f in ls_result.stdout.strip().split('\n') if f and '/' in f]
                    updates["generated_data_files"] = files
                    print(f"--- [Modeling Subgraph] 发现 {len(files)} 个产物文件: {files} ---")
                else:
                    updates["generated_data_files"] = []
            except Exception as e:
                print(f"--- [Modeling Subgraph] 扫描文件失败: {e} ---")
                updates["generated_data_files"] = []


    # 7. 返回状态 (直接把 LLM 的回复追加到主 messages 列表中)
    return updates