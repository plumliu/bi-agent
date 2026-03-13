import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.core.prompts_config import load_prompts_config
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import create_llm, apply_retry

# Initialize LLM (主模型使用Anthropic)
llm = apply_retry(create_llm(use_flash=False))


def planner_node(state: CustomModelingState):
    """
    Planner 节点：根据用户需求和数据 Schema 生成初始任务计划
    """
    print("--- [Planner] 正在规划任务列表... ---")

    # Load prompts
    prompts = load_prompts_config("modeling", "custom")
    planner_instruction = prompts["planner_instruction"]

    # Construct messages
    system_message = SystemMessage(content=planner_instruction)

    files_metadata_str = json.dumps(state["files_metadata"], ensure_ascii=False, indent=2)

    merge_recs_str = ""
    if state.get("merge_recommendations"):
        merge_recs_str = (
            "\n\n【合并建议（Profiler 验证通过）】\n"
            + json.dumps(state["merge_recommendations"], ensure_ascii=False, indent=2)
        )

    human_content = f"""【用户的原始需求】
{state['user_input']}

【当前任务的环境信息】
原始文件数量: {len(state.get('raw_file_paths', []))}
原始文件路径: {state.get('raw_file_paths', [])}
原始文件名: {state.get('original_filenames', [])}

【文件元信息】
{files_metadata_str}{merge_recs_str}

注意：
- 单文件时，files_metadata 长度为 1，无 merge_recommendations
- 多文件时，files_metadata 包含所有文件的元信息，merge_recommendations 包含验证通过的合并建议
- 合并建议仅供参考，executor 应基于 EDA 结果做最终决策
"""
    human_message = HumanMessage(content=human_content)

    # Call LLM
    response = llm.invoke([system_message, human_message])

    # Parse JSON output
    raw_text = extract_text_from_content(response.content)

    # Extract JSON
    match = re.search(r"```(?:json)?(.*?)```", raw_text, re.DOTALL)
    if match:
        content_str = match.group(1).strip()
    else:
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content_str = raw_text[start_idx:end_idx + 1]
        else:
            content_str = raw_text.strip()

    try:
        parsed = json.loads(content_str)
    except json.JSONDecodeError as e:
        print(f"[Planner] JSON 解析失败: {e}")
        print(f"[Planner] LLM 原始响应（前2000字符）:\n{raw_text[:2000]}")
        print(f"[Planner] 提取的 JSON 字符串（前2000字符）:\n{content_str[:2000]}")
        raise RuntimeError(f"Planner JSON 解析失败: {e}") from e

    phase_tasks = parsed["phase_tasks"]
    followup_playbook = parsed.get("followup_playbook", [])

    print(f"--- [Planner] 生成了 {len(phase_tasks)} 个任务 ---")
    for i, task in enumerate(phase_tasks, 1):
        print(f"  [Task {i}] {task['description']}")

    # Initialize state
    return {
        "initial_plan": {"phase_tasks": phase_tasks, "followup_playbook": followup_playbook},
        "remaining_tasks": phase_tasks[1:] if len(phase_tasks) > 1 else [],
        "completed_tasks": [],
        "current_task": phase_tasks[0]["description"] if phase_tasks else None,
        "followup_playbook": followup_playbook,
        "confirmed_findings": [],
        "working_hypotheses": [],
        "open_questions": [],
        "observer_history": [],
        "execution_trace": [],
        "generated_files": {},
    }
