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
            "\n\n【Merge/Concat Recommendation (validated by the Profiler)】\n"
            + json.dumps(state["merge_recommendations"], ensure_ascii=False, indent=2)
        )

    human_content = f"""【user_input】
{state['user_input']}

【Environment information for the current_task】
raw_file count: {len(state.get('raw_file_paths', []))}
raw_file_paths: {state.get('raw_file_paths', [])}
original_filenames: {state.get('original_filenames', [])}

【files_metadata】
{files_metadata_str}{merge_recs_str}

Note:
- For a single file, files_metadata has length 1 and there are no merge_recommendations.
- For multiple files, files_metadata contains metadata for all files, and merge_recommendations contains the merge recommendations that passed validation.
- Merge recommendations are for reference only; the executor should make the final decision based on the EDA results.
"""
    human_message = HumanMessage(content=human_content)

    # Call LLM with retry loop for format errors
    messages = [system_message, human_message]
    max_retries = 3
    phase_tasks = []
    followup_playbook = []

    for attempt in range(max_retries):
        response = llm.invoke(messages)
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

        # Try JSON parse
        try:
            parsed = json.loads(content_str)
        except json.JSONDecodeError as e:
            print(f"[Planner] JSON 解析失败 (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                messages.append(response)
                messages.append(HumanMessage(content=(
                    f"Your output cannot be parsed as JSON. Error: {e}\n"
                    f"Please re-output strictly in the required JSON format and do not include any additional text."
                )))
                continue
            raise RuntimeError(f"Planner JSON 解析失败（已重试 {max_retries} 次）: {e}") from e

        # Validate structure: phase_tasks must be list of dicts with 'description'
        raw_tasks = parsed.get("phase_tasks", [])
        if not isinstance(raw_tasks, list) or len(raw_tasks) == 0:
            error_msg = f"phase_tasks must be a non-empty list. Current value: {raw_tasks}"
        elif not isinstance(raw_tasks[0], dict) or "description" not in raw_tasks[0]:
            error_msg = (
                f"Each element in phase_tasks must be a dictionary containing the key 'description'. "
                f"Current first element type: {type(raw_tasks[0]).__name__}, value: {raw_tasks[0]}"
            )
        else:
            error_msg = None

        if error_msg:
            print(f"[Planner] 格式校验失败 (attempt {attempt + 1}/{max_retries}): {error_msg}")
            if attempt < max_retries - 1:
                messages.append(response)
                messages.append(HumanMessage(content=(
                    f"Your output format is incorrect: {error_msg}\n"
                    f"Please ensure `phase_tasks` is a list, and each element is a dictionary containing the 'description' field, for example:\n"
                    f'{{"phase_tasks": [{{"description": "Task description", "...": "..."}}]}}'
                )))
                continue
            raise RuntimeError(f"Planner 格式校验失败（已重试 {max_retries} 次）: {error_msg}")

        phase_tasks = raw_tasks
        followup_playbook = parsed.get("followup_playbook", [])
        break

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
