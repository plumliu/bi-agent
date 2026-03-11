## 新的建模子图结构  
### 路由结构  
``` textplanner -> executor  
  
executor -> tool  
  
tool -> executor     # 仅当代码执行失败  
tool -> observer     # 仅当代码执行成功  
  
observer -> executor   # CONTINUE / FOLLOW_UP  
observer -> replanner  # REPLAN  
observer -> aggregator # STOP  
  
replanner -> executor  
  
aggregator -> END  
```  
  
新的modeling_custom 子图的具体的state如下：  
### InputState  
- `scenario`  
- `remote_file_path`  
- `data_schema`  
- `user_input`  
  
### InternalState  
// planner  
- `initial_plan`  
- `remaining_tasks`  
- `completed_tasks`  
- `current_task`  
- `followup_playbook`（“粗粒度、先验式、可为空”的 follow-up playbook，基于 schema 和用户目标，预先给出“如果后面发现某类主现象，通常可以沿哪些维度继续深挖”的通用追问手册。由planner初始化、replanner可以进行更新。）  
// executor  
- `latest_ai_message`  
- `latest_execution`  
- `last_error`  
- `execution_trace` （完整的代码运行轨迹，只要是在沙盒中执行成功，那么就追加到这里）  
// observer  
- `latest_control_signal`(四大枚举值CONTINUE / FOLLOW_UP / REPLAN / STOP)  
- `confirmed_findings`  
- `open_questions`（候选问题池，它总是由observer更新；会让replanner在重新规划时查阅）  
- `observer_history`用于构造observer的`[AIMessage, ...]`  
- `replan_reason`  
- `stop_reason`  
// tool  
- `generated_files`在提示词中要求executor使用4个helper方法（create_main_table、append_columns_to_main_table、append_rows_to_main_table、create_artifact），来持久化文件到沙盒中，并且登记元信息，它会自动的将元信息登记到“registered_files.json“中，在tool节点执行代码成功后，需要重新获取沙盒中的registered_files.json读入到`generated_files`。  
### OutputState  
- `modeling_summary`  
- `generated_data_files`  
- `file_metadata`  
- `modeling_artifacts`  
  
---  
  
## 1. Planner  
### 输入  
来自 `InputState`：  
- `scenario`  
- `remote_file_path`  
- `data_schema`  
- `user_input`  
### 输出  
写入 `InternalState`：  
- `initial_plan`  
- `current_task`  
- `remaining_tasks` (`initial_plan`-`current_task`，也就是从第二条任务开始剩下的任务)  
- `followup_playbook`（是planner 初始化的粗粒度追问维度库）  
### 输出的json 格式  
``` json{  
  "phase_tasks": [    {"description": "..."},    {"description": "..."}  ],  "followup_playbook": [    {      "trigger": "...",      "axes": ["location", "time", "entity"]    }  ]}  
```  
  
## 2. Executor  
### 输入  
来自 `InputState`：  
- `user_input`  
- `remote_file_path`  
来自 `InternalState`：  
- `completed_tasks`  
- `current_task`  
- `confirmed_findings`  
- `generated_files`（给executor看现在的主表和JSON产物）  
- `last_error`  
- `latest_control_signal`  
- `execution_trace`  
### 输出  
写入 `InternalState`：  
- `latest_ai_message  
### messages构造  
`[SystemMessage, AIMessage, ToolMessage, AIMessage, ToolMessage, ..., (AIMessage, ToolMessage),HumanMessage]`  
- `SystemMessage`：静态 executor prompt  
- tool执行成功的 `[AIMessage, ToolMessage]` 对。这是一组循环，`[AIMessage, ToolMessage, AIMessage, ToolMessage, ...]`，从`execution_trace`中提取  
- 若本轮来自 tool error，再附加当前失败的 `[AIMessage, ToolMessage]`，从`last_error`中提取  
- `HumanMessage`：本轮执行包  
  - 已完成任务`completed_tasks`  
    - 当前任务`current_task  
  - 已确认发现`confirmed_findings`  
    - 已有文件注册表 `generated_files`  
    - 如果是工具调用失败了（本轮来自 tool error，从tool路由到了executor），那么就在构造HumanMessage时，显式的添加一句“当前的jupyter code cell执行失败！请重新编写当前任务current_task的代码！”  
### 关于产物  
executor需要调用 helper 函数，来间接性的维护沙盒中的/home/user/registered_files.json（本质上是helper内部实现维护的）。executor想要保存文件时总会调用 helper 函数，而这些helper函数会自动的持久化文件，并且将形如下面格式的json同步到registered_files.json中  
``` json{  
  "file_name": "...", // 全量文件名xxxx.feather 或者 xxxx.json  "description": "...", // 一句话描述  
  "columns": [ // 统一使用columns这个名字，它的含义不仅包含了feather列名，也代表json的顶级键的键名  
      { "name": "n_tasks", "description": "任务数" },  
      { "name": "critical_path", "description": "一条关键路径任务序列" }  
  ],}  
  
```  
## 3. Tool（无智能节点，是一个工具节点）  
### 输入  
来自 `InternalState`：  
- `latest_ai_message`  
### 输出  
写入 `InternalState`：  
- `latest_execution`（如果正确，`latest_ai_message的整个代码的高保真执行包，执行错误则置空；latest_execution只保留正确的执行包）  
       ``` json  
       {    
          "task_description": "...",    
          "code": "...",    
          "stdout": "...",    
       }  
       ```  
- `last_error`（如果错了，返回`latest_ai_message` 以及它的报错信息，包含`[AIMessage, ToolMessage]`，执行正确则置空）  
- `generated_files`（在执行的代码中会使用 helper 自动登记主表和JSON产物到registered_files.json 中，始终表示 `/home/user/registered_files.json` 的**全量镜像**，而**不是**本轮新增文件的 delta）  
- `execution_trace`（追加当前 AI/Tool 对，当且仅当执行成功。我们只保留没有任何语法错误的执行轨迹）  
### 路由  
- 若代码执行失败 → `executor`  
- 若代码执行成功 → `observer`  
## 4. Observer  
### 输入  
来自 `InputState`：  
- `user_input`  
来自 `InternalState`：  
- `initial_plan`  
- `completed_tasks`  
- `current_task`  
- `remaining_tasks`  
- `followup_playbook`（由 planner 初始化的粗粒度追问维度库）  
- `open_questions`  
- `confirmed_findings`  
- `generated_files`（主表和JSON产物也要给observer看）  
- `latest_execution`  
- `observer_history`  
### 输出  
  
写入 `InternalState`：  
- `latest_control_signal`（CONTINUE | FOLLOW_UP | REPLAN | STOP）  
- `completed_tasks`（CONTINUE | FOLLOW_UP情况下追加当前任务）  
- `current_task` (CONTINUE：取出下一个任务；FOLLOW_UP：重新拟定一个任务。如果是CONITINUE，则直接从remaining_tasks中取出下一个任务，并更新remaining_tasks；如果是FOLLOW_UP，从大模型输出的`[NEXT_TASK]`中取值)  
- `remaining_tasks`（CONTINUE：取出下一个任务后的任务清单；FOLLOW_UP：保持不变）  
- `confirmed_findings`（追加本轮 delta）  
- `open_questions`（候选问题池，需要在observer结束时更新，可能删除了一些问题，也可能添加一些问题。需要直接使用大模型输出的`[OPEN_QUESTIONS]` 进行覆盖）  
- `observer_history`（追加摘要。从大模型输出的`[TASK_SUMMARY]`取值，并封装成一个AIMessage，append到`observer_history`最后）  
- `replan_reason`（如果REPLAN，则从`[REPLAN_REASON]`中取值）  
- `stop_reason`（如果STOP，则从`[STOP_REASON]`中取值  
  
### messages数组构造`  
`[SystemMessage, HumanMessage, AIMessage, AIMessage, ..., HumanMessage]`  
- `SystemMessage`：静态 observer prompt  
- 紧跟着`SystemMessage`的`HumanMessage`：用户原始输入  
- `[AIMessage,...]`：使用`observer_history`，它应当是一个AIMessage数组，每轮会从大模型输出的`[TASK_SUMMARY]`向内追加  
- 最后构造一个`HumanMessage`：本轮观察者包  
  - 初始任务`initial_plan`  
    - 已完成任务`completed_tasks`  
    - 当前任务`current_task`  
    - 剩下的任务`remaining_tasks`  
    - planner/replanner给出的手册`followup_playbook`   
- 未解问题池`open_questions`  
    - 已确认发现`confirmed_findings`  
    - 已有文件注册表 `generated_files`  
    - 评判上一次的代码执行`latest_execution`：  
       ``` json       {    
          "task_description": "...",    
          "code": "...",    
          "stdout": "...",    
       }  
       ```  
  
## 5. Replanner  
### 输入  
来自 `InputState`：  
- `user_input`  
来自 `InternalState`：  
- `initial_plan`（最初的planner提供）  
- `completed_tasks`（observer审核后更新）  
- `remaining_tasks`（当前的、待更新的剩下的计划，这个是replanner主要需要更新的内容）  
- `confirmed_findings`（observer确定的发现）  
- `open_questions`（observer认为的一系列候选问题）  
- `generated_files`（由executor在代码中给出的、tool节点通过代码解析出的目前生成的文件元信息，包括文件名、列名（顶级键名）、列描述）  
- `replan_reason`（observer触发REPLAN时给出的理由）  
- `latest_execution`（上一次成功执行的高保真执行包）  
### 输出  
  
写入 `InternalState`：  
- `remaining_tasks`（重写剩余的计划）  
- `current_task`（取出剩下的计划中的第一个）  
- `followup_playbook`（由planner初始化，由replanner简单修改）  
## 6. Aggregator（无智能节点）  
### 输入  
来自 `InputState`：  
- `user_input`  
  
来自 `InternalState`：  
- `completed_tasks`  
- `confirmed_findings`  
- `generated_files`  
- `observer_history`  
- `stop_reason`  
  
### 输出  
  
写入 `OutputState`：  
- `modeling_summary`  
- `generated_data_files`  
- `file_metadata`  
- `modeling_artifacts`  
  
我们已经规定了：  
- executor使用提前注入到沙箱中的 helper 负责持久化文件并登记元信息  
- tool 负责把 `registered_files.json` 镜像到`InternalState`的 `generated_files`  
aggregator是无智能节点（没有大模型参与），它只需要：  
1. 读取`stop_reason`  
2. 读取来自 `InternalState`的 `generated_files`，收集其中 `.json` 文件，合成 `modeling_artifacts`  
3. 提取所有 `file_name` 作为 `generated_data_files`  
4. 直接把 `generated_files` 作为 `file_metadata`  
5. 再把 `completed_tasks + confirmed_findings + observer_history + generated_files` 拼接成成 `modeling_summary`