# BI Agent - 智能数据分析系统

基于 LangGraph 构建的多 Agent 协作数据分析平台，通过工作流编排实现从数据上传到可视化的全自动分析流程。

## 项目概述

BI Agent 是一个企业级智能数据分析系统，能够理解用户的自然语言需求，自动完成数据清洗、分析建模和可视化呈现。系统采用多 Agent 协作架构，通过 LangGraph 进行工作流编排，实现了高度自动化和智能化的数据分析能力。

### 核心能力

- **自然语言交互**：用户用自然语言描述分析需求，系统自动理解并执行
- **智能路由**：根据用户意图和数据特征，自动选择最合适的分析算法
- **自动 ETL**：支持多文件上传，LLM 驱动的智能数据合并
- **双模式建模**：标准算法（SOP）+ 自定义分析（Custom）
- **实时反馈**：SSE 流式响应，实时展示分析进度
- **沙盒隔离**：每个请求独立的代码执行环境，确保安全性

## 技术架构

### 技术栈

- **工作流编排**：LangGraph v1.0 - 状态机驱动的 Agent 工作流
- **Agent 框架**：LangChain v1.0 - ReAct Agent 实现
- **后端服务**：FastAPI - 异步 API 服务
- **代码执行**：PPIO Sandbox - 隔离的 Python 运行环境

### 系统架构

```
用户请求
    ↓
FastAPI 入口
    ↓
LangGraph 工作流
    ├─ Auto-ETL 节点（多文件智能合并）
    ├─ Router 节点（意图识别与条件路由）
    │
    ├─ SOP 分支（标准算法）
    │    ├─ Modeling 节点（ReAct 调用预定义算法）
    │    ├─ Fetch Artifacts 节点（下载产物文件）
    │    ├─ Viz 节点（LLM 生成图表配置）
    │    └─ Summary 节点（生成分析总结）
    │        ↓
    │    SSE 流式响应
    │        ↓
    │    前端展示
    │
    └─ Custom 分支（自定义分析）
        ├─ Modeling Custom 子图（智能建模）
        │    ├─ Planner（任务规划）
        │    ├─ Executor（代码生成）
        │    ├─ Tool（沙盒执行）
        │    ├─ Observer（结果观察与决策）
        │    ├─ Replanner（动态重规划）
        │    └─ Aggregator（产物汇总）
        ├─ Viz Custom 子图（并发可视化）
        │    ├─ Viz Planner（生成可视化任务）
        │    ├─ Viz Executor（并发执行）
        │    └─ Viz Aggregator（汇总可视化结果）
        └─ Summary 节点（生成分析总结）
            ↓
        SSE 流式响应
            ↓
        前端展示
```

## 工作流编排

### 1. 智能路由机制

系统通过 Router 节点分析用户需求和数据特征，自动选择最合适的分析路径：

- **SOP 路径**：预定义的标准算法（聚类、异常检测、时序分解、关联分析、预测、分类）
- **Custom 路径**：开放式分析，支持复杂的多步骤任务

**优势**：
- 快速响应标准场景（SOP 场景，有预定义好的标准算法库）
- 灵活处理复杂需求（Custom 多轮迭代）
- 自动识别用户意图，无需手动选择

### 2. 双模式建模架构

#### SOP 模式（标准算法路径）

```
router → modeling → fetch_artifacts → viz → summary → END
```

- **modeling**：ReAct 调用预定义算法工具（如聚类、异常检测）
- **fetch_artifacts**：从沙盒下载生成的数据文件
- **viz**：LLM 生成 ECharts 配置
- **summary**：总结分析结果

**特点**：
- 单次执行，效率高
- 结果标准化，易于可视化
- 适用场景：聚类分析、异常检测、时序分解等

#### Custom 模式（自定义分析路径）

```
router → modeling_custom 子图 → viz_custom 子图 → summary → END
```

**modeling_custom 子图**（智能建模与观察者控制）：

```
planner → executor ⇄ tool ⇄ observer → {executor, replanner, aggregator}
                                ↓
                 CONTINUE / FOLLOW_UP / REPLAN / STOP
```

**核心节点职责**：

1. **Planner**：分析用户需求，生成初始任务列表（2-7个子任务）
2. **Executor**：
   - 使用 React-like 模式（手动管理 messages 数组）
   - 通过 `llm.bind_tools()` 绑定代码执行工具
   - 生成 Python 代码（返回 AIMessage with tool_calls）
3. **Tool**：
   - 在沙盒中执行代码
   - 成功：添加到 execution_trace，路由到 Observer
   - 失败：存入 last_error，路由回 Executor 重试
4. **Observer**：
   - 观察执行结果，做出控制决策
   - **CONTINUE**：任务完成，继续下一个任务
   - **FOLLOW_UP**：需要追加任务（如发现新问题）
   - **REPLAN**：需要重新规划剩余任务
   - **STOP**：所有任务完成或遇到无法解决的问题
5. **Replanner**：根据当前进展重新规划剩余任务
6. **Aggregator**：汇总所有产物（feather 主表 + JSON 产物）

**Helper 函数机制**（主表维护）：

系统在沙盒中预注入 helper 函数，支持增量式数据管理：

```python
# 创建主表
create_main_table(df, "main.feather", "主表描述", columns_desc)

# 向主表添加新列（列级增量）
append_columns_to_main_table("main.feather", df, join_on="id", columns_desc)

# 向主表添加新行（行级增量）
append_rows_to_main_table("main.feather", df)

# 创建 JSON 产物
create_artifact(data, "metrics.json", "指标描述", columns_desc)
```

**优势**：
- 避免重复生成大表，减少沙盒内存占用
- 支持增量式分析（逐步添加特征列）
- 自动维护文件注册表（registered_files.json）

**viz_custom 子图**（按需可视化）：

```
viz_planner → [viz_executor_1, viz_executor_2, ...] → viz_aggregator → END
              (并发执行，使用 LangGraph Send API)
```

**关键改进**：

1. **按需列提取**：
   - Planner 让 LLM 指定每个图表需要的列名
   - Executor 只提取需要的列描述，减少上下文长度
   - 示例：主表有 50 列，散点图只需要 3 列

2. **JSON 产物支持**：
   - 通过 `modeling_artifacts` 传递 JSON 内容
   - Executor 可以同时使用 feather 主表和 JSON 产物

3. **数据需求格式**：
```json
{
  "data_requirements": [
    {
      "file_name": "main.feather",
      "required_columns": ["col1", "col2", "col3"]
    },
    {
      "file_name": "metrics.json",
      "required_columns": null  // null 表示使用全部数据
    }
  ]
}
```

**特点**：
- 支持复杂的多步骤分析
- Observer 控制流支持动态调整和错误自纠正
- 可视化任务并发执行，提升效率
- 零拷贝：直接在沙盒环境中生成的中间文件进行可视化

### 3. LLM 工厂模式

系统使用统一的 LLM 工厂函数，支持多模型配置：

```python
from app.utils.llm_factory import create_llm, apply_retry

# 主模型（Anthropic）- 用于复杂推理
llm = apply_retry(create_llm(use_flash=False))

# FLASH 模型（OpenAI）- 用于快速任务
llm_flash = apply_retry(create_llm(use_flash=True))

# 绑定工具（必须在 apply_retry 之前）
llm_with_tools = apply_retry(llm.bind_tools([tool]))
```

**优势**：
- 统一的重试策略（指数退避 + jitter）
- 灵活切换模型提供商
- 主模型使用 Anthropic 避免中转站缓存问题

### 4. LLM 驱动的自动 ETL

传统 ETL 需要预定义合并规则，本系统通过 LLM 动态生成合并代码：

```python
# LLM 根据文件结构自动生成合并逻辑
merge_code = llm.generate_merge_code(files_info)
sandbox.execute(merge_code)  # 在沙盒中执行
```

**优势**：
- 无需预定义规则，适应任意文件结构
- 支持多 Sheet Excel 文件自动拆分
- 智能识别关联字段，自动生成 JOIN 逻辑

### 5. 请求级沙盒隔离

每个 API 请求创建独立的 PPIO Sandbox：

```python
async def run_workflow_stream(...):
    sandbox = Sandbox.create()  # 创建沙盒
    try:
        # 上传文件、执行工作流
        ...
    finally:
    sandbox.kill()  # 自动清理
```

**优势**：
- 完全隔离，避免数据泄露
- 自动清理，无资源残留
- 支持并发请求，互不干扰

### 6. 流式响应与实时反馈

通过 SSE（Server-Sent Events）实时推送分析进度：

- `log`：普通日志消息
- `log_stream`：AI 打字机输出
- `tool_calling`：工具调用通知
- `tool_log`：沙盒执行结果
- `viz_data`：可视化数据
- `summary`：最终总结

**优势**：
- 用户实时了解分析进度
- 长时间任务不会超时
- 提升用户体验

## 支持的分析场景

### SOP 标准算法

- **聚类分析**（cluster）：K-Means、层次聚类等
- **异常检测**（anomaly）：基于统计方法的异常识别
- **时序分解**（decomposition）：STL 分解（趋势、季节性、残差）
- **关联分析**（association）：Apriori 算法挖掘关联规则
- **预测分析**（forecast）：时间序列预测
- **分类分析**（classification）：分类模型训练与评估

### Custom 自定义分析

支持任意复杂的多步骤分析任务，例如：
- "先进行趋势分解，再检测异常，最后预测未来 12 天"
- "对数据进行聚类，然后分析每个簇的特征"
- "计算关键路径，分析瓶颈任务"

**Observer 控制流示例**：
1. 执行任务 1：数据清洗 → CONTINUE
2. 执行任务 2：特征工程 → FOLLOW_UP（发现需要额外的特征）
3. 执行追加任务：计算新特征 → CONTINUE
4. 执行任务 3：建模分析 → REPLAN（发现数据分布异常，需要调整策略）
5. 重新规划剩余任务 → 继续执行
6. 所有任务完成 → STOP

## 支持的可视化图表

系统支持 11 种图表类型，覆盖常见的数据分析场景：

- **scatter**：散点图（聚类结果、分布分析）
- **line**：折线图（趋势分析、时序数据）
- **bar**：柱状图（对比分析、分类统计）
- **pie**：饼图（占比分析）
- **radar**：雷达图（多维度对比）
- **lineWithConfidence**：带置信区间的折线图（预测结果）
- **boxplot**：箱线图（异常检测、分布分析）
- **lineWithErrorBars**：带误差棒的折线图（实验数据）
- **table**：表格（详细数据展示）
- **heatmap**：热力图（相关性分析）
- **decomposition**：时序分解图（STL 分解结果）

## 快速开始

### 环境要求

- Python 3.10+
- uv（Python 包管理器）

### 安装依赖

```bash
# 使用 uv 安装依赖
uv sync
```

### 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置以下变量：

ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_API_BASE=https://api.anthropic.com 
LLM_MODEL_NAME=claude-sonnet-4-20250514

OPENAI_API_KEY_FLASH=your_openai_key
OPENAI_API_BASE_FLASH=https://api.openai.com/v1
LLM_FLASH_MODEL_NAME=gpt-4o-mini
USE_RESPONSES_API_FLASH=false

# PPIO 沙盒配置
PPIO_API_KEY=your_ppio_key
PPIO_TEMPLATE=your_template_id
```

### 启动服务

```bash
# 启动 FastAPI 服务（热重载模式）
uvicorn main:app --host 0.0.0.0 --port 8009 --reload

# 或直接运行
uv run main.py
```

### API 调用示例

```bash
curl -X POST http://localhost:8009/query_agents_stream \
  -F "file=@data.csv" \
  -F 'request_data={"query":"对这份数据进行聚类分析","scenario":null}'
```

## 项目结构

```
bi-agent/
├── app/
│   ├── core/         # 核心配置
│   │   ├── config.py           # 环境变量配置
│   │   ├── state.py                   # 主工作流状态定义
│   │   ├── prompts_config.py       # 提示词加载器
│   │   ├── modeling_custom_subgraph/  # Custom 子图状态
│   │   │   └── state.py         # CustomModelingState（25+ 字段）
│   │   └── viz_custom_subgraph/       # Viz Custom 子图状态
│   │       ├── state.py             # CustomVizState
│   │       └── schemas.py             # Pydantic schemas
│   ├── graph/                   # LangGraph 工作流定义
│   │   ├── workflow.py                # 主工作流
│   │   ├── modeling_custom_workflow.py # Custom 建模子图
│   │   └── viz_custom_workflow.py     # Custom 可视化子图
│   ├── nodes/                      # 工作流节点
│   │   ├── auto_etl.py          # 自动 ETL 节点
│   │   ├── router.py                  # 路由节点
│   │   ├── modeling.py                # SOP 建模节点
│   │   ├── fetch_artifacts.py         # 产物收集节点
│   │   ├── viz.py             # 可视化规划节点
│   │   ├── summary.py                 # 总结节点
│   │   ├── modeling_custom_subgraph/  # Custom 建模子图节点
│   │   │   ├── planner.py             # 任务规划
│   │   │   ├── executor.py            # 代码生成（React-like）
│   │   │   ├── tool.py              # 沙盒执行
│   │   ├── observer.py            # 结果观察与决策
│   │   │   ├── replanner.py           # 动态重规划
│   │   │   └── aggregator.py      # 产物汇总
│   │   └── viz_custom_subgraph/       # Custom 可视化子图节点
│   │    ├── planner.py             # 可视化任务规划
│   │       ├── executor.py        # 按需提取数据
│   │       └── aggregator.py          # 汇总可视化结果
│   ├── agents/             # Agent 定义
│   │   └── viz_custom_agent.py        # Viz Custom ReAct Agent
│   ├── helpers/              # Helper 函数
│   │   └── modeling_helpers.py        # 主表维护函数
│   ├── tools/                   # LangChain 工具
│   │   ├── python_interpreter.py   # 沙盒代码执行工具
│   │   └── python_script_runner.py    # 脚本执行工具
│   ├── utils/                       # 工具函数
│   │   ├── llm_factory.py             # LLM 工厂函数
│   │   ├── file_parser.py             # 文件解析
│   │   └── extract_text_from_content.py # 内容提取
│   └── prompts/                # 提示词配置
│       ├── scenarios/                 # 场景提示词
│       │   ├── modeling_custom.yaml   # Custom 建模提示词
│       │   └── viz_custom.yaml        # Custom 可视化提示词
│       └── viz_charts/          # 图表提示词
│           ├── scatter.yaml
│           ├── line.yaml
│           └── ...
├── bi-template/              # PPIO 沙盒模板
│   └── sandbox_sdk/                   # 自定义 SDK
├── temp/               # 临时文件目录
├── main.py                # FastAPI 入口
├── .env.example            # 环境变量模板
├── CLAUDE.md              # Claude Code 指南
└── README.md                # 项目文档
```

## 核心设计理念

1. **状态驱动**：使用 LangGraph 的状态机模型，清晰定义数据流转
2. **模块化**：每个节点职责单一，易于测试和维护
3. **可扩展**：新增算法或图表类型只需添加配置文件
4. **安全隔离**：沙盒环境确保代码执行安全
5. **用户友好**：流式响应提供实时反馈
6. **智能控制**：Observer 模式支持动态调整和错误恢复

## 技术亮点总结

- ✅ **LangGraph v1.0 工作流编排**：状态机驱动的多 Agent 协作
- ✅ **双模式架构**：SOP 快速响应 + Custom 灵活分析
- ✅ **智能路由**：自动识别用户意图，选择最优路径
- ✅ **Observer 控制流**：CONTINUE/FOLLOW_UP/REPLAN/STOP 四种决策
- ✅ **React-like 模式**：手动管理 messages，精确控制执行流程
- ✅ **Helper 函数机制**：主表增量维护，减少内存占用
- ✅ **按需数据提取**：viz_custom 只提取需要的列，减少上下文
- ✅ **多模型支持**：Anthropic（主模型）+ OpenAI（FLASH）
- ✅ **LLM 工厂模式**：统一的模型创建和重试策略
- ✅ **LLM 驱动 ETL**：无需预定义规则的智能数据合并
- ✅ **请求级隔离**：独立沙盒环境，确保安全性
- ✅ **流式响应**：SSE 实时推送，提升用户体验
- ✅ **丰富的可视化**：11 种图表类型，覆盖常见场景

## 关键技术决策

### 为什么使用 React-like 而不是 create_agent()？

1. **精确控制**：手动管理 messages 数组，完全掌控执行流程
2. **状态持久化**：execution_trace 可以完整保存到 LangGraph state
3. **灵活路由**：Tool 节点可以根据执行结果灵活路由（成功/失败）
4. **Observer 集成**：需要在每次执行后插入 Observer 节点

### 为什么使用 Helper 函数而不是每次生成完整表？

1. **内存效率**：避免重复生成大表，减少沙盒内存占用
2. **增量分析**：支持逐步添加特征列，符合数据分析的自然流程
3. **文件管理**：自动维护注册表，方便下游节点使用


