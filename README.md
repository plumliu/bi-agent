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

- **工作流编排**：LangGraph - 状态机驱动的 Agent 工作流
- **Agent 框架**：LangChain - ReAct Agent 实现
- **后端服务**：FastAPI - 异步 API 服务
- **代码执行**：PPIO Sandbox - 隔离的 Python 运行环境
- **LLM 能力**：OpenAI API
- **数据处理**：Pandas, NumPy, Scikit-learn

### 系统架构

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
    │    └─ Summary 节点（基于 SOP 路径结果生成总结）
    │        ↓
    │    SSE 流式响应
    │        ↓
    │    前端展示
    │
    └─ Custom 分支（自定义分析）
        ├─ Modeling Custom 子图
        │    ├─ Planner（任务分解）
        │    └─ Executor（ReAct 循环）
        ├─ Viz Custom 子图
        │    ├─ Viz Planner（生成可视化任务）
        │    ├─ Viz Executor（并发执行）
        │    └─ Viz Aggregator（汇总可视化结果）
        └─ Summary 节点（基于 Custom 路径结果生成总结）
            ↓
        SSE 流式响应
            ↓
        前端展示

Summary 是两条路径的“公共能力节点”（同名同职责），但分别在 SOP / Custom 分支内执行。并不代表承担跨分支汇总的职责。

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

系统通过 Router 节点的条件路由，将请求分发到两条独立的处理路径：

#### SOP 模式（标准算法路径）

router → modeling → fetch_artifacts → viz → summary → END


- **modeling**：ReAct调用预定义算法工具（如聚类、异常检测）
- **fetch_artifacts**：从沙盒下载生成的数据文件
- **viz**：LLM 生成 ECharts 配置
- **summary**：总结分析结果

**特点**：
- 单次执行，效率高
- 结果标准化，易于可视化
- 适用场景：聚类分析、异常检测、时序分解等

#### Custom 模式（自定义分析路径）

router → modeling_custom 子图 → viz_custom 子图 → summary → END

**modeling_custom 子图**（任务规划与执行）：
planner → executor → END

- **planner**：将用户需求分解为 2-7 个子任务
- **executor**：ReAct Agent 逐个执行任务（内部循环调用沙盒工具）

**viz_custom 子图**（并发可视化）：
viz_planner → [viz_executor_1, viz_executor_2, ...] → viz_aggregator → END
              (并发执行，使用 LangGraph Send API)

- **viz_planner**：根据建模产物生成多个可视化任务
- **viz_executor**：并发执行每个可视化任务
- **viz_aggregator**：汇总所有图表配置

**特点**：
- 支持复杂的多步骤分析
- Executor 内部的 ReAct 循环支持动态调整和错误自纠正
- 可视化任务并发执行，提升效率
- 不流经 fetch_artifacts，直接在沙盒环境中生成的中间文件进行可视化，实现零拷贝


### 3. LLM 驱动的自动 ETL

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

### 4. 请求级沙盒隔离

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

### 5. 流式响应与实时反馈

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

## 支持的可视化图表

系统支持 11 种图表类型，覆盖常见的数据分析场景：

- **scatter**：散点图（聚类结果、分布分析）
- **line**：折线图（趋势分析、时序数据）
- **bar**：柱状图（对比分析、分类统计）
- **pie**：饼图（占比分析）
- **radar**：雷达图（多维度对比）
- **line_with_confidence**：带置信区间的折线图（预测结果）
- **boxplot**：箱线图（异常检测、分布分析）
- **line_with_error_bars**：带误差棒的折线图（实验数据）
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
# - OPENAI_API_KEY / OPENAI_API_BASE：主 LLM（用于建模）
# - OPENAI_API_KEY_FLASH / OPENAI_API_BASE_FLASH：快速 LLM（用于路由）
# - LLM_MODEL_NAME / LLM_FLASH_MODEL_NAME：模型标识符
# - PPIO_API_KEY / PPIO_TEMPLATE：沙盒环境凭证
```

### 启动服务

```bash
# 启动 FastAPI 服务（热重载模式）
uvicorn main:app --host 0.0.0.0 --port 8009 --reload

# 或直接运行
python main.py
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
│   ├── core/                          # 核心配置
│   │   ├── config.py                  # 环境变量配置
│   │   ├── state.py                   # 主工作流状态定义
│   │   ├── prompts_config.py          # 提示词加载器
│   │   ├── modeling_custom_subgraph/  # Custom 子图状态
│   │   └── viz_custom_subgraph/       # Viz Custom 子图状态
│   ├── graph/                         # LangGraph 工作流定义
│   │   ├── workflow.py                # 主工作流
│   │   ├── modeling_custom_workflow.py # Custom 建模子图
│   │   └── viz_custom_workflow.py     # Custom 可视化子图
│   ├── nodes/                         # 工作流节点
│   │   ├── auto_etl.py                # 自动 ETL 节点
│   │   ├── router.py                  # 路由节点
│   │   ├── modeling.py                # SOP 建模节点
│   │   ├── fetch_artifacts.py         # 产物收集节点
│   │   ├── viz.py                     # 可视化规划节点
│   │   ├── viz_execution.py           # 可视化执行节点
│   │   ├── summary.py                 # 总结节点
│   │   ├── modeling_custom_subgraph/  # Custom 建模子图节点
│   │   │   ├── planner.py             # 任务规划节点
│   │   │   └── executor.py            # 任务执行节点
│   │   └── viz_custom_subgraph/       # Custom 可视化子图节点
│   ├── agents/                        # Agent 定义
│   │   └── modeling_custom_agent.py   # Custom 建模 ReAct Agent
│   ├── tools/                         # LangChain 工具
│   │   └── python_interpreter.py      # 沙盒代码执行工具
│   ├── prompts/                       # 提示词配置
│   │   ├── scenarios/                 # 场景提示词
│   │   │   ├── modeling_custom.yaml   # Custom 建模提示词
│   │   │   └── ...
│   │   └── viz_charts/                # 图表提示词
│   │       ├── scatter.yaml
│   │       ├── line.yaml
│   │       ├── boxplot.yaml
│   │       └── ...
│   └── utils/                         # 工具函数
│       ├── file_parser.py             # 文件解析
│       └── alias_generator.py         # 语义别名生成
├── bi-template/                       # PPIO 沙盒模板
│   └── sandbox_sdk/                   # 自定义 SDK
├── temp/                              # 临时文件目录
├── main.py                            # FastAPI 入口
├── .env.example                       # 环境变量模板
└── README.md                          # 项目文档
```

## 核心设计理念

1. **状态驱动**：使用 LangGraph 的状态机模型，清晰定义数据流转
2. **模块化**：每个节点职责单一，易于测试和维护
3. **可扩展**：新增算法或图表类型只需添加配置文件
4. **安全隔离**：沙盒环境确保代码执行安全
5. **用户友好**：流式响应提供实时反馈

## 技术亮点总结

- ✅ **LangGraph 工作流编排**：状态机驱动的多 Agent 协作
- ✅ **双模式架构**：SOP 快速响应 + Custom 灵活分析
- ✅ **智能路由**：自动识别用户意图，选择最优路径
- ✅ **ReAct Agent**：支持复杂推理和错误自纠正
- ✅ **LLM 驱动 ETL**：无需预定义规则的智能数据合并
- ✅ **请求级隔离**：独立沙盒环境，确保安全性
- ✅ **流式响应**：SSE 实时推送，提升用户体验
- ✅ **丰富的可视化**：11 种图表类型，覆盖常见场景

---

**项目状态**：✅ 基本完工

**开发团队**：BI Agent Team

**技术支持**：详见 CLAUDE.md
