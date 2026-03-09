# DABench 评估实验

用于测试 modeling_custom 子图在 DABench 上的表现。

## 评估指标

本实验使用 DABench 官方评估指标：

1. **ABQ (Accuracy by Questions)** - 问题级准确率：所有子问题都正确才算正确
2. **PSAQ (Proportional by Sub-Question)** - 按子问题比例加权：每个问题得分 = 正确子问题数/总子问题数
3. **UASQ (Accuracy by Sub-Question)** - 子问题级准确率：每个子问题独立计算准确率

详见官方论文和 `official_implementation(reference)/README_eval.md`

## 架构说明

本实验框架直接测试 modeling_custom 子图，不经过完整的 workflow pipeline：

```
文件上传 → auto_etl → modeling_custom 子图 → 结果保存为 JSONL → 官方脚本评估
```

**跳过的组件**：
- router 节点（直接设置 scenario="custom"）
- viz_custom 节点
- summary 节点

**使用的组件**：
- auto_etl：数据整合和 schema 生成
- modeling_custom 子图：核心测试目标（planner + executor）

## 目录结构

```
experiment/modeling_custom/da-bench/
├── README.md                              # 本文档
├── config/
│   └── eval_config.yaml                   # 评估配置
├── src/
│   ├── __init__.py
│   ├── utils.py                           # 辅助函数
│   ├── sandbox_manager.py                 # Sandbox 池管理
│   └── runner.py                          # 核心运行器
├── scripts/
│   ├── run_eval.py                        # 生成 responses.jsonl
│   └── calculate_metrics.py               # 计算评估指标
├── results/
│   ├── responses_TIMESTAMP.jsonl          # 模型输出结果
│   └── metrics/
│       └── responses_TIMESTAMP_metrics.json  # 评估指标
├── official_implementation(reference)/
│   ├── eval_closed_form.py                # 官方评估脚本
│   └── README_eval.md                     # 官方评估说明
└── test_debug.py                          # 调试脚本
```

## 使用方法

### 1. 环境准备

确保已安装项目依赖：
```bash
cd /Users/plumliu/Desktop/python_workspace/bi-agent
uv sync
```

### 2. 运行评估（生成 responses.jsonl）

```bash
cd experiment/modeling_custom/da-bench

# 测试模式（只评估前5个问题）
python scripts/run_eval.py --limit 5

# 完整评估（257个问题）
python scripts/run_eval.py

# 自定义配置
python scripts/run_eval.py \
  --benchmark-dir /path/to/infiagent-DABench \
  --output results/my_responses.jsonl \
  --max-concurrent 30
```

### 3. 计算评估指标

```bash
# 使用我们的评估脚本（推荐）
python scripts/calculate_metrics.py \
  --questions_file_path ../../../../infiagent-DABench/da-dev-questions.jsonl \
  --labels_file_path ../../../../infiagent-DABench/da-dev-labels.jsonl \
  --responses_file_path results/responses_20260308_123456.jsonl

# 或使用官方评估脚本（需要额外依赖）
python official_implementation\(reference\)/eval_closed_form.py \
  --questions_file_path ../../../../infiagent-DABench/da-dev-questions.jsonl \
  --labels_file_path ../../../../infiagent-DABench/da-dev-labels.jsonl \
  --responses_file_path results/responses_20260308_123456.jsonl
```

输出示例：
```
================================================================================
DABench 评估结果
================================================================================
ABQ  (Accuracy by Question):                    59.75%
PSAQ (Proportional by Sub-Question):            66.05%
UASQ (Accuracy by Sub-Question):                65.26%
================================================================================

详细结果已保存至: results/metrics/responses_20260308_123456_metrics.json
```

## 技术细节

### Sandbox 管理
- 每个问题使用独立的 sandbox 实例（完全隔离）
- 默认最多并发 45 个 sandbox（可通过 `--max-concurrent` 调整）
- 自动创建和清理，避免资源泄漏

### 错误处理
- 遇到错误记录并继续评估
- 不会中断整个评估流程
- 失败的问题输出空字符串

### 输出格式
- 结果保存为 JSONL 格式（每行一个 JSON 对象）
- 格式：`{"id": 0, "response": "@mean_fare[34.65]"}`
- 与官方评估脚本完全兼容

### Prompt 修改
本实验使用修改后的 executor prompt（`app/prompts/scenarios/modeling_custom.yaml`）：
- 原版本：输出自然语言总结 + JSON 产物清单
- 修改版本：直接输出 `@key[value]` 格式的答案
- 备份文件：`modeling_custom.yaml.backup`

## 注意事项

1. **Sandbox 配额**：确保 PPIO 账户有足够的 sandbox 配额（至少 50 个）
2. **并发控制**：默认 45 个并发，可通过 `--max-concurrent` 调整
3. **超时设置**：单个 sandbox 超时 3600 秒（1小时）
4. **结果保存**：responses.jsonl 包含每个问题的 ID 和模型输出
5. **评估分离**：先生成 responses.jsonl，再用官方脚本评估（避免重复运行模型）

## 参考

- DABench 官方仓库: https://github.com/InfiAgent/InfiAgent
- 官方评估脚本: `official_implementation(reference)/eval_closed_form.py`
- 评估指标说明: `official_implementation(reference)/README_eval.md`
