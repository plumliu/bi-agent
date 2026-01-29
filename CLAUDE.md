# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BI Agent** is a Python-based intelligent business intelligence system that uses a multi-agent workflow architecture powered by LangGraph and OpenAI-compatible APIs (supports Alibaba Dashscope/Qwen models). It provides automated data analysis and visualization capabilities with support for multiple analysis scenarios.

## Development Commands

```bash
# Setup and installation
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the development server
python main.py
# Server starts on http://0.0.0.0:8009

# Run specific test
python test_subgraph.py
```

## Architecture

The system follows a **multi-agent workflow** pattern using LangGraph with two main execution paths:

### Main Workflow (Standard Scenarios)

```
Start → Router → Modeling → Tools/Fetch → Viz → Viz Execution → Summary → End
```

**Key Components:**
- **Router** (`app/nodes/router.py`): Analyzes user input and data schema to determine analysis scenario
- **Modeling** (`app/nodes/modeling.py`): Generates and executes analysis code using LLM tools
- **Fetch Artifacts** (`app/nodes/fetch_artifacts.py`): Retrieves generated files from sandbox
- **Viz** (`app/nodes/viz.py`): Generates ECharts visualization configuration
- **Viz Execution** (`app/nodes/viz_execution.py`): Validates visualization code
- **Summary** (`app/nodes/summary.py`): Creates final analysis summary

### Custom Subgraph (Complex Analysis)

For the "custom" scenario, a separate subgraph handles iterative planning-execution-reflection:

```
Planner → Executor → Reflector ─┐
         ↑          │            │
         └──────────┴────────────┘
```

**Subgraph Nodes** (`app/nodes/subgraph/`):
- **Planner**: Creates task lists (5-7 tasks)
- **Executor**: Executes Python code in sandbox
- **Reflector**: Reviews results, manages retries/insertions

See `app/graph/modeling_custom_workflow.py` for subgraph routing logic.

## Scenario System

Scenarios are configured via YAML files in `app/prompts/scenarios/` with naming pattern `{step}_{scenario}.yaml`:

**Supported Scenarios:**
- `clustering`: Customer segmentation, pattern finding
- `anomaly`: Fraud detection, outlier identification
- `decomposition`: Time series decomposition, trend analysis
- `association`: Relationship mining between variables
- `forecast`: Future value prediction
- `classification`: Category prediction
- `custom`: Multi-step complex analysis (uses subgraph)

**Configuration Structure:**
- `modeling_{scenario}.yaml`: Analysis instructions and code examples
- `viz_{scenario}.yaml`: Visualization generation prompts
- `summary_{scenario}.yaml`: Final summary instructions

The `custom` scenario uses `app/prompts/scenarios/modeling_custom.yaml` for its planner prompts.

## State Management

**AgentState** (`app/core/state.py`): Main state object passed between all nodes
- `messages`: Conversation history (LangChain messages)
- `user_input`: Original user query
- `data_schema`: Structured dict with column names, dtypes, summary
- `remote_file_path`: Path to CSV in sandbox (`/home/user/data.csv`)
- `scenario`: Determined analysis scenario
- `modeling_artifacts`: Analysis outputs (e.g., cluster assignments)
- `modeling_summary`: Text summary of analysis
- `viz_config`: ECharts configuration JSON
- `viz_success`: Boolean flag for visualization retry loop
- `final_summary`: Final conclusion
- `error_count`: Error tracking

**CustomModelingState** (`app/core/subgraph/state.py`): State for custom subgraph
- `plan`: List of PlanItem tasks
- `current_task_index`: Current task pointer
- `scratchpad`: Accumulated findings

## Sandbox Integration

Uses **PPIO Sandbox** (not E2B) for secure code execution via `ppio_sandbox.code_interpreter.Sandbox`.

**Lifecycle** (managed in `main.py`):
1. Created at FastAPI startup via `lifespan()` context manager
2. One global instance shared across requests (with async lock)
3. Cleaned and reset before each request (`%reset -f`, file removal)
4. CSV uploaded to `/home/user/data.csv`
5. Killed at server shutdown

**Tool** (`app/tools/sandbox.py`): `create_code_interpreter_tool()` binds a sandbox instance to a LangChain tool for LLM code execution.

## API Endpoints

- **`POST /query_agents_stream`**: SSE streaming - real-time updates with event types:
  - `log`: Progress messages (cleaned via `clean_log_content()`)
  - `viz_data`: ECharts configuration
  - `summary`: Final analysis text
  - `done`: Complete result object
  - `error`: Error messages

Both endpoints accept:
- `file`: CSV file upload (multipart/form-data)
- `request_data`: JSON with `query` (string) and optional `scenario` (string)

## Configuration

Copy `.env.example` to `.env`:

```
OPENAI_API_KEY=sk-xxx-your-api-key-here-xxx
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen3-coder-plus
LLM_FLASH_MODEL_NAME=qwen-flash
PPIO_API_KEY=your-ppio-api-key
PPIO_TEMPLATE=your-ppio-template
```

Note: `E2B_API_KEY` is not used (the project migrated to PPIO).

## Router Output Structure

The router uses Pydantic with `Literal` types to prevent hallucination:

```python
ScenarioType = Literal[
    "clustering", "anomaly", "decomposition",
    "association", "forecast", "classification",
    "unknown", "custom"
]
```

## Working with Prompts

To add or modify a scenario:
1. Create YAML files in `app/prompts/scenarios/` with proper naming
2. Use `load_prompts_config(step, scenario)` in nodes to load configs
3. Add scenario to `ScenarioType` Literal in `app/nodes/router.py`
4. Update routing logic if needed

## Key Files

- `main.py`: FastAPI app, sandbox lifecycle, API endpoints
- `app/graph/workflow.py`: Main workflow graph construction
- `app/graph/modeling_custom_workflow.py`: Custom subgraph for iterative analysis
- `app/core/state.py`: Main state definitions
- `app/core/prompts_config.py`: YAML config loader
- `app/nodes/router.py`: Intent classification
- `app/tools/sandbox.py`: Code execution tool factory
