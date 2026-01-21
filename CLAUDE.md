# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BI Agent is a multi-agent Business Intelligence system that uses LangGraph orchestration to analyze CSV data and automatically generate visualizations. The system accepts user queries, intelligently routes them to appropriate analysis algorithms, executes code in a secure sandbox, and returns ECharts configurations.

## Development Commands

```bash
# Install dependencies (uses UV package manager)
uv sync

# Run development server (auto-reload)
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8011 --reload
```

The server runs on port 8011 by default.

## Architecture

### Core Workflow (LangGraph)

The system uses a directed graph of agent nodes in `app/graph/workflow.py`:

```
START -> Router -> Modeling -> Tools/Fetch -> Viz Config -> Viz Execution -> Summary -> END
                (loop back if viz fails)
```

**Flow breakdown:**
1. **Router Node** (`app/nodes/router.py`): Analyzes user query + data schema, determines analysis scenario (clustering, anomaly, trend, forecast, classification, association)
2. **Modeling Node** (`app/nodes/modeling.py`): Executes ML algorithms using the python_interpreter tool
3. **Tools Node** (LangGraph ToolNode): Executes code in E2B sandbox
4. **Fetch Artifacts** (`app/nodes/fetch_artifacts.py`): Downloads processed data from sandbox
5. **Viz Config** (`app/nodes/viz.py`): Generates ECharts configuration
6. **Viz Execution** (`app/nodes/viz_execution.py`): Runs visualization generator, loops back on failure
7. **Summary** (`app/nodes/summary.py`): Creates final explanation

### State Management

`AgentState` in `app/core/state.py` is a TypedDict that flows through all nodes:
- `messages`: Conversation history
- `user_input`: Original query
- `data_schema`: CSV column metadata
- `remote_file_path`: Path in E2B sandbox
- `scenario`: Analysis type (clustering, anomaly, etc.)
- `modeling_artifacts`: JSON metrics from analysis
- `viz_config`: ECharts configuration
- `viz_success`: Boolean flag for control flow
- `final_summary`: Generated text explanation

### Sandbox Pattern

The E2B sandbox lifecycle is managed in `main.py`:
- Single global sandbox instance created at startup (lifespan manager)
- Pre-warmed with pandas, scikit-learn, pyarrow
- Concurrent requests use `asyncio.Lock()` for exclusive access
- Each workflow execution: clears old data, uploads new CSV, runs analysis

### Scenario Configuration

Each analysis scenario has YAML configs in `app/prompts/scenarios/`:
- `modeling_*.yaml`: Instructions for the Modeling agent (EDA, algorithm selection, output format)
- `viz_*.yaml`: Visualization configuration templates
- `summary_*.yaml`: Summary generation prompts

The YAML files define the data contract between agents. For example, clustering expects:
- Processed feather file with `pca_x`, `pca_y`, `cluster_label` columns
- JSON artifacts with `k_value`, `silhouette_score`, `cluster_counts`, `centroids`

### Visualization System

`app/tools/viz/generator.py` contains handler functions for each chart type:
- `scatter`: PCA visualization with grouping
- `radar`: Cluster centroids comparison
- `bar`: Cluster counts or feature statistics
- `pie`: Distribution overview
- `boxplot`: Feature distribution with outliers (anomaly detection)

Each handler takes DataFrame + config + artifacts, returns ECharts-compatible JSON.

## API Endpoints

**POST /query_agents** (synchronous)
- Upload: CSV file + JSON `{"query": "...", "scenario": "..."}`
- Returns: `{"success": true, "message": "...", "echarts": [...]}`

**POST /query_agents_stream** (SSE streaming)
- Same input
- Yields events: `log`, `viz_data`, `summary`, `done`, `error`
- Format: `data: {"type": "...", "data": ...}\n\n`

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=sk-... (Aliyun DashScope)
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen3-coder-plus
LLM_FLASH_MODEL_NAME=qwen-flash
E2B_API_KEY=e2b_...
```

## Key Patterns

**Adding a new scenario:**
1. Add scenario type to `ScenarioType` literal in `app/nodes/router.py`
2. Create 3 YAML files in `app/prompts/scenarios/`
3. Add handler in `app/prompts/scenarios/__init__.py` if needed
4. Update router prompt to recognize the scenario

**Chart handler contract:**
```python
def handle_chart(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    # Return chart-specific dict that will be wrapped as:
    # {"type": "chart_name", "data": <returned dict>}
```

**Modeling agent output contract:**
- Must save processed data to `/home/user/processed_data.feather`
- Must save artifacts to `/home/user/analysis_artifacts.json`
- For clustering: requires `pca_x`, `pca_y`, `cluster_label` columns
- Labels must start from 1, not 0

## File Structure Notes

- `temp/`: Local storage for uploaded CSV and generated visualization JSON
- `app/api/`: Minimal (routes defined directly in main.py)
- Code comments are in Chinese
- Router uses cheaper/faster model (qwen-flash), other agents use qwen3-coder-plus