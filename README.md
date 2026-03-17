# BI Agent (Modeling-Only)

A LangGraph-based BI analysis service focused on modeling workflows only.

## Architecture

Workflow is fixed to:

`START -> profiler -> router -> modeling_custom -> summary -> END`

- No SOP branch
- No visualization branch
- No sandbox runtime
- No legacy artifact registry protocol

## Runtime Model

- Uploaded files are stored under:
  - `/Users/plumliu/Desktop/python_workspace/agent_workspace/sessions/{session_id}`
- Each request starts an isolated local IPython kernel bound to:
  - `/Users/plumliu/Desktop/python_workspace/agent_workspace/.venv/bin/python`
- Kernel is shut down at request end
- Session directory files are preserved

## API

`POST /query_agents_stream`

SSE event types include:
- `log`
- `log_stream`
- `tool_calling`
- `tool_log`
- `summary`
- `done`
- `error`

`done` payload keeps `echarts` for compatibility, always as `[]`.

## Environment Variables

See `.env.example`.

Key local-runtime variables:
- `AGENT_WORKSPACE_DIR`
- `AGENT_WORKSPACE_SESSIONS_DIR`
- `AGENT_WORKSPACE_PYTHON`

## Development

```bash
uv sync
python main.py
```
