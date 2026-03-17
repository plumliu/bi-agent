from typing import Any, Dict

from app.core.modeling_custom_subgraph.state import CustomModelingState


def create_modeling_aggregator_node():
    """Aggregate custom modeling execution into a plain modeling_summary."""

    def modeling_aggregator_node(state: CustomModelingState) -> Dict[str, Any]:
        print("--- [Aggregator] 聚合结果 ---")

        completed_tasks = state.get("completed_tasks") or []
        confirmed_findings = state.get("confirmed_findings") or []
        observer_history = state.get("observer_history") or []
        stop_reason = state.get("stop_reason", "")

        summary_parts = []
        if stop_reason:
            summary_parts.append(f"Stop reason: {stop_reason}\n")

        summary_parts.append("Completed tasks:")
        if completed_tasks:
            for task in completed_tasks:
                summary_parts.append(f"- {task['description']}")
        else:
            summary_parts.append("- none")

        summary_parts.append("\nConfirmed findings:")
        if confirmed_findings:
            for finding in confirmed_findings:
                summary_parts.append(f"- {finding}")
        else:
            summary_parts.append("- none")

        summary_parts.append("\nObserver history:")
        if observer_history:
            for item in observer_history:
                summary_parts.append(f"- {item}")
        else:
            summary_parts.append("- none")

        modeling_summary = "\n".join(summary_parts)
        return {"modeling_summary": modeling_summary}

    return modeling_aggregator_node
