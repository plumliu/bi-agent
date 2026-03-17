from typing import Literal

from pydantic import BaseModel, Field

from app.core.state import WorkflowState

ScenarioType = Literal["custom", "unknown"]


class RouterOutput(BaseModel):
    scenario: ScenarioType = Field(..., description="Routing decision")
    reasoning: str = Field(..., description="Reason for routing")


def router_node(state: WorkflowState) -> dict:
    """Route to custom by default. Empty/invalid input routes to unknown."""
    user_input = state.get("user_input")

    if not isinstance(user_input, str) or not user_input.strip():
        result = RouterOutput(
            scenario="unknown",
            reasoning="user_input is empty or invalid",
        )
    else:
        result = RouterOutput(
            scenario="custom",
            reasoning="all supported analysis runs through modeling_custom",
        )

    print(f"--- [Router] 场景: {result.scenario} | 理由: {result.reasoning} ---")
    return result.model_dump()
