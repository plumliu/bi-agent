from typing import TypedDict, List, Optional, Any, Dict


class WorkflowState(TypedDict):
    """Main workflow state shared across profiler/router/modeling_custom/summary."""

    # User input
    user_input: str

    # Uploaded file paths inside current local session directory
    raw_file_paths: List[str]

    # Original user-facing names for each uploaded fragment
    original_filenames: List[str]

    # Local paths used for profiler metadata extraction
    local_file_paths: List[str]

    # Profiler outputs
    files_metadata: List[Dict[str, Any]]
    merge_recommendations: Optional[List[Dict[str, Any]]]

    # Router outputs
    scenario: Optional[str]
    reasoning: Optional[str]

    # Modeling output
    modeling_summary: Optional[str]

    # Summary output
    final_summary: Optional[str]

    # Control
    error_count: int
