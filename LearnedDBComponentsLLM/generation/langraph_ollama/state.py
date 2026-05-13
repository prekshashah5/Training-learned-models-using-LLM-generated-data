"""
state.py
LangGraph pipeline state definition.
"""

from typing import TypedDict, List, Optional, Dict, Any
from pathlib import Path


class PipelineState(TypedDict, total=False):
    # config
    mode: str
    output_folder: Path
    schema_text: str

    # data
    queries: List[Dict[str, Any]]
    latest_json_path: Optional[Path]

    # control flags
    has_exec_errors: bool
    skip_generation: bool
    iteration_count: int
    max_fix_iterations: int
