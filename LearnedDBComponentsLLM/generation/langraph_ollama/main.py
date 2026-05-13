"""
main.py
LangGraph pipeline entry point.
Builds and runs the directed graph pipeline for SQL query generation,
validation, fixing, and metrics computation.

Refactored imports to use unified package structure.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from generation.langraph_ollama.nodes import (
    init_node,
    generate_queries_node,
    calculate_rows_node,
    fix_queries_node,
    cleanup_sql_node,
    metrics_node,
)
from generation.langraph_ollama.state import PipelineState

# Load .env from project root
root = Path(__file__).resolve().parent.parent.parent
load_dotenv(root / ".env")

PIPELINE_MODE = "generate"

graph = StateGraph(PipelineState)

graph.add_node("init", init_node)
graph.add_node("generate", generate_queries_node)
graph.add_node("calculate_rows", calculate_rows_node)
graph.add_node("fix_queries", fix_queries_node)
graph.add_node("cleanup_sql", cleanup_sql_node)
graph.add_node("metrics", metrics_node)

graph.set_entry_point("init")

graph.add_edge("init", "generate")
graph.add_edge("generate", "calculate_rows")


# Conditional loop: allowed up to max_fix_iterations
def needs_fix(state: PipelineState):
    has_errors = state.get("has_exec_errors", False)
    current_iter = state.get("iteration_count", 0)
    max_iter = state.get("max_fix_iterations", 1)

    if has_errors and current_iter < max_iter:
        return "fix"
    return "metrics"


graph.add_conditional_edges(
    "calculate_rows",
    needs_fix,
    {
        "fix": "fix_queries",
        "metrics": "metrics",
    },
)

graph.add_edge("fix_queries", "cleanup_sql")
graph.add_edge("cleanup_sql", "calculate_rows")

graph.add_edge("metrics", END)

app = graph.compile()


if __name__ == "__main__":
    print(app.get_graph().draw_mermaid())

    initial_state: PipelineState = {
        "output_folder": Path(os.getenv("OUTPUT_FOLDER", "../output")),
        "mode": PIPELINE_MODE,
        "iteration_count": 0,
        "max_fix_iterations": int(os.getenv("MAX_FIX_ITERATIONS", 3)),
    }

    final_state = app.invoke(initial_state)

    session_dir = final_state.get("output_folder")
    if session_dir and session_dir.name.startswith("session_"):
        print(f"\n[info] Automatically running metrics comparison for {session_dir.name}...")
        import subprocess
        compare_script = root / "metrics" / "compare_models.py"
        subprocess.run([sys.executable, str(compare_script), session_dir.name])
