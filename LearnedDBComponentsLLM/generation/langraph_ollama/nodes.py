"""
nodes.py
LangGraph pipeline node functions.
Refactored imports to use unified package structure.
"""

import os
from generation.langraph_ollama.state import PipelineState
from generation.langraph_ollama.generate_queries import run_generation, get_schema
from generation.langraph_ollama.calculate import execute_queries
from generation.langraph_ollama.fix_queries import fix_queries_in_place, cleanup_sql
from utils.session_utils import get_latest_json_path
from utils.io_utils import read_json_file


def init_node(state: PipelineState) -> PipelineState:
    mode = state.get("mode", "load")

    if mode == "load":
        output_folder = state["output_folder"]
        json_path = get_latest_json_path(output_folder)
        queries = read_json_file(json_path)

        return {
            **state,
            "queries": queries,
            "latest_json_path": json_path,
            "skip_generation": True,
        }

    return {
        **state,
        "skip_generation": False,
    }


def generate_queries_node(state: PipelineState) -> PipelineState:
    if state.get("skip_generation"):
        return state
    queries, json_path, output_folder = run_generation()
    return {
        **state,
        "queries": queries,
        "latest_json_path": json_path,
        "output_folder": output_folder,
    }


def load_existing_queries_node(state: PipelineState) -> PipelineState:
    output_folder = state["output_folder"]
    json_path = get_latest_json_path(output_folder)
    queries = read_json_file(json_path)

    return {
        **state,
        "queries": queries,
        "latest_json_path": json_path,
    }


def calculate_rows_node(state: PipelineState) -> PipelineState:
    return {
        **state,
        "queries": state["queries"],
        "has_exec_errors": False,
    }


def fix_queries_node(state: PipelineState) -> PipelineState:
    state["schema_text"] = get_schema()
    fix_queries_in_place(
        queries=state["queries"],
        json_path=state["latest_json_path"],
        schema=state["schema_text"],
        model_name=os.getenv("FIX_MODEL", "llama3"),
        temperature=float(os.getenv("FIX_TEMPERATURE", 0.0)),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    )

    return {
        **state,
        "fix_attempted": True,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def cleanup_sql_node(state: PipelineState) -> PipelineState:
    cleanup_sql(state["latest_json_path"])
    return state


def metrics_node(state: PipelineState) -> PipelineState:
    from metrics.main import (
        validty_pipeline,
        run_complexity_pipeline,
        run_selective_non_selective_pipeline,
    )

    validty_pipeline()
    run_complexity_pipeline(recompute=True)
    run_selective_non_selective_pipeline()

    return state
