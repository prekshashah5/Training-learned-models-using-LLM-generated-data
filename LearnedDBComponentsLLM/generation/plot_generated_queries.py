"""
plot_generated_queries.py
Generate structural query graphs from generated query files without running
the full training pipeline.

Supports two input styles:
1. Structured queries already containing tables/joins/predicates
2. Raw generated query files containing SQL strings under a "sql" field
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.pipeline_graphs import (
    plot_joins_distribution,
    plot_predicates_distribution,
    plot_structural_features,
    plot_tables_distribution,
)
from generation.format_converter import parse_sql_to_mscn
from utils.io_utils import read_json_file
from utils.session_utils import get_latest_json_path


def load_queries(input_path: Optional[str], output_folder: Optional[str]) -> tuple[List[Dict], Path, Path]:
    """Load queries from a direct file path or from the latest generation session."""
    if input_path:
        query_path = Path(input_path)
    elif output_folder:
        query_path = get_latest_json_path(Path(output_folder))
    else:
        raise ValueError("Provide either --input or --output-folder")

    if not query_path.exists():
        raise FileNotFoundError(f"Could not find query file: {query_path}")

    raw_queries = read_json_file(str(query_path))
    if not isinstance(raw_queries, list):
        raise ValueError(f"Expected a list of queries in {query_path}")

    run_dir = query_path.parent
    plots_dir = run_dir / "graphs_generation_only"
    return raw_queries, query_path, plots_dir


def normalize_queries(raw_queries: List[Dict]) -> List[Dict]:
    """Convert mixed query records into the structure expected by pipeline_graphs."""
    normalized = []

    for item in raw_queries:
        if isinstance(item, str):
            parsed = parse_sql_to_mscn(item)
            if parsed is None:
                continue
            normalized.append({
                "tables": parsed.get("tables", []),
                "joins": parsed.get("joins", []),
                "predicates": parsed.get("predicates", []),
            })
            continue

        if not isinstance(item, dict):
            continue

        if all(key in item for key in ("tables", "joins", "predicates")):
            normalized.append({
                "tables": item.get("tables", []),
                "joins": item.get("joins", []),
                "predicates": item.get("predicates", []),
            })
            continue

        sql = item.get("sql")
        if not sql:
            continue

        parsed = parse_sql_to_mscn(sql)
        if parsed is None:
            continue

        normalized.append({
            "tables": parsed.get("tables", []),
            "joins": parsed.get("joins", []),
            "predicates": parsed.get("predicates", []),
        })

    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Generate structural graphs from generated query files only"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a query JSON/JSONL file",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Generation output folder; the latest queries.jsonl will be used",
    )
    args = parser.parse_args()

    raw_queries, query_path, plots_dir = load_queries(args.input, args.output_folder)
    queries = normalize_queries(raw_queries)

    if not queries:
        raise RuntimeError(
            "No plottable queries found. Expected structured queries or records with a sql field."
        )

    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(queries)} queries from {query_path}")
    print(f"Saving graphs to {plots_dir}")

    plot_tables_distribution(queries, str(plots_dir))
    plot_joins_distribution(queries, str(plots_dir))
    plot_predicates_distribution(queries, str(plots_dir))
    plot_structural_features(queries, str(plots_dir))

    print("Generated structural query graphs successfully.")


if __name__ == "__main__":
    main()