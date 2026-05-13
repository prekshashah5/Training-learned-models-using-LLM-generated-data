"""
generate_and_plot.py

One-command helper to:
1) Generate queries using generation.query_generator.generate_all_queries
2) Plot structural graphs from the generated queries

Usage:
    python generate_and_plot.py 50
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

from evaluation.pipeline_graphs import (
    plot_joins_distribution,
    plot_predicates_distribution,
    plot_structural_features,
    plot_tables_distribution,
)
from generation.format_converter import parse_sql_to_mscn
from generation.query_generator import generate_all_queries


def choose_ollama_url(cli_url: str | None) -> str:
    """Pick Ollama URL from CLI override or environment."""
    if cli_url:
        return cli_url
    # Prefer OLLAMA_URL if set, then OLLAMA_HOST used by Ollama CLI setups.
    return (
        os.getenv("OLLAMA_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://localhost:11434"
    )


def load_text_file(path_str: str, kind: str) -> str:
    """Load a text file with a clear error for missing paths."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_graphs(queries: List[dict], output_dir: Path) -> None:
    """Generate structural query graphs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_tables_distribution(queries, str(output_dir))
    plot_joins_distribution(queries, str(output_dir))
    plot_predicates_distribution(queries, str(output_dir))
    plot_structural_features(queries, str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N queries with updated logic and plot structural graphs"
    )
    parser.add_argument("queries", type=int, help="Number of queries to generate")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model name (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Generation batch size (default: 10)",
    )
    parser.add_argument(
        "--schema-file",
        type=str,
        default="schema/IMDB_schema.txt",
        help="Path to schema DDL text file",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default="",
        help="Optional stats file path",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Override Ollama URL; otherwise use OLLAMA_URL/OLLAMA_HOST",
    )
    args = parser.parse_args()

    schema_text = load_text_file(args.schema_file, "Schema")
    stats_text = load_text_file(args.stats_file, "Stats") if args.stats_file else ""
    ollama_url = choose_ollama_url(args.ollama_url)

    print(f"[generate_and_plot] Generating {args.queries} queries using {args.model}")
    print(f"[generate_and_plot] Ollama URL: {ollama_url}")

    sql_queries = generate_all_queries(
        total_queries=args.queries,
        schema_text=schema_text,
        stats_text=stats_text,
        batch_size=args.batch_size,
        model_name=args.model,
        ollama_url=ollama_url,
    )

    structured_queries = []
    for sql in sql_queries:
        parsed = parse_sql_to_mscn(sql)
        if parsed is not None:
            structured_queries.append(parsed)

    if not structured_queries:
        raise RuntimeError("No plottable queries found after generation.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    graphs_dir = Path("generated_queries") / f"graphs_{timestamp}"
    build_graphs(structured_queries, graphs_dir)

    join_counts = Counter(len(q.get("joins", [])) for q in structured_queries)
    print(f"[generate_and_plot] Parsed queries for plotting: {len(structured_queries)}")
    print(f"[generate_and_plot] Join distribution: {dict(sorted(join_counts.items()))}")
    print(f"[generate_and_plot] Graphs saved to: {graphs_dir}")


if __name__ == "__main__":
    main()
