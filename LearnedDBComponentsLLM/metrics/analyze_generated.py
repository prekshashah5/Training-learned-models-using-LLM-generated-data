import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent dir to path

from utils.session_utils import get_latest_json_path
from utils.io_utils import read_json_file
from metrics.SQL_Complexity import compute_complexity_matrix, clean_df_before_complexity
from metrics.plotting import (
    plot_tables_distribution,
    plot_joins_distribution,
    plot_complexity_score_distribution,
    plot_complexity_distribution,
    plot_type_vs_complexity,
    plot_explain_vs_execution_per_query,
    plot_q_error_distribution,
    plot_q_error_comparison,
    plot_execution_time_comparison,
    plot_selective_vs_non_selective_count,
    plot_predicates_distribution,
    plot_column_usage_frequency,
    plot_structural_features
)

def assign_complexity_bucket(score):
    if score < 10:
        return "0-10"
    elif score < 20:
        return "10-20"
    elif score < 30:
        return "20-30"
    elif score < 40:
        return "30-40"
    elif score < 50:
        return "40-50"
    else:
        return "50+"

def get_selectivity_class(rows):
    if rows is None:
        return None
    if rows <= 50:
        return "high"
    elif rows <= 10000:
        return "medium"
    else:
        return "low"

def main():
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    output_folder = project_root / "output"
    plots_dir = output_folder / "plots" / "analysis"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    try:
        gen_file = get_latest_json_path(output_folder)
        print(f"Loading queries from {gen_file}...")
        queries = read_json_file(gen_file)
    except Exception as e:
        print(f"Error loading queries: {e}")
        return
    
    if not queries:
        print("No queries found.")
        return

    # 2. Pre-process Query Data (Complexity & Selectivity)
    print("Computing metrics...")
    df = pd.DataFrame(queries)
    
    # Compute Structural Complexity (Joins, Tables, Predicates, ComplexityScore, UsedColumns)
    if "Joins" not in df.columns or "Tables" not in df.columns or "UsedColumns" not in df.columns:
        print("Running SQL Complexity analysis...")
        df = clean_df_before_complexity(df)
        df = compute_complexity_matrix(df)
    
    # Assign Complexity Buckets
    if "ComplexityScore" in df.columns:
        df["ComplexityBucket"] = df["ComplexityScore"].apply(assign_complexity_bucket)

    # Assign Selectivity Class
    # Check for row count keys: exec_row_count -> rows -> actual_cardinality
    def get_rows(row):
        if pd.notnull(row.get("exec_row_count")): return row["exec_row_count"]
        if pd.notnull(row.get("rows")): return row["rows"]
        if pd.notnull(row.get("actual_cardinality")): return row["actual_cardinality"]
        return None

    df["rows_for_class"] = df.apply(get_rows, axis=1)
    df["selectivity_class"] = df["rows_for_class"].apply(get_selectivity_class)

    # Convert back to list of dicts for plotting functions that expect it
    # (Updated queries list with new fields)
    full_queries = df.to_dict(orient="records")
    
    # Split Selective / Non-Selective
    selective = [q for q in full_queries if q.get("selectivity_class") == "high"]
    non_selective = [q for q in full_queries if q.get("selectivity_class") == "low"]

    # DEBUG: Log Data Summary
    print(f"\n[DEBUG] Total Queries: {len(full_queries)}")
    print(f"[DEBUG] Selective Queries: {len(selective)}")
    print(f"[DEBUG] Non-Selective Queries: {len(non_selective)}")
    
    # Check for execution data presence
    # Filter None and NaN
    exec_times = [
        q.get("exec_time_ms") 
        for q in full_queries 
        if q.get("exec_time_ms") is not None and not pd.isna(q.get("exec_time_ms"))
    ]
    q_errors = [q.get("q_error") for q in full_queries if q.get("q_error") is not None]
    
    print(f"[DEBUG] Queries with 'exec_time_ms': {len(exec_times)}")
    if exec_times:
        print(f"[DEBUG] Execution Time Range: {min(exec_times)}ms - {max(exec_times)}ms")
        print(f"[DEBUG] Avg Exec Time: {sum(exec_times)/len(exec_times):.2f}ms")
    
    print(f"[DEBUG] Queries with 'q_error': {len(q_errors)}")
    if q_errors:
        print(f"[DEBUG] Q-Error Range: {min(q_errors)} - {max(q_errors)}")

    # 3. Generating Plots
    print(f"Generating plots in {plots_dir}...")

    # Structural Distributions (Integer Axis enforced by plotting.py)
    if "Tables" in df.columns:
        plot_tables_distribution(df, plots_dir)
    if "Joins" in df.columns:
        plot_joins_distribution(df, plots_dir)
    if "Predicates" in df.columns:
        plot_predicates_distribution(df, plots_dir)
    if "UsedColumns" in df.columns:
        plot_column_usage_frequency(df, plots_dir)
    
    # Combined structural features
    plot_structural_features(df, plots_dir)
    
    # Complexity Plots
    if "ComplexityScore" in df.columns:
        plot_complexity_score_distribution(df, plots_dir)
        plot_complexity_distribution(df, plots_dir)
        if "type" in df.columns:
            # Clean up type labels for better plotting
            # e.g. "multi_join|low_selectivity" -> "Multi Join & Low Selectivity"
            df["type"] = df["type"].astype(str).apply(
                lambda x: x.replace("|", " & ").replace("_", " ").title()
            )
            plot_type_vs_complexity(df, plots_dir)

    # Execution & Accuracy Plots (Require execution data)
    # Check if we have valid q_error or exec info
    has_exec_data = df.get("q_error", pd.Series(dtype=object)).notnull().any() or df.get("exec_time_ms", pd.Series(dtype=object)).notnull().any()
    
    if has_exec_data:
        plot_explain_vs_execution_per_query(full_queries, plots_dir)
        plot_q_error_distribution(full_queries, plots_dir)
        
        # Selective vs Non-Selective Comparisons
        if selective or non_selective:
            plot_selective_vs_non_selective_count(selective, non_selective, plots_dir)
            plot_q_error_comparison(selective, non_selective, plots_dir)
            plot_execution_time_comparison(selective, non_selective, plots_dir)
    else:
        print("Skipping Execution/Accuracy plots (missing 'q_error' or 'exec_time_ms').")
        print("Tip: Run 'execute_queries.py' or equivalent to populate execution metrics.")

    # 4. Query Length (Simple histogram, keeping local or using matplotlib direct)
    print("Analyzing Query Length...")
    lengths = [len(str(q.get("sql", "")).split()) for q in full_queries]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.title("Query Length Distribution (Tokens)")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.savefig(plots_dir / "dist_query_length.png")
    plt.close()

    print("Analysis complete.")

if __name__ == "__main__":
    main()
