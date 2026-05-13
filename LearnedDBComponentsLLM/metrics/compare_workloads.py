import json
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, wasserstein_distance
from sqlalchemy import create_engine
import sqlglot


from config.db_config import get_connection, count_rows, explain_cardinality
from metrics.SQL_Complexity import extract_features, compute_complexity_matrix, clean_df_before_complexity
from metrics.plotting import save_and_close_plot
from utils.session_utils import get_latest_json_path
from utils.io_utils import read_json_file

# ... existing code ...

def load_real_workload(sql_path):
    print(f"Loading real workload from {sql_path}...")
    with open(sql_path, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
    return [{"sql": q, "id": f"real_{i}", "source": "real"} for i, q in enumerate(queries)]

def run_workload_analysis(queries, description="workload", limit=100):
    if len(queries) > limit:
        import random
        print(f"Sampling {limit} queries from {len(queries)} for {description}...")
        queries = random.sample(queries, limit)
    
    print(f"Analyzing {description} ({len(queries)} queries)...")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        db_available = True
    except Exception as e:
        print(f"Warning: DB Connection failed ({e}). Skipping execution metrics.")
        db_available = False
        cursor = None
    
    results = []
    for q in queries:
        try:
            # Structure
            features = extract_features(q["sql"])
            
            # Execution Metrics
            act_rows = None
            est_rows = None
            q_error = None

            if db_available:
                try:
                    # Check if already computed in generated file
                    if "actual_cardinality" in q:
                         act_rows = q["actual_cardinality"]
                    elif "rows" in q: # sometimes stored as 'rows'
                         act_rows = q["rows"]
                    else:
                         act_rows, _ = count_rows(cursor, q["sql"])
                    
                    est_rows = explain_cardinality(cursor, q["sql"])
                    
                    if act_rows is not None and est_rows is not None:
                         q_error = max(act_rows / max(est_rows, 1), est_rows / max(act_rows, 1))

                except Exception as e:
                    # print(f"Error executing {q['id']}: {e}")
                    pass
            
            # Update query dict
            q_data = {
                **q,
                **features,
                "actual_cardinality": act_rows,
                "estimated_cardinality": est_rows,
                "q_error": q_error
            }
            results.append(q_data)
        except Exception as e:
            # print(f"Error processing {q['id']}: {e}")
            pass
            
    if db_available:
        cursor.close()
        conn.close()
    return pd.DataFrame(results)

def plot_overlay_distribution(df_real, df_gen, column, label, output_dir, log_scale=False):
    plt.figure(figsize=(8, 5))
    
    val_real = df_real[column].dropna()
    val_gen = df_gen[column].dropna()
    
    if len(val_real) == 0 or len(val_gen) == 0:
        print(f"Skipping plot for {label}: Insufficient data.")
        plt.close()
        return

    if log_scale:
        val_real = np.log10(val_real + 1)
        val_gen = np.log10(val_gen + 1)
        xlabel = f"Log10({label})"
    else:
        xlabel = label

    plt.hist(val_real, bins=20, alpha=0.5, label='Real (JOB-Light)', density=True, color='blue')
    plt.hist(val_gen, bins=20, alpha=0.5, label='Generated', density=True, color='orange')
    
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(f"Distribution Comparison: {label}")
    plt.legend()
    
    filename = f"compare_{column}.png"
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()

def main():
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    real_workload_path = project_root.parent / "ActiveLearningSample" / "workloads" / "job-light.sql"
    output_folder = project_root / "output"
    plots_dir = output_folder / "plots" / "comparison"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    gen_file = get_latest_json_path(output_folder)
    gen_queries = read_json_file(gen_file)
    # Ensure source is labeled
    for q in gen_queries: q["source"] = "generated"

    real_queries = load_real_workload(real_workload_path)

    # 2. Analyze
    df_gen = run_workload_analysis(gen_queries, "Generated")
    df_real = run_workload_analysis(real_queries, "Real (Job-Light)") # Warning: might be slow if DB cold

    # 3. Compute Complexity Scores
    df_gen = compute_complexity_matrix(df_gen)
    df_real = compute_complexity_matrix(df_real)

    # 4. Statistical Tests
    stats_report = []
    
    # helper for tests
    def compare_metric(metric_name, col_name, log=False):
        if col_name not in df_real.columns or col_name not in df_gen.columns:
            stats_report.append(f"### {metric_name} Comparison\n- Data missing for this metric.\n")
            return

        v1 = df_real[col_name].dropna()
        v2 = df_gen[col_name].dropna()
        
        if len(v1) == 0 or len(v2) == 0:
            stats_report.append(f"### {metric_name} Comparison\n- Insufficient data (Real: {len(v1)}, Gen: {len(v2)}).\n")
            return

        if log:
            v1 = np.log10(v1 + 1)
            v2 = np.log10(v2 + 1)
            
        ks_stat, ks_p = ks_2samp(v1, v2)
        wd = wasserstein_distance(v1, v2)
        
        stats_report.append(f"### {metric_name} Comparison")
        stats_report.append(f"- **KS Statistic**: {ks_stat:.3f} (p={ks_p:.3f})")
        stats_report.append(f"- **Wasserstein Distance**: {wd:.3f}")
        stats_report.append(f"- **Real Mean**: {v1.mean():.2f}, **Gen Mean**: {v2.mean():.2f}")
        stats_report.append("")

    stats_report.append("# Workload Comparison Report\n")
    compare_metric("Cardinality (Log10)", "actual_cardinality", log=True)
    compare_metric("Complexity Score", "ComplexityScore")
    compare_metric("Number of Joins", "Joins")
    compare_metric("Number of Tables", "Tables")

    # 5. Visualizations
    plot_overlay_distribution(df_real, df_gen, "actual_cardinality", "Cardinality", plots_dir, log_scale=True)
    plot_overlay_distribution(df_real, df_gen, "ComplexityScore", "SQL Complexity Score", plots_dir)
    plot_overlay_distribution(df_real, df_gen, "Joins", "Number of Joins", plots_dir)
    plot_overlay_distribution(df_real, df_gen, "Tables", "Number of Tables", plots_dir)

    # Save Stats Report
    report_path = output_folder / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(stats_report))
    
    print(f"Comparison complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
