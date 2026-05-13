from pathlib import Path
import os 
import json
from dotenv import load_dotenv
from utils.io_utils import read_json_file
from utils.session_utils import get_latest_json_path
import math
import psycopg2
from collections import Counter
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from config.db_config import get_connection
from metrics.plotting import plot_kl_divergence_comparison

load_dotenv()

output_folder = Path(os.getenv("OUTPUT_FOLDER", "../output"))
latest_json_path = get_latest_json_path(output_folder)

EPSILON = 1e-8


# ---------------- DB UTILS ----------------

def explain_cardinality(cursor, sql):
    cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    plan = cursor.fetchone()[0][0]
    node = plan["Plan"]
    # Drill down past Aggregate/Gather nodes to find actual scan/join rows
    while node.get("Node Type") in ("Aggregate", "Gather", "Gather Merge") and "Plans" in node:
        node = node["Plans"][0]
    return node["Plan Rows"]

# ---------------- REAL WORKLOAD ----------------

def load_real_workload(sql_path):
    """Load a real SQL workload from a .sql file (one query per line)."""
    with open(sql_path, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
    return [{"sql": q, "id": f"real_{i}", "source": "real"} for i, q in enumerate(queries)]

# ---------------- SELECTIVITY ----------------

def log_bucket(rows):
    return int(math.log10(max(rows, 1)))

def bucket_label(log_value):
    if log_value <= 1:
        return "high"
    elif log_value <= 3:
        return "medium"
    else:
        return "low"

def extract_selectivity_distribution(queries):
    conn = get_connection()
    cur = conn.cursor()

    buckets = Counter()
    raw_rows = []

    for q in queries:
        try:
            rows = explain_cardinality(cur, q["sql"])
            raw_rows.append(rows)
            bucket = bucket_label(log_bucket(rows))
            buckets[bucket] += 1
        except Exception as e:
            buckets["invalid"] += 1

    cur.close()
    conn.close()

    return buckets, raw_rows

# ---------------- NORMALIZATION ----------------

def normalize(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}

def align_distributions(p, q):
    keys = set(p) | set(q)
    p = {k: p.get(k, EPSILON) for k in keys}
    q = {k: q.get(k, EPSILON) for k in keys}
    return p, q

def kl_divergence(p, q):
    return entropy(list(p.values()), list(q.values()))

# ---------------- PLOTTING ----------------
def plot_selectivity_distribution(dist):
    labels = ["high", "medium", "low"]
    values = [dist.get(l, 0) for l in labels]

    plt.figure()
    plt.bar(labels, values)
    plt.xlabel("Selectivity class")
    plt.ylabel("Number of queries")
    plt.title("Selectivity Distribution of Generated Queries")
    plt.tight_layout()
    plt.show()

# ---------------- MAIN ----------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    plots_dir = project_root / "output" / "plots" / "kl_divergence"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Loading generated queries")
    gen_queries = read_json_file(latest_json_path)

    print("[info] Evaluating selectivity using EXPLAIN")
    gen_dist, gen_rows = extract_selectivity_distribution(gen_queries)
    gen_dist_norm = normalize(gen_dist)

    plot_selectivity_distribution(gen_dist_norm)

    print("\n[Generated Selectivity Distribution]")
    for k, v in gen_dist_norm.items():
        print(f"{k}: {v:.3f}")

    # Compare with real workload
    real_workload_path = project_root.parent / "ActiveLearningSample" / "workloads" / "job-light.sql"

    if real_workload_path.exists():
        print("\n[info] Loading real workload from", real_workload_path)
        real_queries = load_real_workload(real_workload_path)

        print(f"[info] Real workload contains {len(real_queries)} queries")
        real_dist, _ = extract_selectivity_distribution(real_queries)
        real_dist_norm = normalize(real_dist)

        print("\n[Real Workload Selectivity Distribution]")
        for k, v in real_dist_norm.items():
            print(f"{k}: {v:.3f}")

        real_dist_aligned, gen_dist_aligned = align_distributions(
            real_dist_norm, gen_dist_norm
        )

        kl = kl_divergence(real_dist_aligned, gen_dist_aligned)
        print(f"\nKL(real || generated) = {kl:.4f}")

        # Generate the KL divergence comparison graph
        plot_kl_divergence_comparison(gen_dist_norm, real_dist_norm, kl, plots_dir)
        print(f"[info] KL divergence graph saved to {plots_dir}")
    else:
        print(f"\n[warn] Real workload file not found at {real_workload_path}")
        print("[warn] Skipping KL divergence comparison. Place job-light.sql at the expected path.")

    print("\n[done]")

if __name__ == "__main__":
    main()