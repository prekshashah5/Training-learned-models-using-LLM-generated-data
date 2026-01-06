from pathlib import Path
import os 
import json
from dotenv import load_dotenv
from utils import read_json_file, get_latest_json_path
import math
import psycopg2
from collections import Counter
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

load_dotenv()

output_folder = Path(os.getenv("OUTPUT_FOLDER", "../output"))
latest_json_path = get_latest_json_path(output_folder)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "mydatabase"),
    "user": os.getenv("DB_USER", "myuser"),
    "password": os.getenv("DB_PASSWORD", "mypassword"),
}

EPSILON = 1e-8


# ---------------- DB UTILS ----------------

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def explain_cardinality(cursor, sql):
    cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    plan = cursor.fetchone()[0][0]
    return plan["Plan"]["Plan Rows"]

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
    print("[info] Loading generated queries")
    gen_queries = read_json_file(latest_json_path)

    print("[info] Evaluating selectivity using EXPLAIN")
    gen_dist, gen_rows = extract_selectivity_distribution(gen_queries)
    gen_dist_norm = normalize(gen_dist)

    plot_selectivity_distribution(gen_dist_norm)

    print("\n[Generated Selectivity Distribution]")
    for k, v in gen_dist_norm.items():
        print(f"{k}: {v:.3f}")

    # # Optional: compare with real workload
    # if REAL_WORKLOAD_FILE.exists():
    #     print("\n[info] Loading real workload")
    #     real_queries = load_queries(REAL_WORKLOAD_FILE)

    #     real_dist, _ = extract_selectivity_distribution(real_queries)
    #     real_dist_norm = normalize(real_dist)

    #     real_dist_norm, gen_dist_norm = align_distributions(
    #         real_dist_norm, gen_dist_norm
    #     )

    #     kl = kl_divergence(real_dist_norm, gen_dist_norm)
    #     print(f"\nKL(real || generated) = {kl:.4f}")

    print("\n[done]")

if __name__ == "__main__":
    main()