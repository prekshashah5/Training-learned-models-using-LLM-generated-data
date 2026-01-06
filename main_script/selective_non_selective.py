from pathlib import Path
import os 
from dotenv import load_dotenv
from utils import read_json_file, get_latest_json_path
import json
import psycopg2
import numpy as np
from plotting import plot_q_error_distribution, plot_q_error_ecdf, plot_execution_time_comparison, plot_explain_vs_execution_per_query
import time

# ---------------- DB UTILS ----------------

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def explain_cardinality(cursor, sql):
    start = time.perf_counter()
    cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    plan = cursor.fetchone()[0][0]
    elapsed = (time.perf_counter() - start) * 1000
    return max(plan["Plan"].get("Plan Rows", 0), 1), elapsed

def execute_actual_cardinality(cursor, sql):
    wrapped = f"SELECT COUNT(*) FROM ({sql}) AS subq"
    start = time.perf_counter()
    cursor.execute(wrapped)
    elapsed = (time.perf_counter() - start) * 1000
    return max(cursor.fetchone()[0], 1), elapsed

def compute_q_error(estimate, actual):
    estimate = max(estimate, 1)
    actual = max(actual, 1)
    return round(max(estimate / actual, actual / estimate), 2)

# ---------------- SELECTIVITY ----------------

def selectivity_class(rows):
    if rows <= 100:
        return "high"          
    elif rows <= 10000:
        return "medium"
    else:
        return "low"           
    
def evaluate_selectivity(queries):
    conn = get_connection()
    cur = conn.cursor()

    for q in queries:
        # Skip if already evaluated
        if all(k in q for k in ["estimated_rows", "actual_rows", "q_error"]):
            continue

        try:
            est, explain_time = explain_cardinality(cur, q["sql"])
            act, exec_time = execute_actual_cardinality(cur, q["sql"])
            qerr = compute_q_error(est, act)

            q["estimated_rows"] = est
            q["actual_rows"] = act
            q["q_error"] = qerr
            q["explain_time_ms"] = round(explain_time, 2)
            q["execution_time_ms"] = round(exec_time, 2)

            if q.get("actual_rows") is not None:
                q["selectivity_class"] = selectivity_class(q["actual_rows"])

        except Exception as e:
            q["estimated_rows"] = None
            q["actual_rows"] = None
            q["q_error"] = None
            q["explain_time_ms"] = None
            q["execution_time_ms"] = None
            q["error"] = str(e)

    conn.close()
    return queries

def q_error_stats(queries):
    q_errors = [q["q_error"] for q in queries if q.get("q_error") is not None]

    return {
        "mean": float(np.mean(q_errors)),
        "median": float(np.median(q_errors)),
        "p90": float(np.percentile(q_errors, 90)),
        "p99": float(np.percentile(q_errors, 99)),
    }

def split_by_selectivity(queries):
    selective = [q for q in queries if q.get("selectivity_class") == "high"]
    non_selective = [q for q in queries if q.get("selectivity_class") == "low"]
    return selective, non_selective

# ---------------- MAIN ----------------

load_dotenv()

output_folder = Path(os.getenv("OUTPUT_FOLDER", "../output"))
latest_json_path = get_latest_json_path(output_folder)
json_dir = latest_json_path.parent / "plots"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "mydatabase"),
    "user": os.getenv("DB_USER", "myuser"),
    "password": os.getenv("DB_PASSWORD", "mypassword"),
}

EPSILON = 1e-8

print("[info] Loading generated queries")
gen_queries = read_json_file(latest_json_path)

print("[info] Calculating selectivity distribution for generated queries")
gen_queries = evaluate_selectivity(gen_queries)
with open(latest_json_path, "w") as f:
    json.dump(gen_queries, f, indent=2)

stats = q_error_stats(gen_queries)
print("\n[Q-error statistics]")
for k, v in stats.items():
    print(f"{k}: {v:.3f}")

plot_q_error_distribution(gen_queries, json_dir)

selective, non_selective = split_by_selectivity(gen_queries)

plot_q_error_ecdf(selective, non_selective, json_dir)
plot_execution_time_comparison(selective, non_selective, json_dir)
plot_explain_vs_execution_per_query(gen_queries, json_dir)

print("\n[done]")