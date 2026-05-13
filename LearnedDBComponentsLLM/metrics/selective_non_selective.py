from pathlib import Path
import os
from utils.io_utils import read_json_file, write_json_file
from utils.session_utils import get_latest_json_path
import json
import psycopg2
import numpy as np
from metrics.plotting import (plot_q_error_distribution, plot_q_error_comparison, plot_execution_time_comparison, plot_explain_vs_execution_per_query, plot_selective_vs_non_selective_count)
import time
from config.db_config import get_connection, count_rows

# ---------------- DB UTILS ----------------

def explain_cardinality(cursor, sql, timeout=6000):
    cursor.execute("SET statement_timeout = %s;", (timeout,))
    start = time.perf_counter()
    cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    plan = cursor.fetchone()[0][0]
    elapsed = (time.perf_counter() - start) * 1000
    return max(plan["Plan"].get("Plan Rows", 0), 1), elapsed

def execute_actual_cardinality(cursor, sql, timeout=6000):
    count, elapsed = count_rows(cursor, sql, timeout=timeout)
    return count, elapsed

def compute_q_error(estimate, actual):
    estimate = max(estimate, 1)
    actual = max(actual, 1)
    return round(max(estimate / actual, actual / estimate), 2)

# ---------------- SELECTIVITY ----------------

def selectivity_class(rows):
    if rows <= 50:
        return "high"          
    elif rows <= 10000:
        return "medium"
    else:
        return "low"           
    
def evaluate_selectivity(queries, DB_TIMEOUT, DB_CONFIG=None):
    conn = get_connection(DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("SET statement_timeout = %s;", (DB_TIMEOUT,))
    
    for q in queries:
        print(f"[info] Evaluating query ID: {q['id']}")
        try:
            est, explain_time = explain_cardinality(cur, q["sql"])
            act, exec_time = execute_actual_cardinality(cur, q["sql"])
            qerr = compute_q_error(est, act)

            q["estimated_rows"] = est
            q["exec_row_count"] = act
            q["q_error"] = qerr
            q["explain_time_ms"] = round(explain_time, 2)
            q["exec_time_ms"] = round(exec_time, 2)

            if q.get("exec_row_count") is not None:
                q["selectivity_class"] = selectivity_class(q["exec_row_count"])

            conn.commit()
        except psycopg2.errors.QueryCanceled:
            conn.rollback()
            q["estimated_rows"] = None
            q["exec_row_count"] = None
            q["q_error"] = None
            q["explain_time_ms"] = None
            q["exec_time_ms"] = DB_TIMEOUT 
            q["exec_error_msg"] = "Query timed out after {} ms".format(DB_TIMEOUT)
            print(f"[timeout] Query {q.get('id')} was killed by DB")
        except Exception as e:
            conn.rollback()

            q["estimated_rows"] = None
            q["exec_row_count"] = None
            q["q_error"] = None
            q["explain_time_ms"] = None
            q["exec_time_ms"] = None
            q["exec_error_msg"] = str(e)

    cur.close()
    conn.close()
    return queries

def q_error_stats(queries):
    q_errors = [q["q_error"] for q in queries if q.get("q_error") is not None]

    if len(q_errors) == 0:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
        }

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


def run_selective_non_selective_pipeline(recompute_selectivity=False):
    # load_dotenv()

    output_folder = Path(os.getenv("OUTPUT_FOLDER", "/TryingModels/output"))
    # output_folder = Path("/Users/prekshashah/Documents/Masters/Thesis/msc-thesis/code/TryingModels/output")
    latest_json_path = get_latest_json_path(output_folder)
    json_dir = latest_json_path.parent / "plots"

    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "imdb"),
        "user": os.getenv("DB_USER", "prekshashah"),
        "password": os.getenv("DB_PASSWORD", "admin"),
    }
    DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", 6000))  

    EPSILON = 1e-8

    print("[info] Loading generated queries")
    gen_queries = read_json_file(latest_json_path)

    print("[info] Calculating selectivity distribution for generated queries")

    if recompute_selectivity:
        gen_queries = evaluate_selectivity(gen_queries, DB_TIMEOUT, DB_CONFIG=DB_CONFIG)
        with open(latest_json_path, "w") as f:
            json.dump(gen_queries, f, indent=2)

    stats = q_error_stats(gen_queries)
    print("\n[Q-error statistics]")
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

    plot_q_error_distribution(gen_queries, json_dir)

    selective, non_selective = split_by_selectivity(gen_queries)

    plot_q_error_comparison(selective, non_selective, json_dir)
    plot_execution_time_comparison(selective, non_selective, json_dir)
    plot_explain_vs_execution_per_query(gen_queries, json_dir)

    plot_selective_vs_non_selective_count(selective, non_selective, json_dir)

    print("\n[done]")