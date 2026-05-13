from pathlib import Path
import os
import json
import time
from psycopg2 import errors
from utils.io_utils import read_json_file, write_json_file
from utils.session_utils import get_latest_json_path
from logger import logger
from metrics.plotting import plot_query_error_overview
from config.db_config import get_connection, count_rows

# ---------------- CONFIG ----------------

MAX_RETRIES = int(os.getenv("MAX_DB_RETRIES", 2))
SLEEP_BETWEEN_QUERIES = float(os.getenv("DB_THROTTLE_SEC", 0.5))
SLEEP_ON_ERROR = float(os.getenv("DB_ERROR_BACKOFF_SEC", 3.0))
DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", 6000))

# ---------------- CORE ----------------

def calculate_rows(queries, json_path: Path = None):

    # initialize tracking fields
    for q in queries:
        q.setdefault("exec_status", "pending")
        q.setdefault("exec_attempts", 0)

        # Skip permanently broken queries
        if q.get("fix_status") == "failed_fix":
            q["exec_status"] = "skipped"

    conn = get_connection()
    cur = conn.cursor()

    try:
        for q in queries:
            if q["exec_status"] == "done":
                continue
            if q["exec_status"] == "skipped":
                continue
            if q["exec_attempts"] >= MAX_RETRIES:
                q["exec_status"] = "failed"
                continue

            logger.info(f"Counting rows for {q['id']} (attempt {q['exec_attempts'] + 1})")

            try:
                q["exec_attempts"] += 1
                rows, elapsed = count_rows(cur, q["sql"], timeout=DB_TIMEOUT)

                q.update({
                    "exec_row_count": max(rows, 1),
                    "exec_time_ms": elapsed,
                    "exec_status": "done",
                })

                conn.commit()

            except errors.QueryCanceled:
                conn.rollback()
                q["exec_error_msg"] = f"Timeout after {DB_TIMEOUT} ms"

                if q["exec_attempts"] < MAX_RETRIES:
                    logger.warning(f"Timeout on {q['id']} - retrying")
                    time.sleep(SLEEP_ON_ERROR)
                else:
                    q["exec_status"] = "failed"

            except Exception as e:
                conn.rollback()
                q["exec_error_msg"] = str(e)

                if q["exec_attempts"] < MAX_RETRIES:
                    logger.warning(f"Error on {q['id']} - retrying: {e}")
                    time.sleep(SLEEP_ON_ERROR)
                else:
                    q["exec_status"] = "failed"

            time.sleep(SLEEP_BETWEEN_QUERIES)

        # persist ONCE after all queries are processed
        if json_path:
            write_json_file(json_path, queries)

    finally:
        cur.close()
        conn.close()

    logger.info("Row counting completed")

# ---------------- ENTRY ----------------

def calculate_rows_pipeline():
    output_folder = Path(os.getenv("OUTPUT_FOLDER", "/TryingModels/output"))
    latest_json = get_latest_json_path(output_folder)
    queries = read_json_file(latest_json)
    calculate_rows(queries, latest_json)
    plot_dir = latest_json.parent / "plots"
    plot_query_error_overview(queries, plot_dir)
