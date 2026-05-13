"""
calculate.py
Database query execution for row counting.
Refactored imports to use unified package structure.
"""

import time
import os
from pathlib import Path
from typing import List, Dict, Any
from psycopg2 import errors

from utils.logger import logger
from utils.io_utils import write_json_file
from config.db_config import get_connection, count_rows

# ---------------- CONFIG ----------------

MAX_RETRIES = int(os.getenv("MAX_DB_RETRIES", 2))
SLEEP_BETWEEN_QUERIES = float(os.getenv("DB_THROTTLE_SEC", 0.5))
SLEEP_ON_ERROR = float(os.getenv("DB_ERROR_BACKOFF_SEC", 3.0))
DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", 6000))

# ---------------- CORE HELPERS ----------------


def execute_queries(
    queries: List[Dict[str, Any]],
    json_path: Path,
) -> None:
    """
    Executes SQL queries IN-PLACE.
    Optimized: Writes to disk only after all queries are processed.
    """
    # initialize tracking fields
    for q in queries:
        q.setdefault("exec_status", "pending")
        q.setdefault("exec_attempts", 0)

        if q.get("fix_status") == "failed_fix":
            q["exec_status"] = "skipped"

    conn = get_connection()
    cur = conn.cursor()

    try:
        for q in queries:
            if q["exec_status"] in ("done", "skipped"):
                continue

            if q["exec_attempts"] >= MAX_RETRIES:
                q["exec_status"] = "failed"
                continue

            logger.info(
                f"Counting rows for {q['id']} "
                f"(attempt {q['exec_attempts'] + 1})"
            )

            try:
                q["exec_attempts"] += 1
                rows, elapsed = count_rows(cur, q["sql"], timeout=DB_TIMEOUT)

                q.update({
                    "exec_row_count": max(rows, 1),
                    "exec_time_ms": elapsed,
                    "exec_status": "done",
                    "exec_error_msg": None,
                })

                conn.commit()

            except errors.QueryCanceled:
                conn.rollback()
                q["exec_error_msg"] = f"Timeout after {DB_TIMEOUT} ms"

                if q["exec_attempts"] < MAX_RETRIES:
                    logger.warning(f"Timeout on {q['id']} - retrying")
                    q["exec_status"] = "failed"
                    time.sleep(SLEEP_ON_ERROR)
                else:
                    q["exec_status"] = "failed"

            except Exception as e:
                conn.rollback()
                q["exec_error_msg"] = str(e)

                if "syntax error" in str(e).lower():
                    q["exec_status"] = "failed"
                elif q["exec_attempts"] >= MAX_RETRIES:
                    q["exec_status"] = "failed"
                else:
                    q["exec_status"] = "pending"

            time.sleep(SLEEP_BETWEEN_QUERIES)

        # persist ONCE after all queries are processed
        write_json_file(json_path, queries)

    finally:
        cur.close()
        conn.close()

    logger.info("Row counting completed")
