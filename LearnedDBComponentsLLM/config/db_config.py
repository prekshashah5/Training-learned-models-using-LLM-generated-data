"""
db_config.py
Unified database configuration module.

Supports both .env-based defaults (for LangGraph pipeline)
and explicit parameter overrides (for training pipeline).
"""

import os
import psycopg2
from typing import Dict, Any, Optional, Tuple
import time

try:
    from dotenv import load_dotenv
    from pathlib import Path
    # Load .env from project root
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed; use explicit params


def get_db_config(
    db_host: Optional[str] = None,
    db_port: Optional[int] = None,
    db_name: Optional[str] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get database configuration.

    Explicit parameters take precedence over .env variables,
    which take precedence over hardcoded defaults.
    """
    return {
        "host": db_host or os.getenv("DB_HOST", "localhost"),
        "port": int(db_port or os.getenv("DB_PORT", 5432)),
        "database": db_name or os.getenv("DB_NAME", "imdb"),
        "user": db_user or os.getenv("DB_USER", "postgres"),
        "password": db_password or os.getenv("DB_PASSWORD", "admin"),
    }


def get_connection(
    db_host: Optional[str] = None,
    db_port: Optional[int] = None,
    db_name: Optional[str] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
) -> psycopg2.extensions.connection:
    """
    Establish a connection to the PostgreSQL database.

    Supports both explicit parameter passing (training pipeline)
    and .env-based config (LangGraph pipeline).
    """
    config = get_db_config(db_host, db_port, db_name, db_user, db_password)
    conn = psycopg2.connect(**config)
    conn.autocommit = False
    return conn


def count_rows(cursor, sql: str, timeout: int = 6000) -> Tuple[int, float]:
    """
    Execute a COUNT(*) query for a given SQL string and return (row_count, elapsed_ms).
    """
    cursor.execute("SET statement_timeout = %s;", (timeout,))

    sql = sql.rstrip().rstrip(";")
    wrapped = f"SELECT COUNT(*) FROM ({sql}) AS subq"

    start = time.perf_counter()
    cursor.execute(wrapped)
    elapsed = (time.perf_counter() - start) * 1000

    count = int(cursor.fetchone()[0])
    return max(count, 1), round(elapsed, 2)


def load_column_stats(csv_path: str = None) -> str:
    """
    Read column statistics from CSV and return a formatted string for LLM prompts.
    """
    import csv
    from pathlib import Path

    if csv_path is None:
        csv_path = str(Path(__file__).resolve().parent.parent / "tools" / "get_stats" / "col_stats.csv")

    if not os.path.exists(csv_path):
        return "No column statistics available."

    stats_lines = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats_lines.append(
                f"- {row['name']}: min={row['min']}, max={row['max']}, "
                f"cardinality={row['cardinality']}, unique={row['num_unique_values']}"
            )

    return "\n".join(stats_lines)


def explain_cardinality(cursor, sql: str) -> int:
    """
    Execute EXPLAIN (FORMAT JSON) and return the estimated rows.
    """
    cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    plan = cursor.fetchone()[0][0]
    return plan["Plan"]["Plan Rows"]
