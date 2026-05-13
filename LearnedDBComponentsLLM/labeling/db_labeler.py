"""
db_labeler.py
Handles database labeling of queries - executes SELECT COUNT(*) to get
true cardinalities for queries selected by the active learning loop.
"""

import time
import psycopg2
from psycopg2 import errors
from typing import Dict, List, Any, Optional

from config.db_config import get_connection


def reconstruct_sql(tables: List[str], joins: List[str], predicates: List) -> str:
    """
    Reconstruct a SQL SELECT COUNT(*) query from MSCN components.

    Args:
        tables: ["title t", "movie_info mi"]
        joins: ["t.id=mi.movie_id"]
        predicates: [("t.kind_id", "=", "7"), ...]

    Returns:
        "SELECT COUNT(*) FROM title t, movie_info mi WHERE t.id=mi.movie_id AND t.kind_id = 7"
    """
    from_clause = ", ".join(tables)

    conditions = []
    for join in joins:
        if join:
            left, right = join.split("=", 1)
            conditions.append(f"{left} = {right}")

    for pred in predicates:
        if len(pred) == 3:
            col, op, val = pred
            try:
                float(val)
                conditions.append(f"{col} {op} {val}")
            except (ValueError, TypeError):
                conditions.append(f"{col} {op} '{val}'")

    sql = f"SELECT COUNT(*) FROM {from_clause}"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    return sql


def label_single_query(cursor, sql: str, timeout: int = 6000) -> Optional[int]:
    """
    Execute a query and return its cardinality (row count).

    Args:
        cursor: psycopg2 cursor
        sql: SQL query string
        timeout: Timeout in milliseconds

    Returns:
        Row count, or None if execution fails.
    """
    try:
        cursor.execute("SET statement_timeout = %s;", (timeout,))

        sql_clean = sql.rstrip().rstrip(";")

        if sql_clean.upper().strip().startswith("SELECT COUNT"):
            cursor.execute(sql_clean)
        else:
            wrapped = f"SELECT COUNT(*) FROM ({sql_clean}) AS subq"
            cursor.execute(wrapped)

        count = int(cursor.fetchone()[0])

        cursor.execute("SET statement_timeout = 0;")
        return max(count, 1)  # MSCN requires cardinality >= 1

    except errors.QueryCanceled:
        cursor.connection.rollback()
        cursor.execute("SET statement_timeout = 0;")
        print(f"[db_labeler] Timeout after {timeout}ms on: {sql[:100]}...")
        return None
    except psycopg2.errors.UndefinedTable as e:
        cursor.connection.rollback()
        cursor.execute("SET statement_timeout = 0;")
        print(f"[db_labeler] Table not found: {e.diag.message_primary}")
        return None
    except Exception as e:
        cursor.connection.rollback()
        try:
            cursor.execute("SET statement_timeout = 0;")
        except Exception:
            pass
        print(f"[db_labeler] Error: {type(e).__name__}: {e}")
        return None


def label_queries(cursor,
                  queries: List[Dict[str, Any]],
                  timeout: int = 6000,
                  max_retries: int = 2,
                  sleep_between: float = 0.1) -> int:
    """
    Label a batch of queries by executing them on the database.

    Args:
        cursor: psycopg2 cursor
        queries: List of query dicts with "tables", "joins", "predicates" keys.
                 Updated in-place with "cardinality" key.
        timeout: Per-query timeout in milliseconds.
        max_retries: Maximum retry attempts per query.
        sleep_between: Sleep between queries (seconds).

    Returns:
        Number of successfully labeled queries.
    """
    labeled_count = 0

    for i, q in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"  [db_labeler] Labeled {i+1}/{len(queries)} queries...", flush=True)
            
        if q.get("cardinality") is not None:
            labeled_count += 1
            continue

        sql = reconstruct_sql(q["tables"], q["joins"], q["predicates"])

        for attempt in range(max_retries):
            cardinality = label_single_query(cursor, sql, timeout=timeout)

            if cardinality is not None:
                q["cardinality"] = str(cardinality)
                labeled_count += 1
                try:
                    cursor.connection.commit()
                except Exception as e:
                    print(f"[db_labeler] Commit failed for query {i}: {e}")
                    try:
                        cursor.connection.rollback()
                        cursor.execute("SET statement_timeout = 0;")
                    except Exception:
                        pass
                break
            else:
                if attempt < max_retries - 1:
                    time.sleep(1.0)

        if q.get("cardinality") is None:
            print(f"[db_labeler] Failed to label query {i}, setting cardinality=1")
            q["cardinality"] = "1"
            labeled_count += 1

        time.sleep(sleep_between)

    return labeled_count


def label_queries_from_indices(cursor,
                               all_queries: List[Dict],
                               indices: List[int],
                               timeout: int = 6000) -> int:
    """
    Label only the queries at specific indices.
    """
    subset = [all_queries[i] for i in indices]
    return label_queries(cursor, subset, timeout=timeout)

def get_pg_estimates(cursor, queries):
    """Fetch native optimizer estimates using EXPLAIN."""
    estimates = []
    print("  [db_labeler] Fetching PostgreSQL Optimizer native estimates...")
    for i, q in enumerate(queries):
        if (i + 1) % 50 == 0:
            print(f"    [pg_estimates] EXPLAIN {i+1}/{len(queries)} queries...", flush=True)
            
        sql = reconstruct_sql(q.get("tables", []), q.get("joins", []), q.get("predicates", []))
        sql_explain = sql.replace("SELECT COUNT(*)", "SELECT *")
        try:
            cursor.execute(f"EXPLAIN (FORMAT JSON) {sql_explain}")
            plan = cursor.fetchone()[0][0]
            est = plan['Plan'].get('Plan Rows', 1)
            estimates.append(max(est, 1))
        except Exception as e:
            estimates.append(1)
            cursor.connection.rollback()
    return estimates
