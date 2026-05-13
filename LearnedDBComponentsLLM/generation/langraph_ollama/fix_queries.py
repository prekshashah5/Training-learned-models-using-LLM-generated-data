"""
fix_queries.py
SQL query fixing using heuristic rules, SQLGlot, and LLM fallback.
Refactored imports to use unified package structure.
"""

from pathlib import Path
import time
import re
from typing import List, Dict, Any

import sqlglot
from sqlglot import exp

from utils.logger import logger
from utils.io_utils import read_json_file, write_json_file
from utils.sql_utils import normalize_sql
from generation.langraph_ollama.generate_queries import get_generator_llm
from generation.langraph_ollama.prompt import FIX_PROMPT

# ---------------- CONFIG ----------------

MAX_FIX_ATTEMPTS = 3
SLEEP_BETWEEN_FIXES = 0.3

# ---------------- ERROR CLASSIFIERS ----------------


def is_ambiguous_error(msg: str) -> bool:
    return "ambiguous" in msg.lower()


def is_fake_fk_error(msg: str) -> bool:
    return "fk_" in msg.lower() and "does not exist" in msg.lower()


# ---------------- FIXERS ----------------


def fix_fake_fk_columns(sql: str) -> str:
    """Remove hallucinated fk_* columns."""
    if "fk_" not in sql.lower():
        return sql
    return re.sub(
        r"AND\s+\w+\.\w+\s*=\s*fk_[a-zA-Z0-9_]+",
        "",
        sql,
        flags=re.IGNORECASE,
    )


def fix_ambiguous_columns(sql: str) -> str:
    """Fully qualify unqualified columns when joins are present."""
    try:
        tree = sqlglot.parse_one(sql)
    except Exception:
        return sql

    tables = [t.alias_or_name for t in tree.find_all(exp.Table)]
    if len(tables) < 2:
        return sql

    primary_table = tables[0]

    for col in tree.find_all(exp.Column):
        if col.table is None:
            col.set("table", exp.to_identifier(primary_table))

    return tree.sql()


def apply_llm_fix(sql: str, error_msg: str, schema: str, llm) -> str:
    prompt = FIX_PROMPT.format(
        SCHEMA=schema,
        SQL=sql,
        ERROR=error_msg,
    )
    response = llm.invoke(prompt)
    return normalize_sql(response.content)


def cleanup_sql(json_path: Path):
    queries = read_json_file(json_path)
    cleaned = 0

    for q in queries:
        original = q.get("sql", "")
        normalized = normalize_sql(original)

        if original != normalized:
            q["sql"] = normalized
            q["fix_normalized"] = True
            cleaned += 1

    write_json_file(json_path, queries)
    logger.info(f"Normalized SQL for {cleaned} queries")


# ---------------- CORE HELPER ----------------


def fix_queries_in_place(
    queries: List[Dict[str, Any]],
    json_path: Path,
    schema: str,
    model_name: str,
    temperature: float,
    base_url: str,
) -> None:
    """
    Fixes SQL queries IN-PLACE.
    Optimized: Writes to disk only after all fixed queries are processed.
    """
    logger.info("Starting to fix SQL queries with errors")
    llm = get_generator_llm(
        model_name=model_name,
        temperature=temperature,
        BASEURL=base_url,
    )

    any_fixed = False
    for q in queries:
        error_msg = q.get("exec_error_msg")
        if not error_msg:
            continue

        if q.get("fix_attempts", 0) >= MAX_FIX_ATTEMPTS:
            q["fix_status"] = "failed_fix"
            continue

        logger.info(f"Fixing query {q['id']}")
        any_fixed = True

        sql = normalize_sql(q["sql"])
        original_sql = q.get("original_sql", q["sql"])
        fixed_sql = sql

        # -------- STEP 1: fake FK fix --------
        if is_fake_fk_error(error_msg):
            logger.info(f"Applying fake-FK fix to {q['id']}")
            fixed_sql = fix_fake_fk_columns(fixed_sql)

        # -------- STEP 2: ambiguity fix --------
        fixed_sql = fix_ambiguous_columns(fixed_sql)

        # -------- STEP 3: LLM fallback --------
        if fixed_sql == sql:
            try:
                logger.info(f"Applying LLM fix to {q['id']}")
                fixed_sql = apply_llm_fix(
                    fixed_sql,
                    error_msg,
                    schema,
                    llm,
                )
                fixed_sql = fix_ambiguous_columns(fixed_sql)
            except Exception as e:
                logger.error(f"LLM fix failed for {q['id']}: {e}")

        # -------- UPDATE QUERY --------
        q["original_sql"] = original_sql
        q["sql"] = fixed_sql
        q["fix_attempts"] = q.get("fix_attempts", 0) + 1
        q["exec_error_msg"] = None
        q["fix_status"] = "fixed"

        # reset execution state
        q["exec_status"] = "pending"
        q["exec_attempts"] = 0

        time.sleep(SLEEP_BETWEEN_FIXES)

    if any_fixed:
        write_json_file(json_path, queries)

    logger.info("All fixable SQL errors have been processed")
