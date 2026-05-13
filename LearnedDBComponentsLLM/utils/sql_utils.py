"""
sql_utils.py
SQL normalization, JSON extraction, and parsing utilities.
"""

import json
import re
from typing import Any, List, Optional


def normalize_sql(sql: str) -> str:
    """
    Remove markdown fences and normalize whitespace.
    """
    if not sql:
        return sql

    sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)
    sql = sql.strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";")
    return sql


def _strip_code_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(\w+)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def coerce_json_array(
    content: str,
    expected_len: Optional[int] = None,
) -> List[Any]:
    """
    Robustly parse model output into a JSON array.
    - Strips ```json fences
    - Handles quoted JSON with escapes
    - Enforces array type
    - Optionally enforces expected length
    """
    if not content or not content.strip():
        raise ValueError("Empty model output")

    t = _strip_code_fences(content)

    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        if (t.startswith('"') and t.endswith('"')) or (
            t.startswith("'") and t.endswith("'")
        ):
            inner = t[1:-1].encode("utf-8").decode("unicode_escape")
            obj = json.loads(inner)
        else:
            raise

    if not isinstance(obj, list):
        raise TypeError(f"Expected JSON array, got {type(obj)}")

    if expected_len is not None and len(obj) != expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(obj)}")

    return obj


def extract_json_array(text: str):
    """Extract structured items from LLM output text using regex."""
    items = []

    item_pattern = re.compile(
        r'"id"\s*:\s*"(?P<id>.*?)"\s*,\s*'
        r'"sql"\s*:\s*"(?P<sql>.*?)"\s*,\s*'
        r'"type"\s*:\s*"(?P<type>.*?)"\s*,\s*'
        r'"reasoning"\s*:\s*"(?P<reasoning>.*?)(?=(\n\s*"id"\s*:|$))',
        re.DOTALL,
    )

    for match in item_pattern.finditer(text):
        reasoning = match.group("reasoning").strip()
        reasoning = reasoning.rstrip('", \n')

        item = {
            "id": match.group("id").strip(),
            "sql": match.group("sql").strip(),
            "type": match.group("type").strip(),
            "reasoning": reasoning,
        }
        items.append(item)

    if not items:
        return _extract_by_state_machine(text)

    return items


def _extract_by_state_machine(lines: str):
    """Fallback extraction of JSON items using a state machine approach."""
    items = []
    current = {}

    for line in lines.splitlines():
        line = line.strip().rstrip(",")

        if line.startswith('"id"'):
            if current:
                items.append(current)
                current = {}

        for key in ("id", "sql", "type", "reasoning"):
            if line.startswith(f'"{key}"'):
                value = line.split(":", 1)[1].strip().strip('"')
                current[key] = value

    if current:
        items.append(current)

    if not items:
        raise ValueError("No items found")

    return items
