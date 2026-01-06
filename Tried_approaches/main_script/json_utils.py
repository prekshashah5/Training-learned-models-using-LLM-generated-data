
import json
import re
from typing import Any, List


def _strip_code_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(\w+)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def coerce_json_array(
    content: str,
    expected_len: int | None = None,
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
        # Handle quoted JSON string
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
