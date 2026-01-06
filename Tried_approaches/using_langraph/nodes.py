# nodes.py
import json
import re
from state import GraphState

MAX_RETRIES = 0


def prompt_node(state: GraphState):
    print("prompt_node called")
    start = state["batch_id"] * state["batch_size"] + 1
    end = start + state["batch_size"] - 1

    avoid_lines = [
        f"- {k}: {', '.join(sorted(v))}"
        for k, v in state["avoid"].items()
        if v
    ]

    prompt = f"""
Generate SQL SELECT queries as JSON array.

Keys:
id(Q{start}..Q{end}), sql, type, reasoning (≤12 words)

Schema:
{state["schema"]}

Rules:
- SELECT-FROM-WHERE only
- NO subqueries
- NO GROUP BY / ORDER BY / HAVING
- NO aggregate functions
- Explicit JOIN ... ON
- AND-only predicates
- Valid SQL
- JSON array only

Avoid repeating:
{chr(10).join(avoid_lines)}

Generate exactly {state["batch_size"]} queries.
""".strip()

    return {**state, "prompt": prompt}


def llm_node(state: GraphState, llm):
    print("llm_node called")
    out = llm(state["prompt"])[0]["generated_text"].strip()
    return {**state, "raw": out}


def _strip_fences(s: str) -> str:
    print("Stripping fences")
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(\w+)?", "", s)
        s = re.sub(r"```$", "", s)
    return s.strip()


def parse_node(state: GraphState):
    try:
        print("parse_node called")
        parsed = json.loads(_strip_fences(state["raw"]))
        if not isinstance(parsed, list):
            raise ValueError
        if len(parsed) != state["batch_size"]:
            raise ValueError
        return {**state, "parsed": parsed, "retries": 0}
    except Exception:
        return {**state, "parsed": [], "retries": state["retries"] + 1}


def retry_router(state: GraphState) -> str:
    print("retry_router called")
    if state["parsed"]:
        return "update"
    if state["retries"] > MAX_RETRIES:
        raise RuntimeError("Generation failed")
    return "llm"


def update_node(state: GraphState):
    print("update_node called")
    avoid = state["avoid"]

    for q in state["parsed"]:
        sql = q["sql"].upper()
        avoid["types"].add(q["type"])

        for op in ["=", ">", "<", "LIKE", "IN", "BETWEEN"]:
            if op in sql:
                avoid["predicates"].add(op)

        for col in ["MOVIE_ID", "TITLE", "RELEASE_YEAR", "RATING"]:
            if col in sql:
                avoid["columns"].add(col.lower())

    return {
        **state,
        "queries": state["queries"] + state["parsed"],
        "batch_id": state["batch_id"] + 1
    }


def continue_router(state: GraphState) -> str:
    print("continue_router called")
    if state["batch_id"] >= state["total_batches"]:
        return "end"
    return "prompt"
