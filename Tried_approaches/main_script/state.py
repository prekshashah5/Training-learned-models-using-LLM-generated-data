# state.py

def init_state():
    return {
        "types": set(),
        "columns": set(),
        "aggregations": set(),
        "predicates": set(),
    }


def render_avoid_block(state) -> str:
    lines = []
    for k, v in state.items():
        if v:
            lines.append(f"- {k}: {', '.join(sorted(v))}")
    return "\n".join(lines)


def update_state(state, queries):
    for q in queries:
        state["types"].add(q["type"])

        sql = q["sql"].upper()

        if "COUNT(" in sql: state["aggregations"].add("COUNT")
        if "AVG(" in sql: state["aggregations"].add("AVG")
        if "MIN(" in sql: state["aggregations"].add("MIN")
        if "MAX(" in sql: state["aggregations"].add("MAX")

        for op in ["=", ">", "<", "BETWEEN", "LIKE", "IN"]:
            if op in sql:
                state["predicates"].add(op)

        for col in ["MOVIE_ID", "TITLE", "RELEASE_YEAR", "RATING"]:
            if col in sql:
                state["columns"].add(col.lower())
