# prompt.py

BASE_PROMPT = """
Generate SQL SELECT queries as JSON.

Keys:
id(Q{start_id}..Q{end_id}), sql, type(point|range|agg|multi_predicate|join|multi_join), reasoning(≤12w)

Schema:
{SCHEMA}

Strict rules (must follow):
- SELECT-FROM-WHERE only
- NO subqueries
- NO GROUP BY, HAVING, ORDER BY
- NO aggregate functions (COUNT, AVG, SUM, MIN, MAX)
- Predicates only in WHERE
- Joins must be explicit JOIN ... ON
- Use AND to combine predicates

General rules:
- Valid standard SQL
- Use only this schema
- Self-joins allowed
- Return JSON array only
""".strip()


def render_prompt(schema: str, batch_size: int, batch_id: int) -> str:
    start_id = batch_id * batch_size + 1
    end_id = start_id + batch_size - 1
    
    prompt = BASE_PROMPT.replace("{start_id}", str(start_id)).replace("{end_id}", str(end_id)).replace("{SCHEMA}", schema)

    # if avoid_block:
    #     prompt += f"""

    #     Avoid repeating the following patterns:
    #     {avoid_block}
    #     """

    prompt += f"""
        Generate exactly {batch_size} new queries.
        """.strip()

    return prompt
