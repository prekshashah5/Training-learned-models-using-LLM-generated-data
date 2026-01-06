# prompt.py

OLDER_PROMPT = """
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

You are SQL Query Generator. 
""".strip()

BASE_PROMPT = """
Generate {batch_size} unique SQL SELECT queries as a JSON array of {batch_size} objects with keys:
"id": Q{start_id}..Q{end_id}, "sql": "<SQL SELECT>", "type": "point|range|agg|multi_predicate", "reasoning": "<short reasoning behind the query logic>" 
Schema:
{SCHEMA}

Rules:
1. Only valid standard SQL SELECTs.
2. No subqueries, GROUP BY, HAVING, ORDER BY, aggregate functions.
3. Include all types: point, range, multi_predicate, joins, multi_joins
4. Use varied predicates: =, >, <, BETWEEN, LIKE, IN.
5. Return meaningful columns per query.
6. Keep "reasoning" concise.
7. Output valid JSON array only — no text or markdown.
""".strip()


def render_prompt(schema: str, batch_size: int, batch_id: int) -> str:
    start_id = batch_id * batch_size + 1
    end_id = start_id + batch_size - 1

    prompt = BASE_PROMPT.replace("{batch_size}", str(batch_size)).replace("{start_id}", str(start_id)).replace("{end_id}", str(end_id)).replace("{SCHEMA}", schema)

    # if avoid_block:
    #     prompt += f"""

    #     Avoid repeating the following patterns:
    #     {avoid_block}
    #     """

    prompt += f"""
        Generate exactly {batch_size} new queries.
        """.strip()

    return prompt
