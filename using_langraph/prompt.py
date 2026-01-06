BASE_PROMPT = """
Generate {batch_size} unique SQL SELECT queries as a JSON array of {batch_size} objects with keys:
"id": Q{start_id}..Q{end_id}, "sql": "<SQL SELECT>", "type": "point|range|agg|multi_predicate", "reasoning": "<short reasoning behind the query logic>" 
Schema:
{SCHEMA}

Rules:
1. Only valid standard SQL SELECTs.
2. Include all types: point, range, multi_predicate, aggregation, joins, multi_joins
3. Use varied predicates: =, >, <, BETWEEN, LIKE, IN.
4. Return meaningful columns per query.
5. Keep "reasoning" concise (≤12 words).
6. Output valid JSON array only — no text or markdown.
""".strip()


def render_prompt(schema: str, batch_size: int, batch_id: int) -> str:
    start_id = batch_id * batch_size + 1
    end_id = start_id + batch_size - 1

    prompt = BASE_PROMPT.replace("{batch_size}", str(batch_size)).replace("{start_id}", str(start_id)).replace("{end_id}", str(end_id)).replace("{SCHEMA}", schema)

    prompt += f"""
        Generate exactly {batch_size} new queries.
        """.strip()

    return prompt