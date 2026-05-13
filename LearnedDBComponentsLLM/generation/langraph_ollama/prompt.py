"""
prompt.py
LLM prompt templates for query generation and fixing.
"""

BASE_PROMPT = """
Generate {batch_size} SQL SELECT queries as a JSON array of {batch_size} objects with keys:
"id": Q1..Q{batch_size}, "sql": "<SQL SELECT>", "type": "point|range|multi_predicate|join|multi_join|low_selectivity", "reasoning": "<short factual reasoning>"

Schema:
{SCHEMA}

Column Statistics (Use these to generate REALISTIC filter constants):
{STATS}

CRITICAL UNIQUENESS CONSTRAINT:
- Queries are duplicates if they use the same tables and joins, even with different constants.
- Queries that differ only by literal values are NOT unique.
- Each query must represent a distinct access pattern over the schema.

Rules:
1. Only valid standard SQL SELECT statements.
2. PROHIBITED: No subqueries, GROUP BY, HAVING, ORDER BY, or aggregate functions (COUNT, SUM, AVG, etc.).
3. Do NOT use SELECT * - select meaningful columns only.
4. Use explicit JOIN ... ON syntax only.
5. You MUST generate exactly:
   - 3 highly selective queries (point predicates on keys)
   - 3 range predicate queries (>, <, BETWEEN)
   - 2 low selective queries (LIKE, IN, OR)
   - 2 very low selective queries (multi-table joins with minimal filtering)
6. Use varied predicates across queries: =, >, <, BETWEEN, LIKE, IN, IS NULL.
7. Include both single-table and multi-table queries.
8. REALISM: Use the Column Statistics to ensure filters return a non-zero number of rows.
9. PRODUCTION STANDARDS:
   - Always fully qualify columns (table.column) when more than one table is involved.
   - Use meaningful aliases for tables.
10. Output a valid JSON array only - no extra text.

FINAL VALIDATION (perform silently before output):
- Verify all tables and columns exist in the schema.
- Verify no aggregates or grouping exists.
- Verify query type counts are exact.
- Verify all queries are structurally distinct from each other and previous ones.
""".strip()

FIX_PROMPT = """
The following SQL query is invalid for the given schema.

Schema:
{SCHEMA}

SQL:
{SQL}

Error message from PostgreSQL:
{ERROR}

Rules:
- Fix ONLY what is necessary.
- If a column name exists in multiple joined tables, it MUST be fully qualified using table.column syntax in SELECT and WHERE clauses.
- Preserve the original query intent.
- Do NOT introduce subqueries, GROUP BY, ORDER BY, or aggregates.
- Output ONLY the corrected SQL.
Provide the corrected SQL query below:
""".strip()
