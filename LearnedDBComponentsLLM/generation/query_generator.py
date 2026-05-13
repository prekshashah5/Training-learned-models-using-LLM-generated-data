"""
query_generator.py
Generates unlabeled SQL queries using Ollama LLM and converts them
to MSCN-compatible structured format.

Does NOT execute any query on the database - only generates query structures.
Includes schema-aware validation to reject invalid queries before they
enter the pipeline.
"""

import json
import re
import time
import os
import csv
from datetime import datetime
import requests
from typing import List, Dict, Optional, Tuple, Set

from generation.format_converter import parse_sql_to_mscn


# ═══════════════════════════════════════════════════════════════════════════
# Schema Validator - parses DDL and validates queries without DB access
# ═══════════════════════════════════════════════════════════════════════════

class SchemaValidator:
    """
    Parses a DDL schema text (CREATE TABLE statements) into a structured
    representation and validates SQL queries against it.

    No database connection is needed - all validation is purely in-memory.
    """

    # Column types considered numeric (predicates allowed)
    NUMERIC_TYPES = {
        'integer', 'int', 'smallint', 'bigint', 'serial', 'bigserial',
        'numeric', 'decimal', 'real', 'float', 'double precision',
        'boolean', 'bool',
    }

    # Column types considered text (predicates NOT allowed for MSCN)
    TEXT_TYPES = {
        'text', 'varchar', 'char', 'character varying', 'character',
        'bytea', 'uuid', 'json', 'jsonb', 'xml',
    }

    def __init__(self, schema_text: str, stats_text: str = ""):
        """
        Parse DDL schema text and optional stats text.

        Args:
            schema_text: SQL DDL containing CREATE TABLE statements.
            stats_text:  Optional column statistics (min, max, cardinality).
        """
        # table_name -> {"columns": {col_name: col_type}, "pk": str|None, "fks": {col: "ref_table.ref_col"}}
        self.tables: Dict[str, Dict] = {}
        # Set of "table_name.column_name" for numeric columns
        self.numeric_columns: Set[str] = set()
        # Set of "table_name.column_name" for text columns
        self.text_columns: Set[str] = set()
        # Set of valid join condition strings, e.g. {"title_principals.tconst=title_basics.tconst"}
        self.valid_joins: Set[Tuple[str, str]] = set()
        # Column value ranges from stats: "table.column" -> (min, max)
        self.column_ranges: Dict[str, Tuple[float, float]] = {}

        self._parse_ddl(schema_text)
        if stats_text:
            self._parse_stats(stats_text)

    def _parse_ddl(self, ddl: str):
        """Parse CREATE TABLE statements from DDL text."""
        # Find all CREATE TABLE blocks
        pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);'
        for match in re.finditer(pattern, ddl, re.IGNORECASE | re.DOTALL):
            table_name = match.group(1).strip().lower()
            body = match.group(2).strip()

            columns = {}
            pk = None
            fks = {}

            # Split body into individual definitions
            # We need to handle multi-line CONSTRAINT ... FOREIGN KEY blocks
            defs = self._split_column_defs(body)

            for col_def in defs:
                col_def = col_def.strip()
                if not col_def:
                    continue

                # Check for CONSTRAINT ... FOREIGN KEY
                fk_match = re.search(
                    r'FOREIGN\s+KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)',
                    col_def, re.IGNORECASE
                )
                if fk_match:
                    fk_col = fk_match.group(1).lower()
                    ref_table = fk_match.group(2).lower()
                    ref_col = fk_match.group(3).lower()
                    fks[fk_col] = f"{ref_table}.{ref_col}"
                    continue

                # Check for standalone CONSTRAINT (non-FK, e.g. CHECK)
                if re.match(r'CONSTRAINT', col_def, re.IGNORECASE):
                    continue

                # Check for PRIMARY KEY constraint
                pk_match = re.match(r'PRIMARY\s+KEY\s*\((.+?)\)', col_def, re.IGNORECASE)
                if pk_match:
                    pk = pk_match.group(1).strip().lower()
                    continue

                # Parse column definition: column_name TYPE [constraints...]
                col_match = re.match(r'(\w+)\s+(.+)', col_def)
                if col_match:
                    col_name = col_match.group(1).lower()
                    col_type_raw = col_match.group(2).strip().lower()

                    # Extract base type (remove size specs like VARCHAR(10))
                    col_type = re.split(r'[\s(]', col_type_raw)[0]

                    # Handle "double precision" as a special case
                    if col_type == 'double' and 'precision' in col_type_raw:
                        col_type = 'double precision'

                    # Handle "character varying"
                    if col_type == 'character' and 'varying' in col_type_raw:
                        col_type = 'character varying'

                    columns[col_name] = col_type

                    # Check for inline PRIMARY KEY
                    if 'primary key' in col_type_raw:
                        pk = col_name

                    # Classify column type
                    full_col = f"{table_name}.{col_name}"
                    if col_type in self.NUMERIC_TYPES:
                        self.numeric_columns.add(full_col)
                    else:
                        self.text_columns.add(full_col)

            self.tables[table_name] = {
                "columns": columns,
                "pk": pk,
                "fks": fks,
            }

        # Build valid join pairs from FK relationships
        self._build_valid_joins()

    def _split_column_defs(self, body: str) -> List[str]:
        """
        Split CREATE TABLE body into individual column/constraint definitions.
        Handles multi-line CONSTRAINT blocks by tracking parenthesis depth.
        """
        defs = []
        current = []
        depth = 0

        for char in body:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                defs.append(''.join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            defs.append(''.join(current).strip())

        return defs

    def _build_valid_joins(self):
        """Build the set of valid join conditions from FK relationships."""
        for table_name, info in self.tables.items():
            for fk_col, ref in info["fks"].items():
                ref_table, ref_col = ref.split('.')
                # Store both directions as valid
                self.valid_joins.add((f"{table_name}.{fk_col}", f"{ref_table}.{ref_col}"))
                self.valid_joins.add((f"{ref_table}.{ref_col}", f"{table_name}.{fk_col}"))

    def _parse_stats(self, stats_text: str):
        """Parse column statistics text for value ranges."""
        # Try to parse lines like: "table.column: min=X, max=Y"
        # or "column (table): min X, max Y"
        for line in stats_text.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Pattern: "table.column: min=X, max=Y" or similar
            match = re.search(
                r'(\w+\.\w+)\s*:?\s*min\s*[=:]?\s*([\d.]+)\s*,?\s*max\s*[=:]?\s*([\d.]+)',
                line, re.IGNORECASE
            )
            if match:
                col = match.group(1).lower()
                min_val = float(match.group(2))
                max_val = float(match.group(3))
                self.column_ranges[col] = (min_val, max_val)

    def get_table_names(self) -> List[str]:
        """Return list of all known table names."""
        return list(self.tables.keys())

    def get_numeric_columns_for_table(self, table_name: str) -> List[str]:
        """Return numeric column names for a given table."""
        result = []
        table_name = table_name.lower()
        if table_name in self.tables:
            for col_name, col_type in self.tables[table_name]["columns"].items():
                if f"{table_name}.{col_name}" in self.numeric_columns:
                    result.append(col_name)
        return result

    def get_valid_joins_for_tables(self, table_names: List[str]) -> List[Tuple[str, str]]:
        """Return valid join conditions between the given tables."""
        table_set = {t.lower() for t in table_names}
        result = []
        seen = set()
        for left, right in self.valid_joins:
            left_table = left.split('.')[0]
            right_table = right.split('.')[0]
            if left_table in table_set and right_table in table_set:
                key = tuple(sorted([left, right]))
                if key not in seen:
                    seen.add(key)
                    result.append((left, right))
        return result

    def validate_query(self, sql: str) -> Tuple[bool, str]:
        """
        Validate a SQL query against the schema.

        Returns:
            (is_valid, rejection_reason)
            If valid, rejection_reason is empty string.
        """
        sql_clean = sql.strip().rstrip(';')
        sql_upper = sql_clean.upper()

        # ── Check 1: Must be SELECT COUNT(*) ──
        if not sql_upper.startswith('SELECT COUNT'):
            return False, "Not a SELECT COUNT(*) query"

        # ── Check 2: Forbidden SQL patterns ──
        bad_patterns = [' OR ', ' IN (', ' IN(', ' LIKE ', ' BETWEEN ', ' NOT ',
                        ' IS NULL', ' IS NOT NULL', ' EXISTS',
                        ' UNION ', ' HAVING ', ' GROUP BY ', ' ORDER BY ',
                        ' LIMIT ', ' DISTINCT ', ' CASE ', ' WHEN ']
        where_idx = sql_upper.find('WHERE')
        if where_idx > 0:
            where_clause = sql_upper[where_idx:]
            for pattern in bad_patterns:
                if pattern in where_clause:
                    return False, f"Forbidden SQL pattern: {pattern.strip()}"

        # ── Check 3: No subqueries ──
        if sql_upper.count('SELECT') > 1:
            return False, "Contains subquery"

        # ── Check 4: Parse tables from FROM clause ──
        from_match = re.search(
            r'\bFROM\b\s+(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)',
            sql_clean, re.IGNORECASE | re.DOTALL
        )
        if not from_match:
            return False, "Cannot parse FROM clause"

        from_clause = from_match.group(1).strip()
        # Split by comma and JOIN keywords
        table_parts = re.split(
            r'\b(?:INNER\s+)?(?:LEFT\s+)?(?:RIGHT\s+)?(?:CROSS\s+)?(?:FULL\s+)?JOIN\b|,',
            from_clause, flags=re.IGNORECASE
        )

        alias_to_table = {}
        for part in table_parts:
            part = part.strip()
            if not part:
                continue
            # Remove ON clause
            on_idx = re.search(r'\bON\b', part, re.IGNORECASE)
            if on_idx:
                part = part[:on_idx.start()].strip()

            tokens = part.split()
            if not tokens:
                continue

            table_name = tokens[0].lower()
            if len(tokens) >= 2 and tokens[1].upper() not in ('ON', 'WHERE', 'JOIN', 'INNER', 'LEFT'):
                alias = tokens[-1].lower()
                if tokens[1].upper() == 'AS' and len(tokens) > 2:
                    alias = tokens[2].lower()
            else:
                alias = table_name

            # Validate table exists
            if table_name not in self.tables:
                return False, f"Unknown table: {table_name}"

            alias_to_table[alias] = table_name

        if not alias_to_table:
            return False, "No tables found in FROM clause"

        # ── Check 5: Parse WHERE clause and validate conditions ──
        where_match = re.search(
            r'\bWHERE\b\s+(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)',
            sql_clean, re.IGNORECASE | re.DOTALL
        )

        if where_match:
            where_clause_raw = where_match.group(1).strip()
            conditions = re.split(r'\bAND\b', where_clause_raw, flags=re.IGNORECASE)

            join_count = 0
            for cond in conditions:
                cond = cond.strip().strip('()')
                if not cond:
                    continue

                # Check if join condition (col = col)
                join_match = re.match(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)$', cond.strip())
                if join_match:
                    left_alias = join_match.group(1).lower()
                    left_col = join_match.group(2).lower()
                    right_alias = join_match.group(3).lower()
                    right_col = join_match.group(4).lower()

                    # Validate aliases exist
                    if left_alias not in alias_to_table:
                        return False, f"Unknown alias in join: {left_alias}"
                    if right_alias not in alias_to_table:
                        return False, f"Unknown alias in join: {right_alias}"

                    left_table = alias_to_table[left_alias]
                    right_table = alias_to_table[right_alias]

                    # Validate columns exist
                    if left_col not in self.tables[left_table]["columns"]:
                        return False, f"Unknown column in join: {left_table}.{left_col}"
                    if right_col not in self.tables[right_table]["columns"]:
                        return False, f"Unknown column in join: {right_table}.{right_col}"

                    # Validate this is a valid FK join
                    left_full = f"{left_table}.{left_col}"
                    right_full = f"{right_table}.{right_col}"
                    if (left_full, right_full) not in self.valid_joins:
                        return False, f"Invalid join (not a FK relationship): {left_full} = {right_full}"

                    join_count += 1
                    continue

                # Check if filter predicate (col op value)
                pred_match = re.match(r'(\w+)\.(\w+)\s*(=|!=|<>|<=|>=|<|>)\s*(.+)$', cond.strip())
                if pred_match:
                    alias = pred_match.group(1).lower()
                    col = pred_match.group(2).lower()
                    op = pred_match.group(3)
                    val = pred_match.group(4).strip().strip("'\"")

                    # Validate alias exists
                    if alias not in alias_to_table:
                        return False, f"Unknown alias in predicate: {alias}"

                    table_name = alias_to_table[alias]

                    # Validate column exists
                    if col not in self.tables[table_name]["columns"]:
                        return False, f"Unknown column: {table_name}.{col}"

                    # Validate column is numeric
                    full_col = f"{table_name}.{col}"
                    if full_col in self.text_columns:
                        return False, f"Predicate on text column: {full_col}"

                    # Validate value is numeric
                    try:
                        float(val)
                    except ValueError:
                        return False, f"Non-numeric predicate value: {val} on {full_col}"

                    # Validate operator is supported
                    if op not in ('=', '<', '>', '<=', '>=', '!=', '<>'):
                        return False, f"Unsupported operator: {op}"

                    continue

                # If we get here, the condition didn't match any known pattern
                return False, f"Unparseable condition: {cond.strip()}"

            # Check that multi-table queries have join conditions
            if len(alias_to_table) > 1 and join_count == 0:
                return False, "Multi-table query without join conditions"

        else:
            # No WHERE clause - only valid for single-table queries
            if len(alias_to_table) > 1:
                return False, "Multi-table query without WHERE clause"

        return True, ""

    def get_schema_summary_for_prompt(self) -> str:
        """
        Generate a concise schema summary with column types clearly annotated.
        This helps the LLM understand which columns are numeric vs text.
        """
        lines = []
        for table_name, info in self.tables.items():
            cols = []
            for col_name, col_type in info["columns"].items():
                full_col = f"{table_name}.{col_name}"
                type_label = "NUMERIC" if full_col in self.numeric_columns else "TEXT"
                pk_label = " [PK]" if col_name == info.get("pk") else ""
                fk_label = ""
                if col_name in info.get("fks", {}):
                    fk_label = f" [FK -> {info['fks'][col_name]}]"
                cols.append(f"    {col_name} ({col_type}) [{type_label}]{pk_label}{fk_label}")

            lines.append(f"TABLE: {table_name}")
            lines.extend(cols)
            lines.append("")

        # Add join info
        lines.append("VALID JOIN CONDITIONS:")
        seen = set()
        for left, right in self.valid_joins:
            key = tuple(sorted([left, right]))
            if key not in seen:
                seen.add(key)
                lines.append(f"    {left} = {right}")

        return "\n".join(lines)


# ── Dynamic Schema Prompt ────────────────────────────────────────────────

GENERATION_PROMPT = """You are a SQL query workload generator for a database.

Here is the database schema with column types clearly marked:
{schema}

Here are the column statistics (min, max values):
{stats}

Generate exactly {batch_size} SQL queries. Each query MUST follow these STRICT rules:
1. Use SELECT COUNT(*) FROM ... WHERE ... format ONLY
2. Use 1-5 tables from the schema above
3. Use proper table aliases (e.g., title_basics tb)
4. For multi-table queries, ALWAYS include join conditions using ONLY the VALID JOIN CONDITIONS listed above
5. Include 0-4 filter predicates using ONLY =, <, > operators
6. Use ONLY [NUMERIC] columns for filter predicates - NEVER use [TEXT] columns in predicates
7. Use ONLY integer numeric values in predicates - NEVER use string values like 'Drama' or 'Action'
8. ALL conditions in WHERE must be connected with AND only
9. Do NOT use OR, IN, LIKE, IS NULL, IS NOT NULL, BETWEEN, NOT, subqueries, DISTINCT, HAVING, GROUP BY, ORDER BY, LIMIT
10. Each predicate must be a simple: alias.column operator integer_value (e.g., tb.startYear > 2000)

EXAMPLES OF VALID QUERIES:
{{"sql": "SELECT COUNT(*) FROM title_basics tb WHERE tb.startYear > 2000 AND tb.runtimeMinutes < 120"}}
{{"sql": "SELECT COUNT(*) FROM title_basics tb, title_ratings tr WHERE tb.tconst = tr.tconst AND tr.num_votes > 1000"}}
{{"sql": "SELECT COUNT(*) FROM title_basics tb, title_principals tp, name_basics nb WHERE tb.tconst = tp.tconst AND tp.nconst = nb.nconst AND tb.startYear > 1990"}}

EXAMPLES OF INVALID QUERIES (DO NOT GENERATE THESE):
- Using text columns: WHERE tb.genres = 'Drama'  (WRONG - genres is TEXT)
- Using string values: WHERE tb.primaryTitle = 'Inception'  (WRONG - string value)
- Missing join: SELECT COUNT(*) FROM title_basics tb, title_ratings tr WHERE tr.num_votes > 100  (WRONG - no join condition)
- Using OR: WHERE tb.startYear > 2000 OR tb.runtimeMinutes < 90  (WRONG - uses OR)

STRUCTURAL TARGETS:
{structure_hint}

{diversity_hint}

Return ONLY a JSON array of objects, each with a "sql" field:
[
  {{"sql": "SELECT COUNT(*) FROM title_basics tb WHERE tb.startYear > 2000 AND tb.runtimeMinutes < 120"}},
  {{"sql": "SELECT COUNT(*) FROM title_basics tb, title_ratings tr WHERE tb.tconst = tr.tconst AND tr.num_votes > 1000"}},
  ...
]
"""


JOIN_PRIORITY_ORDER = [1, 2, 0, 3, 4]
JOIN_TARGET_RATIOS = {
    1: 0.36,
    2: 0.28,
    0: 0.18,
    3: 0.12,
    4: 0.06,
}


def get_join_count(sql: str) -> Optional[int]:
    """Return the number of join predicates in a SQL query."""
    parsed = parse_sql_to_mscn(sql)
    if not parsed:
        return None
    return len(parsed.get("joins", []))


def build_structure_hint(batch_size: int, current_join_counts: Optional[Dict[int, int]] = None) -> str:
    """Build an explicit join-count target for the next batch."""
    current_join_counts = current_join_counts or {}
    total_existing = sum(current_join_counts.values())
    target_total_after_batch = total_existing + batch_size

    desired_batch = {}
    assigned = 0
    for join_count in JOIN_PRIORITY_ORDER:
        desired_total = round(target_total_after_batch * JOIN_TARGET_RATIOS[join_count])
        needed = max(desired_total - current_join_counts.get(join_count, 0), 0)
        desired_batch[join_count] = min(needed, batch_size - assigned)
        assigned += desired_batch[join_count]

    for join_count in JOIN_PRIORITY_ORDER:
        if assigned >= batch_size:
            break
        desired_batch[join_count] += 1
        assigned += 1

    lines = [
        "Prefer joined queries over single-table queries.",
        "Desired frequency order by join count: 1 join > 2 joins > 0 joins > 3 joins > 4 joins.",
        "When possible, match this approximate batch composition:",
    ]
    for join_count in JOIN_PRIORITY_ORDER:
        table_count = join_count + 1
        lines.append(
            f"- {desired_batch[join_count]} queries with {join_count} joins ({table_count} tables)"
        )
    lines.append("If the schema cannot support a target exactly, stay as close as possible while preserving validity and diversity.")
    return "\n".join(lines)


def select_queries_by_join_priority(
    sqls: List[str],
    batch_size: int,
    current_join_counts: Optional[Dict[int, int]] = None,
) -> Tuple[List[str], Dict[int, int]]:
    """Pick queries from a candidate batch to match the preferred join-count mix."""
    current_join_counts = current_join_counts or {}
    target_total_after_batch = sum(current_join_counts.values()) + batch_size

    desired_totals = {
        join_count: round(target_total_after_batch * JOIN_TARGET_RATIOS[join_count])
        for join_count in JOIN_PRIORITY_ORDER
    }
    desired_batch = {
        join_count: max(desired_totals[join_count] - current_join_counts.get(join_count, 0), 0)
        for join_count in JOIN_PRIORITY_ORDER
    }

    buckets: Dict[int, List[str]] = {join_count: [] for join_count in JOIN_PRIORITY_ORDER}
    overflow: List[str] = []
    for sql in sqls:
        join_count = get_join_count(sql)
        if join_count in buckets:
            buckets[join_count].append(sql)
        else:
            overflow.append(sql)

    selected = []
    seen = set()

    for join_count in JOIN_PRIORITY_ORDER:
        for sql in buckets[join_count][:desired_batch[join_count]]:
            if sql not in seen:
                selected.append(sql)
                seen.add(sql)
            if len(selected) >= batch_size:
                break
        if len(selected) >= batch_size:
            break

    for join_count in JOIN_PRIORITY_ORDER:
        for sql in buckets[join_count]:
            if sql in seen:
                continue
            selected.append(sql)
            seen.add(sql)
            if len(selected) >= batch_size:
                break
        if len(selected) >= batch_size:
            break

    for sql in overflow:
        if sql in seen:
            continue
        selected.append(sql)
        seen.add(sql)
        if len(selected) >= batch_size:
            break

    selected_counts = {}
    for sql in selected:
        jc = get_join_count(sql)
        if jc is not None:
            selected_counts[jc] = selected_counts.get(jc, 0) + 1

    deficits = {}
    for join_count in JOIN_PRIORITY_ORDER:
        want = desired_batch.get(join_count, 0)
        have = selected_counts.get(join_count, 0)
        if want > have:
            deficits[join_count] = want - have

    return selected, deficits


def generate_targeted_queries_for_join_count(
    join_count: int,
    need: int,
    schema_text: str,
    stats_text: str,
    model_name: str,
    ollama_url: str,
    schema_validator: Optional[SchemaValidator],
    max_attempts: int = 3,
) -> List[str]:
    """Generate additional queries targeting an exact join count to fill deficits."""
    if need <= 0:
        return []

    if schema_validator is not None:
        schema_for_prompt = schema_validator.get_schema_summary_for_prompt()
    else:
        schema_for_prompt = schema_text

    table_count = join_count + 1
    targeted_hint = (
        f"Critical target: return EXACTLY {need} queries with EXACTLY {join_count} joins "
        f"(i.e., about {table_count} joined tables) as much as schema allows."
    )

    prompt = GENERATION_PROMPT.format(
        schema=schema_for_prompt,
        stats=stats_text or "No specific stats available.",
        batch_size=max(int(need * 2.5), need + 4),
        structure_hint=targeted_hint,
        diversity_hint="Hint: prioritize complex valid join paths and keep filters simple.",
    )

    accepted = []
    seen = set()
    for _ in range(max_attempts):
        raw_response = call_ollama(prompt, model_name, ollama_url)
        if raw_response is None:
            continue

        queries_json = extract_json_array(raw_response)
        if queries_json is None:
            continue

        for q in queries_json:
            if not (isinstance(q, dict) and "sql" in q):
                continue
            sql = q["sql"]
            if sql in seen:
                continue
            if schema_validator is not None:
                is_valid, _ = schema_validator.validate_query(sql)
                if not is_valid:
                    continue
            jc = get_join_count(sql)
            if jc != join_count:
                continue

            seen.add(sql)
            accepted.append(sql)
            if len(accepted) >= need:
                return accepted

    return accepted


def validate_sql(sql: str, schema_validator: Optional['SchemaValidator'] = None) -> bool:
    """
    Validate a SQL query. Uses schema-aware validation if a SchemaValidator
    is provided, otherwise falls back to basic keyword filtering.

    Returns True if the query is valid.
    """
    if schema_validator is not None:
        is_valid, reason = schema_validator.validate_query(sql)
        if not is_valid:
            print(f"  [schema-reject] {reason}: {sql[:100]}...")
        return is_valid

    # Fallback: basic validation (no schema context)
    sql_upper = sql.upper().strip()

    if not sql_upper.startswith("SELECT COUNT"):
        return False

    bad_patterns = [' OR ', ' IN (', ' LIKE ', ' BETWEEN ', ' NOT ',
                    ' IS NULL', ' IS NOT NULL', ' EXISTS',
                    ' UNION ', ' HAVING ', ' GROUP BY ']
    where_idx = sql_upper.find('WHERE')
    if where_idx > 0:
        where_clause = sql_upper[where_idx:]
        for pattern in bad_patterns:
            if pattern in where_clause:
                return False

    if sql_upper.count('SELECT') > 1:
        return False

    return True


def call_ollama(prompt: str,
                model_name: str = "llama3.2",
                ollama_url: str = "http://localhost:11434",
                temperature: float = 0.8) -> Optional[str]:
    """Call Ollama API to generate a response."""
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 4096,
                }
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"[query_generator] Ollama error: {e}")
        return None


def extract_json_array(text: str) -> Optional[List[Dict]]:
    """Extract a JSON array from LLM output text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _distribution(values: List[int]) -> Dict[str, int]:
    """Build a frequency map with string keys for JSON serialization."""
    freq: Dict[str, int] = {}
    for v in values:
        k = str(v)
        freq[k] = freq.get(k, 0) + 1
    return dict(sorted(freq.items(), key=lambda kv: int(kv[0])))


def _summarize_sqls(sqls: List[str]) -> Dict[str, object]:
    """Compute structural metrics from generated SQL strings."""
    table_counts: List[int] = []
    join_counts: List[int] = []
    predicate_counts: List[int] = []
    parsed_ok = 0

    for sql in sqls:
        parsed = parse_sql_to_mscn(sql)
        if not parsed:
            continue
        parsed_ok += 1
        table_counts.append(len(parsed.get("tables", [])))
        join_counts.append(len(parsed.get("joins", [])))
        predicate_counts.append(len(parsed.get("predicates", [])))

    def _avg(values: List[int]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "parsed_query_count": parsed_ok,
        "avg_tables_per_query": _avg(table_counts),
        "avg_joins_per_query": _avg(join_counts),
        "avg_predicates_per_query": _avg(predicate_counts),
        "tables_distribution": _distribution(table_counts),
        "joins_distribution": _distribution(join_counts),
        "predicates_distribution": _distribution(predicate_counts),
    }


def _write_generation_metrics(save_dir: str,
                              timestamp: str,
                              summary: Dict[str, object],
                              structure: Dict[str, object],
                              stats: Dict[str, object]) -> None:
    """Write generation metrics artifacts for thesis/reporting."""
    metrics = {
        "run_summary": summary,
        "generation_stats": stats,
        "structure_metrics": structure,
    }

    metrics_json_path = os.path.join(save_dir, f"generation_metrics_{timestamp}.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_csv_path = os.path.join(save_dir, f"generation_summary_{timestamp}.csv")
    csv_row = {
        "timestamp": summary.get("timestamp"),
        "model_name": summary.get("model_name"),
        "ollama_url": summary.get("ollama_url"),
        "requested_queries": summary.get("requested_queries"),
        "generated_queries": summary.get("generated_queries"),
        "batch_size": summary.get("batch_size"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "batches_attempted": stats.get("batches_attempted", 0),
        "ollama_calls": stats.get("ollama_calls", 0),
        "json_parse_failures": stats.get("json_parse_failures", 0),
        "schema_rejections": stats.get("schema_rejections", 0),
        "accepted_before_selection": stats.get("accepted_before_selection", 0),
        "selected_after_priority": stats.get("selected_after_priority", 0),
        "empty_batches": stats.get("empty_batches", 0),
        "avg_tables_per_query": structure.get("avg_tables_per_query", 0.0),
        "avg_joins_per_query": structure.get("avg_joins_per_query", 0.0),
        "avg_predicates_per_query": structure.get("avg_predicates_per_query", 0.0),
    }

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writeheader()
        writer.writerow(csv_row)

    print(f"[query_generator] Metrics saved to: {metrics_json_path}")
    print(f"[query_generator] Summary CSV saved to: {summary_csv_path}")


def generate_queries_batch(batch_size: int,
                            schema_text: str,
                            stats_text: str,
                            model_name: str = "llama3.2",
                            ollama_url: str = "http://localhost:11434",
                            existing_count: int = 0,
                            max_retries: int = 3,
                            schema_validator: Optional[SchemaValidator] = None,
                            current_join_counts: Optional[Dict[int, int]] = None,
                            generation_stats: Optional[Dict[str, object]] = None) -> List[str]:
    """
    Generate a batch of SQL queries using Ollama.
    If a SchemaValidator is provided, each query is validated against the schema
    and invalid queries are filtered out.
    """
    diversity_hints = [
        "Focus on single-table queries with various filters.",
        "Focus on 2-table join queries with 1-2 predicates.",
        "Focus on 3-table join queries with multiple predicates.",
        "Mix simple and complex queries. Include point lookups (=) and range scans (<, >).",
        "Generate queries with extreme selectivities: some returning very few rows, some returning millions.",
    ]

    hint_idx = (existing_count // batch_size) % len(diversity_hints)
    diversity_hint = f"Hint: {diversity_hints[hint_idx]}"
    structure_hint = build_structure_hint(batch_size, current_join_counts)

    # Use the enhanced schema summary if validator is available
    if schema_validator is not None:
        schema_for_prompt = schema_validator.get_schema_summary_for_prompt()
    else:
        schema_for_prompt = schema_text

    # Request extra queries to compensate for validation rejections
    request_size = max(int(batch_size * 1.8), batch_size + 4) if schema_validator else batch_size

    prompt = GENERATION_PROMPT.format(
        schema=schema_for_prompt,
        stats=stats_text or "No specific stats available.",
        batch_size=request_size,
        structure_hint=structure_hint,
        diversity_hint=diversity_hint,
    )

    for attempt in range(max_retries):
        if generation_stats is not None:
            generation_stats["ollama_calls"] = generation_stats.get("ollama_calls", 0) + 1
        raw_response = call_ollama(prompt, model_name, ollama_url)
        if raw_response is None:
            time.sleep(2)
            continue

        queries_json = extract_json_array(raw_response)
        if queries_json is None:
            print(f"[query_generator] Failed to parse JSON (attempt {attempt + 1})")
            if generation_stats is not None:
                generation_stats["json_parse_failures"] = generation_stats.get("json_parse_failures", 0) + 1
            time.sleep(1)
            continue

        sqls = []
        rejected = 0
        for q in queries_json:
            if isinstance(q, dict) and "sql" in q:
                sql = q["sql"]
                if schema_validator is not None:
                    is_valid, reason = schema_validator.validate_query(sql)
                    if not is_valid:
                        rejected += 1
                        if generation_stats is not None:
                            generation_stats["schema_rejections"] = generation_stats.get("schema_rejections", 0) + 1
                            reason_counts = generation_stats.setdefault("rejection_reasons", {})
                            reason_counts[reason] = reason_counts.get(reason, 0) + 1
                        continue
                sqls.append(sql)

        if generation_stats is not None:
            generation_stats["accepted_before_selection"] = generation_stats.get("accepted_before_selection", 0) + len(sqls)

        if rejected > 0:
            print(f"[query_generator] Batch {attempt + 1}: accepted {len(sqls)}, rejected {rejected}")

        if sqls:
            selected_sqls, deficits = select_queries_by_join_priority(sqls, batch_size, current_join_counts)

            # Targeted refill for missing buckets, prioritizing more informative join counts first.
            refill_priority = [2, 1, 3, 4, 0]
            for target_jc in refill_priority:
                if len(selected_sqls) >= batch_size:
                    break
                need = deficits.get(target_jc, 0)
                if need <= 0:
                    continue
                extra = generate_targeted_queries_for_join_count(
                    join_count=target_jc,
                    need=min(need, batch_size - len(selected_sqls)),
                    schema_text=schema_text,
                    stats_text=stats_text,
                    model_name=model_name,
                    ollama_url=ollama_url,
                    schema_validator=schema_validator,
                )
                for sql in extra:
                    if sql not in selected_sqls:
                        selected_sqls.append(sql)
                    if len(selected_sqls) >= batch_size:
                        break

            # Final backfill if targeted generation still couldn't fill the batch.
            if len(selected_sqls) < batch_size:
                for sql in sqls:
                    if sql in selected_sqls:
                        continue
                    selected_sqls.append(sql)
                    if len(selected_sqls) >= batch_size:
                        break

            selected_join_counts = {}
            for sql in selected_sqls:
                join_count = get_join_count(sql)
                if join_count is not None:
                    selected_join_counts[join_count] = selected_join_counts.get(join_count, 0) + 1
            if generation_stats is not None:
                generation_stats["selected_after_priority"] = generation_stats.get("selected_after_priority", 0) + len(selected_sqls)
            print(f"[query_generator] Selected join mix: {selected_join_counts}")
            return selected_sqls

    print(f"[query_generator] All {max_retries} attempts failed for batch")
    return []


def generate_all_queries(total_queries: int,
                          schema_text: str,
                          stats_text: str,
                          batch_size: int = 20,
                          model_name: str = "llama3.2",
                          ollama_url: str = "http://localhost:11434",
                          schema_validator: Optional[SchemaValidator] = None) -> List[str]:
    """
    Generate the full set of unlabeled SQL queries.
    
    If schema_validator is provided, queries are validated against the schema
    during generation (not after), ensuring only valid queries are returned.
    """
    run_start = time.time()
    all_sqls = []
    num_batches = (total_queries + batch_size - 1) // batch_size
    generation_stats: Dict[str, object] = {
        "batches_attempted": 0,
        "empty_batches": 0,
        "ollama_calls": 0,
        "json_parse_failures": 0,
        "schema_rejections": 0,
        "accepted_before_selection": 0,
        "selected_after_priority": 0,
        "rejection_reasons": {},
    }

    # Build schema validator from DDL if not provided
    if schema_validator is None and schema_text:
        try:
            schema_validator = SchemaValidator(schema_text, stats_text)
            print(f"[query_generator] Built schema validator: {len(schema_validator.tables)} tables, "
                  f"{len(schema_validator.numeric_columns)} numeric cols, "
                  f"{len(schema_validator.valid_joins)//2} join conditions")
        except Exception as e:
            print(f"[query_generator] WARNING: Could not build schema validator: {e}")
            schema_validator = None

    print(f"[query_generator] Generating {total_queries} queries in {num_batches} batches...")

    consecutive_empty = 0
    current_join_counts: Dict[int, int] = {}
    for b in range(num_batches * 2):  # Allow extra batches to compensate for rejections
        generation_stats["batches_attempted"] = generation_stats.get("batches_attempted", 0) + 1
        remaining = total_queries - len(all_sqls)
        current_batch_size = min(batch_size, remaining)

        if current_batch_size <= 0:
            break

        if consecutive_empty >= 5:
            print(f"[query_generator] WARNING: 5 consecutive empty batches. Stopping generation.")
            break

        print(f"[query_generator] Batch {b + 1} (have {len(all_sqls)}, need {remaining} more)")

        sqls = generate_queries_batch(
            batch_size=current_batch_size,
            schema_text=schema_text,
            stats_text=stats_text,
            model_name=model_name,
            ollama_url=ollama_url,
            existing_count=len(all_sqls),
            schema_validator=schema_validator,
            current_join_counts=current_join_counts,
            generation_stats=generation_stats,
        )

        if sqls:
            all_sqls.extend(sqls)
            for sql in sqls:
                join_count = get_join_count(sql)
                if join_count is not None:
                    current_join_counts[join_count] = current_join_counts.get(join_count, 0) + 1
            consecutive_empty = 0
            print(f"[query_generator] Got {len(sqls)} valid queries (total: {len(all_sqls)})")
            print(f"[query_generator] Running join mix: {current_join_counts}")
        else:
            consecutive_empty += 1
            generation_stats["empty_batches"] = generation_stats.get("empty_batches", 0) + 1
            print(f"[query_generator] Empty batch ({consecutive_empty} consecutive)")

        time.sleep(0.5)

    all_sqls = all_sqls[:total_queries]
    print(f"[query_generator] Generation complete: {len(all_sqls)} valid queries")

    # Save generated queries to disk for inspection / reuse
    save_dir = os.path.join("generated_queries")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"queries_{timestamp}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_sqls, f, indent=2)
    print(f"[query_generator] Queries saved to: {save_path}")

    elapsed = round(time.time() - run_start, 3)
    summary = {
        "timestamp": timestamp,
        "model_name": model_name,
        "ollama_url": ollama_url,
        "requested_queries": total_queries,
        "generated_queries": len(all_sqls),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "queries_path": save_path,
    }
    structure = _summarize_sqls(all_sqls)
    _write_generation_metrics(save_dir, timestamp, summary, structure, generation_stats)

    return all_sqls


def generate_synthetic_queries(num_queries: int, seed: int = 42) -> List[Dict]:
    """
    Generate synthetic queries programmatically (no LLM needed).
    Uses the IMDB schema tables that match the actual database.
    Useful for testing or when Ollama is not available.
    """
    import random
    import numpy as np

    random.seed(seed)
    np_rng = np.random.RandomState(seed)

    table_defs = {
        "title_basics tb": {
            "columns": [
                ("tb.startyear", 1880, 2025),
                ("tb.runtimeminutes", 1, 600),
            ],
        },
        "title_ratings tr": {
            "columns": [
                ("tr.average_rating", 1, 10),
                ("tr.num_votes", 1, 2500000),
            ],
        },
        "title_principals tp": {
            "columns": [
                ("tp.ordering", 1, 50),
            ],
        },
        "name_basics nb": {
            "columns": [
                ("nb.birthyear", 1800, 2005),
                ("nb.deathyear", 1900, 2025),
            ],
        },
    }

    join_map = {
        "title_ratings tr": "tb.tconst=tr.tconst",
        "title_principals tp": "tb.tconst=tp.tconst",
        "name_basics nb": "tp.nconst=nb.nconst",
    }

    # name_basics requires title_principals as intermediate join
    join_requires = {
        "name_basics nb": "title_principals tp",
    }

    all_tables = list(table_defs.keys())
    # Only allow single-table queries on tables that work standalone
    single_table_choices = ["title_basics tb", "title_ratings tr"]
    # For multi-table, join only with title_basics as hub
    multi_table_extras = ["title_ratings tr", "title_principals tp"]
    operators = ["=", "<", ">"]

    queries = []

    for _ in range(num_queries):
        num_tables = random.choices([1, 2], weights=[0.4, 0.6])[0]

        if num_tables == 1:
            chosen = [random.choice(single_table_choices)]
        else:
            extra = [random.choice(multi_table_extras)]
            chosen = ["title_basics tb"] + extra

        joins = []
        for t in chosen:
            if t in join_map and ("title_basics tb" in chosen or "title_principals tp" in chosen):
                joins.append(join_map[t])

        num_preds = random.randint(0, min(4, sum(len(table_defs[t]["columns"]) for t in chosen)))
        predicates = []

        available_cols = []
        for t in chosen:
            available_cols.extend(table_defs[t]["columns"])

        if available_cols and num_preds > 0:
            chosen_cols = random.sample(available_cols, min(num_preds, len(available_cols)))
            for col_name, min_val, max_val in chosen_cols:
                op = random.choice(operators)
                val = np_rng.randint(min_val, max_val + 1)
                predicates.append((col_name, op, str(val)))

        queries.append({
            "tables": chosen,
            "joins": joins,
            "predicates": predicates,
            "cardinality": None,
        })

    return queries

