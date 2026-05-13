"""
format_converter.py
Converts SQL queries into the MSCN CSV format:
    tables_str#joins_str#predicates_str#cardinality

The MSCN format uses table aliases and compact representations:
    title t,movie_info mi#t.id=mi.movie_id#t.kind_id,=,7,mi.info_type_id,>,16#12345
"""

import re
from typing import Dict, List, Optional, Tuple


# Operators to standardize
OPERATOR_MAP = {
    "=": "=",
    "!=": "!=",
    "<>": "!=",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
}


def parse_sql_to_mscn(sql: str) -> Optional[Dict]:
    """
    Parse a SQL query string into the MSCN component dictionary.

    Returns:
        {
            "tables": ["title t", "movie_info mi"],
            "joins": ["t.id=mi.movie_id"],
            "predicates": [("t.kind_id", "=", "7"), ("mi.info_type_id", ">", "16")],
            "sql": "SELECT ... FROM ...",
        }
    or None if parsing fails.
    """
    try:
        sql_clean = sql.strip().rstrip(";")
        sql_upper = sql_clean.upper()

        # ── Extract FROM clause ─────────────────────────────────────────
        from_match = re.search(r'\bFROM\b\s+(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)',
                               sql_clean, re.IGNORECASE | re.DOTALL)
        if not from_match:
            return None

        from_clause = from_match.group(1).strip()

        # ── Parse tables and aliases ────────────────────────────────────
        # Remove JOIN keywords and ON clauses for table extraction
        # Split by commas and JOIN keywords
        tables_raw = re.split(r'\b(?:INNER\s+)?(?:LEFT\s+)?(?:RIGHT\s+)?(?:CROSS\s+)?(?:FULL\s+)?JOIN\b|,',
                              from_clause, flags=re.IGNORECASE)

        table_list = []
        join_conditions_from_on = []
        alias_map = {}  # alias -> table_name

        for t in tables_raw:
            t = t.strip()
            if not t:
                continue

            # Remove ON clause if present, capture the condition
            on_match = re.search(r'\bON\b\s+(.+)', t, re.IGNORECASE)
            if on_match:
                on_cond = on_match.group(1).strip()
                join_conditions_from_on.append(on_cond)
                t = t[:on_match.start()].strip()

            # Parse table name and optional alias
            parts = t.split()
            if len(parts) >= 2 and parts[1].upper() not in ('ON', 'WHERE', 'JOIN', 'INNER', 'LEFT'):
                table_name = parts[0].strip().lower()
                alias = parts[-1].strip().lower()
                # Remove AS keyword if present
                if parts[1].upper() == 'AS':
                    alias = parts[2].strip().lower() if len(parts) > 2 else parts[0].strip().lower()
            elif len(parts) == 1:
                table_name = parts[0].strip().lower()
                alias = table_name  # If no alias, the table name IS the alias
            else:
                continue

            alias_map[alias] = table_name
            table_list.append(f"{table_name} {alias}")

        if not table_list:
            return None

        # ── Extract WHERE clause ────────────────────────────────────────
        where_match = re.search(r'\bWHERE\b\s+(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)',
                                sql_clean, re.IGNORECASE | re.DOTALL)

        joins = []
        predicates = []

        if where_match:
            where_clause = where_match.group(1).strip()
            # Split by AND (not inside parentheses)
            conditions = re.split(r'\bAND\b', where_clause, flags=re.IGNORECASE)

            for cond in conditions:
                cond = cond.strip().strip("()")
                if not cond:
                    continue

                # Check if it's a join condition (col = col pattern)
                join_match = re.match(
                    r'(\w+\.\w+)\s*=\s*(\w+\.\w+)',
                    cond.strip()
                )
                if join_match:
                    left = join_match.group(1).lower()
                    right = join_match.group(2).lower()

                    # Verify both sides reference known aliases
                    left_alias = left.split('.')[0]
                    right_alias = right.split('.')[0]

                    if left_alias in alias_map and right_alias in alias_map:
                        joins.append(f"{left}={right}")
                        continue

                # Check if it's a predicate (col op value)
                pred_match = re.match(
                    r'(\w+\.\w+)\s*(=|!=|<>|<=|>=|<|>)\s*(.+)',
                    cond.strip()
                )
                if pred_match:
                    col = pred_match.group(1).lower()
                    op = OPERATOR_MAP.get(pred_match.group(2), pred_match.group(2))
                    val = pred_match.group(3).strip().strip("'\"")

                    # Try to convert to numeric
                    try:
                        if '.' in val:
                            val = str(float(val))
                        else:
                            val = str(int(val))
                    except ValueError:
                        pass

                    predicates.append((col, op, val))

        # Also add join conditions from ON clauses
        for on_cond in join_conditions_from_on:
            parts = re.split(r'\bAND\b', on_cond, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                join_match = re.match(r'(\w+\.\w+)\s*=\s*(\w+\.\w+)', part)
                if join_match:
                    joins.append(f"{join_match.group(1).lower()}={join_match.group(2).lower()}")

        return {
            "tables": table_list,
            "joins": joins,
            "predicates": predicates,
            "sql": sql_clean,
        }

    except Exception as e:
        print(f"[format_converter] Failed to parse SQL: {e}")
        return None


def query_dict_to_csv_line(query_dict: Dict, cardinality: Optional[int] = None) -> str:
    """
    Convert a parsed query dict to MSCN CSV format line.

    Format: tables#joins#predicates#cardinality
    Example: title t,movie_info mi#t.id=mi.movie_id#t.kind_id,=,7#12345
    """
    tables_str = ",".join(query_dict["tables"])
    joins_str = ",".join(query_dict["joins"])

    pred_parts = []
    for col, op, val in query_dict["predicates"]:
        pred_parts.extend([col, op, val])
    predicates_str = ",".join(pred_parts)

    if cardinality is not None:
        return f"{tables_str}#{joins_str}#{predicates_str}#{cardinality}"
    else:
        return f"{tables_str}#{joins_str}#{predicates_str}"


def csv_line_to_components(csv_line: str) -> Tuple[List[str], List[str], List[Tuple], str]:
    """
    Parse MSCN CSV line back into components.
    Returns: (tables, joins, predicates_as_tuples, cardinality_str)
    """
    parts = csv_line.strip().split('#')
    tables = parts[0].split(',') if parts[0] else []
    joins = parts[1].split(',') if len(parts) > 1 and parts[1] else []
    
    predicates = []
    if len(parts) > 2 and parts[2]:
        pred_flat = parts[2].split(',')
        for i in range(0, len(pred_flat), 3):
            if i + 2 < len(pred_flat):
                predicates.append((pred_flat[i], pred_flat[i+1], pred_flat[i+2]))
    
    cardinality = parts[3] if len(parts) > 3 else None
    return tables, joins, predicates, cardinality


def build_column_min_max_from_db(cursor, table_columns: Dict[str, List[str]]) -> Dict[str, Tuple[float, float]]:
    """
    Query the database for actual min/max values for numeric columns.

    Args:
        cursor: psycopg2 cursor
        table_columns: {table_name: [column_name, ...]}

    Returns:
        {"alias.column": (min_val, max_val), ...}
    """
    column_min_max = {}

    for table_name, columns in table_columns.items():
        # Alias doesn't strictly matter for querying DB directly, but we need
        # to format it for MSCN expectations
        alias = table_name  # We'll just prefix it properly based on input mapping later
        for col in columns:
            try:
                cursor.execute(f"SELECT MIN({col}), MAX({col}) FROM {table_name}")
                row = cursor.fetchone()
                if row and row[0] is not None and row[1] is not None:
                    column_min_max[col] = (float(row[0]), float(row[1]))
            except Exception as e:
                print(f"[format_converter] Could not get min/max for {table_name}.{col}: {e}")
                cursor.connection.rollback()

    return column_min_max
