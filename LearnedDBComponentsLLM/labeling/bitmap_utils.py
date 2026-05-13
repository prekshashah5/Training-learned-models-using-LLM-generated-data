"""
bitmap_utils.py
Runtime bitmap generation for MSCN model training.

A "bitmap" for a query is a bit-vector per table, where bit i = 1 if the i-th
materialized sample row satisfies the query's predicates on that table.

These materialized samples are random rows drawn once at pipeline start.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def get_primary_keys(cursor) -> Dict[str, str]:
    """
    Dynamically fetch primary key column names for all tables in the database.
    Falls back to using the first column of each table if no explicit PKs exist.
    """
    pk_query = """
    SELECT
        kcu.table_name,
        kcu.column_name
    FROM
        information_schema.table_constraints tco
    JOIN information_schema.key_column_usage kcu 
      ON kcu.constraint_name = tco.constraint_name
      AND kcu.constraint_schema = tco.constraint_schema
    WHERE tco.constraint_type = 'PRIMARY KEY'
      AND kcu.table_schema = 'public';
    """
    try:
        cursor.execute(pk_query)
        rows = cursor.fetchall()
        cursor.connection.commit()
        if rows:
            pks = {row[0]: row[1] for row in rows}
            print(f"[bitmap_utils] Auto-detected PKs for {len(pks)} tables: {pks}")
            return pks
    except Exception as e:
        print(f"[bitmap_utils] PK constraint query failed: {e}")
        cursor.connection.rollback()

    print("[bitmap_utils] No explicit PRIMARY KEY constraints found. Falling back to first-column heuristic.")
    fallback_query = """
    SELECT c.table_name, c.column_name
    FROM information_schema.columns c
    JOIN information_schema.tables t
      ON c.table_name = t.table_name AND c.table_schema = t.table_schema
    WHERE c.table_schema = 'public'
      AND t.table_type = 'BASE TABLE'
      AND c.ordinal_position = 1
    ORDER BY c.table_name;
    """
    try:
        cursor.execute(fallback_query)
        rows = cursor.fetchall()
        cursor.connection.commit()
        if rows:
            pks = {row[0]: row[1] for row in rows}
            print(f"[bitmap_utils] Fallback: using first column as PK for {len(pks)} tables: {pks}")
            return pks
    except Exception as e:
        print(f"[bitmap_utils] Fallback column query also failed: {e}")
        cursor.connection.rollback()

    return {}


def create_materialized_samples(cursor,
                                 table_primary_keys: Dict[str, str],
                                 num_samples: int = 1000) -> Dict[str, np.ndarray]:
    """
    Create materialized samples by selecting random primary key values from each table.
    """
    samples = {}

    for table_name, pk_col in table_primary_keys.items():
        try:
            cursor.execute(
                f"SELECT {pk_col} FROM {table_name} ORDER BY RANDOM() LIMIT %s",
                (num_samples,)
            )
            rows = cursor.fetchall()
            pks = np.array([r[0] for r in rows])
            samples[table_name] = pks
            cursor.connection.commit()
            print(f"[bitmap_utils] Sampled {len(pks)} rows from {table_name}")
        except Exception as e:
            print(f"[bitmap_utils] Error sampling {table_name}: {e}")
            cursor.connection.rollback()
            samples[table_name] = np.array([])

    return samples


def generate_bitmap_for_query(cursor,
                               query_tables: List[str],
                               query_predicates: List[Tuple],
                               materialized_samples: Dict[str, np.ndarray],
                               table_primary_keys: Dict[str, str],
                               num_samples: int = 1000,
                               timeout_ms: int = 60000) -> np.ndarray:
    """
    Generate bitmap for a single query.

    For each table mentioned in the query, generate a bit-vector of length num_samples:
    - bit i = 1 if the i-th sample row of that table satisfies the query predicates.
    - bit i = 0 otherwise.
    """
    bitmaps = []

    for table_entry in query_tables:
        parts = table_entry.strip().split()
        table_name = parts[0].lower()
        alias = parts[1].lower() if len(parts) > 1 else table_name

        pk_col = table_primary_keys.get(table_name, "id")
        sample_pks = materialized_samples.get(table_name, np.array([]))

        if len(sample_pks) == 0:
            bitmaps.append(np.zeros(num_samples, dtype=np.float32))
            continue

        table_predicates = []
        for pred in query_predicates:
            if len(pred) == 3:
                col, op, val = pred
                col_alias = col.split('.')[0]
                if col_alias == alias:
                    actual_col = col.split('.')[1]
                    table_predicates.append((actual_col, op, val))

        if not table_predicates:
            bitmap = np.ones(num_samples, dtype=np.float32)
            if len(bitmap) < num_samples:
                bitmap = np.pad(bitmap, (0, num_samples - len(bitmap)), 'constant')
            bitmaps.append(bitmap[:num_samples])
            continue

        where_parts = [f"{pk_col} = ANY(%s)"]
        for actual_col, op, val in table_predicates:
            try:
                float(val)
                where_parts.append(f"{actual_col} {op} {val}")
            except (ValueError, TypeError):
                where_parts.append(f"{actual_col} {op} '{val}'")

        where_clause = " AND ".join(where_parts)

        try:
            sql = f"SELECT {pk_col} FROM {table_name} WHERE {where_clause}"
            cursor.execute("SET statement_timeout = %s;", (timeout_ms,))
            cursor.execute(sql, (sample_pks.tolist(),))
            matching_pks = set(r[0] for r in cursor.fetchall())
            cursor.execute("SET statement_timeout = 0;")
            cursor.connection.commit()

            bitmap = np.array(
                [1.0 if pk in matching_pks else 0.0 for pk in sample_pks],
                dtype=np.float32
            )

            if len(bitmap) < num_samples:
                bitmap = np.pad(bitmap, (0, num_samples - len(bitmap)), 'constant')

            bitmaps.append(bitmap[:num_samples])

        except Exception as e:
            print(f"[bitmap_utils] Error generating bitmap for {table_name}: {e}")
            cursor.connection.rollback()
            try:
                cursor.execute("SET statement_timeout = 0;")
            except Exception:
                pass
            bitmaps.append(np.zeros(num_samples, dtype=np.float32))

    return np.array(bitmaps, dtype=np.float32)


def generate_bitmaps_for_queries(cursor,
                                  queries: List[Dict],
                                  materialized_samples: Dict[str, np.ndarray],
                                  table_primary_keys: Dict[str, str],
                                  num_samples: int = 1000,
                                  timeout_ms: int = 60000) -> List[np.ndarray]:
    """
    Generate bitmaps for a batch of queries.
    """
    bitmaps = []

    for i, q in enumerate(queries):
        if i % 100 == 0 and i > 0:
            print(f"[bitmap_utils] Generated bitmaps for {i}/{len(queries)} queries")

        bitmap = generate_bitmap_for_query(
            cursor,
            q["tables"],
            q["predicates"],
            materialized_samples,
            table_primary_keys,
            num_samples,
            timeout_ms
        )
        bitmaps.append(bitmap)

    print(f"[bitmap_utils] Generated bitmaps for all {len(queries)} queries")
    return bitmaps
