import psycopg2
import csv

import sys
from pathlib import Path

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from db_utils import get_db_config

OUTPUT_CSV = Path(__file__).resolve().parent / "col_stats.csv"

def main():
    config = get_db_config()
    conn = psycopg2.connect(**config)
    conn.autocommit = False  # important for safe rollback
    cur = conn.cursor()

    # Fetch all columns and their types
    cur.execute(
        """
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, column_name
        """
    )

    columns = cur.fetchall()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["name", "min", "max", "cardinality", "num_unique_values"]
        )

        for schema, table, column, dtype in columns:
            full_name = f"{table}.{column}"
            print(f"Processing {full_name} ({dtype})...")

            # Determine if we can get min/max based on data type
            numeric_types = ('integer', 'numeric', 'bigint', 'smallint', 'real', 'double precision')
            is_numeric = any(t in dtype.lower() for t in numeric_types)

            if is_numeric:
                query = f"""
                    SELECT
                        MIN("{column}")::text,
                        MAX("{column}")::text,
                        COUNT(*),
                        COUNT(DISTINCT "{column}")
                    FROM "{schema}"."{table}"
                """
            else:
                # For text/categorical, skip min/max to avoid cast errors
                query = f"""
                    SELECT
                        'N/A',
                        'N/A',
                        COUNT(*),
                        COUNT(DISTINCT "{column}")
                    FROM "{schema}"."{table}"
                """

            try:
                cur.execute(query)
                min_v, max_v, card, ndv = cur.fetchone()
                conn.commit()

                writer.writerow([full_name, min_v, max_v, card, ndv])

            except Exception as e:
                print(f"  Error processing {full_name}: {e}")
                conn.rollback()
                continue

    cur.close()
    conn.close()
    print(f"Done. Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()