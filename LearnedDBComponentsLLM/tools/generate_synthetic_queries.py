"""
generate_synthetic_queries.py

Generates synthetic SQL queries that mimic the structural pattern of
LLM-generated queries from this project:
    SELECT COUNT(*) FROM <tables> WHERE <joins> AND <predicates>

Uses the same IMDB schema (title_basics, name_basics, title_principals,
title_ratings), same join relationships (FK-based), and same numeric
predicate columns with realistic value ranges.

Output: JSON file (list of SQL strings) in generated_queries/ — same
format as LLM-generated queries, ready to use as --reference in
kl_convergence_plot.py.

Usage:
    python tools/generate_synthetic_queries.py --n 5000
    python tools/generate_synthetic_queries.py --n 1000 --seed 42 --out-dir synthetic_queries
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Tables with their aliases
TABLES = {
    "title_basics":     "tb",
    "title_principals": "tp",
    "name_basics":      "nb",
    "title_ratings":    "tr",
}

# FK join conditions as (table_a, alias_a, col_a, table_b, alias_b, col_b)
# Only valid FK-based joins
FK_JOINS = [
    ("title_basics",     "tb", "tconst", "title_principals", "tp", "tconst"),
    ("title_basics",     "tb", "tconst", "title_ratings",    "tr", "tconst"),
    ("title_principals", "tp", "nconst", "name_basics",      "nb", "nconst"),
]

# All valid connected table subsets and their required joins
# Key: frozenset of table names → list of join tuples to use
CONNECTED_SUBSETS = {
    frozenset({"title_basics"}): [],
    frozenset({"title_principals"}): [],
    frozenset({"name_basics"}): [],
    frozenset({"title_ratings"}): [],
    frozenset({"title_basics", "title_principals"}): [
        ("title_basics", "tb", "tconst", "title_principals", "tp", "tconst"),
    ],
    frozenset({"title_basics", "title_ratings"}): [
        ("title_basics", "tb", "tconst", "title_ratings", "tr", "tconst"),
    ],
    frozenset({"title_principals", "name_basics"}): [
        ("title_principals", "tp", "nconst", "name_basics", "nb", "nconst"),
    ],
    frozenset({"title_basics", "title_principals", "name_basics"}): [
        ("title_basics",     "tb", "tconst", "title_principals", "tp", "tconst"),
        ("title_principals", "tp", "nconst", "name_basics",      "nb", "nconst"),
    ],
    frozenset({"title_basics", "title_principals", "title_ratings"}): [
        ("title_basics", "tb", "tconst", "title_principals", "tp", "tconst"),
        ("title_basics", "tb", "tconst", "title_ratings",    "tr", "tconst"),
    ],
    frozenset({"title_basics", "title_principals", "name_basics", "title_ratings"}): [
        ("title_basics",     "tb", "tconst", "title_principals", "tp", "tconst"),
        ("title_basics",     "tb", "tconst", "title_ratings",    "tr", "tconst"),
        ("title_principals", "tp", "nconst", "name_basics",      "nb", "nconst"),
    ],
}

# Weights for how often each subset is chosen (matches LLM output skew)
SUBSET_WEIGHTS = {
    frozenset({"title_basics"}): 15,
    frozenset({"title_principals"}): 3,
    frozenset({"name_basics"}): 3,
    frozenset({"title_ratings"}): 8,
    frozenset({"title_basics", "title_principals"}): 18,
    frozenset({"title_basics", "title_ratings"}): 20,
    frozenset({"title_principals", "name_basics"}): 10,
    frozenset({"title_basics", "title_principals", "name_basics"}): 12,
    frozenset({"title_basics", "title_principals", "title_ratings"}): 7,
    frozenset({"title_basics", "title_principals", "name_basics", "title_ratings"}): 4,
}

# Numeric predicate columns per table: (alias, column, operators, values_fn)
# values_fn returns a realistic random value for that column
PREDICATE_COLUMNS = [
    # title_basics
    ("tb", "startYear",       [">", "<"], lambda rng: rng.choice([1970, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015])),
    ("tb", "runtimeMinutes",  [">", "<"], lambda rng: rng.choice([60, 90, 100, 120, 150, 180])),
    ("tb", "isAdult",         ["="],      lambda rng: rng.choice([0, 1])),
    # name_basics
    ("nb", "birthYear",       [">", "<"], lambda rng: rng.choice([1920, 1940, 1950, 1960, 1970, 1980])),
    ("nb", "deathYear",       [">", "<"], lambda rng: rng.choice([1960, 1970, 1980, 1990, 2000, 2010])),
    # title_principals
    ("tp", "ordering",        [">", "<"], lambda rng: rng.choice([1, 2, 3, 5])),
    # title_ratings
    ("tr", "average_rating",  [">", "<"], lambda rng: round(rng.choice([5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0]), 1)),
    ("tr", "num_votes",       [">", "<"], lambda rng: rng.choice([100, 500, 1000, 2000, 5000, 10000])),
]

# Alias → table name lookup
ALIAS_TO_TABLE = {v: k for k, v in TABLES.items()}


def _pick_subset(rng: random.Random) -> frozenset:
    keys = list(SUBSET_WEIGHTS.keys())
    weights = [SUBSET_WEIGHTS[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def _pick_predicates(rng: random.Random, available_aliases: set, n_preds: int) -> list:
    """Pick n_preds predicate conditions from columns whose table alias is in available_aliases."""
    eligible = [p for p in PREDICATE_COLUMNS if p[0] in available_aliases]
    if not eligible:
        return []

    chosen = []
    seen_cols = set()
    attempts = 0
    while len(chosen) < n_preds and attempts < 40:
        attempts += 1
        alias, col, ops, val_fn = rng.choice(eligible)
        key = (alias, col)
        if key in seen_cols:
            continue
        seen_cols.add(key)
        op = rng.choice(ops)
        val = val_fn(rng)
        chosen.append(f"{alias}.{col} {op} {val}")
    return chosen


def generate_query(rng: random.Random) -> str:
    subset = _pick_subset(rng)
    joins = CONNECTED_SUBSETS[subset]
    available_aliases = {TABLES[t] for t in subset}

    # FROM clause
    from_parts = []
    for tname in sorted(subset):  # sorted for determinism
        alias = TABLES[tname]
        from_parts.append(f"{tname} {alias}")
    from_clause = ", ".join(from_parts)

    # WHERE conditions
    conditions = []

    # Join conditions
    for (ta, aa, ca, tb_, ab, cb) in joins:
        conditions.append(f"{aa}.{ca} = {ab}.{cb}")

    # Predicate conditions: pick 1-3
    n_preds = rng.choices([1, 2, 3, 4], weights=[40, 35, 18, 7], k=1)[0]
    predicates = _pick_predicates(rng, available_aliases, n_preds)
    conditions.extend(predicates)

    if conditions:
        where_clause = " AND ".join(conditions)
        return f"SELECT COUNT(*) FROM {from_clause} WHERE {where_clause}"
    else:
        return f"SELECT COUNT(*) FROM {from_clause}"


def generate_all(n: int, seed: int) -> list:
    rng = random.Random(seed)
    queries = []
    seen = set()
    attempts = 0
    max_attempts = n * 10

    while len(queries) < n and attempts < max_attempts:
        attempts += 1
        sql = generate_query(rng)
        if sql not in seen:
            seen.add(sql)
            queries.append(sql)

    if len(queries) < n:
        print(f"Warning: only generated {len(queries)} unique queries (requested {n})")
    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IMDB SQL queries matching LLM query structure")
    parser.add_argument("--n",       type=int, default=5000,              help="Number of queries to generate (default: 5000)")
    parser.add_argument("--seed",    type=int, default=42,                help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out-dir", type=str, default="synthetic_queries", help="Output directory (default: synthetic_queries)")
    args = parser.parse_args()

    print(f"Generating {args.n} synthetic queries (seed={args.seed})...")
    queries = generate_all(args.n, args.seed)
    print(f"Generated {len(queries)} unique queries")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"synthetic_queries_{timestamp}.json"
    out_path.write_text(json.dumps(queries, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")

    # Print structural summary
    from collections import Counter
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from generation.format_converter import parse_sql_to_mscn
        table_counts = Counter()
        join_counts = Counter()
        pred_counts = Counter()
        for sql in queries:
            parsed = parse_sql_to_mscn(sql)
            if parsed:
                table_counts[len(parsed.get("tables", []))] += 1
                join_counts[len(parsed.get("joins", []))] += 1
                pred_counts[len(parsed.get("predicates", []))] += 1
        print(f"\nStructural summary:")
        print(f"  Tables distribution: {dict(sorted(table_counts.items()))}")
        print(f"  Joins  distribution: {dict(sorted(join_counts.items()))}")
        print(f"  Preds  distribution: {dict(sorted(pred_counts.items()))}")
    except Exception:
        pass

    print(f"\nTo compare with LLM-generated queries:")
    print(f"  python tools/kl_convergence_plot.py \\")
    print(f"    --reference {out_path} \\")
    print(f"    --generated generated_queries/queries_<timestamp>.json \\")
    print(f"    --step 250")


if __name__ == "__main__":
    main()
