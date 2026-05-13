"""
generate_kl_plot.py
Standalone script to generate the KL divergence comparison plot.
Uses the actual IMDB database to compute selectivity distributions 
for both real-style and LLM-style query patterns, then plots the comparison.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Setup path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

import math
import numpy as np
from collections import Counter
from scipy.stats import entropy
from config.db_config import get_connection
from metrics.plotting import plot_kl_divergence_comparison

EPSILON = 1e-8

def explain_cardinality(cursor, sql):
    cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    plan = cursor.fetchone()[0][0]
    node = plan["Plan"]
    # Drill down past Aggregate/Gather nodes to find actual scan/join rows
    while node.get("Node Type") in ("Aggregate", "Gather", "Gather Merge") and "Plans" in node:
        node = node["Plans"][0]
    return node["Plan Rows"]

def log_bucket(rows):
    return int(math.log10(max(rows, 1)))

def bucket_label(log_value):
    if log_value <= 1:
        return "high"
    elif log_value <= 3:
        return "medium"
    else:
        return "low"

def normalize(counter):
    total = sum(counter.values())
    if total == 0:
        return counter
    return {k: v / total for k, v in counter.items()}

def align_distributions(p, q):
    keys = set(p) | set(q)
    p = {k: p.get(k, EPSILON) for k in keys}
    q = {k: q.get(k, EPSILON) for k in keys}
    return p, q

def kl_divergence(p, q):
    return entropy(list(p.values()), list(q.values()))


# Real workload queries - typical analytical queries on the actual schema
# Column names: title_basics(tconst, titletype, primarytitle, originaltitle, isadult, startyear, endyear, runtimeminutes, genres)
# title_ratings(tconst, average_rating, num_votes), title_akas(titleid, ordering, title, region, language, types, attributes, isoriginaltitle)
# title_principals(tconst, ordering, nconst, category, job, characters), title_episode(const, parenttconst, seasonnumber, episodenumber)
# name_basics(nconst, primaryname, birthyear, deathyear, primaryprofession, knownfortitles), title_crew(tconst, directors, writers)
REAL_WORKLOAD_QUERIES = [
    "SELECT COUNT(*) FROM title_basics WHERE startyear > 2015",
    "SELECT COUNT(*) FROM title_basics WHERE startyear BETWEEN 2000 AND 2010",
    "SELECT COUNT(*) FROM title_basics WHERE titletype = 'movie'",
    "SELECT COUNT(*) FROM title_basics WHERE titletype = 'tvSeries'",
    "SELECT COUNT(*) FROM title_basics WHERE runtimeminutes > 120",
    "SELECT COUNT(*) FROM title_basics WHERE runtimeminutes < 30",
    "SELECT COUNT(*) FROM title_ratings WHERE average_rating > 8.0",
    "SELECT COUNT(*) FROM title_ratings WHERE average_rating < 3.0",
    "SELECT COUNT(*) FROM title_ratings WHERE num_votes > 10000",
    "SELECT COUNT(*) FROM title_ratings WHERE num_votes > 100000",
    "SELECT COUNT(*) FROM title_basics tb JOIN title_ratings tr ON tb.tconst = tr.tconst WHERE tr.average_rating > 7.0",
    "SELECT COUNT(*) FROM title_basics tb JOIN title_ratings tr ON tb.tconst = tr.tconst WHERE tb.startyear > 2010 AND tr.average_rating > 8.0",
    "SELECT COUNT(*) FROM title_basics tb JOIN title_ratings tr ON tb.tconst = tr.tconst WHERE tb.titletype = 'movie' AND tr.num_votes > 5000",
    "SELECT COUNT(*) FROM title_basics tb JOIN title_episode te ON tb.tconst = te.const WHERE tb.startyear > 2015",
    "SELECT COUNT(*) FROM title_basics WHERE startyear > 2020 AND titletype = 'movie'",
    "SELECT COUNT(*) FROM title_basics tb JOIN title_principals tp ON tb.tconst = tp.tconst WHERE tb.startyear > 2010",
    "SELECT COUNT(*) FROM title_principals WHERE category = 'actor'",
    "SELECT COUNT(*) FROM title_principals WHERE category = 'director'",
    "SELECT COUNT(*) FROM title_akas WHERE region = 'US'",
    "SELECT COUNT(*) FROM title_akas WHERE region = 'JP'",
    "SELECT COUNT(*) FROM name_basics WHERE birthyear > 1990",
    "SELECT COUNT(*) FROM name_basics WHERE birthyear BETWEEN 1950 AND 1970",
    "SELECT COUNT(*) FROM title_basics WHERE startyear = 2023",
    "SELECT COUNT(*) FROM title_basics tb JOIN title_crew tc ON tb.tconst = tc.tconst WHERE tb.startyear > 2010",
    "SELECT COUNT(*) FROM title_ratings WHERE average_rating BETWEEN 5.0 AND 7.0",
]

# Simulated LLM-generated queries - LLMs tend to generate simpler, 
# more uniform queries that often have medium selectivity
LLM_GENERATED_QUERIES = [
    "SELECT COUNT(*) FROM title_basics WHERE startyear > 2000",
    "SELECT COUNT(*) FROM title_basics WHERE titletype = 'movie'",
    "SELECT COUNT(*) FROM title_ratings WHERE average_rating > 5.0",
    "SELECT COUNT(*) FROM title_ratings WHERE num_votes > 100",
    "SELECT COUNT(*) FROM title_basics WHERE runtimeminutes > 60",
    "SELECT COUNT(*) FROM name_basics WHERE birthyear > 1980",
    "SELECT COUNT(*) FROM title_akas WHERE region = 'US'",
    "SELECT COUNT(*) FROM title_principals WHERE category = 'actor'",
    "SELECT COUNT(*) FROM title_basics WHERE startyear BETWEEN 1990 AND 2020",
    "SELECT COUNT(*) FROM title_basics WHERE titletype = 'short'",
    "SELECT COUNT(*) FROM title_ratings WHERE average_rating > 6.0",
    "SELECT COUNT(*) FROM title_basics WHERE runtimeminutes > 90",
    "SELECT COUNT(*) FROM title_basics WHERE startyear > 1990",
    "SELECT COUNT(*) FROM title_ratings WHERE num_votes > 1000",
    "SELECT COUNT(*) FROM name_basics WHERE birthyear > 1960",
    "SELECT COUNT(*) FROM title_basics tb JOIN title_ratings tr ON tb.tconst = tr.tconst WHERE tr.average_rating > 5.0",
    "SELECT COUNT(*) FROM title_basics WHERE startyear > 1980 AND titletype = 'movie'",
    "SELECT COUNT(*) FROM title_akas WHERE region = 'GB'",
    "SELECT COUNT(*) FROM title_principals WHERE category = 'actress'",
    "SELECT COUNT(*) FROM title_basics WHERE startyear > 2005",
    "SELECT COUNT(*) FROM title_ratings WHERE average_rating > 4.0 AND num_votes > 500",
    "SELECT COUNT(*) FROM title_basics WHERE runtimeminutes BETWEEN 30 AND 120",
    "SELECT COUNT(*) FROM title_episode WHERE seasonnumber = '1'",
    "SELECT COUNT(*) FROM title_basics WHERE titletype = 'tvEpisode'",
    "SELECT COUNT(*) FROM name_basics WHERE birthyear BETWEEN 1970 AND 2000",
]


def compute_distribution(queries, cursor, label):
    """Compute selectivity distribution for a list of SQL queries."""
    buckets = Counter()
    errors = 0
    for i, sql in enumerate(queries):
        try:
            rows = explain_cardinality(cursor, sql)
            bucket = bucket_label(log_bucket(rows))
            buckets[bucket] += 1
        except Exception as e:
            errors += 1
            print(f"  [{label}] Error on query {i+1}: {e}")
            try:
                cursor.connection.rollback()
            except:
                pass
    if errors > 0:
        print(f"  [{label}] {errors} queries failed out of {len(queries)}")
    return buckets


def main():
    plots_dir = project_root / "output" / "plots" / "kl_divergence"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Connecting to database...")
    conn = get_connection()
    cur = conn.cursor()

    # Compute real workload distribution
    print(f"[info] Evaluating {len(REAL_WORKLOAD_QUERIES)} real workload queries...")
    real_dist = compute_distribution(REAL_WORKLOAD_QUERIES, cur, "real")
    real_dist_norm = normalize(real_dist)

    print("\n[Real Workload Selectivity Distribution]")
    for k in ["high", "medium", "low"]:
        print(f"  {k}: {real_dist.get(k, 0)} queries ({real_dist_norm.get(k, 0):.3f})")

    # Compute LLM-generated distribution
    print(f"\n[info] Evaluating {len(LLM_GENERATED_QUERIES)} LLM-generated queries...")
    gen_dist = compute_distribution(LLM_GENERATED_QUERIES, cur, "generated")
    gen_dist_norm = normalize(gen_dist)

    print("\n[LLM-Generated Selectivity Distribution]")
    for k in ["high", "medium", "low"]:
        print(f"  {k}: {gen_dist.get(k, 0)} queries ({gen_dist_norm.get(k, 0):.3f})")

    # Align and compute KL divergence
    real_aligned, gen_aligned = align_distributions(real_dist_norm, gen_dist_norm)
    kl = kl_divergence(real_aligned, gen_aligned)
    print(f"\nKL(real || generated) = {kl:.4f}")

    # Generate the plot
    plot_kl_divergence_comparison(gen_dist_norm, real_dist_norm, kl, plots_dir)

    plot_path = plots_dir / "kl_divergence_comparison.png"
    print(f"\n[done] KL divergence plot saved to: {plot_path}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
