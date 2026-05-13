import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from generation.query_generator import generate_synthetic_queries
from labeling.db_labeler import label_queries, reconstruct_sql
from labeling.bitmap_utils import generate_bitmaps_for_queries, get_primary_keys, create_materialized_samples
from training.pipeline import (
    build_vocabularies, build_column_min_max, encode_single_query, 
    safe_normalize_labels, unnormalize_labels, train_model, predict, 
    compute_qerrors, make_dataset
)
from config.db_config import get_connection
from mscn.model import SetConv
from torch.utils.data import DataLoader

def get_pg_estimates(cursor, queries):
    estimates = []
    print("Getting PostgreSQL optimizer estimates via EXPLAIN...")
    for i, q in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"  [pg_estimates] EXPLAIN {i+1}/{len(queries)} queries...", flush=True)
            
        sql = reconstruct_sql(q.get("tables", []), q.get("joins", []), q.get("predicates", []))
        # Replace COUNT(*) with * to get the row estimate before aggregation
        sql_explain = sql.replace("SELECT COUNT(*)", "SELECT *")
        try:
            cursor.execute(f"EXPLAIN (FORMAT JSON) {sql_explain}")
            plan = cursor.fetchone()[0][0]
            est = plan['Plan'].get('Plan Rows', 1)
            estimates.append(max(est, 1))
        except Exception as e:
            print(f"Failed EXPLAIN on {sql_explain[:50]}... Error: {e}")
            estimates.append(1)
            cursor.connection.rollback()
    return estimates



def main():
    parser = argparse.ArgumentParser(description="Compare PostgreSQL Optimizer vs MSCN Model")
    parser.add_argument("--total-queries", type=int, default=150, help="Number of queries to generate (default: 150)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of MSCN training epochs (default: 20)")
    parser.add_argument("--db-timeout", type=int, default=5000, help="Per-query database timeout in ms (default: 5000)")
    args = parser.parse_args()

    print("="*60)
    print(" COMPARISON: PostgreSQL Optimizer vs MSCN Model")
    print("="*60)

    # 1. Generate synthetic queries
    num_queries = args.total_queries
    print(f"Generating {num_queries} synthetic queries...")
    all_queries = generate_synthetic_queries(num_queries=num_queries)
    
    # 2. Assign True Labels
    conn = get_connection()
    cursor = conn.cursor()
    print("Executing COUNT(*) to get true cardinalities...")
    label_queries(cursor, all_queries, timeout=args.db_timeout)
    conn.commit()

    # Filter out timeouts
    valid_queries = [q for q in all_queries if q.get("cardinality") is not None and int(q["cardinality"]) > 1]
    print(f"Kept {len(valid_queries)} valid queries.")

    # 3. Validation / Test Split
    split_idx = int(len(valid_queries) * 0.6)
    train_queries = valid_queries[:split_idx]
    test_queries = valid_queries[split_idx:]
    
    # 4. Get Postgres Estimates for test set
    pg_estimates = get_pg_estimates(cursor, test_queries)
    
    # 5. Build MSCN Model and Train
    print("Building vocabularies...")
    table2vec, column2vec, op2vec, join2vec = build_vocabularies(valid_queries)
    column_min_max = build_column_min_max(valid_queries)
    
    table_primary_keys = get_primary_keys(cursor)
    materialized_samples = create_materialized_samples(cursor, table_primary_keys, 1000)
    
    print("Generating bitmaps for training set...")
    train_bitmaps = generate_bitmaps_for_queries(cursor, train_queries, materialized_samples, table_primary_keys, 1000)
    print("Generating bitmaps for test set...")
    test_bitmaps = generate_bitmaps_for_queries(cursor, test_queries, materialized_samples, table_primary_keys, 1000)
    
    train_samples, train_preds, train_joins = [], [], []
    test_samples, test_preds, test_joins = [], [], []
    
    for i, q in enumerate(train_queries):
        s, p, j = encode_single_query(q, train_bitmaps[i], table2vec, column2vec, op2vec, join2vec, column_min_max, 1000)
        train_samples.append(s)
        train_preds.append(p)
        train_joins.append(j)
        
    for i, q in enumerate(test_queries):
        s, p, j = encode_single_query(q, test_bitmaps[i], table2vec, column2vec, op2vec, join2vec, column_min_max, 1000)
        test_samples.append(s)
        test_preds.append(p)
        test_joins.append(j)
        
    train_labels = [int(float(q["cardinality"])) for q in train_queries]
    test_labels = [int(float(q["cardinality"])) for q in test_queries]
    
    train_labels_norm, min_val, max_val = safe_normalize_labels(train_labels)
    test_labels_norm, _, _ = safe_normalize_labels(test_labels)
    
    max_joins = max([len(j) for j in train_joins] + [len(j) for j in test_joins] + [1])
    max_preds = max([len(p) for p in train_preds] + [len(p) for p in test_preds] + [1])
    max_tables = max([len(s) for s in train_samples] + [len(s) for s in test_samples] + [1])
    
    train_dataset = make_dataset(train_samples, train_preds, train_joins, train_labels_norm, max_joins, max_preds, max_tables)
    test_dataset = make_dataset(test_samples, test_preds, test_joins, test_labels_norm, max_joins, max_preds, max_tables)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    sample_feats = len(table2vec) + 1000
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)
    
    model = SetConv(sample_feats, predicate_feats, join_feats, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining MSCN for {args.epochs} epochs...")
    train_model(model, train_loader, min_val, max_val, args.epochs, optimizer)
    
    print("\nEvaluating MSCN on test set...")
    mscn_preds_norm, _ = predict(model, test_loader)
    mscn_estimates = unnormalize_labels(mscn_preds_norm, min_val, max_val)
    
    # 6. Compute Q-Errors (Manual calculation for raw cardinalities to avoid double-unnorm)
    pg_qerrors = []
    for est, true in zip(pg_estimates, test_labels):
        pg_qerrors.append(max(est/true, true/est) if est > 0 and true > 0 else 1.0)
        
    mscn_qerrors = []
    for est, true in zip(mscn_estimates, test_labels):
        mscn_qerrors.append(max(est/true, true/est) if est > 0 and true > 0 else 1.0)
    
    pg_qerrors = np.array(pg_qerrors)
    mscn_qerrors = np.array(mscn_qerrors)
    
    print(f"\nResults on {len(test_queries)} test queries:")
    print(f"  PostgreSQL Median Q-error: {np.median(pg_qerrors):.4f}")
    print(f"  MSCN Median Q-error:       {np.median(mscn_qerrors):.4f}")
    print(f"  PostgreSQL 90th Q-error:   {np.percentile(pg_qerrors, 90):.4f}")
    print(f"  MSCN 90th Q-error:         {np.percentile(mscn_qerrors, 90):.4f}")

    # 7. Plot Comparison CDF
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort for CDF
    pg_qerrors_sorted = np.sort(pg_qerrors)
    mscn_qerrors_sorted = np.sort(mscn_qerrors)
    cdf = np.arange(1, len(pg_qerrors) + 1) / len(pg_qerrors)
    
    ax.plot(pg_qerrors_sorted, cdf, linewidth=2.5, color='#e74c3c', label='PostgreSQL Estimates')
    ax.plot(mscn_qerrors_sorted, cdf, linewidth=2.5, color='#3498db', label='MSCN Estimates')
    
    ax.set_xscale('log')
    ax.set_xlabel('Q-Error (log scale)')
    ax.set_ylabel('CDF')
    ax.set_title('Cardinality Estimation Accuracy: PostgreSQL vs MSCN')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    out_file = 'compare_pg_vs_mscn_cdf.png'
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {out_file}")

    # Plot Scatter
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    max_val_plot = max(max(test_labels), max(pg_estimates), max(mscn_estimates)) * 2
    min_val_plot = max(1, min(min(test_labels), min(pg_estimates), min(mscn_estimates)) / 2)
    
    # PG Scatter
    ax1.scatter(test_labels, pg_estimates, alpha=0.5, color='#e74c3c')
    ax1.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'k--', alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(min_val_plot, max_val_plot)
    ax1.set_ylim(min_val_plot, max_val_plot)
    ax1.set_xlabel('True Cardinality')
    ax1.set_ylabel('Predicted Cardinality (PostgreSQL)')
    ax1.set_title('PostgreSQL Optimizer')
    
    # MSCN Scatter
    ax2.scatter(test_labels, mscn_estimates, alpha=0.5, color='#3498db')
    ax2.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'k--', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(min_val_plot, max_val_plot)
    ax2.set_ylim(min_val_plot, max_val_plot)
    ax2.set_xlabel('True Cardinality')
    ax2.set_ylabel('Predicted Cardinality (MSCN)')
    ax2.set_title('Learned MSCN Model')
    
    out_file2 = 'compare_pg_vs_mscn_scatter.png'
    fig2.savefig(out_file2, dpi=150, bbox_inches='tight')
    print(f"Saved scatter plot to {out_file2}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
