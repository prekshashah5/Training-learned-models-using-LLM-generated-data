"""
pipeline.py
Complete parameterized pipeline: Data Generation → Active Learning → Model Training

Workflow:
  1. Generate unlabeled SQL queries (via Ollama LLM or synthetic generation)
  2. Convert to MSCN-compatible format
  3. Label an initial batch by executing on PostgreSQL
  4. Generate bitmaps at runtime using materialized samples
  5. Run Active Learning loop:
     a. Encode labeled data → MSCN tensors
     b. Train model
     c. Acquire most informative samples from unlabeled pool
     d. Label acquired samples on DB
     e. Generate bitmaps for newly labeled samples
     f. Repeat
  6. Output learning curves, metrics, and results
"""

import argparse
import os
import sys
import csv
import time
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from mscn.model import SetConv
from mscn.util import (
    get_set_encoding,
    unnormalize_labels,
    normalize_data,
    idx_to_onehot,
)

from generation.format_converter import parse_sql_to_mscn, query_dict_to_csv_line
from config.db_config import get_connection
from labeling.db_labeler import label_queries, reconstruct_sql, get_pg_estimates
from labeling.bitmap_utils import (
    get_primary_keys,
    create_materialized_samples,
    generate_bitmaps_for_queries,
)
from generation.query_generator import generate_all_queries, generate_synthetic_queries, validate_sql, SchemaValidator
from evaluation.pipeline_graphs import generate_all_graphs, plot_pg_vs_mscn_comparison
from utils.io_utils import read_json_file

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


# ── Global ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")


import pickle

def save_bitmaps(bitmaps, filepath):
    """Save constructed bitmaps using pickle."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(bitmaps, f)
        print(f"  [bitmaps] Saved to {filepath}")
    except Exception as e:
        print(f"  [error] Failed to save bitmaps to {filepath}: {e}")

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_normalize_labels(labels_raw, min_val=None, max_val=None):
    labels_log = np.array([np.log(float(l)) for l in labels_raw])
    if min_val is None:
        min_val = labels_log.min()
    if max_val is None:
        max_val = labels_log.max()
    if min_val == max_val:
        max_val = min_val + 1.0  # Prevent divide-by-zero
    labels_norm = (labels_log - min_val) / (max_val - min_val)
    return labels_norm, min_val, max_val


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    preds = unnormalize_torch(preds, min_val, max_val).flatten()
    targets = unnormalize_torch(targets, min_val, max_val).flatten()
    qerror = torch.max(preds / targets, targets / preds)
    return torch.mean(qerror)


# ═══════════════════════════════════════════════════════════════════════════
# DATA ENCODING (adapted from mscn/data.py for in-memory use)
# ═══════════════════════════════════════════════════════════════════════════

def build_vocabularies(queries):
    """
    Build encoding vocabularies from the full set of queries.
    Returns dicts for table, column, operator, and join encoding.
    """
    all_tables = set()
    all_columns = set()
    all_operators = set()
    all_joins = set()

    for q in queries:
        for t in q["tables"]:
            all_tables.add(t.strip())
        for j in q["joins"]:
            if j.strip():
                all_joins.add(j.strip())
        for pred in q["predicates"]:
            if len(pred) == 3:
                all_columns.add(pred[0])
                all_operators.add(pred[1])

    table2vec, _ = get_set_encoding(all_tables)
    column2vec, _ = get_set_encoding(all_columns)
    op2vec, _ = get_set_encoding(all_operators)
    join2vec, _ = get_set_encoding(all_joins)

    return table2vec, column2vec, op2vec, join2vec


def build_column_min_max(queries):
    """
    Build column min/max values from predicate values across all queries.
    """
    column_min_max = {}
    for q in queries:
        for pred in q["predicates"]:
            if len(pred) == 3:
                col, op, val = pred
                try:
                    fval = float(val)
                except ValueError:
                    continue
                if col not in column_min_max:
                    column_min_max[col] = [fval, fval]
                else:
                    column_min_max[col][0] = min(column_min_max[col][0], fval)
                    column_min_max[col][1] = max(column_min_max[col][1], fval)

    return column_min_max


def encode_single_query(query, bitmaps, table2vec, column2vec, op2vec, join2vec,
                         column_min_max, num_materialized_samples):
    """
    Encode a single query into feature vectors for the MSCN model.

    Returns:
        samples_enc: list of np.array (one per table, table_onehot + bitmap)
        predicates_enc: list of np.array (one per predicate)
        joins_enc: list of np.array (one per join)
    """
    # ── Encode samples (table one-hot + bitmap) ─────────────────────────
    samples_enc = []
    for j, table_entry in enumerate(query["tables"]):
        table_entry = table_entry.strip()
        sample_vec = []

        # Table one-hot
        if table_entry in table2vec:
            sample_vec.append(table2vec[table_entry])
        else:
            sample_vec.append(np.zeros(len(table2vec), dtype=np.float32))

        # Bitmap for this table
        if bitmaps is not None and j < len(bitmaps):
            sample_vec.append(bitmaps[j][:num_materialized_samples].astype(np.float32))
        else:
            sample_vec.append(np.zeros(num_materialized_samples, dtype=np.float32))

        samples_enc.append(np.hstack(sample_vec))

    # ── Encode predicates ───────────────────────────────────────────────
    predicates_enc = []
    for pred in query["predicates"]:
        if len(pred) == 3:
            col, op, val = pred
            pred_vec = []

            # Column one-hot
            if col in column2vec:
                pred_vec.append(column2vec[col])
            else:
                pred_vec.append(np.zeros(len(column2vec), dtype=np.float32))

            # Operator one-hot
            if op in op2vec:
                pred_vec.append(op2vec[op])
            else:
                pred_vec.append(np.zeros(len(op2vec), dtype=np.float32))

            # Normalized value
            try:
                fval = float(val)
                if col in column_min_max:
                    min_v, max_v = column_min_max[col]
                    norm_val = (fval - min_v) / (max_v - min_v) if max_v > min_v else 0.0
                else:
                    norm_val = 0.0
                pred_vec.append(np.array([norm_val], dtype=np.float32))
            except ValueError:
                pred_vec.append(np.array([0.0], dtype=np.float32))

            predicates_enc.append(np.hstack(pred_vec))

    # If no predicates, add a zero vector
    if not predicates_enc:
        zero_pred = np.zeros(len(column2vec) + len(op2vec) + 1, dtype=np.float32)
        predicates_enc.append(zero_pred)

    # ── Encode joins ────────────────────────────────────────────────────
    joins_enc = []
    for j in query["joins"]:
        j = j.strip()
        if j and j in join2vec:
            joins_enc.append(join2vec[j])

    # If no joins, add a zero vector
    if not joins_enc:
        zero_join = np.zeros(len(join2vec), dtype=np.float32) if len(join2vec) > 0 else np.zeros(1, dtype=np.float32)
        joins_enc.append(zero_join)

    return samples_enc, predicates_enc, joins_enc


def make_dataset(samples_list, predicates_list, joins_list, labels,
                 max_num_joins, max_num_predicates, max_num_tables=None):
    """
    Add zero-padding and wrap as TensorDataset.
    Equivalent to mscn/data.py make_dataset but works with in-memory data.
    """
    if max_num_tables is None:
        max_num_tables = max_num_joins + 1

    sample_feat_size = samples_list[0][0].shape[0]
    pred_feat_size = predicates_list[0][0].shape[0]
    join_feat_size = joins_list[0][0].shape[0]

    sample_masks = []
    sample_tensors = []
    for sample in samples_list:
        sample_tensor = np.vstack(sample)
        num_pad = max(0, max_num_tables - sample_tensor.shape[0])
        sample_mask = np.zeros((max_num_tables, 1), dtype=np.float32)
        sample_mask[:sample_tensor.shape[0]] = 1.0
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))

    sample_tensors = torch.FloatTensor(np.vstack(sample_tensors))
    sample_masks = torch.FloatTensor(np.vstack(sample_masks))

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates_list:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.zeros((max_num_predicates, 1), dtype=np.float32)
        predicate_mask[:predicate_tensor.shape[0]] = 1.0
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))

    predicate_tensors = torch.FloatTensor(np.vstack(predicate_tensors))
    predicate_masks = torch.FloatTensor(np.vstack(predicate_masks))

    join_masks = []
    join_tensors = []
    for join in joins_list:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.zeros((max_num_joins, 1), dtype=np.float32)
        join_mask[:join_tensor.shape[0]] = 1.0
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))

    join_tensors = torch.FloatTensor(np.vstack(join_tensors))
    join_masks = torch.FloatTensor(np.vstack(join_masks))

    target_tensor = torch.FloatTensor(labels)

    return TensorDataset(
        sample_tensors, predicate_tensors, join_tensors,
        target_tensor, sample_masks, predicate_masks, join_masks
    )


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, min_val, max_val, epochs, optimizer=None):
    """Train the MSCN model for given epochs. Returns (optimizer, epoch_losses)."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        loss_total = 0.0
        for data_batch in train_loader:
            samples, predicates, joins, targets, s_mask, p_mask, j_mask = data_batch
            if DEVICE.type == "cuda":
                samples, predicates, joins, targets = (
                    samples.to(DEVICE), predicates.to(DEVICE),
                    joins.to(DEVICE), targets.to(DEVICE)
                )
                s_mask, p_mask, j_mask = (
                    s_mask.to(DEVICE), p_mask.to(DEVICE), j_mask.to(DEVICE)
                )
            optimizer.zero_grad()
            outputs = model(samples, predicates, joins, s_mask, p_mask, j_mask)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        avg_loss = loss_total / max(len(train_loader), 1)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs}, loss: {avg_loss:.4f}")

    return optimizer, epoch_losses


def predict(model, data_loader):
    """Run model prediction, return (preds, actuals)."""
    preds = []
    actuals = []
    model.eval()
    with torch.no_grad():
        for data_batch in data_loader:
            samples, predicates, joins, targets, s_mask, p_mask, j_mask = data_batch
            if DEVICE.type == "cuda":
                samples, predicates, joins = (
                    samples.to(DEVICE), predicates.to(DEVICE), joins.to(DEVICE)
                )
                s_mask, p_mask, j_mask = (
                    s_mask.to(DEVICE), p_mask.to(DEVICE), j_mask.to(DEVICE)
                )
            outputs = model(samples, predicates, joins, s_mask, p_mask, j_mask)
            for i in range(outputs.shape[0]):
                preds.append(outputs[i].item())
                actuals.append(targets[i].item())
    return preds, actuals


def compute_qerrors(preds_norm, labels_norm, min_val, max_val):
    """Compute Q-errors from normalized predictions and labels."""
    preds_unnorm = unnormalize_labels(preds_norm, min_val, max_val)
    labels_unnorm = unnormalize_labels(labels_norm, min_val, max_val)

    preds_unnorm = np.array(preds_unnorm, dtype=np.float64).flatten()
    labels_unnorm = np.array(labels_unnorm, dtype=np.float64).flatten()
    preds_unnorm = np.maximum(preds_unnorm, 1e-10)
    labels_unnorm = np.maximum(labels_unnorm, 1e-10)

    return np.maximum(preds_unnorm / labels_unnorm, labels_unnorm / preds_unnorm)


def _get_boolean_columns(cursor):
    """Return set of public boolean columns as {'table.column', ...}."""
    sql = """
    SELECT table_name, column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND data_type = 'boolean';
    """
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.connection.commit()
        return {f"{tbl.lower()}.{col.lower()}" for tbl, col in rows}
    except Exception as e:
        print(f"WARNING: Could not load boolean column metadata: {e}")
        try:
            cursor.connection.rollback()
        except Exception:
            pass
        return set()


def _normalize_boolean_predicates(queries, boolean_columns):
    """
    Normalize boolean predicate values for DB compatibility.
    Converts 0/1-like values to FALSE/TRUE for known boolean columns.
    """
    if not boolean_columns:
        return 0

    true_vals = {"1", "true", "t", "yes", "y"}
    false_vals = {"0", "false", "f", "no", "n"}
    converted = 0

    for q in queries:
        alias_to_table = {}
        for table_entry in q.get("tables", []):
            parts = table_entry.strip().split()
            if not parts:
                continue
            table_name = parts[0].lower()
            alias = parts[1].lower() if len(parts) > 1 else table_name
            alias_to_table[alias] = table_name

        new_preds = []
        for pred in q.get("predicates", []):
            if len(pred) != 3:
                new_preds.append(pred)
                continue

            col, op, val = pred
            col_parts = str(col).split('.', 1)
            if len(col_parts) != 2:
                new_preds.append(pred)
                continue

            alias = col_parts[0].lower()
            column_name = col_parts[1].lower()
            table_name = alias_to_table.get(alias)
            full_col = f"{table_name}.{column_name}" if table_name else None

            if full_col in boolean_columns and op in ("=", "!=", "<>"):
                v = str(val).strip().strip("'\"").lower()
                if v in true_vals:
                    new_preds.append((col, op, "TRUE"))
                    converted += 1
                    continue
                if v in false_vals:
                    new_preds.append((col, op, "FALSE"))
                    converted += 1
                    continue

            new_preds.append(pred)

        q["predicates"] = new_preds

    return converted


def _find_latest_generated_query_file(generated_queries_dir):
    """Return latest generated query file path, or None if not found."""
    base_dir = Path(generated_queries_dir)
    if not base_dir.exists():
        return None

    candidates = list(base_dir.glob("queries_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_queries_from_file(file_path, max_queries=None):
    """
    Load and parse generated queries from JSON/JSONL formats.
    Returns (parsed_queries, raw_count).
    """
    raw_queries = read_json_file(str(file_path))
    if not isinstance(raw_queries, list):
        raise ValueError(f"Query file must contain a list, got: {type(raw_queries).__name__}")

    parsed_queries = []
    for item in raw_queries:
        parsed = None
        if isinstance(item, str):
            parsed = parse_sql_to_mscn(item)
            if parsed is not None:
                parsed["sql"] = item
        elif isinstance(item, dict) and all(k in item for k in ("tables", "joins", "predicates")):
            parsed = {
                "tables": item["tables"],
                "joins": item["joins"],
                "predicates": item["predicates"],
                "sql": item.get("sql"),
            }
        elif isinstance(item, dict) and "sql" in item:
            parsed = parse_sql_to_mscn(item["sql"])
            if parsed is not None:
                parsed["sql"] = item["sql"]

        if parsed is not None:
            parsed["cardinality"] = None
            parsed_queries.append(parsed)

    if max_queries is not None and max_queries > 0:
        parsed_queries = parsed_queries[:max_queries]

    return parsed_queries, len(raw_queries)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    """Execute the complete pipeline."""
    global DEVICE
    DEVICE = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed: {args.seed}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(args.out, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")

    # ── Database connection ─────────────────────────────────────────────
    print("\n=== STEP 1: Database Connection ===")
    conn = get_connection(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
    )
    cursor = conn.cursor()
    print("Connected to PostgreSQL")

    # ── Materialized Samples ────────────────────────────────────────────
    print("\n=== STEP 2: Creating Materialized Samples ===")
    
    table_primary_keys = get_primary_keys(cursor)
    if not table_primary_keys:
        print("WARNING: Could not auto-detect primary keys. Bitmaps may fail.")
        
    materialized_samples = create_materialized_samples(
        cursor, table_primary_keys, args.num_materialized_samples
    )

    # ── Query Generation ────────────────────────────────────────────────
    print(f"\n=== STEP 3: Generating {args.total_queries} Queries ===")

    if args.synthetic:
        print("Using SYNTHETIC query generation (no Ollama needed)")
        all_queries = generate_synthetic_queries(args.total_queries, seed=args.seed or 42)
        total_generated = len(all_queries)
        skipped_validation = 0
        skipped_parse = 0
    else:
        all_queries = []
        total_generated = 0
        skipped_validation = 0
        skipped_parse = 0
        used_generated_file = None

        if args.use_latest_generated:
            query_file = Path(args.generated_queries_file) if args.generated_queries_file else _find_latest_generated_query_file(args.generated_queries_dir)
            if query_file is not None and query_file.exists():
                try:
                    # Honor --total-queries only when it is explicitly passed in CLI.
                    # If not explicitly passed, use all queries from the generated file.
                    max_queries = args.total_queries if getattr(args, "total_queries_explicit", False) else None
                    loaded_queries, raw_count = _load_queries_from_file(query_file, max_queries=max_queries)
                    if loaded_queries:
                        all_queries = loaded_queries
                        total_generated = raw_count
                        skipped_parse = max(raw_count - len(loaded_queries), 0)
                        used_generated_file = str(query_file)
                        print(f"Using pre-generated queries from: {query_file}")
                        print(f"  Loaded {len(loaded_queries)} parsed queries from {raw_count} raw entries")
                    else:
                        print(f"WARNING: No parseable queries found in {query_file}; falling back to fresh generation")
                except Exception as e:
                    print(f"WARNING: Failed to load pre-generated queries from {query_file}: {e}")
                    print("         Falling back to fresh LLM generation")
            elif args.generated_queries_file:
                print(f"WARNING: --generated-queries-file not found: {args.generated_queries_file}")
                print("         Falling back to fresh LLM generation")

        if not all_queries:
            print(f"Using LLM generation via Ollama ({args.model_name})")

            # Load schema and optionally stats
            try:
                with open(args.schema_file, 'r') as f:
                    schema_text = f.read()
            except Exception as e:
                print(f"ERROR: Could not read schema file '{args.schema_file}': {e}")
                return

            stats_text = ""
            if args.stats_file:
                try:
                    with open(args.stats_file, 'r') as f:
                        stats_text = f.read()
                except Exception as e:
                    print(f"WARNING: Could not read stats file '{args.stats_file}': {e}")

            # Build schema validator for in-memory query validation (no DB needed)
            schema_validator = SchemaValidator(schema_text, stats_text)
            print(f"  Schema validator: {len(schema_validator.tables)} tables, "
                  f"{len(schema_validator.numeric_columns)} numeric cols, "
                  f"{len(schema_validator.valid_joins)//2} FK joins")

            raw_sqls = generate_all_queries(
                total_queries=args.total_queries,
                schema_text=schema_text,
                stats_text=stats_text,
                batch_size=args.batch_size_gen,
                model_name=args.model_name,
                ollama_url=args.ollama_url,
                schema_validator=schema_validator,
            )

            # Convert SQL strings to structured format
            # (queries are already schema-validated during generation,
            #  this is a safety net + format conversion step)
            all_queries = []
            skipped_validation = 0
            for sql in raw_sqls:
                if not validate_sql(sql, schema_validator):
                    skipped_validation += 1
                    continue
                parsed = parse_sql_to_mscn(sql)
                if parsed:
                    parsed["cardinality"] = None  # unlabeled
                    all_queries.append(parsed)
                else:
                    print(f"  [skip-parse] Could not parse: {sql[:80]}...")

            if skipped_validation > 0:
                print(f"  Post-generation safety net filtered {skipped_validation} additional queries")

            total_generated = len(raw_sqls)
            valid_count = len(all_queries)
            skipped_parse = total_generated - valid_count - skipped_validation
        elif used_generated_file:
            print("Skipping fresh query generation because reusable generated queries were found")

    print(f"Generated {len(all_queries)} valid queries")

    if len(all_queries) == 0:
        print("ERROR: No valid queries generated. Exiting.")
        cursor.close()
        conn.close()
        return

    # Normalize predicates for DB compatibility (important for reused files)
    boolean_columns = _get_boolean_columns(cursor)
    bool_converted = _normalize_boolean_predicates(all_queries, boolean_columns)
    if bool_converted > 0:
        print(f"Normalized {bool_converted} boolean predicate values for PostgreSQL")

    # ── Build vocabularies from ALL queries (before labeling) ───────────
    print("\n=== STEP 4: Building Vocabularies ===")
    table2vec, column2vec, op2vec, join2vec = build_vocabularies(all_queries)
    column_min_max = build_column_min_max(all_queries)

    print(f"  Tables: {len(table2vec)}")
    print(f"  Columns: {len(column2vec)}")
    print(f"  Operators: {len(op2vec)}")
    print(f"  Joins: {len(join2vec)}")

    sample_feats = len(table2vec) + args.num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = max(len(join2vec), 1)

    # ── Split into train pool and validation set ────────────────────────
    n_total = len(all_queries)
    n_val = max(int(n_total * 0.1), 1)
    n_pool = n_total - n_val

    indices = np.random.permutation(n_total)
    pool_indices = list(indices[:n_pool])
    val_indices = list(indices[n_pool:])

    val_queries = [all_queries[i] for i in val_indices]
    print(f"  Pool size: {n_pool}, Validation size: {n_val}")

    # ── Label validation set ────────────────────────────────────────────
    print("\n=== STEP 5: Labeling Validation Set ===")
    label_queries(cursor, val_queries, timeout=args.db_timeout)
    conn.commit()

    # Generate bitmaps for validation set
    print("Generating bitmaps for validation set...")
    val_bitmaps = generate_bitmaps_for_queries(
        cursor, val_queries, materialized_samples, table_primary_keys,
        args.num_materialized_samples, timeout_ms=args.db_timeout
    )
    
    val_bitmaps_path = os.path.join(results_dir, "val_bitmaps.bitmap")
    save_bitmaps(val_bitmaps, val_bitmaps_path)

    # Encode validation set
    val_samples, val_preds, val_joins_enc = [], [], []
    for i, q in enumerate(val_queries):
        s, p, j = encode_single_query(
            q, val_bitmaps[i], table2vec, column2vec, op2vec, join2vec,
            column_min_max, args.num_materialized_samples
        )
        val_samples.append(s)
        val_preds.append(p)
        val_joins_enc.append(j)

    val_labels_raw = [q["cardinality"] for q in val_queries]
    val_labels_norm, min_val, max_val = safe_normalize_labels(val_labels_raw)

    max_num_joins_val = max(len(j) for j in val_joins_enc)
    max_num_preds_val = max(len(p) for p in val_preds)
    max_num_tables_val = max(len(s) for s in val_samples)

    # ═══════════════════════════════════════════════════════════════════
    # ACTIVE LEARNING LOOP
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n=== STEP 6: Active Learning Loop ({args.strategy}) ===")
    print(f"  Rounds: {args.rounds}, Acquire: {args.acquire}, Epochs: {args.epochs}")

    # Initial labeled set (20% of total queries)
    n_initial = min(int(n_total * 0.20), len(pool_indices))
    labeled_pool_idx = list(pool_indices[:n_initial])
    unlabeled_pool_idx = list(pool_indices[n_initial:])

    # Label initial set
    print(f"\nLabeling initial {n_initial} queries...")
    initial_queries = [all_queries[i] for i in labeled_pool_idx]
    t_label_start = time.perf_counter()
    label_queries(cursor, initial_queries, timeout=args.db_timeout)
    t_label_initial = time.perf_counter() - t_label_start
    conn.commit()

    # Generate bitmaps for initial set
    print("Generating bitmaps for initial set...")
    initial_bitmaps = generate_bitmaps_for_queries(
        cursor, initial_queries, materialized_samples, table_primary_keys,
        args.num_materialized_samples, timeout_ms=args.db_timeout
    )
    
    init_bitmaps_path = os.path.join(results_dir, "initial_bitmaps.bitmap")
    save_bitmaps(initial_bitmaps, init_bitmaps_path)

    # Store bitmaps for all labeled queries
    query_bitmaps = {}  # idx -> bitmap
    for i, pool_i in enumerate(labeled_pool_idx):
        query_bitmaps[pool_i] = initial_bitmaps[i]

    # Initialize model
    model = SetConv(sample_feats, predicate_feats, join_feats, args.hid_units)
    if DEVICE.type == "cuda":
        model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    labeled_sizes = []
    median_errors = []
    all_epoch_losses = []       # [(round, epoch, loss), ...]
    all_round_qerrors = []      # [np.array per round]
    labeling_stats = {"success": 0, "failed": 0}
    labeling_times = []         # [(round, num_queries_labeled, elapsed_seconds)]
    labeling_times.append((0, len(initial_queries), t_label_initial))
    final_preds_unnorm = []
    final_labels_unnorm = []

    for r in range(args.rounds):
        print(f"\n{'='*60}")
        print(f"  AL ROUND {r + 1}/{args.rounds}")
        print(f"  Labeled: {len(labeled_pool_idx)}, Unlabeled pool: {len(unlabeled_pool_idx)}")
        print(f"{'='*60}")

        # ── Encode labeled data ─────────────────────────────────────────
        train_samples, train_preds_enc, train_joins_enc = [], [], []
        train_labels_raw = []

        for idx in labeled_pool_idx:
            q = all_queries[idx]
            bmp = query_bitmaps.get(idx)
            s, p, j = encode_single_query(
                q, bmp, table2vec, column2vec, op2vec, join2vec,
                column_min_max, args.num_materialized_samples
            )
            train_samples.append(s)
            train_preds_enc.append(p)
            train_joins_enc.append(j)
            train_labels_raw.append(q["cardinality"])

        train_labels_norm, _, _ = safe_normalize_labels(train_labels_raw, min_val, max_val)

        # Compute max dimensions across train + val for consistent padding
        max_num_joins = max(max(len(j) for j in train_joins_enc), max_num_joins_val)
        max_num_preds = max(max(len(p) for p in train_preds_enc), max_num_preds_val)
        max_num_tables = max(max(len(s) for s in train_samples), max_num_tables_val)

        # Build datasets
        train_dataset = make_dataset(
            train_samples, train_preds_enc, train_joins_enc,
            train_labels_norm, max_num_joins, max_num_preds, max_num_tables
        )
        val_dataset = make_dataset(
            val_samples, val_preds, val_joins_enc,
            val_labels_norm, max_num_joins, max_num_preds, max_num_tables
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size_train)

        # ── Train ───────────────────────────────────────────────────────
        print(f"\n  Training for {args.epochs} epochs...")
        optimizer, epoch_losses = train_model(model, train_loader, min_val, max_val, args.epochs, optimizer)
        for ep_idx, ep_loss in enumerate(epoch_losses):
            all_epoch_losses.append((r + 1, ep_idx + 1, ep_loss))

        # ── Evaluate on validation ──────────────────────────────────────
        preds_val, labels_val = predict(model, val_loader)
        qerrors = compute_qerrors(preds_val, labels_val, min_val, max_val)
        median_q = np.median(qerrors)
        print(f"  Validation Median Q-error: {median_q:.4f}")
        print(f"  Validation 90th Q-error: {np.percentile(qerrors, 90):.4f}")
        print(f"  Validation 95th Q-error: {np.percentile(qerrors, 95):.4f}")

        labeled_sizes.append(len(labeled_pool_idx))
        median_errors.append(median_q)
        all_round_qerrors.append(qerrors)

        # Store final predictions for scatter plot
        final_preds_unnorm = unnormalize_labels(preds_val, min_val, max_val)
        final_labels_unnorm = unnormalize_labels(labels_val, min_val, max_val)

        # ── Acquisition ─────────────────────────────────────────────────
        if len(unlabeled_pool_idx) == 0:
            print("  Pool exhausted. Stopping.")
            break

        acquire_now = min(args.acquire, len(unlabeled_pool_idx))

        if args.strategy == "mc_dropout":
            print(f"  Acquiring {acquire_now} samples using MC Dropout (T=25)...")
            # Need to label + encode unlabeled pool temporarily for uncertainty
            # For MC Dropout, we need bitmaps for the unlabeled queries too
            # We generate dummy bitmaps (zeros) for speed - the model still
            # needs input tensors
            model.train()  # Keep dropout active
            all_mc_preds = []
            T = 25

            # Encode unlabeled pool with dummy bitmaps
            ul_samples, ul_preds_enc, ul_joins_enc = [], [], []
            for idx in unlabeled_pool_idx:
                q = all_queries[idx]
                dummy_bmp = np.zeros((len(q["tables"]), args.num_materialized_samples), dtype=np.float32)
                s, p, j = encode_single_query(
                    q, dummy_bmp, table2vec, column2vec, op2vec, join2vec,
                    column_min_max, args.num_materialized_samples
                )
                ul_samples.append(s)
                ul_preds_enc.append(p)
                ul_joins_enc.append(j)

            # Dummy labels for dataset creation
            dummy_labels = np.zeros(len(unlabeled_pool_idx), dtype=np.float32) + 0.5

            ul_max_joins = max(max(len(j) for j in ul_joins_enc), max_num_joins)
            ul_max_preds = max(max(len(p) for p in ul_preds_enc), max_num_preds)
            ul_max_tables = max(max(len(s) for s in ul_samples), max_num_tables)

            ul_dataset = make_dataset(
                ul_samples, ul_preds_enc, ul_joins_enc,
                dummy_labels, ul_max_joins, ul_max_preds, ul_max_tables
            )
            ul_loader = DataLoader(ul_dataset, batch_size=args.batch_size_train)

            for _ in range(T):
                mc_preds = []
                for batch in ul_loader:
                    s, p, j, tgt, sm, pm, jm = batch
                    if DEVICE.type == "cuda":
                        s, p, j = s.to(DEVICE), p.to(DEVICE), j.to(DEVICE)
                        sm, pm, jm = sm.to(DEVICE), pm.to(DEVICE), jm.to(DEVICE)
                    outputs = model(s, p, j, sm, pm, jm)
                    mc_preds.extend(outputs.detach().cpu().numpy().flatten())
                all_mc_preds.append(mc_preds)

            all_mc_preds = np.array(all_mc_preds)  # [T, pool_size]
            pool_variances = np.var(all_mc_preds, axis=0)
            sorted_rel_idx = np.argsort(pool_variances)[::-1]
            new_rel_indices = sorted_rel_idx[:acquire_now]
            new_indices = [unlabeled_pool_idx[i] for i in new_rel_indices]

        elif args.strategy == "uncertainty":
            print(f"  Acquiring {acquire_now} samples using uncertainty sampling...")
            # Similar approach but using single pass prediction error
            ul_samples, ul_preds_enc, ul_joins_enc = [], [], []
            for idx in unlabeled_pool_idx:
                q = all_queries[idx]
                dummy_bmp = np.zeros((len(q["tables"]), args.num_materialized_samples), dtype=np.float32)
                s, p, j = encode_single_query(
                    q, dummy_bmp, table2vec, column2vec, op2vec, join2vec,
                    column_min_max, args.num_materialized_samples
                )
                ul_samples.append(s)
                ul_preds_enc.append(p)
                ul_joins_enc.append(j)

            dummy_labels = np.zeros(len(unlabeled_pool_idx), dtype=np.float32) + 0.5

            ul_max_joins = max(max(len(j) for j in ul_joins_enc), max_num_joins)
            ul_max_preds = max(max(len(p) for p in ul_preds_enc), max_num_preds)
            ul_max_tables = max(max(len(s) for s in ul_samples), max_num_tables)

            ul_dataset = make_dataset(
                ul_samples, ul_preds_enc, ul_joins_enc,
                dummy_labels, ul_max_joins, ul_max_preds, ul_max_tables
            )
            ul_loader = DataLoader(ul_dataset, batch_size=args.batch_size_train)

            model.eval()
            ul_preds = []
            with torch.no_grad():
                for batch in ul_loader:
                    s, p, j, tgt, sm, pm, jm = batch
                    if DEVICE.type == "cuda":
                        s, p, j = s.to(DEVICE), p.to(DEVICE), j.to(DEVICE)
                        sm, pm, jm = sm.to(DEVICE), pm.to(DEVICE), jm.to(DEVICE)
                    outputs = model(s, p, j, sm, pm, jm)
                    ul_preds.extend(outputs.cpu().numpy().flatten())

            # Use prediction uncertainty (distance from 0.5 = max uncertainty)
            ul_preds = np.array(ul_preds)
            uncertainties = -np.abs(ul_preds - 0.5)  # Higher = more uncertain
            sorted_rel_idx = np.argsort(uncertainties)[::-1]
            new_rel_indices = sorted_rel_idx[:acquire_now]
            new_indices = [unlabeled_pool_idx[i] for i in new_rel_indices]

        else:  # random
            print(f"  Acquiring {acquire_now} samples using random sampling...")
            new_indices = list(np.random.choice(
                unlabeled_pool_idx, acquire_now, replace=False
            ))

        # ── Label and generate bitmaps for acquired queries ─────────────
        print(f"  Labeling {len(new_indices)} acquired queries...")
        acquired_queries = [all_queries[i] for i in new_indices]
        t_label_start = time.perf_counter()
        label_queries(cursor, acquired_queries, timeout=args.db_timeout)
        t_label_round = time.perf_counter() - t_label_start
        conn.commit()
        labeling_times.append((r + 1, len(acquired_queries), t_label_round))

        print(f"  Generating bitmaps for acquired queries...")
        acquired_bitmaps = generate_bitmaps_for_queries(
            cursor, acquired_queries, materialized_samples, table_primary_keys,
            args.num_materialized_samples, timeout_ms=args.db_timeout
        )
        
        acquired_bitmaps_path = os.path.join(results_dir, f"acquired_bitmaps_R{r+1}.bitmap")
        save_bitmaps(acquired_bitmaps, acquired_bitmaps_path)

        for i, pool_i in enumerate(new_indices):
            query_bitmaps[pool_i] = acquired_bitmaps[i]

        # Update pools
        labeled_pool_idx.extend(new_indices)
        unlabeled_pool_idx = list(set(unlabeled_pool_idx) - set(new_indices))

    # ═══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n=== STEP 7: Saving Results ===")

    # Save learning curve CSV
    csv_path = os.path.join(results_dir, "learning_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["labeled_size", "median_qerror", "round"])
        for i, (sz, err) in enumerate(zip(labeled_sizes, median_errors)):
            writer.writerow([sz, err, i + 1])
    print(f"  Saved: {csv_path}")

    # Save labeling times CSV
    labeling_csv_path = os.path.join(results_dir, "labeling_times.csv")
    with open(labeling_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "num_queries_labeled", "elapsed_seconds"])
        for rnd, nq, elapsed in labeling_times:
            writer.writerow([rnd, nq, round(elapsed, 4)])
    print(f"  Saved: {labeling_csv_path}")

    # Save pipeline config
    config_path = os.path.join(results_dir, "pipeline_config.txt")
    with open(config_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"  Saved: {config_path}")

    # Save model
    model_path = os.path.join(results_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Saved: {model_path}")

    # ═══════════════════════════════════════════════════════════════════
    # COMPREHENSIVE GRAPH GENERATION
    # ═══════════════════════════════════════════════════════════════════
    # Count labeling stats from all queries
    for q in all_queries:
        if q.get("cardinality") is not None and q["cardinality"] != "1":
            labeling_stats["success"] += 1
        elif q.get("cardinality") == "1":
            labeling_stats["failed"] += 1

    generate_all_graphs(
        queries=all_queries,
        labeled_sizes=labeled_sizes,
        median_errors=median_errors,
        all_epoch_losses=all_epoch_losses,
        all_round_qerrors=all_round_qerrors,
        final_preds_unnorm=final_preds_unnorm,
        final_labels_unnorm=final_labels_unnorm,
        labeling_stats=labeling_stats,
        strategy=args.strategy,
        total_generated=total_generated,
        valid_count=len(all_queries),
        skipped_validation=skipped_validation,
        skipped_parse=skipped_parse,
        output_dir=results_dir,
        labeling_times=labeling_times,
        total_pool_size=n_pool,
    )

    print(f"\n=== STEP 8: PostgreSQL Benchmark Comparison ===")
    pg_estimates = get_pg_estimates(cursor, val_queries)
    
    # Generate comparative plots directly into the graphs directory
    graphs_dir = os.path.join(results_dir, "graphs")
    plot_pg_vs_mscn_comparison(pg_estimates, final_preds_unnorm, final_labels_unnorm, graphs_dir)

    # Cleanup
    cursor.close()
    conn.close()
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"  Final Median Q-error: {median_errors[-1]:.4f}")
    print(f"  Total labeled queries: {labeled_sizes[-1]}")
    print(f"  Results saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: Data Generation → Active Learning → Model Training"
    )

    # Query generation
    gen_group = parser.add_argument_group("Query Generation")
    gen_group.add_argument("--total-queries", type=int, default=5000,
                           help="Number of total queries to generate (default: 5000)")
    gen_group.add_argument("--batch-size-gen", type=int, default=20,
                           help="LLM generation batch size (default: 20)")
    gen_group.add_argument("--model-name", type=str, default="llama3.2:3b",
                           help="Ollama model for generation (default: llama3.2:3b)")
    gen_group.add_argument("--ollama-url", type=str, default="http://206.1.53.104:11434",
                           help="Ollama API URL")
    gen_group.add_argument("--schema-file", type=str, default="../TryingModels/schema/IMDB_schema.txt",
                           help="Path to the SQL schema file (default: ../TryingModels/schema/IMDB_schema.txt)")
    gen_group.add_argument("--stats-file", type=str, default="",
                           help="Optional path to DB column stats (e.g. from db_utils.py)")
    gen_group.add_argument("--synthetic", action="store_true",
                           help="Use synthetic query generation instead of LLM")
    gen_group.add_argument("--generated-queries-file", type=str, default="",
                           help="Optional path to a generated queries file to reuse")
    gen_group.add_argument("--generated-queries-dir", type=str, default="generated_queries",
                           help="Directory to search for latest generated queries (default: generated_queries)")
    gen_group.add_argument("--use-latest-generated", dest="use_latest_generated", action="store_true", default=True,
                           help="Reuse latest generated queries when available (default: enabled)")
    gen_group.add_argument("--no-use-latest-generated", dest="use_latest_generated", action="store_false",
                           help="Disable reusing generated queries and force fresh generation")

    # Database
    db_group = parser.add_argument_group("Database")
    db_group.add_argument("--db-host", type=str, default="localhost")
    db_group.add_argument("--db-port", type=int, default=5432)
    db_group.add_argument("--db-name", type=str, default="imdb")
    db_group.add_argument("--db-user", type=str, default="postgres")
    db_group.add_argument("--db-password", type=str, default="1111")
    db_group.add_argument("--db-timeout", type=int, default=60000,
                           help="Per-query timeout in ms (default: 60000)")

    # Model & Training
    train_group = parser.add_argument_group("Model & Training")
    train_group.add_argument("--num-materialized-samples", type=int, default=1000,
                              help="Bitmap sample count (default: 1000)")
    train_group.add_argument("--strategy", type=str, default="random",
                              choices=["random", "uncertainty", "mc_dropout"],
                              help="Active learning strategy (default: random)")
    train_group.add_argument("--rounds", type=int, default=5,
                              help="Number of AL rounds (default: 5)")
    train_group.add_argument("--acquire", type=int, default=200,
                              help="Queries to acquire per round (default: 200)")
    train_group.add_argument("--epochs", type=int, default=10,
                              help="Training epochs per round (default: 10)")
    train_group.add_argument("--batch-size-train", type=int, default=1024,
                              help="Training batch size (default: 1024)")
    train_group.add_argument("--hid-units", type=int, default=256,
                              help="MSCN hidden units (default: 256)")
    train_group.add_argument("--cuda", action="store_true",
                              help="Use GPU if available")

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--out", type=str, default="pipeline_results",
                            help="Output directory (default: pipeline_results)")
    out_group.add_argument("--seed", type=int, default=None,
                            help="Random seed for reproducibility")

    # Env flag
    parser.add_argument("--env", action="store_true",
                        help="Load argument defaults from .env file")

    args = parser.parse_args()

    # Track whether --total-queries was explicitly provided by user.
    # This lets generated-query reuse load all queries unless user asks for a cap.
    args.total_queries_explicit = any(
        arg == "--total-queries" or arg.startswith("--total-queries=")
        for arg in sys.argv[1:]
    )

    # If --env is passed, override defaults with .env values
    if args.env:
        from dotenv import load_dotenv
        load_dotenv()

        _defaults = parser.parse_args([])
        if args.total_queries == _defaults.total_queries:
            args.total_queries = int(os.getenv("TOTAL_QUERIES", args.total_queries))
        if args.batch_size_gen == _defaults.batch_size_gen:
            args.batch_size_gen = int(os.getenv("BATCH_SIZE", args.batch_size_gen))
        if args.model_name == _defaults.model_name:
            args.model_name = os.getenv("MODELS", args.model_name)
        if args.ollama_url == _defaults.ollama_url:
            args.ollama_url = os.getenv("OLLAMA_URL", args.ollama_url)
        if args.schema_file == _defaults.schema_file:
            args.schema_file = os.getenv("SCHEMA_FILE", args.schema_file)
        if args.db_host == _defaults.db_host:
            args.db_host = os.getenv("DB_HOST", args.db_host)
        if args.db_port == _defaults.db_port:
            args.db_port = int(os.getenv("DB_PORT", args.db_port))
        if args.db_name == _defaults.db_name:
            args.db_name = os.getenv("DB_NAME", args.db_name)
        if args.db_user == _defaults.db_user:
            args.db_user = os.getenv("DB_USER", args.db_user)
        if args.db_password == _defaults.db_password:
            args.db_password = os.getenv("DB_PASSWORD", args.db_password)
        if args.db_timeout == _defaults.db_timeout:
            args.db_timeout = int(os.getenv("DB_TIMEOUT", args.db_timeout))
        if args.num_materialized_samples == _defaults.num_materialized_samples:
            args.num_materialized_samples = int(os.getenv("NUM_MATERIALIZED_SAMPLES", args.num_materialized_samples))
        if args.rounds == _defaults.rounds:
            args.rounds = int(os.getenv("AL_ROUNDS", args.rounds))
        if args.acquire == _defaults.acquire:
            args.acquire = int(os.getenv("AL_ACQUIRE", args.acquire))
        if args.epochs == _defaults.epochs:
            args.epochs = int(os.getenv("AL_EPOCHS", args.epochs))
        if args.batch_size_train == _defaults.batch_size_train:
            args.batch_size_train = int(os.getenv("BATCH_SIZE", args.batch_size_train))
        if args.hid_units == _defaults.hid_units:
            args.hid_units = int(os.getenv("HIDDEN_UNITS", args.hid_units))

        print(f"[env] Loaded defaults from .env file")

    run_pipeline(args)


if __name__ == "__main__":
    main()

