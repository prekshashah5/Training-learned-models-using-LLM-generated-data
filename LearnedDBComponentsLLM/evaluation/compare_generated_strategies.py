import argparse
import copy
import csv
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import training.pipeline as pipeline_mod
from config.db_config import get_connection
from generation.format_converter import parse_sql_to_mscn
from labeling.bitmap_utils import (
    create_materialized_samples,
    generate_bitmaps_for_queries,
    get_primary_keys,
)
from labeling.db_labeler import label_queries
from mscn.model import SetConv
from utils.io_utils import read_json_file


VALID_STRATEGIES = {"random", "uncertainty", "mc_dropout"}
STRATEGY_DISPLAY = {
    "random": "Random Sampling",
    "uncertainty": "Uncertainty Sampling",
    "mc_dropout": "MC Dropout",
}


def strategy_label(name):
    return STRATEGY_DISPLAY.get(name, name)


def load_generated_queries(input_path=None):
    if input_path:
        query_path = Path(input_path)
    else:
        candidates = sorted(Path("generated_queries").glob("queries_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No generated query files found in generated_queries/")
        query_path = candidates[-1]

    raw_queries = read_json_file(str(query_path))
    parsed_queries = []

    for item in raw_queries:
        parsed = None
        if isinstance(item, str):
            parsed = parse_sql_to_mscn(item)
        elif isinstance(item, dict) and all(key in item for key in ("tables", "joins", "predicates")):
            parsed = {
                "tables": item["tables"],
                "joins": item["joins"],
                "predicates": item["predicates"],
                "sql": item.get("sql"),
            }
        elif isinstance(item, dict) and "sql" in item:
            parsed = parse_sql_to_mscn(item["sql"])

        if parsed is not None:
            parsed["cardinality"] = None
            parsed_queries.append(parsed)

    if not parsed_queries:
        raise RuntimeError(f"No parseable queries found in {query_path}")

    return parsed_queries, query_path


def prepare_shared_context(args, all_queries):
    pipeline_mod.DEVICE = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    conn = get_connection(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
    )
    cursor = conn.cursor()

    table_primary_keys = get_primary_keys(cursor)
    materialized_samples = create_materialized_samples(
        cursor, table_primary_keys, args.num_materialized_samples
    )

    table2vec, column2vec, op2vec, join2vec = pipeline_mod.build_vocabularies(all_queries)
    column_min_max = pipeline_mod.build_column_min_max(all_queries)

    sample_feats = len(table2vec) + args.num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = max(len(join2vec), 1)

    n_total = len(all_queries)
    n_val = max(int(n_total * args.validation_fraction), 1)
    indices = np.random.permutation(n_total)
    pool_indices = list(indices[:-n_val])
    val_indices = list(indices[-n_val:])

    val_queries = copy.deepcopy([all_queries[i] for i in val_indices])
    label_queries(cursor, val_queries, timeout=args.db_timeout)
    conn.commit()

    val_bitmaps = generate_bitmaps_for_queries(
        cursor, val_queries, materialized_samples, table_primary_keys, args.num_materialized_samples
    )

    val_samples, val_preds, val_joins = [], [], []
    for i, query in enumerate(val_queries):
        samples, predicates, joins = pipeline_mod.encode_single_query(
            query,
            val_bitmaps[i],
            table2vec,
            column2vec,
            op2vec,
            join2vec,
            column_min_max,
            args.num_materialized_samples,
        )
        val_samples.append(samples)
        val_preds.append(predicates)
        val_joins.append(joins)

    val_labels_raw = [q["cardinality"] for q in val_queries]
    val_labels_norm, min_val, max_val = pipeline_mod.safe_normalize_labels(val_labels_raw)

    return {
        "conn": conn,
        "cursor": cursor,
        "table_primary_keys": table_primary_keys,
        "materialized_samples": materialized_samples,
        "table2vec": table2vec,
        "column2vec": column2vec,
        "op2vec": op2vec,
        "join2vec": join2vec,
        "column_min_max": column_min_max,
        "sample_feats": sample_feats,
        "predicate_feats": predicate_feats,
        "join_feats": join_feats,
        "pool_indices": pool_indices,
        "val_indices": val_indices,
        "val_samples": val_samples,
        "val_preds": val_preds,
        "val_joins": val_joins,
        "val_labels_norm": val_labels_norm,
        "val_labels_raw": [float(v) for v in val_labels_raw],
        "min_val": min_val,
        "max_val": max_val,
        "max_num_joins_val": max(len(j) for j in val_joins),
        "max_num_preds_val": max(len(p) for p in val_preds),
        "max_num_tables_val": max(len(s) for s in val_samples),
    }


def select_new_indices(strategy, unlabeled_pool_idx, queries, shared, max_num_joins, max_num_preds, max_num_tables, batch_size):
    if strategy == "random":
        return list(np.random.choice(unlabeled_pool_idx, batch_size, replace=False))

    ul_samples, ul_preds_enc, ul_joins_enc = [], [], []
    for idx in unlabeled_pool_idx:
        query = queries[idx]
        dummy_bmp = np.zeros((len(query["tables"]), shared["sample_feats"] - len(shared["table2vec"])), dtype=np.float32)
        samples, predicates, joins = pipeline_mod.encode_single_query(
            query,
            dummy_bmp,
            shared["table2vec"],
            shared["column2vec"],
            shared["op2vec"],
            shared["join2vec"],
            shared["column_min_max"],
            shared["sample_feats"] - len(shared["table2vec"]),
        )
        ul_samples.append(samples)
        ul_preds_enc.append(predicates)
        ul_joins_enc.append(joins)

    dummy_labels = np.zeros(len(unlabeled_pool_idx), dtype=np.float32) + 0.5
    ul_dataset = pipeline_mod.make_dataset(
        ul_samples,
        ul_preds_enc,
        ul_joins_enc,
        dummy_labels,
        max(max(len(j) for j in ul_joins_enc), max_num_joins),
        max(max(len(p) for p in ul_preds_enc), max_num_preds),
        max(max(len(s) for s in ul_samples), max_num_tables),
    )
    ul_loader = DataLoader(ul_dataset, batch_size=batch_size)

    model = shared["model"]
    if strategy == "mc_dropout":
        model.train()
        mc_predictions = []
        for _ in range(25):
            round_preds = []
            for batch in ul_loader:
                s, p, j, _, sm, pm, jm = batch
                if pipeline_mod.DEVICE.type == "cuda":
                    s, p, j = s.to(pipeline_mod.DEVICE), p.to(pipeline_mod.DEVICE), j.to(pipeline_mod.DEVICE)
                    sm, pm, jm = sm.to(pipeline_mod.DEVICE), pm.to(pipeline_mod.DEVICE), jm.to(pipeline_mod.DEVICE)
                outputs = model(s, p, j, sm, pm, jm)
                round_preds.extend(outputs.detach().cpu().numpy().flatten())
            mc_predictions.append(round_preds)
        variances = np.var(np.array(mc_predictions), axis=0)
        order = np.argsort(variances)[::-1]
    else:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in ul_loader:
                s, p, j, _, sm, pm, jm = batch
                if pipeline_mod.DEVICE.type == "cuda":
                    s, p, j = s.to(pipeline_mod.DEVICE), p.to(pipeline_mod.DEVICE), j.to(pipeline_mod.DEVICE)
                    sm, pm, jm = sm.to(pipeline_mod.DEVICE), pm.to(pipeline_mod.DEVICE), jm.to(pipeline_mod.DEVICE)
                outputs = model(s, p, j, sm, pm, jm)
                preds.extend(outputs.cpu().numpy().flatten())
        uncertainties = -np.abs(np.array(preds) - 0.5)
        order = np.argsort(uncertainties)[::-1]

    return [unlabeled_pool_idx[i] for i in order[:batch_size]]


def run_strategy(strategy, base_queries, shared, args):
    if args.seed is not None:
        pipeline_mod.set_seed(args.seed)

    queries = copy.deepcopy(base_queries)
    labeled_pool_idx = list(shared["pool_indices"][:min(int(len(base_queries) * args.initial_fraction), len(shared["pool_indices"]))])
    unlabeled_pool_idx = list(shared["pool_indices"][len(labeled_pool_idx):])

    initial_queries = [queries[i] for i in labeled_pool_idx]
    label_queries(shared["cursor"], initial_queries, timeout=args.db_timeout)
    shared["conn"].commit()
    initial_bitmaps = generate_bitmaps_for_queries(
        shared["cursor"],
        initial_queries,
        shared["materialized_samples"],
        shared["table_primary_keys"],
        args.num_materialized_samples,
    )
    query_bitmaps = {pool_idx: initial_bitmaps[i] for i, pool_idx in enumerate(labeled_pool_idx)}

    model = SetConv(shared["sample_feats"], shared["predicate_feats"], shared["join_feats"], args.hid_units)
    if pipeline_mod.DEVICE.type == "cuda":
        model.to(pipeline_mod.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    labeled_sizes = []
    median_errors = []
    p90_errors = []
    p95_errors = []
    all_epoch_losses = []
    final_qerrors = np.array([])
    final_preds_unnorm = []
    final_labels_unnorm = []

    for round_idx in range(args.rounds):
        train_samples, train_preds, train_joins, train_labels_raw = [], [], [], []
        for idx in labeled_pool_idx:
            query = queries[idx]
            samples, predicates, joins = pipeline_mod.encode_single_query(
                query,
                query_bitmaps[idx],
                shared["table2vec"],
                shared["column2vec"],
                shared["op2vec"],
                shared["join2vec"],
                shared["column_min_max"],
                args.num_materialized_samples,
            )
            train_samples.append(samples)
            train_preds.append(predicates)
            train_joins.append(joins)
            train_labels_raw.append(query["cardinality"])

        train_labels_norm, _, _ = pipeline_mod.safe_normalize_labels(train_labels_raw, shared["min_val"], shared["max_val"])
        max_num_joins = max(max(len(j) for j in train_joins), shared["max_num_joins_val"])
        max_num_preds = max(max(len(p) for p in train_preds), shared["max_num_preds_val"])
        max_num_tables = max(max(len(s) for s in train_samples), shared["max_num_tables_val"])

        train_dataset = pipeline_mod.make_dataset(train_samples, train_preds, train_joins, train_labels_norm, max_num_joins, max_num_preds, max_num_tables)
        val_dataset = pipeline_mod.make_dataset(shared["val_samples"], shared["val_preds"], shared["val_joins"], shared["val_labels_norm"], max_num_joins, max_num_preds, max_num_tables)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size_train)

        optimizer, epoch_losses = pipeline_mod.train_model(model, train_loader, shared["min_val"], shared["max_val"], args.epochs, optimizer)
        for epoch_idx, loss in enumerate(epoch_losses, start=1):
            all_epoch_losses.append((round_idx + 1, epoch_idx, loss))

        preds_val, labels_val = pipeline_mod.predict(model, val_loader)
        qerrors = pipeline_mod.compute_qerrors(preds_val, labels_val, shared["min_val"], shared["max_val"])
        labeled_sizes.append(len(labeled_pool_idx))
        median_errors.append(float(np.median(qerrors)))
        p90_errors.append(float(np.percentile(qerrors, 90)))
        p95_errors.append(float(np.percentile(qerrors, 95)))
        final_qerrors = qerrors
        final_preds_unnorm = pipeline_mod.unnormalize_labels(preds_val, shared["min_val"], shared["max_val"])
        final_labels_unnorm = pipeline_mod.unnormalize_labels(labels_val, shared["min_val"], shared["max_val"])

        if not unlabeled_pool_idx:
            break

        shared["model"] = model
        acquire_now = min(args.acquire, len(unlabeled_pool_idx))
        new_indices = select_new_indices(strategy, unlabeled_pool_idx, queries, shared, max_num_joins, max_num_preds, max_num_tables, acquire_now)
        acquired_queries = [queries[i] for i in new_indices]
        label_queries(shared["cursor"], acquired_queries, timeout=args.db_timeout)
        shared["conn"].commit()
        acquired_bitmaps = generate_bitmaps_for_queries(
            shared["cursor"],
            acquired_queries,
            shared["materialized_samples"],
            shared["table_primary_keys"],
            args.num_materialized_samples,
        )
        for i, idx in enumerate(new_indices):
            query_bitmaps[idx] = acquired_bitmaps[i]
        labeled_pool_idx.extend(new_indices)
        unlabeled_pool_idx = [idx for idx in unlabeled_pool_idx if idx not in set(new_indices)]

    return {
        "strategy": strategy,
        "labeled_sizes": labeled_sizes,
        "median_errors": median_errors,
        "p90_errors": p90_errors,
        "p95_errors": p95_errors,
        "all_epoch_losses": all_epoch_losses,
        "final_qerrors": final_qerrors,
        "final_preds_unnorm": np.array(final_preds_unnorm, dtype=np.float64),
        "final_labels_unnorm": np.array(final_labels_unnorm, dtype=np.float64),
    }


def save_round_metrics(results, output_dir):
    csv_path = output_dir / "strategy_round_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "round", "labeled_size", "median_qerror", "p90_qerror", "p95_qerror"])
        for result in results:
            for idx, (size, median, p90, p95) in enumerate(zip(result["labeled_sizes"], result["median_errors"], result["p90_errors"], result["p95_errors"]), start=1):
                writer.writerow([result["strategy"], idx, size, median, p90, p95])


def save_validation_predictions(results, output_dir):
    csv_path = output_dir / "validation_predictions.csv"
    labels = results[0]["final_labels_unnorm"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["validation_index", "actual_cardinality"] + [f"pred_{r['strategy']}" for r in results]
        writer.writerow(header)
        for idx in range(len(labels)):
            row = [idx, labels[idx]]
            for result in results:
                row.append(result["final_preds_unnorm"][idx])
            writer.writerow(row)


def save_strategy_summary(results, output_dir):
    csv_path = output_dir / "strategy_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy",
            "display_name",
            "final_labeled_size",
            "final_median_qerror",
            "final_p90_qerror",
            "final_p95_qerror",
            "best_median_qerror",
            "round_of_best_median",
            "final_validation_mae",
        ])
        for result in results:
            final_preds = np.array(result["final_preds_unnorm"], dtype=np.float64)
            final_labels = np.array(result["final_labels_unnorm"], dtype=np.float64)
            mae = float(np.mean(np.abs(final_preds - final_labels))) if len(final_preds) else float("nan")

            medians = result["median_errors"]
            best_idx = int(np.argmin(medians)) if medians else 0

            writer.writerow([
                result["strategy"],
                strategy_label(result["strategy"]),
                result["labeled_sizes"][-1] if result["labeled_sizes"] else None,
                result["median_errors"][-1] if result["median_errors"] else None,
                result["p90_errors"][-1] if result["p90_errors"] else None,
                result["p95_errors"][-1] if result["p95_errors"] else None,
                medians[best_idx] if medians else None,
                best_idx + 1 if medians else None,
                mae,
            ])


def save_chart_guide(results, output_dir):
    guide_path = output_dir / "chart_guide.md"
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write("# Strategy Comparison Chart Guide\n\n")
        f.write("These figures compare strategies on the same generated query set, same train/pool split, and same validation set.\n\n")
        f.write("## Key Metric\n")
        f.write("- Q-error = max(pred/actual, actual/pred). Lower is better.\n")
        f.write("- Median Q-error summarizes typical error.\n")
        f.write("- p90/p95 Q-error summarize tail risk (bad-case behavior).\n\n")

        f.write("## Files\n")
        f.write("- `comparison_learning_curves.png`: Median Q-error vs labeled samples (log scale).\n")
        f.write("- `comparison_round_stats.png`: Round-wise median/p90/p95 Q-error (log scale).\n")
        f.write("- `comparison_qerror_cdf.png`: Final validation CDF of Q-error (log-x).\n")
        f.write("- `comparison_actual_validation_output.png`: Actual validation cardinalities vs predictions.\n")
        f.write("- `comparison_predicted_vs_actual_scatter.png`: Predicted vs actual scatter per strategy.\n")
        f.write("- `strategy_round_metrics.csv`: Per-round metrics table.\n")
        f.write("- `strategy_summary.csv`: Final/best summary metrics table.\n")
        f.write("- `validation_predictions.csv`: Per-query predictions for downstream analysis.\n\n")

        f.write("## Final Snapshot\n")
        for result in results:
            f.write(
                f"- {strategy_label(result['strategy'])}: "
                f"final median={result['median_errors'][-1]:.4f}, "
                f"p90={result['p90_errors'][-1]:.4f}, "
                f"p95={result['p95_errors'][-1]:.4f}, "
                f"labeled={result['labeled_sizes'][-1]}\n"
            )


def plot_learning_curves(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in results:
        label = strategy_label(result["strategy"])
        ax.plot(result["labeled_sizes"], result["median_errors"], marker='o', linewidth=2, label=label)
        if result["labeled_sizes"]:
            ax.annotate(
                f"{result['median_errors'][-1]:.2f}",
                (result["labeled_sizes"][-1], result["median_errors"][-1]),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=9,
            )
    ax.set_yscale('log')
    ax.set_xlabel("Number of Labeled Samples")
    ax.set_ylabel("Validation Median Q-error (log scale, lower is better)")
    ax.set_title("Learning Curves on Shared Data")
    ax.legend(title="Strategy")
    ax.grid(True, which="both", ls='-', alpha=0.3)
    fig.savefig(output_dir / "comparison_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_round_stats(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in results:
        name = strategy_label(result["strategy"])
        rounds = list(range(1, len(result["median_errors"]) + 1))
        ax.plot(rounds, result["median_errors"], marker='o', linewidth=2, label=f"{name} median")
        ax.plot(rounds, result["p90_errors"], linestyle='--', alpha=0.8, label=f"{name} p90")
        ax.plot(rounds, result["p95_errors"], linestyle=':', alpha=0.8, label=f"{name} p95")
    ax.set_yscale('log')
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Q-error (log scale)")
    ax.set_title("Round-wise Validation Q-error (Median, p90, p95)")
    ax.legend(fontsize=8, ncol=2, title="Metrics")
    ax.grid(True, which='both', ls='-', alpha=0.3)
    fig.savefig(output_dir / "comparison_round_stats.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_qerror_cdf(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in results:
        name = strategy_label(result["strategy"])
        qerrors = np.sort(result["final_qerrors"])
        cdf = np.arange(1, len(qerrors) + 1) / len(qerrors) * 100
        ax.plot(qerrors, cdf, linewidth=2, label=name)

        p50 = float(np.percentile(qerrors, 50))
        p90 = float(np.percentile(qerrors, 90))
        ax.scatter([p50, p90], [50, 90], s=20)
        ax.annotate(f"p50={p50:.2f}", (p50, 50), textcoords="offset points", xytext=(4, -10), fontsize=8)
        ax.annotate(f"p90={p90:.2f}", (p90, 90), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xscale('log')
    ax.set_xlabel("Q-error (log scale)")
    ax.set_ylabel("Cumulative % of Validation Queries")
    ax.set_title("Final Validation Q-error CDF")
    ax.legend(title="Strategy")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "comparison_qerror_cdf.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_actual_vs_predictions(results, output_dir):
    labels = results[0]["final_labels_unnorm"]
    order = np.argsort(labels)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(labels[order], color='black', linewidth=2.5, label='Actual validation output')
    for result in results:
        name = strategy_label(result["strategy"])
        preds = result["final_preds_unnorm"][order]
        mae = float(np.mean(np.abs(preds - labels[order])))
        ax.plot(preds, linewidth=1.8, label=f"Predicted - {name} (MAE={mae:.1f})")
    ax.set_yscale('log')
    ax.set_xlabel("Validation queries (sorted by actual cardinality)")
    ax.set_ylabel("Cardinality (log scale)")
    ax.set_title("Actual Validation Output vs Strategy Predictions")
    ax.legend(title="Curves", fontsize=9)
    ax.grid(True, which='both', ls='-', alpha=0.3)
    fig.savefig(output_dir / "comparison_actual_validation_output.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_predicted_vs_actual_scatter(results, output_dir):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        name = strategy_label(result["strategy"])
        labels = np.maximum(result["final_labels_unnorm"], 1)
        preds = np.maximum(result["final_preds_unnorm"], 1)
        ax.scatter(labels, preds, alpha=0.35, s=18)
        low = max(min(labels.min(), preds.min()), 1)
        high = max(labels.max(), preds.max())
        ax.plot([low, high], [low, high], '--', color='red')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Actual cardinality")
        ax.set_ylabel("Predicted cardinality")
        qerrors = np.maximum(preds / labels, labels / preds)
        median_q = float(np.median(qerrors))
        ax.set_title(f"{name}\nmedian Q-error={median_q:.2f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Predicted vs Actual on Shared Validation Set")
    fig.savefig(output_dir / "comparison_predicted_vs_actual_scatter.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare AL strategies on the same generated query set")
    parser.add_argument("--strategies", nargs='+', required=True, help="Strategies to compare, e.g. random uncertainty")
    parser.add_argument("--input", type=str, default=None, help="Path to a generated_queries JSON file; defaults to latest")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--acquire", type=int, default=200)
    parser.add_argument("--batch-size-train", type=int, default=1024)
    parser.add_argument("--hid-units", type=int, default=256)
    parser.add_argument("--num-materialized-samples", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--initial-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--db-host", type=str, default='localhost')
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", type=str, default='imdb')
    parser.add_argument("--db-user", type=str, default='postgres')
    parser.add_argument("--db-password", type=str, default='1111')
    parser.add_argument("--db-timeout", type=int, default=60000)
    args = parser.parse_args()

    invalid = [s for s in args.strategies if s not in VALID_STRATEGIES]
    if invalid:
        raise ValueError(f"Unsupported strategies: {invalid}. Valid: {sorted(VALID_STRATEGIES)}")

    if args.seed is not None:
        pipeline_mod.set_seed(args.seed)

    all_queries, query_path = load_generated_queries(args.input)
    results_base = Path(args.out) if args.out else Path("comparisons_generated") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_base.mkdir(parents=True, exist_ok=True)

    shared = prepare_shared_context(args, all_queries)
    try:
        results = []
        for strategy in args.strategies:
            print(f"\n>>> Running shared-data strategy: {strategy}")
            results.append(run_strategy(strategy, all_queries, shared, args))

        save_round_metrics(results, results_base)
        save_validation_predictions(results, results_base)
        save_strategy_summary(results, results_base)
        plot_learning_curves(results, results_base)
        plot_round_stats(results, results_base)
        plot_qerror_cdf(results, results_base)
        plot_actual_vs_predictions(results, results_base)
        plot_predicted_vs_actual_scatter(results, results_base)
        save_chart_guide(results, results_base)

        with open(results_base / "comparison_config.txt", "w", encoding="utf-8") as f:
            f.write(f"queries_file: {query_path}\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")

        print(f"\nShared-data comparison complete. Results saved to: {results_base}")
    finally:
        shared["cursor"].close()
        shared["conn"].close()


if __name__ == "__main__":
    main()