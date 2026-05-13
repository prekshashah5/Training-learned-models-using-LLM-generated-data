"""
pipeline_graphs.py
Comprehensive graph generation for the complete pipeline.

Produces 14 graphs organized into 3 categories:
  - Data Generation Analysis (6 graphs)
  - Model Training & Testing (6 graphs)
  - Analysis Summary (2 graphs)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, output_dir, filename, dpi=150):
    """Save figure and close."""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [graph] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 1: DATA GENERATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_tables_distribution(queries, output_dir):
    """1. Distribution of number of tables per query."""
    counts = [len(q.get("tables", [])) for q in queries]
    if not counts:
        return

    values, freqs = np.unique(counts, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(values, freqs, color='#3498db', edgecolor='white')
    ax.set_xlabel("Number of Tables")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Tables per Query Distribution")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis='y')
    _save(fig, output_dir, "data_tables_distribution.png")


def plot_joins_distribution(queries, output_dir):
    """2. Distribution of number of joins per query."""
    counts = [len(q.get("joins", [])) for q in queries]
    if not counts:
        return

    values, freqs = np.unique(counts, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(values, freqs, color='#2ecc71', edgecolor='white')
    ax.set_xlabel("Number of Joins")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Joins per Query Distribution")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis='y')
    _save(fig, output_dir, "data_joins_distribution.png")


def plot_predicates_distribution(queries, output_dir):
    """3. Distribution of number of predicates per query."""
    counts = [len(q.get("predicates", [])) for q in queries]
    if not counts:
        return

    values, freqs = np.unique(counts, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(values, freqs, color='#e74c3c', edgecolor='white')
    ax.set_xlabel("Number of Predicates")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Predicates per Query Distribution")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis='y')
    _save(fig, output_dir, "data_predicates_distribution.png")


def plot_structural_features(queries, output_dir):
    """4. Combined structural overview (tables, joins, predicates) in one figure."""
    tables = [len(q.get("tables", [])) for q in queries]
    joins = [len(q.get("joins", [])) for q in queries]
    preds = [len(q.get("predicates", [])) for q in queries]

    if not tables:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    data = [(tables, "Tables", '#3498db'), (joins, "Joins", '#2ecc71'), (preds, "Predicates", '#e74c3c')]

    for ax, (vals, label, color) in zip(axes, data):
        v, c = np.unique(vals, return_counts=True)
        ax.bar(v, c, color=color, edgecolor='white')
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label} per Query")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Query Structural Features", fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, output_dir, "data_structural_features.png")


def plot_cardinality_distribution(queries, output_dir):
    """5. Histogram of true cardinalities (log scale)."""
    cards = []
    for q in queries:
        c = q.get("cardinality")
        if c is not None:
            try:
                cards.append(max(float(c), 1))
            except (ValueError, TypeError):
                pass

    if not cards:
        return

    log_cards = np.log10(np.array(cards))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(log_cards, bins=30, color='#9b59b6', edgecolor='white', alpha=0.85)
    ax.set_xlabel("log₁₀(True Cardinality)")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Cardinality Distribution (log scale)")
    ax.grid(True, alpha=0.3, axis='y')
    _save(fig, output_dir, "data_cardinality_distribution.png")


def plot_query_validation_summary(total_generated, valid_count, skipped_validation, skipped_parse, output_dir):
    """6. Query validation summary (valid vs rejected)."""
    labels = ["Valid Queries", "Rejected (validation)", "Rejected (parse)"]
    values = [valid_count, skipped_validation, skipped_parse]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    # Only show non-zero slices
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        return

    labels_f, values_f, colors_f = zip(*filtered)

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        values_f, labels=labels_f, colors=colors_f, autopct='%1.1f%%',
        startangle=90, pctdistance=0.85
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax.set_title(f"Query Validation Summary (Total: {total_generated})")
    _save(fig, output_dir, "data_query_validation.png")


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 2: MODEL TRAINING & TESTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_learning_curve(labeled_sizes, median_errors, strategy, output_dir):
    """7. Active Learning curve (labeled samples vs Q-error)."""
    if not labeled_sizes or not median_errors:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labeled_sizes, median_errors, marker='o', linewidth=2, color='#2c3e50', markersize=6)
    ax.set_xlabel("Number of Labeled Samples")
    ax.set_ylabel("Validation Median Q-error")
    ax.set_yscale('log')
    ax.set_title(f"{strategy.upper()} Active Learning Curve")
    ax.grid(True, which="both", ls="-", alpha=0.3)

    # Annotate improvement
    if len(median_errors) >= 2:
        improvement = (median_errors[0] - median_errors[-1]) / median_errors[0] * 100
        ax.annotate(f"{improvement:.1f}% improvement in median Q-error",
                    xy=(labeled_sizes[-1], median_errors[-1]),
                    xytext=(labeled_sizes[-1] * 0.7, median_errors[0] * 0.8),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, color='green', fontweight='bold')

    _save(fig, output_dir, f"train_learning_curve_{strategy}.png")


def plot_training_loss(all_epoch_losses, output_dir):
    """8. Training loss curve across all rounds."""
    if not all_epoch_losses:
        return

    # all_epoch_losses: [(round, epoch, loss), ...]
    rounds = sorted(set(r for r, _, _ in all_epoch_losses))

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(rounds)))

    for i, rnd in enumerate(rounds):
        rnd_data = [(e, l) for r, e, l in all_epoch_losses if r == rnd]
        epochs = [e for e, _ in rnd_data]
        losses = [l for _, l in rnd_data]
        ax.plot(epochs, losses, marker='.', color=cmap[i], label=f"Round {rnd}", linewidth=1.5)

    ax.set_xlabel("Epoch (within round)")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Across AL Rounds")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, "train_loss_curve.png")


def plot_predicted_vs_actual(preds_unnorm, labels_unnorm, output_dir):
    """9. Predicted vs Actual cardinality scatter (log-log)."""
    preds = np.array(preds_unnorm, dtype=np.float64).flatten()
    labels = np.array(labels_unnorm, dtype=np.float64).flatten()
    preds = np.maximum(preds, 1)
    labels = np.maximum(labels, 1)

    if len(preds) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(labels, preds, alpha=0.4, s=15, color='#3498db')

    max_val = max(labels.max(), preds.max())
    min_val = max(min(labels.min(), preds.min()), 1)
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='red', linewidth=1.5, label='Ideal (y=x)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("True Cardinality")
    ax.set_ylabel("Predicted Cardinality")
    ax.set_title("Predicted vs Actual Cardinality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, "test_predicted_vs_actual.png")


def plot_qerror_cdf(qerrors, output_dir):
    """10. Q-error cumulative distribution function (CDF)."""
    if not len(qerrors):
        return

    sorted_q = np.sort(qerrors)
    cdf = np.arange(1, len(sorted_q) + 1) / len(sorted_q) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_q, cdf, linewidth=2, color='#e74c3c')
    ax.set_xlabel("Q-error")
    ax.set_ylabel("Cumulative % of Queries")
    ax.set_title("Q-error CDF (Cumulative Distribution)")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Mark key percentiles
    for p, ls in [(50, '--'), (90, ':'), (95, '-.')]:
        val = np.percentile(sorted_q, p)
        ax.axhline(y=p, color='gray', linestyle=ls, alpha=0.5)
        ax.axvline(x=val, color='gray', linestyle=ls, alpha=0.5)
        ax.annotate(f"p{p}={val:.2f}", xy=(val, p), fontsize=8,
                    xytext=(val * 1.5, p - 3), color='#7f8c8d')

    _save(fig, output_dir, "test_qerror_cdf.png")


def plot_qerror_boxplot_per_round(all_round_qerrors, output_dir):
    """11. Q-error boxplot per AL round."""
    if not all_round_qerrors:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(all_round_qerrors, patch_artist=True, showfliers=False)

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(all_round_qerrors)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("AL Round")
    ax.set_ylabel("Q-error")
    ax.set_yscale('log')
    ax.set_title("Q-error Distribution per AL Round")
    ax.set_xticklabels([f"R{i+1}" for i in range(len(all_round_qerrors))])
    ax.grid(True, alpha=0.3, axis='y')
    _save(fig, output_dir, "test_qerror_per_round.png")


def plot_qerror_stats_per_round(all_round_qerrors, output_dir):
    """12. Per-round Q-error statistics (median, 90th, 95th)."""
    if not all_round_qerrors:
        return

    rounds = list(range(1, len(all_round_qerrors) + 1))
    medians = [np.median(q) for q in all_round_qerrors]
    p90s = [np.percentile(q, 90) for q in all_round_qerrors]
    p95s = [np.percentile(q, 95) for q in all_round_qerrors]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, medians, 'o-', label='Median', linewidth=2, color='#2ecc71')
    ax.plot(rounds, p90s, 's--', label='90th percentile', linewidth=1.5, color='#f39c12')
    ax.plot(rounds, p95s, '^:', label='95th percentile', linewidth=1.5, color='#e74c3c')

    ax.set_xlabel("AL Round")
    ax.set_ylabel("Q-error")
    ax.set_yscale('log')
    ax.set_title("Q-error Statistics per Round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _save(fig, output_dir, "test_qerror_stats.png")


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 3: ANALYSIS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def plot_labeling_success_rate(labeling_stats, output_dir):
    """13. Labeling success rate (labeled vs failed)."""
    success = labeling_stats.get("success", 0)
    failed = labeling_stats.get("failed", 0)

    if success + failed == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    sizes = [success, failed]
    labels = [f"Success\n({success})", f"Failed\n({failed})"]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)

    ax.pie(sizes, labels=labels, colors=colors, explode=explode,
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax.set_title("Query Labeling Success Rate")
    _save(fig, output_dir, "analysis_labeling_rate.png")


def plot_pipeline_summary(labeled_sizes, median_errors, all_epoch_losses,
                          labeling_stats, queries, strategy, output_dir):
    """14. Multi-panel pipeline summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Learning Curve
    ax = axes[0, 0]
    if labeled_sizes and median_errors:
        ax.plot(labeled_sizes, median_errors, 'o-', linewidth=2, color='#2c3e50')
        ax.set_xlabel("Labeled Samples")
        ax.set_ylabel("Median Q-error")
        ax.set_yscale('log')
        ax.set_title("Active Learning Curve")
        ax.grid(True, alpha=0.3)

    # Panel 2: Training Loss (last round)
    ax = axes[0, 1]
    if all_epoch_losses:
        last_round = max(r for r, _, _ in all_epoch_losses)
        last_data = [(e, l) for r, e, l in all_epoch_losses if r == last_round]
        if last_data:
            epochs, losses = zip(*last_data)
            ax.plot(epochs, losses, 'o-', color='#e74c3c', linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training Loss (Round {last_round})")
            ax.grid(True, alpha=0.3)

    # Panel 3: Structural Summary
    ax = axes[1, 0]
    n_tables = [len(q.get("tables", [])) for q in queries]
    n_joins = [len(q.get("joins", [])) for q in queries]
    n_preds = [len(q.get("predicates", [])) for q in queries]
    summary_data = {
        "Avg Tables": np.mean(n_tables) if n_tables else 0,
        "Avg Joins": np.mean(n_joins) if n_joins else 0,
        "Avg Predicates": np.mean(n_preds) if n_preds else 0,
        "Total Queries": len(queries),
    }
    bars = ax.bar(summary_data.keys(), summary_data.values(), color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
    for bar, val in zip(bars, summary_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title("Query Structural Summary")
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Labeling Stats
    ax = axes[1, 1]
    success = labeling_stats.get("success", 0)
    failed = labeling_stats.get("failed", 0)
    if success + failed > 0:
        ax.pie([success, failed], labels=[f"Success ({success})", f"Failed ({failed})"],
               colors=['#2ecc71', '#e74c3c'], autopct='%1.0f%%', startangle=90)
        ax.set_title("Labeling Outcome")
    else:
        ax.text(0.5, 0.5, "No labeling data", ha='center', va='center', fontsize=12)
        ax.set_title("Labeling Outcome")

    fig.suptitle(f"Pipeline Summary - {strategy.upper()} Strategy", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, output_dir, "analysis_pipeline_summary.png")


def plot_labeling_efficiency(labeling_times, median_errors, total_pool_size, strategy, output_dir):
    """15. Labeling Efficiency: AL cumulative time vs supervised all-at-once baseline.

    Args:
        labeling_times: [(round, num_queries_labeled, elapsed_seconds), ...]
        median_errors: List of median Q-errors per round
        total_pool_size: Total number of pool queries (supervised labels all at start)
        strategy: Active learning strategy name
        output_dir: Directory to save the graph
    """
    if not labeling_times:
        return

    rounds = [r for r, _, _ in labeling_times]
    batch_queries = [nq for _, nq, _ in labeling_times]
    batch_times = [t for _, _, t in labeling_times]

    # Cumulative labeling time for AL
    cum_times = np.cumsum(batch_times)
    # Cumulative queries labeled
    cum_queries = np.cumsum(batch_queries)

    # Supervised estimate: all pool queries labeled at once
    total_al_queries = sum(batch_queries)
    total_al_time = sum(batch_times)
    avg_time_per_query = total_al_time / total_al_queries if total_al_queries > 0 else 0
    supervised_time = total_pool_size * avg_time_per_query

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})

    # Add vertical padding so legend and upper lines do not overlap
    max_y = max(supervised_time, cum_times[-1]) if len(cum_times) > 0 else supervised_time
    if max_y > 0:
        ax1.set_ylim(0, max_y * 1.35)

    # Bar chart: per-round batch labeling time (ax1)
    bar_colors = ['#85C1E9' if r == 0 else '#AED6F1' for r in rounds]
    bars = ax1.bar(rounds, batch_times, color=bar_colors, edgecolor='white',
                   alpha=0.6, label='Per-round labeling time', zorder=2)

    # Line: cumulative AL labeling time (ax1)
    ax1.plot(rounds, cum_times, 'o-', color='#2C3E50', linewidth=2.5,
             markersize=7, label=f'AL cumulative ({strategy})', zorder=3)

    # Supervised baseline (ax1)
    ax1.axhline(y=supervised_time, color='#E74C3C', linestyle='--', linewidth=2,
                label=f'Supervised baseline time ({total_pool_size}q)', zorder=3)

    # Annotate bars with only time (decluttered)
    for bar, t in zip(bars, batch_times):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max_y * 0.02),
                 f'{t:.0f}s', ha='center', va='bottom',
                 fontsize=9, color='#5D6D7E')

    # Annotate final cumulative point
    ax1.annotate(
        f'{cum_times[-1]:.0f}s\n({int(cum_queries[-1])} queries)',
        xy=(rounds[-1], cum_times[-1]),
        xytext=(rounds[-1] - 0.5, cum_times[-1] + (max_y * 0.08)),
        arrowprops=dict(arrowstyle='->', color='#2C3E50', relpos=(0.5, 0)),
        fontsize=10, fontweight='bold', color='#2C3E50', ha='center'
    )

    # Annotate supervised baseline on the far left
    ax1.text(rounds[0], supervised_time + (max_y * 0.02),
             f'Supervised Baseline: {supervised_time:.0f}s',
             ha='left', va='bottom', fontsize=10, color='#E74C3C', fontweight='bold')

    ax1.set_ylabel('Labeling Time (seconds)', fontsize=11)
    ax1.set_title(f'Labeling Efficiency: Labeling Time Breakdown', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#BDC3C7')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add secondary x-axis to ax1 (Top) showing cumulative queries
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(rounds)
    ax2.set_xticklabels([f'{int(cq)}' for cq in cum_queries], fontsize=8)
    ax2.set_xlabel('Cumulative Queries Labeled', fontsize=9, color='gray')
    ax2.tick_params(axis='x', colors='gray')

    # Bottom Subplot for Q-Error (ax3)
    if median_errors:
        # median_errors is logged per round (length R), rounds includes init (length R+1)
        plot_rounds = rounds[-len(median_errors):]
        
        ax3.plot(plot_rounds, median_errors, 's-', color='#8E44AD', linewidth=3,
                 markersize=8, label='AL Median Q-Error', zorder=4)
                 
        # Target baseline based on the active learning run
        target_qerror = median_errors[-1]
        ax3.axhline(y=target_qerror, color='#9B59B6', linestyle='--', linewidth=2,
                    label=f'Supervised Target Q-Error ({target_qerror:.1f})', zorder=4)

        ax3.set_ylabel('Median Q-Error', fontsize=11, color='#8E44AD', fontweight='bold')
        ax3.set_title(f'Labeling Efficiency: Q-Error Convergence', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#BDC3C7')
        
    ax3.set_xlabel('Active Learning Round', fontsize=11)
    ax3.set_xticks(rounds)
    ax3.set_xticklabels([f'Init' if r == 0 else f'R{r}' for r in rounds])
    ax3.grid(True, alpha=0.3, axis='both')

    fig.tight_layout()
    _save(fig, output_dir, "analysis_labeling_efficiency.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_graphs(
    queries,
    labeled_sizes,
    median_errors,
    all_epoch_losses,
    all_round_qerrors,
    final_preds_unnorm,
    final_labels_unnorm,
    labeling_stats,
    strategy,
    total_generated,
    valid_count,
    skipped_validation,
    skipped_parse,
    output_dir,
    labeling_times=None,
    total_pool_size=None,
):
    """
    Generate all 15 pipeline graphs.

    Args:
        queries: List of all query dicts
        labeled_sizes: List of labeled set sizes per round
        median_errors: List of median Q-errors per round
        all_epoch_losses: [(round, epoch, loss), ...]
        all_round_qerrors: [np.array of qerrors per round]
        final_preds_unnorm: Final validation predictions (unnormalized)
        final_labels_unnorm: Final validation labels (unnormalized)
        labeling_stats: {"success": int, "failed": int}
        strategy: Active learning strategy name
        total_generated: Total raw queries generated
        valid_count: Queries that passed validation
        skipped_validation: Queries rejected by validate_sql
        skipped_parse: Queries that failed MSCN parsing
        output_dir: Directory to save all graphs
        labeling_times: [(round, num_queries, elapsed_sec), ...] or None
        total_pool_size: Total pool queries for supervised baseline estimate
    """
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    print(f"\n=== Generating Comprehensive Graphs ===")
    print(f"  Output: {graphs_dir}")

    # Category 1: Data Generation Analysis
    print("  Category 1: Data Generation Analysis")
    plot_tables_distribution(queries, graphs_dir)
    plot_joins_distribution(queries, graphs_dir)
    plot_predicates_distribution(queries, graphs_dir)
    plot_structural_features(queries, graphs_dir)
    plot_cardinality_distribution(queries, graphs_dir)
    plot_query_validation_summary(total_generated, valid_count, skipped_validation, skipped_parse, graphs_dir)

    # Category 2: Model Training & Testing
    print("  Category 2: Model Training & Testing")
    plot_learning_curve(labeled_sizes, median_errors, strategy, graphs_dir)
    plot_training_loss(all_epoch_losses, graphs_dir)
    plot_predicted_vs_actual(final_preds_unnorm, final_labels_unnorm, graphs_dir)
    final_qerrors = all_round_qerrors[-1] if all_round_qerrors else np.array([])
    plot_qerror_cdf(final_qerrors, graphs_dir)
    plot_qerror_boxplot_per_round(all_round_qerrors, graphs_dir)
    plot_qerror_stats_per_round(all_round_qerrors, graphs_dir)

    # Category 3: Analysis Summary
    print("  Category 3: Analysis Summary")
    plot_labeling_success_rate(labeling_stats, graphs_dir)
    plot_pipeline_summary(labeled_sizes, median_errors, all_epoch_losses,
                          labeling_stats, queries, strategy, graphs_dir)
    if labeling_times and total_pool_size:
        plot_labeling_efficiency(labeling_times, median_errors, total_pool_size, strategy, graphs_dir)

    # Count generated files
    n_files = len([f for f in os.listdir(graphs_dir) if f.endswith('.png')])
    print(f"  Generated {n_files} graphs in {graphs_dir}")


def plot_pg_vs_mscn_comparison(pg_estimates, mscn_estimates, test_labels, output_dir):
    """
    Plots the final CDF and Scatter comparisons between PostgreSQL native estimator and the learned MSCN model.
    """
    print("\n  Generating final PG vs MSCN comparative plots...")
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Filter out zeros or negatives for log scale safety
    valid_idx = [i for i in range(len(test_labels)) if test_labels[i] > 0 and pg_estimates[i] > 0 and mscn_estimates[i] > 0]
    
    pg_est = np.array([pg_estimates[i] for i in valid_idx])
    mscn_est = np.array([mscn_estimates[i] for i in valid_idx])
    labels = np.array([test_labels[i] for i in valid_idx])

    if len(labels) == 0:
        print("  Warning: No valid labels found to plot comparison. Skipping.")
        return

    # Compute Q-Errors locally for plotting
    pg_qerrors = np.maximum(pg_est / labels, labels / pg_est)
    mscn_qerrors = np.maximum(mscn_est / labels, labels / mscn_est)

    # 1. Plot CDF
    fig, ax = plt.subplots(figsize=(8, 6))
    pg_qerrors_sorted = np.sort(pg_qerrors)
    mscn_qerrors_sorted = np.sort(mscn_qerrors)
    cdf = np.arange(1, len(pg_qerrors) + 1) / len(pg_qerrors)
    
    ax.plot(pg_qerrors_sorted, cdf, linewidth=2.5, color='#e74c3c', label='PostgreSQL Estimates')
    ax.plot(mscn_qerrors_sorted, cdf, linewidth=2.5, color='#3498db', label='Final MSCN Model')
    
    ax.set_xscale('log')
    ax.set_xlabel('Q-Error (log scale)')
    ax.set_ylabel('CDF')
    ax.set_title('Cardinality Estimation Accuracy: PostgreSQL vs MSCN')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    out_file = os.path.join(output_dir, 'compare_pg_vs_mscn_cdf.png')
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Plot Quantile Line Chart (same data perspective as CDF, flipped axes)
    fig_line, ax_line = plt.subplots(figsize=(8, 6))
    quantiles = np.arange(1, len(pg_qerrors_sorted) + 1) / len(pg_qerrors_sorted) * 100.0

    ax_line.plot(quantiles, pg_qerrors_sorted, linewidth=2.5, color='#e74c3c', label='PostgreSQL Estimates')
    ax_line.plot(quantiles, mscn_qerrors_sorted, linewidth=2.5, color='#3498db', label='Final MSCN Model')

    ax_line.set_xlabel('Query Quantile (%)')
    ax_line.set_ylabel('Q-Error (log scale)')
    ax_line.set_title('Q-Error by Query Quantile: PostgreSQL vs MSCN')
    ax_line.set_yscale('log')
    ax_line.grid(True, alpha=0.3, which='both')
    ax_line.legend()

    out_file_line = os.path.join(output_dir, 'compare_pg_vs_mscn_qerror_line.png')
    fig_line.savefig(out_file_line, dpi=150, bbox_inches='tight')
    plt.close(fig_line)

    # 3. Plot Scatter
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    max_val_plot = max(max(labels), max(pg_est), max(mscn_est)) * 2
    min_val_plot = max(1, min(min(labels), min(pg_est), min(mscn_est)) / 2)
    
    # PG Scatter
    ax1.scatter(labels, pg_est, alpha=0.5, color='#e74c3c')
    ax1.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'k--', alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(min_val_plot, max_val_plot)
    ax1.set_ylim(min_val_plot, max_val_plot)
    ax1.set_xlabel('True Cardinality')
    ax1.set_ylabel('Predicted Cardinality (PostgreSQL)')
    ax1.set_title('PostgreSQL Optimizer')
    
    # MSCN Scatter
    ax2.scatter(labels, mscn_est, alpha=0.5, color='#3498db')
    ax2.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'k--', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(min_val_plot, max_val_plot)
    ax2.set_ylim(min_val_plot, max_val_plot)
    ax2.set_xlabel('True Cardinality')
    ax2.set_ylabel('Predicted Cardinality (MSCN)')
    ax2.set_title('Final Learned MSCN Model')
    
    out_file2 = os.path.join(output_dir, 'compare_pg_vs_mscn_scatter.png')
    fig2.savefig(out_file2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"  Saved PG comparison plots to {output_dir}")
