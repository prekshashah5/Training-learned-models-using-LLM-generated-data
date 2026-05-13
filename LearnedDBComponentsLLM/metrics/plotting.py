import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path

def save_and_close_plot(plot_func_name, output_dir, dpi=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{plot_func_name.replace('plot_', '')}.png"
    path = output_dir / filename

    # Only set a fallback title if no custom title has been set
    ax = plt.gca()
    if not ax.get_title():
        plt.title(plot_func_name.replace("plot_", "").replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def plot_q_error_distribution(queries, output_dir):
    q_errors = [
        q["q_error"]
        for q in queries
        if q.get("q_error") is not None and q["q_error"] > 0
    ]

    if not q_errors:
        print("[warn] No valid Q-errors")
        return

    # Sort Q-errors
    q_errors = sorted(q_errors)

    # Cumulative count
    counts = range(1, len(q_errors) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(
        q_errors,
        counts,
        linewidth=2
    )

    plt.xlabel("Q-error")
    plt.ylabel("Number of Queries")
    plt.title("Cumulative Q-error Distribution")
    plt.grid(True, alpha=0.3)

    save_and_close_plot(plot_q_error_distribution.__name__, output_dir)

def plot_complexity_distribution(df, output_dir):
    counts = df["ComplexityBucket"].value_counts().sort_index()

    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.xlabel("Complexity Bucket")
    plt.ylabel("Number of Queries")

    for x, y in zip(counts.index, counts.values):
        plt.text(x, y, str(y), ha="center", va="bottom")

    save_and_close_plot(plot_complexity_distribution.__name__, output_dir)

def plot_type_vs_complexity(df, output_dir):
    grouped = df.groupby(["type", "ComplexityBucket"]).size().unstack(fill_value=0)

    if grouped.empty:
        print(f"[warn] No complexity data for {plot_type_vs_complexity.__name__}")
        return

    x = np.arange(len(grouped.index))
    width = 0.8 / len(grouped.columns)

    plt.figure(figsize=(14, 8))

    for i, bucket in enumerate(grouped.columns):
        plt.bar(
            x + i * width,
            grouped[bucket].values,
            width,
            label=f"Bucket {bucket}"
        )

    plt.xticks(x + width * (len(grouped.columns) - 1) / 2, grouped.index, rotation=90, fontsize=8)
    plt.xlabel("Query Type")
    plt.ylabel("Number of Queries")
    plt.legend()

    save_and_close_plot(plot_type_vs_complexity.__name__, output_dir)

def plot_complexity_score_distribution(df, output_dir):
    values, counts = np.unique(df["ComplexityScore"], return_counts=True)

    plt.figure()
    plt.bar(values, counts)
    plt.xlabel("Complexity Score")
    plt.ylabel("Number of Queries")

    save_and_close_plot(plot_complexity_score_distribution.__name__, output_dir)

def plot_q_error_comparison(selective, non_selective, output_dir):
    sel_q = sorted(
        q["q_error"] for q in selective
        if q.get("q_error") is not None
    )

    non_q = sorted(
        q["q_error"] for q in non_selective
        if q.get("q_error") is not None
    )

    if not sel_q and not non_q:
        print(f"[warn] No valid Q-errors for {plot_q_error_comparison.__name__}")
        return

    plt.figure(figsize=(8, 4))

    if sel_q:
        plt.plot(
            range(len(sel_q)),
            sel_q,
            label="Selective",
            linewidth=2
        )

    if non_q:
        plt.plot(
            range(len(non_q)),
            non_q,
            label="Non-selective",
            linewidth=2,
            linestyle="--"
        )

    plt.xlabel("Query Rank (sorted independently by Q-error)")
    plt.ylabel("Q-error")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_and_close_plot(plot_q_error_comparison.__name__, output_dir)

def plot_execution_time_comparison(selective, non_selective, output_dir):
    sel_t = sorted(
        q["exec_time_ms"]
        for q in selective
        if q.get("exec_time_ms") is not None
    )

    non_t = sorted(
        q["exec_time_ms"]
        for q in non_selective
        if q.get("exec_time_ms") is not None
    )

    if not sel_t and not non_t:
        print(f"[warn] No valid Execution Times for {plot_execution_time_comparison.__name__}")
        return

    if not sel_t and not non_t:
        print(f"[warn] No valid Execution Times for {plot_execution_time_comparison.__name__}")
        return

    # Define Buckets (expanded for high execution times)
    buckets = [0, 1, 10, 100, 1000, 10000, 60000, float('inf')]
    labels = ['<1ms', '1-10ms', '10-100ms', '100ms-1s', '1s-10s', '10s-1min', '>1min']
    
    # Categorize data
    def get_counts(times, bins):
        if not times:
            return pd.Series(0, index=labels)
        # cut returns categorical, value_counts returns counts by category
        # reindex ensures all labels are present (0 count if missing)
        return pd.cut(times, bins=bins, labels=labels, right=False).value_counts().reindex(labels, fill_value=0)

    sel_counts = get_counts(sel_t, buckets)
    non_counts = get_counts(non_t, buckets)
    
    print(f"\n[DEBUG] {plot_execution_time_comparison.__name__}:")
    print(f"  Selective counts per bucket: {sel_counts.to_dict()}")
    print(f"  Non-selective counts per bucket: {non_counts.to_dict()}")

    # Plotting
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    
    # Selective Bars
    plt.bar(x - width/2, sel_counts.values, width, label='Selective (High filtering)', color='#1f77b4', alpha=0.8)
    # Non-selective Bars
    plt.bar(x + width/2, non_counts.values, width, label='Non-selective (Bulk scan)', color='#ff7f0e', alpha=0.8)

    plt.xlabel('Execution Time Range')
    plt.ylabel('Number of Queries')
    plt.title('Execution Time Performance Distribution')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(sel_counts.values):
        if v > 0: plt.text(i - width/2, v, str(v), ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(non_counts.values):
        if v > 0: plt.text(i + width/2, v, str(v), ha='center', va='bottom', fontsize=8)

    save_and_close_plot(plot_execution_time_comparison.__name__, output_dir)

def plot_columns_distribution(df, output_dir):
    values, counts = np.unique(df["Columns"], return_counts=True)

    plt.figure()
    plt.bar(values, counts)
    plt.xlabel("Number of Columns")
    plt.ylabel("Number of Queries")
    plt.xticks(values.astype(int))
    save_and_close_plot(plot_columns_distribution.__name__, output_dir)


def plot_column_usage_frequency(df, output_dir):
    # Flatten list of lists
    all_cols = []
    for cols in df["UsedColumns"]:
        if isinstance(cols, list):
            all_cols.extend(cols)
        elif isinstance(cols, str):
             # Fallback if somehow stored as string repr
             pass

    if not all_cols:
        print(f"[warn] No column usage data for {plot_column_usage_frequency.__name__}")
        return

    # Count frequency
    counts = pd.Series(all_cols).value_counts()
    
    # Plot Top 20 most used columns
    top_n = 20
    top_cols = counts.head(top_n)

    plt.figure(figsize=(10, 6))
    top_cols.sort_values().plot(kind='barh', color='teal') # Sort for better visual order in barh
    plt.xlabel("Frequency (Number of Queries)")
    plt.ylabel("Column Name")
    plt.title(f"Top {top_n} Most Used Columns")
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    save_and_close_plot(plot_column_usage_frequency.__name__, output_dir)


def plot_tables_distribution(df, output_dir):
    values, counts = np.unique(df["Tables"], return_counts=True)

    plt.figure()
    plt.bar(values, counts)
    plt.xlabel("Number of Tables")
    plt.ylabel("Number of Queries")
    plt.xticks(values.astype(int))
    save_and_close_plot(plot_tables_distribution.__name__, output_dir)


def plot_joins_distribution(df, output_dir):
    values, counts = np.unique(df["Joins"], return_counts=True)

    plt.figure()
    plt.bar(values, counts)
    plt.xlabel("Number of Joins")
    plt.ylabel("Number of Queries")
    plt.xticks(values.astype(int))
    save_and_close_plot(plot_joins_distribution.__name__, output_dir)

def plot_predicates_distribution(df, output_dir):
    values, counts = np.unique(df["Predicates"], return_counts=True)

    plt.figure()
    plt.bar(values, counts, color='teal')
    plt.xlabel("Number of Predicates")
    plt.ylabel("Number of Queries")
    plt.xticks(values.astype(int))
    save_and_close_plot(plot_predicates_distribution.__name__, output_dir)

def plot_structural_features(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, col, title in zip(
        axes,
        ["Columns", "Tables", "Joins"],
        ["Columns per Query", "Tables per Query", "Joins per Query"]
    ):
        values, counts = np.unique(df[col], return_counts=True)
        ax.bar(values, counts)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Number of Queries")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    save_and_close_plot(plot_structural_features.__name__, output_dir)


def plot_explain_vs_execution_per_query(queries, output_dir):
    ratios = []
    for q in queries:
        explain = q.get("explain_time_ms")
        execute = q.get("exec_time_ms")

        if explain is None or execute is None or explain <= 0:
            continue

        ratios.append(execute / explain)

    if not ratios:
        print("[warn] No valid queries for EXPLAIN / Execution ratio plot")
        return

    ratios = sorted(ratios)
    ratio_cap = 7000

    # Cap values ONLY for visualization
    capped_ratios = [min(r, ratio_cap) for r in ratios]

    plt.figure(figsize=(8, 4))
    plt.plot(
        range(len(capped_ratios)),
        capped_ratios,
        linewidth=2
    )

    plt.xlabel("Query Rank (sorted by execution / explain ratio)")
    plt.ylabel("Execution / EXPLAIN Time Ratio")
    plt.ylim(0, ratio_cap * 1.05)
    plt.grid(True, alpha=0.3)

    # Indicate capping
    num_capped = sum(r > ratio_cap for r in ratios)
    if num_capped > 0:
        plt.text(
            0.99,
            0.95,
            f"{num_capped} queries capped at {ratio_cap}×",
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
            fontsize=9
        )


    save_and_close_plot(plot_explain_vs_execution_per_query.__name__, output_dir)

def plot_selective_vs_non_selective_count(selective, non_selective, output_dir):
    counts = {
        "Selective": len(selective),
        "Non-selective": len(non_selective)
    }

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values)
    plt.xlabel("Query Category")
    plt.ylabel("Number of Queries")

    # value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha="center", va="bottom")

    save_and_close_plot(plot_selective_vs_non_selective_count.__name__, output_dir)

# ==================== MULTI-MODEL COMPARISON PLOTS ====================

def plot_q_error_comparison_models(model_data_dict, output_dir):
    """
    Plot Q-error comparison across multiple models.
    
    Args:
        model_data_dict: Dictionary mapping model names to lists of query dicts
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, queries in model_data_dict.items():
        q_errors = sorted([q["q_error"] for q in queries if q.get("q_error") is not None])
        if q_errors:
            plt.plot(range(len(q_errors)), q_errors, label=model_name, marker='o', markersize=3, alpha=0.7)
    
    plt.xlabel("Query Index (sorted by Q-error)")
    plt.ylabel("Q-error")
    plt.title("Q-error Distribution Across Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_and_close_plot(plot_q_error_comparison_models.__name__, output_dir)

def plot_execution_time_models(model_data_dict, output_dir):
    """
    Plot execution time comparison across multiple models.
    
    Args:
        model_data_dict: Dictionary mapping model names to lists of query dicts
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    models = list(model_data_dict.keys())
    exec_times_by_model = []
    
    for model_name, queries in model_data_dict.items():
        exec_times = [q["exec_time_ms"] for q in queries if q.get("exec_time_ms") is not None]
        exec_times_by_model.append(exec_times)
    
    # Box plot for better visualization
    bp = plt.boxplot(exec_times_by_model, labels=models, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xlabel("Model")
    plt.ylabel("Execution Time (ms)")
    plt.title("Execution Time Distribution Across Models")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    save_and_close_plot(plot_execution_time_models.__name__, output_dir)

def plot_metric_statistics_models(model_data_dict, metric_key, output_dir, metric_label=None):
    """
    Plot aggregated statistics (mean, median, p90, p99) for a metric across models.
    
    Args:
        model_data_dict: Dictionary mapping model names to lists of query dicts
        metric_key: Key in query dict to extract metric (e.g., "q_error", "exec_time_ms")
        output_dir: Directory to save the plot
        metric_label: Display label for the metric (defaults to metric_key)
    """
    if metric_label is None:
        metric_label = metric_key.replace("_", " ").title()
    
    models = list(model_data_dict.keys())
    means = []
    medians = []
    p90s = []
    p99s = []
    
    for model_name, queries in model_data_dict.items():
        values = [q[metric_key] for q in queries if q.get(metric_key) is not None]
        if values:
            means.append(np.mean(values))
            medians.append(np.median(values))
            p90s.append(np.percentile(values, 90))
            p99s.append(np.percentile(values, 99))
        else:
            means.append(0)
            medians.append(0)
            p90s.append(0)
            p99s.append(0)
    
    x = np.arange(len(models))
    width = 0.2
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, means, width, label='Mean', alpha=0.8)
    plt.bar(x - 0.5*width, medians, width, label='Median', alpha=0.8)
    plt.bar(x + 0.5*width, p90s, width, label='90th Percentile', alpha=0.8)
    plt.bar(x + 1.5*width, p99s, width, label='99th Percentile', alpha=0.8)
    
    plt.xlabel("Model")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} Statistics Across Models")
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    save_and_close_plot(f"{plot_metric_statistics_models.__name__}_{metric_key}", output_dir)

def plot_validity_rate_models(model_data_dict, output_dir):
    """
    Plot validity rate (valid/total queries) across models.
    
    Args:
        model_data_dict: Dictionary mapping model names to lists of query dicts
        output_dir: Directory to save the plot
    """
    models = list(model_data_dict.keys())
    validity_rates = []
    valid_counts = []
    total_counts = []
    
    for model_name, queries in model_data_dict.items():
        total = len(queries)
        valid = sum(1 for q in queries if q.get("query_valid") == True)
        rate = (valid / total * 100) if total > 0 else 0
        validity_rates.append(rate)
        valid_counts.append(valid)
        total_counts.append(total)
    
    x = np.arange(len(models))
    width = 0.6
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    bars = ax1.bar(x, validity_rates, width, label='Validity Rate (%)', alpha=0.8)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Validity Rate (%)", color='blue')
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, rate, valid, total) in enumerate(zip(bars, validity_rates, valid_counts, total_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%\n({valid}/{total})',
                ha='center', va='bottom', fontsize=9)
    
    plt.title("Query Validity Rate Across Models")
    plt.tight_layout()
    
    save_and_close_plot(plot_validity_rate_models.__name__, output_dir)

def plot_complexity_distribution_models(model_data_dict, output_dir):
    """
    Plot complexity distribution comparison across models.
    
    Args:
        model_data_dict: Dictionary mapping model names to lists of query dicts
        output_dir: Directory to save the plot
    """
    bins = [0, 10, 20, 30, 40, 60]
    labels = ["Simple", "Moderate", "Complex", "Very Complex", "Extreme"]
    
    # Prepare data
    model_complexity_counts = {}
    for model_name, queries in model_data_dict.items():
        complexity_scores = [q.get("ComplexityScore", 0) for q in queries if q.get("ComplexityScore") is not None]
        if complexity_scores:
            buckets = pd.cut(complexity_scores, bins=bins, labels=labels, include_lowest=True)
            counts = buckets.value_counts().sort_index()
            model_complexity_counts[model_name] = counts
    
    # Create grouped bar chart
    if not model_complexity_counts:
        print(f"[warn] No complexity data for {plot_complexity_distribution_models.__name__}")
        return

    x = np.arange(len(labels))
    width = 0.8 / len(model_complexity_counts)
    
    plt.figure(figsize=(12, 6))
    
    for i, (model_name, counts) in enumerate(model_complexity_counts.items()):
        values = [counts.get(label, 0) for label in labels]
        plt.bar(x + i * width, values, width, label=model_name, alpha=0.8)
    
    plt.xlabel("Complexity Bucket")
    plt.ylabel("Number of Queries")
    plt.title("Complexity Distribution Across Models")
    plt.xticks(x + width * (len(model_complexity_counts) - 1) / 2, labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    save_and_close_plot(plot_complexity_distribution_models.__name__, output_dir)

def plot_generation_time_models(model_runs_data, output_dir):
    """
    Plot query generation time across models.
    
    Args:
        model_runs_data: Dictionary mapping model names to run metadata dicts with "generation_time_s" and "num_queries"
        output_dir: Directory to save the plot
    """
    models = list(model_runs_data.keys())
    gen_times = []
    num_queries = []
    queries_per_second = []
    
    for model_name, run_data in model_runs_data.items():
        gen_time = run_data.get("generation_time_s", 0)
        num_q = run_data.get("num_queries", 0)
        qps = num_q / gen_time if gen_time > 0 else 0
        
        gen_times.append(gen_time)
        num_queries.append(num_q)
        queries_per_second.append(qps)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(models))
    width = 0.6
    
    # Plot 1: Generation time
    bars1 = ax1.bar(x, gen_times, width, alpha=0.8, color='steelblue')
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Generation Time (seconds)")
    ax1.set_title("Query Generation Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, time, num_q in zip(bars1, gen_times, num_queries):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s\n({num_q} queries)',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Queries per second
    bars2 = ax2.bar(x, queries_per_second, width, alpha=0.8, color='coral')
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Queries per Second")
    ax2.set_title("Generation Throughput")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, qps in zip(bars2, queries_per_second):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{qps:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    save_and_close_plot(plot_generation_time_models.__name__, output_dir)

def plot_query_error_overview(queries, out_dir: Path):

    total = len(queries)
    errors = sum(1 for q in queries if q.get("exec_error_msg") or q.get("exec_status") == "failed")
    success = total - errors

    labels = ["Successful", "Errored"]
    values = [success, errors]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Number of Queries")

    save_and_close_plot(plot_query_error_overview.__name__, out_dir)
    print(f"[info] Total queries: {total}")
    print(f"[info] Errors: {errors}")


def plot_kl_divergence_comparison(gen_dist_norm, real_dist_norm, kl_value, output_dir):
    """
    Plot a grouped bar chart comparing selectivity distributions
    between LLM-generated and real workloads, annotated with the
    KL divergence value.

    Args:
        gen_dist_norm: dict with keys like "high", "medium", "low" → proportions (generated)
        real_dist_norm: dict with keys like "high", "medium", "low" → proportions (real)
        kl_value: float, the KL divergence KL(real || generated)
        output_dir: Path or str, directory to save the plot
    """
    categories = ["high", "medium", "low"]
    gen_vals = [gen_dist_norm.get(c, 0) for c in categories]
    real_vals = [real_dist_norm.get(c, 0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_real = ax.bar(x - width / 2, real_vals, width,
                       label='Real Workload (JOB-Light)', color='#3498db', alpha=0.85)
    bars_gen = ax.bar(x + width / 2, gen_vals, width,
                      label='LLM Generated', color='#e74c3c', alpha=0.85)

    ax.set_xlabel("Selectivity Class")
    ax.set_ylabel("Proportion of Queries")
    ax.set_title("KL Divergence: LLM Generated vs Real Workload")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars_real:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars_gen:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    save_and_close_plot(plot_kl_divergence_comparison.__name__, output_dir)