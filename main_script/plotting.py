import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_and_close_plot(plot_func_name, output_dir, dpi=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{plot_func_name.replace('plot_', '')}.png"
    path = output_dir / filename

    plt.title(plot_func_name.replace("plot_", "").replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def plot_q_error_distribution(queries, output_dir):
    # a little difficult to visualize
    q_errors = [q["q_error"] for q in queries if q.get("q_error") is not None]

    plt.figure()
    plt.hist(q_errors, bins=50, log=True)
    plt.xlabel("Q-error")
    plt.ylabel("Frequency (log scale)")
    save_and_close_plot(plot_q_error_distribution.__name__, output_dir)

def plot_complexity_distribution(df, output_dir):
    # Complexity Distribution
    plt.figure()
    ax = df["ComplexityBucket"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Complexity Bucket")
    plt.ylabel("Number of Queries")

    # Label bars
    for p in ax.patches:
        ax.annotate(
            str(int(p.get_height())),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom"
        )

    save_and_close_plot(plot_complexity_distribution.__name__, output_dir)

def plot_type_vs_complexity(df, output_dir):
    plt.figure()
    df.groupby(["type", "ComplexityBucket"]).size().unstack().plot(kind="bar")
    plt.xlabel("Query Type")
    plt.ylabel("Number of Queries")
    save_and_close_plot(plot_type_vs_complexity.__name__, output_dir)

def plot_complexity_score_distribution(df, output_dir):
    plt.figure()
    plt.hist(df["ComplexityScore"], bins=10)
    plt.xlabel("Complexity Score")
    plt.ylabel("Number of Queries")
    save_and_close_plot(plot_complexity_score_distribution.__name__, output_dir)

def plot_q_error_ecdf(selective, non_selective, output_dir, max_q_error=50):
    sel_q = sorted(q["q_error"] for q in selective if q["q_error"] is not None)
    non_q = sorted(q["q_error"] for q in non_selective if q["q_error"] is not None)

    def ecdf(data):
        y = np.arange(1, len(data) + 1) / len(data)
        return data, y

    plt.figure(figsize=(6, 4))

    if sel_q:
        x, y = ecdf(sel_q)
        plt.step(x, y, where="post", label="Selective")

    if non_q:
        x, y = ecdf(non_q)
        plt.step(x, y, where="post", label="Non-selective")

    plt.xlim(1, max_q_error)
    plt.xlabel("Q-error")
    plt.ylabel("Fraction of queries ≤ x")
    plt.grid(alpha=0.3)

    save_and_close_plot(plot_q_error_ecdf.__name__, output_dir)

def plot_execution_time_comparison(selective, non_selective, output_dir):
    # Each dot is one query
    sel_t = [q["execution_time_ms"] for q in selective if q["execution_time_ms"]]
    non_t = [q["execution_time_ms"] for q in non_selective if q["execution_time_ms"]]

    plt.figure(figsize=(6, 4))

    plt.scatter(
        [1] * len(sel_t), sel_t,
        alpha=0.6, s=20
    )
    plt.scatter(
        [2] * len(non_t), non_t,
        alpha=0.6, s=20
    )

    plt.xticks([1, 2], ["Selective", "Non-selective"])
    plt.ylabel("Execution time (ms)")

    ymin = min(sel_t + non_t) * 0.9
    ymax = max(sel_t + non_t) * 1.1
    plt.ylim(ymin, ymax)

    save_and_close_plot(plot_execution_time_comparison.__name__, output_dir)

def plot_columns_distribution(df, output_dir):
    plt.figure()
    plt.hist(df["Columns"], bins=range(1, df["Columns"].max() + 2))
    plt.xlabel("Number of Columns per Query")
    plt.ylabel("Number of Queries")
    plt.xticks(range(1, df["Columns"].max() + 1))
    save_and_close_plot(plot_columns_distribution.__name__, output_dir)

def plot_tables_distribution(df, output_dir):
    plt.figure()
    plt.hist(df["Tables"], bins=range(1, df["Tables"].max() + 2))
    plt.xlabel("Number of Tables per Query")
    plt.ylabel("Number of Queries")
    plt.xticks(range(1, df["Tables"].max() + 1))
    save_and_close_plot(plot_tables_distribution.__name__, output_dir)

def plot_joins_distribution(df, output_dir):
    plt.figure()
    plt.hist(df["Joins"], bins=range(0, df["Joins"].max() + 2))
    plt.xlabel("Number of Joins per Query")
    plt.ylabel("Number of Queries")
    plt.xticks(range(1, df["Joins"].max() + 1))
    save_and_close_plot(plot_joins_distribution.__name__, output_dir)

def plot_structural_features(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].hist(df["Columns"], bins=range(1, df["Columns"].max() + 2))
    axes[0].set_title("Columns per Query")
    axes[0].set_xlabel("Number of Columns")
    axes[0].set_ylabel("Number of Queries")
    axes[0].set_xticks(range(1, df["Columns"].max() + 1))

    axes[1].hist(df["Tables"], bins=range(1, df["Tables"].max() + 2))
    axes[1].set_title("Tables per Query")
    axes[1].set_xlabel("Number of Tables")
    axes[1].set_xticks(range(1, df["Tables"].max() + 1))

    axes[2].hist(df["Joins"], bins=range(0, df["Joins"].max() + 2))
    axes[2].set_title("Joins per Query")
    axes[2].set_xlabel("Number of Joins")
    axes[2].set_xticks(range(0, df["Joins"].max() + 1))

    save_and_close_plot(plot_structural_features.__name__, output_dir)


def plot_explain_vs_execution_per_query(queries, output_dir):
    explain_times = [
        q["explain_time_ms"] for q in queries
        if q.get("explain_time_ms") is not None
    ]
    execution_times = [
        q["execution_time_ms"] for q in queries
        if q.get("execution_time_ms") is not None
    ]

    x = np.arange(len(explain_times))
    width = 0.4

    plt.figure()
    plt.bar(x - width/2, explain_times, width, label="EXPLAIN time (ms)")
    plt.bar(x + width/2, execution_times, width, label="Execution time (ms)")

    plt.xlabel("Query index")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.tight_layout()

    save_and_close_plot(plot_explain_vs_execution_per_query.__name__, output_dir)