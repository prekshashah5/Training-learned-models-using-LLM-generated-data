#pip install sqlglot pandas numpy matplotlib
import sqlglot
from sqlglot import parse_one
import pandas as pd
from utils import read_json_file, get_latest_json_path
from sqlglot import expressions as exp
from pathlib import Path
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from plotting import plot_complexity_distribution, plot_type_vs_complexity, plot_complexity_score_distribution, plot_columns_distribution, plot_tables_distribution, plot_joins_distribution, plot_structural_features


# Extract features using SQLGlot AST

def extract_features(query):
    try:
        tree = parse_one(query)
    except Exception:
        return {"error": "INVALID SQL"}

    # Number of tables
    tables = tree.find_all(sqlglot.expressions.Table)
    num_tables = len(list(tables))

    # Number of joins
    joins = tree.find_all(sqlglot.expressions.Join)
    num_joins = len(list(joins))

    # Number of predicates
    predicate_ops = ["=", "!=", "<", ">", "<=", ">=", " IN ", " LIKE ", " BETWEEN ", " IS "]
    num_predicates = sum(query.upper().count(op) for op in predicate_ops)

    # Aggregations (COUNT, SUM, AVG, etc.)
    agg_funcs = tree.find_all(sqlglot.expressions.AggFunc)
    num_aggregations = len(list(agg_funcs))

    # DISTINCT
    has_distinct = 1 if tree.find(sqlglot.expressions.Distinct) else 0

    # Window Functions
    window_functions = tree.find_all(sqlglot.expressions.Window)
    num_window_functions = len(list(window_functions))

    # Columns in SELECT
    select = tree.find(exp.Select)
    if select:
        num_columns = len(select.expressions)
    else:
        num_columns = 0

    return {
        "Tables": num_tables,
        "Columns": num_columns,
        "Joins": num_joins,
        "Predicates": num_predicates,
        "Aggregations": num_aggregations,
        "WindowFunctions": num_window_functions,
        "Distinct": has_distinct,
    }

# Normalize columns

def normalize(col):
    return (col - col.min()) / (col.max() - col.min() + 1e-9)


# Main complexity calculation

def compute_complexity_matrix(df):

    feature_rows = df["sql"].apply(extract_features)
    features = pd.DataFrame(feature_rows.tolist())

    df = pd.concat([df, features], axis=1)
    

    # Weighted score
    df["ComplexityScore"] = (
        20 * normalize(df["Tables"]) +
        30 * normalize(df["Joins"]) +
        25 * normalize(df["Predicates"]) +
        5 * normalize(df["Aggregations"]) +
        5 * normalize(df["WindowFunctions"]) +
        5 * df["Distinct"]
    ).round(1)

    return df


load_dotenv()

output_folder = Path(os.getenv("OUTPUT_FOLDER", "../output"))
latest_json_path = get_latest_json_path(output_folder)

df = pd.DataFrame(read_json_file(latest_json_path))


if "ComplexityScore" not in df.columns:
    print("ComplexityScore not found, computing now...")
    df = compute_complexity_matrix(df)
    print(df)

    df.to_json(
        latest_json_path,
        orient="records",
        indent=2
    )

    print(f"ComplexityScore computed and saved to {latest_json_path}")
else:
    print("ComplexityScore already exists, skipping computation")

output_dir = latest_json_path.parent / "plots"


# Visualizing 
bins = [0, 10, 20, 30, 40, 60]
labels = ["Simple", "Moderate", "Complex", "Very Complex", "Extreme"]

df["ComplexityBucket"] = pd.cut(
    df["ComplexityScore"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Complexity Distribution
plot_complexity_distribution(df, output_dir)

# Type vs Complexity
plot_type_vs_complexity(df, output_dir)

# Raw Score Distribution
plot_complexity_score_distribution(df, output_dir)

plot_columns_distribution(df, output_dir)
plot_tables_distribution(df, output_dir)
plot_joins_distribution(df, output_dir)

plot_structural_features(df, output_dir)

print(f"Plots saved to {output_dir}")