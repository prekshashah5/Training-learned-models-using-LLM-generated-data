#pip install sqlglot pandas numpy matplotlib
import sqlglot
from sqlglot import parse_one
import pandas as pd
from utils.io_utils import read_json_file
from utils.session_utils import get_latest_json_path
from sqlglot import expressions as exp
from pathlib import Path
import os
import matplotlib.pyplot as plt
from metrics.plotting import (plot_complexity_distribution, plot_type_vs_complexity, plot_complexity_score_distribution, plot_columns_distribution, plot_tables_distribution, plot_joins_distribution, plot_structural_features)

# Extract features using SQLGlot AST

def extract_features(query):
    try:
        tree = parse_one(query)
    except Exception:
        return {"query_valid": "INVALID SQL"}

    # Number of tables
    tables = tree.find_all(sqlglot.expressions.Table)
    num_tables = len(list(tables))

    # Number of joins
    joins = tree.find_all(sqlglot.expressions.Join)
    num_joins = len(list(joins))

    # Number of predicates
    num_predicates = len(list(tree.find_all(exp.Condition)))

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

    # Extract ALL used columns (SELECT, WHERE, JOIN, GROUP BY, etc.)
    # We want unique columns per query to avoid double counting if used in select + where
    used_columns = set()
    for col in tree.find_all(exp.Column):
        # col.sql() gives "table.column" or "column"
        used_columns.add(col.sql())

    return {
        "Tables": num_tables,
        "Columns": num_columns,
        "Joins": num_joins,
        "Predicates": num_predicates,
        "Aggregations": num_aggregations,
        "WindowFunctions": num_window_functions,
        "Distinct": has_distinct,
        "UsedColumns": list(used_columns) # Store as list
    }

# Normalize columns
def normalize(col, max_value=10):
    return (col.clip(upper=max_value)) / max_value


# Main complexity calculation

def compute_complexity_matrix(df):
    df = clean_df_before_complexity(df)
    feature_rows = df["sql"].apply(extract_features)
    features = pd.DataFrame(feature_rows.tolist()).fillna(0)

    df = pd.concat(
        [df.reset_index(drop=True), features.reset_index(drop=True)],
        axis=1
    )

    
    # Normalize columns
    MAX_VALUES = {
        "Tables": 5,
        "Joins": 5,
        "Predicates": 5,
        "Aggregations": 3,
        "WindowFunctions": 2,
    }

    # Weighted score
    df["ComplexityScore"] = (
        20 * normalize(df["Tables"], MAX_VALUES["Tables"]) +
        30 * normalize(df["Joins"], MAX_VALUES["Joins"]) +
        25 * normalize(df["Predicates"], MAX_VALUES["Predicates"]) +
        5 * normalize(df["Aggregations"], MAX_VALUES["Aggregations"]) +
        5 * normalize(df["WindowFunctions"], MAX_VALUES["WindowFunctions"]) +
        5 * df["Distinct"]
    ).round(1)

    return df

def clean_df_before_complexity(df):
    COMPLEXITY_COLUMNS = [
        "Tables",
        "Columns",
        "Joins",
        "Predicates",
        "Aggregations",
        "WindowFunctions",
        "Distinct",
        "ComplexityScore",
        "ComplexityBucket",
        "UsedColumns"
    ]
    df = df.drop(columns=[c for c in COMPLEXITY_COLUMNS if c in df.columns])
    return df


def run_complexity_pipeline(recompute=False, session_name=None):
    # Depending on where the script is executed from, resolve the output folder
    default_output = Path(os.getenv("OUTPUT_FOLDER", "../output"))
    
    # If we are already running from TryingModels, the output folder is ./output, not ../output
    if Path("output").exists() and Path("output").is_dir():
        output_folder = Path("output")
    else:
        output_folder = default_output
    
    if session_name:
        latest_session = output_folder / session_name
        if not latest_session.exists():
            print(f"[error] Session folder {latest_session} does not exist.")
            return
    else:
        # Get the latest session directory
        session_dirs = [d for d in output_folder.iterdir() if d.is_dir() and d.name.startswith("session_")]
        if not session_dirs:
            print("[error] No session folders found for complexity pipeline.")
            return
            
        latest_session = sorted(session_dirs, key=lambda d: d.name)[-1]
        
    print(f"[info] Running complexity pipeline on session: {latest_session}")

    # Use the centralized data loader to grab every model's folder
    from utils.session_utils import load_all_model_runs
    from utils.io_utils import write_json_file
    model_data, _ = load_all_model_runs(latest_session)
    
    if not model_data:
        print("[warn] No model data loaded for complexity analysis")
        return

    # To maintain backward compatibility with old single-folder plotting functions inside SQL_Complexity,
    # we'll aggregate all queries into one massive DataFrame for the global plots,
    # while ensuring each individual model's JSON file gets its own ComplexityScores saved securely.
    all_queries_for_global_plots = []

    for model_name, queries in model_data.items():
        if not queries:
            continue
            
        print(f"\nEvaluating Complexity for: {model_name}")
        df = pd.DataFrame(queries)

        if recompute or "ComplexityScore" not in df.columns:
            print("ComplexityScore not found, computing now...")
            df = clean_df_before_complexity(df)
            df = compute_complexity_matrix(df)
            print(f"ComplexityScore computed for {model_name}")
        else:
            print(f"ComplexityScore already exists for {model_name}, skipping computation")

        bins = [0, 10, 20, 30, 40, 60]
        labels = ["Simple", "Moderate", "Complex", "Very Complex", "Extreme"]

        df["ComplexityBucket"] = pd.cut(
            df["ComplexityScore"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        # Convert back to list of dicts to save as JSONL
        records = df.to_dict(orient="records")
        
        # We need to find the correct json file path for this specific model to overwrite it
        # Since load_all_model_runs strips the exact file path from the query dicts, 
        # we'll iterate the session_folder directly to match it
        for run_dir in latest_session.iterdir():
            if run_dir.is_dir() and model_name.replace(":", "_") in run_dir.name:
                queries_path = run_dir / "queries.jsonl"
                if queries_path.exists():
                    # We write it as JSONL lines format to be consistent
                    with open(queries_path, "w", encoding="utf-8") as f:
                        for q in records:
                            import json
                            f.write(json.dumps(q) + "\n")
                    print(f"Saved complexity scores back to {queries_path}")
                    break
                    
        all_queries_for_global_plots.extend(records)

    # Now generate the global overarching plots combining ALL the models' queries
    if all_queries_for_global_plots:
        global_df = pd.DataFrame(all_queries_for_global_plots)
        output_dir = latest_session / "complexity_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Complexity Distribution
        plot_complexity_distribution(global_df, output_dir)
        # Type vs Complexity
        plot_type_vs_complexity(global_df, output_dir)
        # Raw Score Distribution
        plot_complexity_score_distribution(global_df, output_dir)
        plot_columns_distribution(global_df, output_dir)
        plot_tables_distribution(global_df, output_dir)
        plot_joins_distribution(global_df, output_dir)
        plot_structural_features(global_df, output_dir)

        print(f"\nComplexity Global Plots saved to {output_dir}")