import argparse
import subprocess
import itertools
import csv
import os
import re
import sys
from datetime import datetime

def run_command(command):
    print(f"Running: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace')
    
    full_output = []
    for line in process.stdout:
        print(line, end="")
        full_output.append(line)
    
    process.wait()
    if process.returncode != 0:
        print(f"Error running command: process exited with return code {process.returncode}")
    return "".join(full_output)

def extract_metrics(output):
    # Extract the last "Validation Median Q-error" from the output
    matches = re.findall(r"Validation Median Q-error: ([\d.]+)", output)
    if matches:
        return float(matches[-1])
    return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run automated benchmarks.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for all runs")
    parser.add_argument("--sup-queries", type=int, default=None, help="Fixed number of queries for supervised baseline")
    parser.add_argument("--env", action="store_true", help="Load argument defaults from .env file")
    args = parser.parse_args()

    # If --env is passed, override defaults with .env values
    if args.env:
        from dotenv import load_dotenv
        load_dotenv()

        if args.sup_queries is None:
            _sup = os.getenv("TOTAL_QUERIES")
            if _sup:
                args.sup_queries = int(_sup)

        print(f"[env] Loaded defaults from .env: sup_queries={args.sup_queries}")

    # ---- HARDCODED PARAMETERS ----
    TESTSETS = ["job-light"]
    QUERIES_LIST = [1000, 5000, 10000]
    EPOCHS_LIST = [50,100]
    ROUNDS_LIST = [10]
    ACQUIRE_LIST = [80,150] 
    # ------------------------------

    combinations = list(itertools.product(TESTSETS, QUERIES_LIST, EPOCHS_LIST, ROUNDS_LIST, ACQUIRE_LIST))
    
    results = []
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compare_script = os.path.join(script_dir, "compare_strategies.py")
    benchmarks_base_dir = os.path.join(script_dir, "benchmarks")

    os.makedirs(benchmarks_base_dir, exist_ok=True)
    session_dir = os.path.join(benchmarks_base_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(session_dir, exist_ok=True)
    
    summary_file = os.path.join(session_dir, "summary.csv")
    
    header = ["testset", "queries", "epochs", "rounds", "acquire", "comparison_path"]
    
    with open(summary_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for i, combo in enumerate(combinations):
            testset, queries, epochs, rounds, acquire = combo
            
            print(f"\n--- Comparison {i+1}/{len(combinations)} ---")
            config_name = f"q{queries}_e{epochs}_r{rounds}_a{acquire}"
            config_dir = os.path.join(session_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)

            command = [
                sys.executable, compare_script, testset,
                "--queries", str(queries),
                "--epochs", str(epochs),
                "--rounds", str(rounds),
                "--acquire", str(acquire),
                "--out", config_dir,
                "--strategies", "mc_dropout", "random"
            ]

            if args.seed:
                command.append("--seed")
                command.append(str(args.seed))
            
            if args.sup_queries:
                command.append("--sup-queries")
                command.append(str(args.sup_queries))
            
            run_command(command)
            
            # Store config details for aggregation (add config_name)
            row = [testset, queries, epochs, rounds, acquire, config_dir, config_name]
            writer.writerow(row[:6]) # Write only compatible columns to CSV
            f.flush() 
            
            results.append(row)
            print(f"Comparison completed for {combo}. Path: {config_dir}")

    print(f"\nBenchmarks completed. Summary saved to {summary_file}")
    
    # ---- AGGREGATION & PLOTTING ----
    print("\ngenerating aggregate plots...")
    import pandas as pd
    import matplotlib.pyplot as plt

    # We want to plot curves for 'mc_dropout' from different runs.
    # We will iterate through 'results', read 'comparison_summary.csv' in each config_dir.
    
    plt.figure(figsize=(12, 8))
    plot_1_valid = False
    
    for res in results:
        # res = [testset, queries, epochs, rounds, acquire, config_dir, config_name]
        c_dir = res[5]
        c_name = res[6]
        
        summary_csv = os.path.join(c_dir, "comparison_summary.csv")
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            # Filter for mc_dropout
            mc_df = df[df['strategy'] == 'mc_dropout']
            
            if not mc_df.empty:
                plot_1_valid = True
                # Plot 1: Labeled Samples vs Q-Error
                plt.plot(mc_df['labeled_size'], mc_df['median_qerror'], marker='o', label=f"MC Dropout ({c_name})")

    if plot_1_valid:
        plt.yscale('log')
        plt.xlabel("Number of Labeled Samples")
        plt.ylabel("Validation Median Q-error (Log Scale)")
        plt.title("Aggregate Benchmark: Samples vs Error (MC Dropout)")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        agg_plot_1 = os.path.join(session_dir, "global_summary_plot_samples.png")
        plt.savefig(agg_plot_1)
        print(f"Saved: {agg_plot_1}")
    else:
        print("No MC Dropout data found for Plot 1")

    # Plot 2: Cost vs Q-Error
    plt.figure(figsize=(12, 8))
    plot_2_valid = False

    for res in results:
        c_dir = res[5]
        c_name = res[6]
        
        summary_csv = os.path.join(c_dir, "comparison_summary.csv")
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            mc_df = df[df['strategy'] == 'mc_dropout']
            
            if not mc_df.empty:
                # We need to recalculate 'cumulative_cost' if it's not in summary CSV
                # compare_strategies.py saves 'step_cost' and 'cumulative_cost' to 'learning_data.csv' 
                # but 'comparison_summary.csv' is a concat of those. 
                # Let's check if 'cumulative_cost' exists.
                
                if 'cumulative_cost' in mc_df.columns:
                    plot_2_valid = True
                    plt.plot(mc_df['cumulative_cost'], mc_df['median_qerror'], marker='o', label=f"MC Dropout ({c_name})")
                else:
                    print(f"Warning: 'cumulative_cost' not found in {summary_csv}")

    if plot_2_valid:
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("Total Training Samples Processed (Cumulative Cost) [Log Scale]")
        plt.ylabel("Validation Median Q-error (Log Scale)")
        plt.title("Aggregate Benchmark: Cost-Efficiency (MC Dropout)")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        agg_plot_2 = os.path.join(session_dir, "global_summary_plot_cost.png")
        plt.savefig(agg_plot_2)
        print(f"Saved: {agg_plot_2}")
    else:
        print("No Global Cost data found for Plot 2")

if __name__ == "__main__":
    main()
