import argparse
import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def run_command(command):
    print(f"Executing: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace')
    
    for line in process.stdout:
        # print(line, end="")
        pass

    # Simple logging of output
    for line in process.stdout:
        # Don't print everything, just keep it alive
        if "Round" in line or "Validation Median Q-error" in line:
            print(f"  {line.strip()}")
    
    process.wait()
    return process.returncode

def calculate_efficiency(strategy_data, results_base):
    targets = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0]
    efficiency_records = []

    for strat, df in strategy_data.items():
        # Ensure sorted by progress
        if 'cumulative_epochs' in df.columns:
            df = df.sort_values("cumulative_epochs")
        
        for t in targets:
            # Find first row where error is below target
            reached = df[df["median_qerror"] <= t]
            if not reached.empty:
                first = reached.iloc[0]
                efficiency_records.append({
                    "strategy": strat,
                    "target_q_error": t,
                    "labeled_samples_needed": int(first["labeled_size"]),
                    "epochs_needed": int(first["cumulative_epochs"]),
                    "achieved_q_error": first["median_qerror"]
                })
            else:
                # Did not reach
                efficiency_records.append({
                    "strategy": strat,
                    "target_q_error": t,
                    "labeled_samples_needed": None,
                    "epochs_needed": None,
                    "achieved_q_error": None
                })
    
    if not efficiency_records:
        return

    eff_df = pd.DataFrame(efficiency_records)
    csv_path = os.path.join(results_base, "efficiency_analysis.csv")
    eff_df.to_csv(csv_path, index=False)
    print(f"\nEfficiency analysis saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare Active Learning strategies head-to-head.")
    parser.add_argument("testset", help="synthetic, scale, or job-light")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--acquire", type=int, default=100)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--out", type=str, default=None, help="Output directory for comparisons")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--plot-only", action="store_true", help="Skip training and only generate plots from existing data")
    parser.add_argument("--strategies", nargs="+", default=["supervised", "random", "uncertainty", "mc_dropout"], help="Strategies to run/plot")
    parser.add_argument("--sup-queries", type=int, default=None, help="Number of queries for supervised strategy")
    parser.add_argument("--env", action="store_true", help="Load argument defaults from .env file")
    args = parser.parse_args()

    # If --env is passed, override defaults with .env values
    if args.env:
        from dotenv import load_dotenv
        load_dotenv()

        _defaults = parser.parse_args([args.testset])
        if args.queries == _defaults.queries:
            args.queries = int(os.getenv("TOTAL_QUERIES", args.queries))
        if args.epochs == _defaults.epochs:
            args.epochs = int(os.getenv("AL_EPOCHS", args.epochs))
        if args.rounds == _defaults.rounds:
            args.rounds = int(os.getenv("AL_ROUNDS", args.rounds))
        if args.acquire == _defaults.acquire:
            args.acquire = int(os.getenv("AL_ACQUIRE", args.acquire))

        print(f"[env] Loaded defaults from .env: queries={args.queries}, epochs={args.epochs}, "
              f"rounds={args.rounds}, acquire={args.acquire}")

    strategies = args.strategies
    if args.out:
        results_base = args.out
    else:
        results_base = os.path.join("comparisons", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(results_base, exist_ok=True)

    strategy_data = {}

    for strat in strategies:
        print(f"\n>>> Running Strategy: {strat.upper()}")
        strat_out = os.path.join(results_base, strat)
        
        cmd = [
            sys.executable, "train.py", args.testset,
            "--queries", str(args.queries),
            "--epochs", str(args.epochs),
            "--rounds", str(args.rounds),
            "--acquire", str(args.acquire),
            "--strategy", strat,
            "--out", strat_out,
        ]

        if args.seed:
            cmd.append("--seed")
            cmd.append(str(args.seed))

        if args.sup_queries and strat == "supervised":
            cmd.append("--sup-queries")
            cmd.append(str(args.sup_queries))

        if args.cuda:
            cmd.append("--cuda")
        
        if args.plot_only:
            print("  (Skipping execution due to --plot-only)")
            ret = 0
            # If skipping, we assume directories exist. If not, error handling below catches it.
        else:
            ret = run_command(cmd)
        if ret != 0:
            print(f"Error running strategy {strat}")
            continue

        # Find the timestamped directory created by train.py inside strat_out
        timestamp_dirs = [d for d in os.listdir(strat_out) if os.path.isdir(os.path.join(strat_out, d))]
        if not timestamp_dirs:
            print(f"Could not find result directory for {strat}")
            continue
        
        # Take the latest one
        timestamp_dirs.sort()
        csv_path = os.path.join(strat_out, timestamp_dirs[-1], "learning_data.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            strategy_data[strat] = df
        else:
            print(f"Could not find learning_data.csv for {strat}")

    # Plotting
    if strategy_data:
        # Calculate cost for all strategies
        for strat in strategy_data:
            df = strategy_data[strat]
            if 'cumulative_epochs' in df.columns:
                df = df.sort_values("cumulative_epochs")
                # Calculate epochs step (difference from previous row)
                # First row step is just its cumulative_epochs
                df['epoch_step'] = df['cumulative_epochs'].diff().fillna(df['cumulative_epochs'].iloc[0])
                
                # Cost = labeled_size * epochs_in_this_step
                df['step_cost'] = df['labeled_size'] * df['epoch_step']
                df['cumulative_cost'] = df['step_cost'].cumsum()
                strategy_data[strat] = df

        # Calculate and save efficiency metrics
        calculate_efficiency(strategy_data, results_base)

        # Plot 1: Standard (Samples vs Error)
        plt.figure(figsize=(10, 6))
        
        # Supervised baseline (Final result only)
        supervised_df = strategy_data.get("supervised")
        if supervised_df is not None:
            baseline_val = supervised_df['median_qerror'].iloc[-1]
            plt.axhline(y=baseline_val, color='r', linestyle='--', label='supervised baseline')

        # Active Learning strategies
        for strat, df in strategy_data.items():
            if strat == "supervised":
                continue
            # Logic handled by input args filtering earlier
            plt.plot(df['labeled_size'], df['median_qerror'], marker='o', label=strat)
        
        plt.yscale('log')
        plt.xlabel("Number of Labeled Samples")
        plt.ylabel("Validation Median Q-error (Log Scale)")
        plt.title(f"Strategy Comparison ({args.testset})")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plot_path = os.path.join(results_base, "comparison_plot.png")
        plt.savefig(plot_path)
        print(f"\nFinal comparison plot saved to: {plot_path}")

        # Plot 2: Cost-Efficiency (Total Training Samples vs Error)
        plt.figure(figsize=(12, 8))
        
        # Supervised annotation
        if supervised_df is not None:
            # Supervised Curve: Cost = cumulative_epochs * labeled_size
            # This shows how the baseline improves as it consumes more compute
            if 'cumulative_epochs' in supervised_df.columns:
                sup_cost = supervised_df['cumulative_epochs'] * supervised_df['labeled_size']
                sup_err = supervised_df['median_qerror']
                plt.plot(sup_cost, sup_err, color='red', linestyle='--', label='Supervised Baseline (Learning Curve)')
                
                # Mark the final point
                final_cost = sup_cost.iloc[-1]
                final_err = sup_err.iloc[-1]
                plt.text(final_cost, final_err, f" {int(supervised_df['labeled_size'].iloc[-1])} samples", verticalalignment='bottom', color='red')
            else:
                # Fallback to single point if no cumulative epochs
                last_row = supervised_df.iloc[-1]
                sup_cost = last_row['labeled_size'] * last_row['epochs'] # Approximate if cumulative missing
                sup_err = last_row['median_qerror']
                plt.plot(sup_cost, sup_err, marker='o', markersize=15, color='green', label='Supervised Baseline')
                plt.text(sup_cost, sup_err, f" {int(last_row['labeled_size'])} samples", verticalalignment='bottom')

        for strat, df in strategy_data.items():
            if strat == "supervised":
                continue
                
            if 'cumulative_cost' in df.columns:
                plt.plot(df['cumulative_cost'], df['median_qerror'], marker='o', label=strat)
                
                # Annotation loop
                for i, row in df.iterrows():
                    # Annotate with number of samples used
                    label = f"{int(row['labeled_size'])}"
                    plt.annotate(label, 
                                 (row['cumulative_cost'], row['median_qerror']),
                                 textcoords="offset points", 
                                 xytext=(0, 5), 
                                 ha='center',
                                 fontsize=8)
        
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("Total Training Samples Processed (Cumulative Cost) [Log Scale]")
        plt.ylabel("Validation Median Q-error (Log Scale)")
        plt.title(f"Cost-Efficiency Comparison ({args.testset})\n(Labels indicate number of active learning samples)")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)

        cost_plot_path = os.path.join(results_base, "comparison_plot_cost.png")
        plt.savefig(cost_plot_path)
        print(f"Cost-efficiency plot saved to: {cost_plot_path}")
        
        # Also save a summary CSV
        summary_path = os.path.join(results_base, "comparison_summary.csv")
        # Combine all dataframes with a strategy column
        combined_list = []
        for strat, df in strategy_data.items():
            df['strategy'] = strat
            combined_list.append(df)
        
        if combined_list:
            pd.concat(combined_list).to_csv(summary_path, index=False)
            print(f"Summary data saved to: {summary_path}")

if __name__ == "__main__":
    main()
