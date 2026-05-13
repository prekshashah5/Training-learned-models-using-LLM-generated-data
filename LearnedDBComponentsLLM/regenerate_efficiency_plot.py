import os
import sys
import csv
import matplotlib.pyplot as plt

# Add current directory to path so we can import from evaluation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluation.pipeline_graphs import plot_labeling_efficiency

def load_csv(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            data.append([float(x) if '.' in x else int(x) for x in row])
    return data

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else 'pipeline_results/2026-04-16_16-17-55'
    
    labeling_times_path = os.path.join(target_dir, 'labeling_times.csv')
    learning_data_path = os.path.join(target_dir, 'learning_data.csv')
    
    if not os.path.exists(labeling_times_path) or not os.path.exists(learning_data_path):
        print(f"Error: Missing CSV files in {target_dir}")
        return
        
    print(f"Loading data from {target_dir}...")
    
    # labeling_times: round, num_queries_labeled, elapsed_seconds
    labeling_times = load_csv(labeling_times_path)
    
    # learning_data: labeled_size, median_qerror, round
    learning_data = load_csv(learning_data_path)
    median_errors = [row[1] for row in learning_data]
    
    total_pool_size = 4500  # Default used in experiments
    strategy = "mc_dropout"

    graphs_dir = os.path.join(target_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    print("Generating updated labeling efficiency plot...")
    plot_labeling_efficiency(labeling_times, median_errors, total_pool_size, strategy, graphs_dir)
    print("Done!")

if __name__ == '__main__':
    main()
