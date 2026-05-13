#!/usr/bin/env python3
"""
Script to compare performance metrics across multiple LLM models.

This script loads query data from multiple model runs and generates
comparison plots for various metrics including:
- Q-error distribution
- Execution time
- Validity rates
- Complexity distributions
- Generation time/throughput
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from utils.session_utils import load_all_model_runs, aggregate_model_runs_metadata
from plotting import (
    plot_q_error_comparison_models,
    plot_execution_time_models,
    plot_metric_statistics_models,
    plot_validity_rate_models,
    plot_complexity_distribution_models,
    plot_generation_time_models
)

def main():
    load_dotenv()
    
    # Get output folder from environment or use default
    default_output = Path(os.getenv("OUTPUT_FOLDER", "../output"))
    
    if Path("output").exists() and Path("output").is_dir():
        output_folder = Path("output")
    else:
        output_folder = default_output
        
    if not output_folder.exists():
        print(f"[error] Output folder does not exist: {output_folder}")
        return
    
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    if session_id:
        # Check if the user passed a direct path like "output/session_X"
        if Path(session_id).exists() and Path(session_id).is_dir():
            target_session = Path(session_id)
        else:
            target_session = output_folder / session_id
            
        if not target_session.exists() or not target_session.is_dir():
            print(f"[error] Target session folder does not exist: {target_session}")
            return
        latest_session = target_session
    else:
        # Find the most recent session folder
        session_dirs = [d for d in output_folder.iterdir() if d.is_dir() and d.name.startswith("session_")]
        if not session_dirs:
            print(f"[error] No session folders found in {output_folder}")
            return
            
        latest_session = sorted(session_dirs, key=lambda d: d.name)[-1]
        
    print(f"[info] Loading data from session: {latest_session}")
    
    # Load all model runs from the specific session
    model_data, model_runs_metadata = load_all_model_runs(latest_session)
    
    if not model_data:
        print("[error] No model run data found in session. Make sure you have run queries with different models.")
        return
    
    # Print summary
    print(f"\n[info] Found {len(model_data)} model(s):")
    for model_name, queries in model_data.items():
        print(f"  - {model_name}: {len(queries)} queries")
    
    # Create comparison output directory inside the session folder
    comparison_output_dir = latest_session / "model_comparison"
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[info] Generating comparison plots in: {comparison_output_dir}")
    
    # Generate all comparison plots
    try:
        # Q-error comparison
        print("[info] Plotting Q-error comparison...")
        plot_q_error_comparison_models(model_data, comparison_output_dir)
        
        # Execution time comparison
        print("[info] Plotting execution time comparison...")
        plot_execution_time_models(model_data, comparison_output_dir)
        
        # Q-error statistics
        print("[info] Plotting Q-error statistics...")
        plot_metric_statistics_models(model_data, "q_error", comparison_output_dir, 
                                     metric_label="Q-error")
        
        # Execution time statistics
        print("[info] Plotting execution time statistics...")
        plot_metric_statistics_models(model_data, "exec_time_ms", comparison_output_dir,
                                     metric_label="Execution Time (ms)")
        
        # Validity rate
        print("[info] Plotting validity rates...")
        plot_validity_rate_models(model_data, comparison_output_dir)
        
        # Complexity distribution (if available)
        has_complexity = any(
            any(q.get("ComplexityScore") is not None for q in queries)
            for queries in model_data.values()
        )
        if has_complexity:
            print("[info] Plotting complexity distribution...")
            plot_complexity_distribution_models(model_data, comparison_output_dir)
        else:
            print("[info] Skipping complexity distribution (no complexity data found)")
        
        # Generation time/throughput
        aggregated_metadata = aggregate_model_runs_metadata(model_runs_metadata)
        if aggregated_metadata and any(m.get("generation_time_s", 0) > 0 for m in aggregated_metadata.values()):
            print("[info] Plotting generation time/throughput...")
            plot_generation_time_models(aggregated_metadata, comparison_output_dir)
        else:
            print("[info] Skipping generation time plots (no generation time data found in runs.xlsx)")
        
        print(f"\n[done] All comparison plots saved to: {comparison_output_dir}")
        
    except Exception as e:
        print(f"[error] Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
