"""
update_thesis_graphs.py
Run the pipeline and copy generated graphs to Thesis/graphs/.

Usage:
    python update_thesis_graphs.py                           # defaults
    python update_thesis_graphs.py --total-queries 1000      # more queries
    python update_thesis_graphs.py --strategy uncertainty     # different strategy
    python update_thesis_graphs.py --rounds 5 --acquire 100  # more rounds

All arguments are passed directly to the pipeline.
"""

import subprocess
import sys
import os
import glob
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
THESIS_GRAPHS = os.path.join(ROOT, "Thesis", "graphs")

# ── Default pipeline arguments (override via CLI) ───────────────────────
DEFAULTS = {
    "--synthetic": None,        # Use synthetic generation (no Ollama needed)
    "--total-queries": "500",
    "--rounds": "3",
    "--acquire": "50",
    "--epochs": "5",
    "--strategy": "mc_dropout",
    "--seed": "42",
    "--out": "pipeline_results",
    "--db-timeout": "30000",
}


def build_args(user_args):
    """Merge defaults with user-provided arguments."""
    merged = dict(DEFAULTS)

    i = 0
    while i < len(user_args):
        key = user_args[i]
        if key.startswith("--"):
            # Check if next arg is a value or another flag
            if i + 1 < len(user_args) and not user_args[i + 1].startswith("--"):
                merged[key] = user_args[i + 1]
                i += 2
            else:
                merged[key] = None  # flag with no value
                i += 1
        else:
            i += 1

    # Build command list
    cmd = []
    for key, val in merged.items():
        cmd.append(key)
        if val is not None:
            cmd.append(val)
    return cmd


def find_latest_graphs_dir(out_dir):
    """Find the most recent pipeline run's graphs/ directory."""
    if not os.path.exists(out_dir):
        return None

    runs = sorted(
        [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))],
        reverse=True,
    )
    for run in runs:
        graphs = os.path.join(out_dir, run, "graphs")
        if os.path.exists(graphs):
            return graphs
    return None


def main():
    user_args = sys.argv[1:]
    pipeline_args = build_args(user_args)

    # Determine output dir
    out_dir = DEFAULTS["--out"]
    for i, arg in enumerate(user_args):
        if arg == "--out" and i + 1 < len(user_args):
            out_dir = user_args[i + 1]

    print("=" * 60)
    print("  UPDATE THESIS GRAPHS")
    print("=" * 60)
    print(f"  Pipeline args: {' '.join(pipeline_args)}")
    print(f"  Output dir:    {out_dir}")
    print(f"  Thesis graphs: {THESIS_GRAPHS}")
    print("=" * 60)

    # ── Run pipeline ────────────────────────────────────────────────────
    cmd = [sys.executable, "-m", "training.pipeline"] + pipeline_args
    print(f"\n>>> Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=ROOT)

    if result.returncode != 0:
        print(f"\n✗ Pipeline failed with exit code {result.returncode}")
        sys.exit(1)

    # ── Copy graphs ─────────────────────────────────────────────────────
    graphs_dir = find_latest_graphs_dir(os.path.join(ROOT, out_dir))

    if not graphs_dir:
        print("\n✗ No graphs directory found in pipeline output!")
        sys.exit(1)

    print(f"\n>>> Copying graphs from: {graphs_dir}")
    os.makedirs(THESIS_GRAPHS, exist_ok=True)

    copied = 0
    for png in glob.glob(os.path.join(graphs_dir, "*.png")):
        dest = os.path.join(THESIS_GRAPHS, os.path.basename(png))
        shutil.copy2(png, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f"  ✓ {os.path.basename(png):45s} ({size_kb:.1f} KB)")
        copied += 1

    print(f"\n{'=' * 60}")
    print(f"  Done! Copied {copied} graphs to Thesis/graphs/")
    print(f"  Recompile thesis (2x) to update figures.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
