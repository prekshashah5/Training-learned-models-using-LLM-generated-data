import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.format_converter import parse_sql_to_mscn

EPSILON = 1e-8


def _load_sql_queries(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sqls: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                sqls.append(item)
            elif isinstance(item, dict):
                if "sql" in item and isinstance(item["sql"], str):
                    sqls.append(item["sql"])
                elif all(k in item for k in ("tables", "joins", "predicates")):
                    # If structured without original SQL, skip for KL over SQL-derived features.
                    continue
    return sqls


def _extract_feature_vectors(sqls: List[str]) -> Dict[str, List[int]]:
    features = {
        "tables": [],
        "joins": [],
        "predicates": [],
    }

    for sql in sqls:
        parsed = parse_sql_to_mscn(sql)
        if not parsed:
            continue
        features["tables"].append(len(parsed.get("tables", [])))
        features["joins"].append(len(parsed.get("joins", [])))
        features["predicates"].append(len(parsed.get("predicates", [])))

    return features


def _pmf(values: List[int], bins: List[int]) -> np.ndarray:
    if not bins:
        return np.array([], dtype=np.float64)

    counts = np.zeros(len(bins), dtype=np.float64)
    index = {b: i for i, b in enumerate(bins)}
    for v in values:
        if v in index:
            counts[index[v]] += 1.0

    counts += EPSILON
    counts /= counts.sum()
    return counts


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # D_KL(P || Q)
    return float(np.sum(p * np.log(p / q)))


def _latest_generated_query_file(generated_dir: Path) -> Optional[Path]:
    candidates = sorted(generated_dir.glob("queries_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def build_kl_convergence(
    reference_sqls: List[str],
    generated_sqls: List[str],
    step: int,
) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    ref_features = _extract_feature_vectors(reference_sqls)
    gen_features = _extract_feature_vectors(generated_sqls)

    ref_count = len(ref_features["joins"])
    gen_count = len(gen_features["joins"])

    bins = {
        name: sorted(set(ref_features[name]) | set(gen_features[name]))
        for name in ("tables", "joins", "predicates")
    }

    ref_p = {name: _pmf(ref_features[name], bins[name]) for name in bins}

    rows: List[Dict[str, float]] = []
    checkpoints = list(range(step, gen_count + 1, step))
    if not checkpoints or checkpoints[-1] != gen_count:
        checkpoints.append(gen_count)

    for t in checkpoints:
        r: Dict[str, float] = {"checkpoint": float(t)}
        kl_values = []

        for name in ("tables", "joins", "predicates"):
            q_t = _pmf(gen_features[name][:t], bins[name])
            if ref_p[name].size == 0 or q_t.size == 0:
                kl = 0.0
            else:
                kl = _kl_divergence(ref_p[name], q_t)
            r[f"kl_{name}"] = kl
            kl_values.append(kl)

        r["kl_mean"] = float(np.mean(kl_values)) if kl_values else 0.0
        rows.append(r)

    meta = {
        "reference_queries_parsed": ref_count,
        "generated_queries_parsed": gen_count,
    }
    return rows, meta, ref_features, gen_features


def save_outputs(
    rows: List[Dict[str, float]],
    out_dir: Path,
    prefix,          # unused, kept for API compatibility
    ref_features: Dict[str, List[int]] = None,
    gen_features: Dict[str, List[int]] = None,
    ref_label: str = "Reference (Real)",
    gen_label: str = "LLM-Generated",
    rows2: List[Dict[str, float]] = None,
    gen2_label: str = None,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "kl_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["checkpoint", "kl_tables", "kl_joins", "kl_predicates", "kl_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── KL convergence chart ──────────────────────────────────────────────
    two = rows2 is not None and gen2_label is not None
    x1 = [int(r["checkpoint"]) for r in rows]

    if two:
        x2 = [int(r["checkpoint"]) for r in rows2]
        title_base = f"KL Divergence Comparison: {gen_label} vs {gen2_label}"
        ylabel = "KL Divergence (lower = more similar)"

        # ── Graph 1: Mean KL only (clean comparison) ─────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x1, [r["kl_mean"] for r in rows],  color="#1565C0", linewidth=2.5,
                linestyle="--", marker="o", markersize=5, label=gen_label)
        ax.plot(x2, [r["kl_mean"] for r in rows2], color="#E53935", linewidth=2.5,
                linestyle="--", marker="s", markersize=5, label=gen2_label)
        ax.set_xlabel("Number of Queries Seen", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Mean KL Divergence: {gen_label} vs {gen2_label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        png_path = out_dir / "kl_convergence_mean.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Graph 2: Per-feature breakdown ────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        fig.suptitle(f"Per-Feature KL Divergence: {gen_label} vs {gen2_label}",
                     fontsize=12, fontweight="bold")
        for ax, fname, flabel in zip(
            axes,
            ["kl_tables", "kl_joins", "kl_predicates"],
            ["Tables", "Joins", "Predicates"],
        ):
            ax.plot(x1, [r[fname] for r in rows],  color="#1565C0", linewidth=2,
                    marker="o", markersize=3, label=gen_label)
            ax.plot(x2, [r[fname] for r in rows2], color="#E53935", linewidth=2,
                    marker="s", markersize=3, label=gen2_label)
            ax.set_title(flabel, fontsize=11, fontweight="bold")
            ax.set_xlabel("Queries Seen", fontsize=10)
            ax.set_ylabel("KL Divergence", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        plt.tight_layout()
        feat_path = out_dir / "kl_convergence_features.png"
        fig.savefig(feat_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        kl_paths = [png_path, feat_path]
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x1, [r["kl_tables"]     for r in rows], label="Tables",     linewidth=2, color="#1565C0")
        ax.plot(x1, [r["kl_joins"]      for r in rows], label="Joins",      linewidth=2, color="#0288D1")
        ax.plot(x1, [r["kl_predicates"] for r in rows], label="Predicates", linewidth=2, color="#00897B")
        ax.plot(x1, [r["kl_mean"]       for r in rows], label="Mean",       linewidth=2.5, linestyle="--", color="#1B5E20")
        ax.set_xlabel("Number of Queries Seen", fontsize=11)
        ax.set_ylabel("KL Divergence (lower = closer to reference)", fontsize=11)
        ax.set_title(f"KL Convergence: {gen_label} vs {ref_label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        png_path = out_dir / "kl_convergence.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        kl_paths = [png_path]

    png_path = kl_paths[0]

    # --- Three separate distribution comparison figures ---
    dist_paths = []
    if ref_features is not None and gen_features is not None:
        features = [
            ("tables",     "Number of Tables per Query",     "dist_tables"),
            ("joins",      "Number of Joins per Query",      "dist_joins"),
            ("predicates", "Number of Predicates per Query", "dist_predicates"),
        ]

        for fname, flabel, fsuffix in features:
            ref_vals = ref_features[fname]
            gen_vals = gen_features[fname]
            all_bins = sorted(set(ref_vals) | set(gen_vals))

            ref_counts = np.array([ref_vals.count(b) for b in all_bins], dtype=float)
            gen_counts = np.array([gen_vals.count(b) for b in all_bins], dtype=float)
            ref_pct = ref_counts / ref_counts.sum() * 100
            gen_pct = gen_counts / gen_counts.sum() * 100

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            x2 = np.arange(len(all_bins))
            w = 0.35
            ax2.bar(x2 - w / 2, ref_pct, width=w, label=ref_label, color="#2196F3", alpha=0.85)
            ax2.bar(x2 + w / 2, gen_pct, width=w, label=gen_label, color="#FF9800", alpha=0.85)

            ax2.set_xticks(x2)
            ax2.set_xticklabels([str(b) for b in all_bins], fontsize=10)
            ax2.set_xlabel(flabel, fontsize=11)
            ax2.set_ylabel("Percentage of Queries (%)", fontsize=11)
            ax2.set_title(f"{flabel}: {ref_label} vs {gen_label}", fontsize=12, fontweight="bold")
            ax2.legend(fontsize=10)
            ax2.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            dist_path = out_dir / f"{fsuffix}.png"
            fig2.savefig(dist_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            dist_paths.append(dist_path)

    return csv_path, png_path, dist_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create KL convergence plot for generated SQL workload")
    parser.add_argument("--reference",     type=str, required=True, help="Reference workload JSON (e.g. JOB-light)")
    parser.add_argument("--generated",     type=str, default="",    help="First workload JSON (default: latest in generated_queries/)")
    parser.add_argument("--generated2",    type=str, default="",    help="(Optional) Second workload JSON to compare on same KL chart")
    parser.add_argument("--generated-dir", type=str, default="generated_queries", help="Folder to scan for latest queries")
    parser.add_argument("--step",          type=int, default=250,   help="Checkpoint interval (default: 250)")
    parser.add_argument("--out-dir",       type=str, default="generated_queries", help="Output directory")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    if args.generated:
        gen_path = Path(args.generated)
    else:
        latest = _latest_generated_query_file(Path(args.generated_dir))
        if latest is None:
            raise FileNotFoundError("No generated queries file found.")
        gen_path = latest

    if not gen_path.exists():
        raise FileNotFoundError(f"Generated file not found: {gen_path}")

    ref_sqls = _load_sql_queries(ref_path)
    gen_sqls = _load_sql_queries(gen_path)
    if not ref_sqls:
        raise RuntimeError("Reference file contains no SQL strings.")
    if not gen_sqls:
        raise RuntimeError("Generated file contains no SQL strings.")

    rows, meta, ref_features, gen_features = build_kl_convergence(ref_sqls, gen_sqls, step=max(args.step, 1))

    # Derive human-readable labels from filenames
    def _label(p: Path) -> str:
        if "job_light" in p.stem:  return "JOB-light (Real)"
        if "synthetic" in p.stem:  return "Synthetic"
        if "queries_"  in p.stem:  return "LLM-Generated"
        return p.stem

    ref_label  = _label(ref_path)
    gen_label  = _label(gen_path)

    rows2 = gen2_label = None
    if args.generated2:
        gen2_path = Path(args.generated2)
        if not gen2_path.exists():
            raise FileNotFoundError(f"Second generated file not found: {gen2_path}")
        gen2_sqls = _load_sql_queries(gen2_path)
        if not gen2_sqls:
            raise RuntimeError("Second generated file contains no SQL strings.")
        rows2, meta2, _, _ = build_kl_convergence(ref_sqls, gen2_sqls, step=max(args.step, 1))
        gen2_label = _label(gen2_path)
        print(f"Second workload:  {gen2_path}  ({meta2['generated_queries_parsed']} queries)")

    csv_path, png_path, dist_paths = save_outputs(
        rows, Path(args.out_dir), None, ref_features, gen_features, ref_label, gen_label,
        rows2, gen2_label,
    )

    print(f"Reference:        {ref_path}  ({meta['reference_queries_parsed']} queries)")
    print(f"Generated (1):    {gen_path}  ({meta['generated_queries_parsed']} queries)")
    print(f"Saved KL plot:    {png_path}")
    kl_feat = Path(args.out_dir) / "kl_convergence_features.png"
    if kl_feat.exists():
        print(f"Saved KL feat:    {kl_feat}")
    print(f"Saved CSV:        {csv_path}")
    for p in dist_paths:
        print(f"Saved dist plot:  {p}")


if __name__ == "__main__":
    main()
