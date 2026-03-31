"""
analyze_seeds.py  --  NIDS 3.0

Computes mean +/- std across seed runs to establish whether the macro F1
difference between FT-Transformer Direct and FT-Transformer + SupCon is
statistically meaningful.

Usage:
    python analyze_seeds.py --results-dir results/ --prefix seed_

Expected folder naming convention from run_seeds.ps1:
    seed_ftt_direct_s1, seed_ftt_direct_s2, seed_ftt_direct_s3
    seed_ftt_contrastive_s1, seed_ftt_contrastive_s2, seed_ftt_contrastive_s3

Prints a summary table and saves:
    results/seed_analysis.json   -- full stats
    results/seed_variance.png    -- error bar plot
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_group(results_dir, prefix, group_key):
    """Load all summary.json files matching prefix + group_key."""
    results = []
    for entry in sorted(os.listdir(results_dir)):
        if entry.startswith(f"{prefix}{group_key}"):
            path = os.path.join(results_dir, entry, "summary.json")
            if os.path.isfile(path):
                with open(path) as f:
                    results.append((entry, json.load(f)))
    return results


def compute_stats(values):
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "n":    len(arr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--prefix",      default="seed_")
    args = parser.parse_args()

    direct_runs      = load_group(args.results_dir, args.prefix, "ftt_direct")
    contrastive_runs = load_group(args.results_dir, args.prefix, "ftt_contrastive")

    if not direct_runs:
        print("No direct seed runs found. Check --prefix and folder names.")
        return
    if not contrastive_runs:
        print("No contrastive seed runs found.")
        return

    print(f"\nLoaded {len(direct_runs)} direct runs, {len(contrastive_runs)} contrastive runs\n")

    # ── Macro F1 comparison ───────────────────────────────────────────────────
    direct_f1      = [d["macro_f1"] for _, d in direct_runs]
    contrastive_f1 = [d["macro_f1"] for _, d in contrastive_runs]

    direct_stats      = compute_stats(direct_f1)
    contrastive_stats = compute_stats(contrastive_f1)
    delta             = contrastive_stats["mean"] - direct_stats["mean"]

    print("=" * 60)
    print("  MACRO F1 COMPARISON ACROSS SEEDS")
    print("=" * 60)
    print(f"  FT-Transformer Direct:      {direct_stats['mean']:.4f} +/- {direct_stats['std']:.4f}")
    print(f"  FT-Transformer + SupCon:    {contrastive_stats['mean']:.4f} +/- {contrastive_stats['std']:.4f}")
    print(f"  Delta (SupCon - Direct):    {delta:+.4f}")
    print()

    # Check if delta is meaningful (> 1 std)
    pooled_std = np.sqrt(direct_stats["std"]**2 + contrastive_stats["std"]**2)
    if abs(delta) < pooled_std:
        print("  VERDICT: Delta is WITHIN noise (|delta| < pooled std).")
        print("           The difference is not statistically meaningful.")
    else:
        print(f"  VERDICT: Delta exceeds pooled std ({pooled_std:.4f}).")
        sign = "favors contrastive" if delta > 0 else "favors direct"
        print(f"           The difference {sign} and may be real.")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    first_direct = direct_runs[0][1]
    class_names  = list(first_direct["per_class_f1"].keys())

    print("\n" + "=" * 60)
    print("  PER-CLASS F1 MEAN +/- STD")
    print("=" * 60)
    print(f"  {'Class':<20s}  {'Direct':>16s}  {'+ SupCon':>16s}  {'Delta':>8s}")
    print("  " + "-" * 64)

    per_class_stats = {}
    for cls in class_names:
        d_vals = [d["per_class_f1"].get(cls, 0) for _, d in direct_runs]
        c_vals = [d["per_class_f1"].get(cls, 0) for _, d in contrastive_runs]
        d_stats = compute_stats(d_vals)
        c_stats = compute_stats(c_vals)
        delta_cls = c_stats["mean"] - d_stats["mean"]
        per_class_stats[cls] = {
            "direct":      d_stats,
            "contrastive": c_stats,
            "delta":       delta_cls,
        }
        print(f"  {cls:<20s}  {d_stats['mean']:.3f} +/- {d_stats['std']:.3f}  "
              f"{c_stats['mean']:.3f} +/- {c_stats['std']:.3f}  "
              f"{delta_cls:+.3f}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "direct":       {"macro_f1": direct_stats, "per_class": {}},
        "contrastive":  {"macro_f1": contrastive_stats, "per_class": {}},
        "macro_f1_delta": delta,
    }
    for cls in class_names:
        summary["direct"]["per_class"][cls]      = per_class_stats[cls]["direct"]
        summary["contrastive"]["per_class"][cls]  = per_class_stats[cls]["contrastive"]

    out_json = os.path.join(args.results_dir, "seed_analysis.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full stats saved to: {out_json}")

    # ── Error bar plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: macro F1 comparison
    ax = axes[0]
    configs = ["Direct", "+ SupCon"]
    means   = [direct_stats["mean"], contrastive_stats["mean"]]
    stds    = [direct_stats["std"],  contrastive_stats["std"]]
    colors  = ["#2E86C1", "#27AE60"]
    bars    = ax.bar(configs, means, yerr=stds, color=colors,
                     capsize=8, edgecolor="black", linewidth=0.8,
                     error_kw={"linewidth": 2, "ecolor": "#333333"}, width=0.5)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + std + 0.003,
                f"{mean:.4f}\n+/-{std:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(min(means) - 0.05, max(means) + max(stds) + 0.04)
    ax.set_ylabel("Macro F1 Score", fontsize=11)
    ax.set_title("FT-Transformer: Direct vs + SupCon\n(Mean +/- Std across 3 seeds)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: per-class delta with error bars
    ax = axes[1]
    cls_labels = list(per_class_stats.keys())
    deltas_cls = [per_class_stats[c]["delta"] for c in cls_labels]
    colors_cls = ["#27AE60" if d >= 0 else "#E74C3C" for d in deltas_cls]
    bars = ax.bar(cls_labels, deltas_cls, color=colors_cls,
                  edgecolor="black", linewidth=0.6, width=0.6)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticklabels(cls_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1 Delta (SupCon - Direct)", fontsize=10)
    ax.set_title("Per-Class F1 Change from Contrastive Pretraining\n(Mean across 3 seeds)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_fig = os.path.join(args.results_dir, "seed_variance.png")
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Variance plot saved to: {out_fig}")


if __name__ == "__main__":
    main()