"""
compare_experiments.py  --  NIDS 3.0

Generate cross-experiment comparison visuals from existing summary.json files.
No reruns needed — reads results already saved to disk.

Usage:
    python compare_experiments.py --results-dir results/

    (Use --prefix exp to load ALL experiments including exp20-exp22)

Generates:
    results/comparison_ablation_bar.png    - grouped bar: Direct vs SupCon per backbone
    results/comparison_delta_bar.png       - contrastive delta per backbone
    results/comparison_radar.png           - per-class F1 radar (ablation set only)
    results/comparison_heatmap.png         - experiment x class heatmap (ablation set only)
    results/comparison_summary.csv         - combined results table
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import csv

PALETTE = {
    "fttransformer": "#2E86C1",
    "bilstm":        "#27AE60",
    "cnn_bilstm_se": "#E74C3C",
    "cnn":           "#F39C12",
}
BACKBONE_ORDER  = ["fttransformer", "bilstm", "cnn_bilstm_se", "cnn"]
BACKBONE_LABELS = {
    "fttransformer": "FT-Transformer",
    "bilstm":        "BiLSTM",
    "cnn_bilstm_se": "CNN+BiLSTM+SE",
    "cnn":           "CNN",
}
DPI = 200


# =============================================================================
# Data Loading
# =============================================================================

def load_experiments(results_dir, prefix="exp"):
    experiments = {}
    for entry in sorted(os.listdir(results_dir)):
        if prefix and not entry.startswith(prefix):
            continue
        summary_path = os.path.join(results_dir, entry, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                data = json.load(f)
            experiments[entry] = data
            backbone  = data.get("backbone", "?")
            con_ep    = data.get("config", {}).get("contrastive_epochs", 0)
            strategy  = "direct" if con_ep == 0 else "+ SupCon"
            print(f"  Loaded: {entry:<35s} {backbone:<15s} {strategy:<10s} "
                  f"macro_f1={data['macro_f1']:.4f}")
    return experiments


def group_by_backbone(experiments):
    """
    Returns dict: backbone -> {"direct": best_data, "contrastive": best_data}
    Picks best macro F1 for each (backbone, strategy) pair across all loaded runs.
    """
    groups = {}
    for name, data in experiments.items():
        backbone = data.get("backbone", "unknown")
        con_ep   = data.get("config", {}).get("contrastive_epochs", 0)
        strategy = "direct" if con_ep == 0 else "contrastive"

        if backbone not in groups:
            groups[backbone] = {}
        if strategy not in groups[backbone]:
            groups[backbone][strategy] = data
        else:
            if data["macro_f1"] > groups[backbone][strategy]["macro_f1"]:
                groups[backbone][strategy] = data
    return groups


def build_ablation_set(groups):
    """
    Returns a clean flat dict of label -> data using only backbones that have
    BOTH direct and contrastive runs. Excludes tuning-only or unpaired runs.
    Ordered by BACKBONE_ORDER for consistency across all plots.
    """
    clean = {}
    for backbone in BACKBONE_ORDER:
        if backbone not in groups:
            continue
        g = groups[backbone]
        if "direct" in g:
            label = f"{BACKBONE_LABELS[backbone]} Direct"
            clean[label] = g["direct"]
        if "contrastive" in g:
            label = f"{BACKBONE_LABELS[backbone]} + SupCon"
            clean[label] = g["contrastive"]
    return clean


# =============================================================================
# Plot 1: Grouped Bar — Direct vs + SupCon per Backbone
# =============================================================================

def plot_ablation_bar(groups, output_path):
    """
    Grouped bar chart. Skips missing strategies gracefully (no 0.0000 bars).
    """
    # Use fixed backbone order
    backbones = [b for b in BACKBONE_ORDER if b in groups]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")

    n          = len(backbones)
    bar_w      = 0.32
    group_gap  = 0.9       # gap between backbone groups
    x_centers  = np.arange(n) * group_gap

    direct_handles     = []
    contrastive_handles = []

    for i, backbone in enumerate(backbones):
        color = PALETTE.get(backbone, "#888888")
        label = BACKBONE_LABELS.get(backbone, backbone)
        g     = groups[backbone]

        direct_f1 = g.get("direct", {}).get("macro_f1", None)
        con_f1    = g.get("contrastive", {}).get("macro_f1", None)

        xc = x_centers[i]

        if direct_f1 is not None and con_f1 is not None:
            x_direct = xc - bar_w / 2
            x_con    = xc + bar_w / 2
        elif direct_f1 is not None:
            x_direct = xc
            x_con    = None
        else:
            x_direct = None
            x_con    = xc

        if x_direct is not None:
            b = ax.bar(x_direct, direct_f1, width=bar_w,
                       color=color, alpha=0.45,
                       edgecolor="black", linewidth=0.7)
            ax.text(x_direct, direct_f1 + 0.004, f"{direct_f1:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#222222")
            if not direct_handles:
                direct_handles = [mpatches.Patch(
                    facecolor="#888888", alpha=0.45,
                    edgecolor="black", label="Direct Training")]

        if x_con is not None:
            b = ax.bar(x_con, con_f1, width=bar_w,
                       color=color, alpha=1.0,
                       edgecolor="black", linewidth=0.7)
            ax.text(x_con, con_f1 + 0.004, f"{con_f1:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#222222")
            if not contrastive_handles:
                contrastive_handles = [mpatches.Patch(
                    facecolor="#888888", alpha=1.0,
                    edgecolor="black", label="+ Supervised Contrastive")]

    ax.set_xticks(x_centers)
    ax.set_xticklabels([BACKBONE_LABELS.get(b, b) for b in backbones],
                       fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_ylim(0.40, 0.75)
    ax.set_title("Ablation Study: Direct Training vs Supervised Contrastive Learning\n"
                 "per Backbone Architecture", fontsize=13, fontweight="bold", pad=12)
    ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_handles = direct_handles + contrastive_handles
    if all_handles:
        ax.legend(handles=all_handles, fontsize=10, loc="lower right",
                  framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Ablation bar chart saved: {output_path}")


# =============================================================================
# Plot 2: Delta Bar — Contrastive Gain per Backbone
# =============================================================================

def plot_delta_bar(groups, output_path):
    """
    Horizontal bar chart of macro F1 delta per backbone.
    Only includes backbones with BOTH strategies. Labels placed outside bars
    with no overlap.
    """
    paired = [(b, groups[b]) for b in BACKBONE_ORDER
              if b in groups
              and "direct" in groups[b]
              and "contrastive" in groups[b]]

    if not paired:
        print("  Delta bar: no paired backbones found, skipping.")
        return

    labels = [BACKBONE_LABELS.get(b, b) for b, _ in paired]
    deltas = [g["contrastive"]["macro_f1"] - g["direct"]["macro_f1"]
              for _, g in paired]
    colors = ["#27AE60" if d >= 0 else "#E74C3C" for d in deltas]

    fig, ax = plt.subplots(figsize=(11, 1.4 * len(labels) + 2))
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, deltas, color=colors,
                    edgecolor="black", linewidth=0.7, height=0.5)

    # Place value labels outside the bar, away from y-axis labels
    x_max = max(abs(d) for d in deltas) * 1.0
    for bar, delta in zip(bars, deltas):
        sign  = "+" if delta >= 0 else ""
        # Offset text well outside the bar end
        offset = x_max * 0.08
        x_txt  = delta + offset if delta >= 0 else delta - offset
        ha     = "left" if delta >= 0 else "right"
        ax.text(x_txt,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{delta:.4f}",
                ha=ha, va="center", fontsize=11, fontweight="bold",
                color="#27AE60" if delta >= 0 else "#E74C3C")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Macro F1 Change (+ SupCon minus Direct)", fontsize=11)
    ax.set_title("Effect of Supervised Contrastive Pretraining per Backbone\n"
                 "(Positive = SupCon helps, Negative = SupCon hurts)",
                 fontsize=13, fontweight="bold", pad=12)

    # Extend x-axis to make room for labels
    x_range = max(abs(d) for d in deltas)
    ax.set_xlim(-(x_range + x_range * 0.35), x_range + x_range * 0.35)

    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Delta bar chart saved: {output_path}")


# =============================================================================
# Plot 3: Radar Chart  (ablation set only)
# =============================================================================

def plot_radar_chart(ablation_set, output_path):
    """
    Radar chart using only the clean ablation set (8 experiments, one per cell).
    Ordered consistently: FT-Transformer Direct, FT-Transformer + SupCon, etc.
    """
    if not ablation_set:
        return

    first       = next(iter(ablation_set.values()))
    class_names = list(first["per_class_f1"].keys())
    n_classes   = len(class_names)
    angles      = np.linspace(0, 2 * np.pi, n_classes, endpoint=False).tolist()
    angles     += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    # Two line styles: solid for + SupCon, dashed for Direct
    style_map = {
        "fttransformer": ("#2E86C1", "o"),
        "bilstm":        ("#27AE60", "s"),
        "cnn_bilstm_se": ("#E74C3C", "D"),
        "cnn":           ("#F39C12", "^"),
    }

    sorted_entries = sorted(ablation_set.items(),
                            key=lambda x: x[1]["macro_f1"], reverse=True)

    for label, data in sorted_entries:
        backbone = data.get("backbone", "fttransformer")
        con_ep   = data.get("config", {}).get("contrastive_epochs", 0)
        is_direct = con_ep == 0
        color, marker = style_map.get(backbone, ("#888888", "o"))
        linestyle = "--" if is_direct else "-"
        alpha     = 0.7 if is_direct else 1.0

        f1_values  = [data["per_class_f1"].get(c, 0) for c in class_names]
        f1_values += f1_values[:1]
        legend_label = f"{label} (F1={data['macro_f1']:.3f})"

        ax.plot(angles, f1_values, linestyle=linestyle, color=color,
                linewidth=2.2, marker=marker, markersize=5,
                label=legend_label, alpha=alpha)
        ax.fill(angles, f1_values, alpha=0.04, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_title("Per-Class F1 Score — Ablation Study\n(solid = + SupCon, dashed = Direct)",
                 fontsize=14, fontweight="bold", pad=40)
    ax.legend(loc="upper right", bbox_to_anchor=(1.50, 1.18),
              fontsize=9, framealpha=0.95, edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Radar chart saved: {output_path}")


# =============================================================================
# Plot 4: Heatmap  (ablation set only)
# =============================================================================

def plot_comparison_heatmap(ablation_set, output_path):
    """
    Heatmap using only the clean ablation set. Sorted by macro F1 descending.
    """
    if not ablation_set:
        return

    first       = next(iter(ablation_set.values()))
    class_names = list(first["per_class_f1"].keys())

    sorted_entries = sorted(ablation_set.items(),
                            key=lambda x: x[1]["macro_f1"], reverse=True)

    exp_labels = [label for label, _ in sorted_entries]
    matrix     = [[data["per_class_f1"].get(c, 0) for c in class_names]
                  for _, data in sorted_entries]
    macro_f1s  = [data["macro_f1"] for _, data in sorted_entries]

    matrix_ext     = np.column_stack([np.array(matrix), macro_f1s])
    col_labels_ext = class_names + ["Macro F1"]

    fig, ax = plt.subplots(figsize=(16, max(5, len(sorted_entries) * 0.95 + 2)))
    sns.heatmap(
        matrix_ext, annot=True, fmt=".3f", cmap="RdYlGn",
        xticklabels=col_labels_ext, yticklabels=exp_labels,
        vmin=0, vmax=1, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "F1 Score", "shrink": 0.8},
        annot_kws={"size": 9},
        ax=ax
    )
    ax.set_title("Ablation Study: Per-Class F1 Scores (sorted by Macro F1)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Attack Class", fontsize=11)
    ax.set_ylabel("Configuration", fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved: {output_path}")


# =============================================================================
# CSV Export
# =============================================================================

def save_comparison_csv(ablation_set, output_path):
    if not ablation_set:
        return
    first       = next(iter(ablation_set.values()))
    class_names = list(first["per_class_f1"].keys())

    sorted_entries = sorted(ablation_set.items(),
                            key=lambda x: x[1]["macro_f1"], reverse=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Configuration", "Backbone", "Strategy", "Accuracy", "Macro F1"]
            + class_names + ["Time (min)"]
        )
        for label, data in sorted_entries:
            backbone = data.get("backbone", "")
            con_ep   = data.get("config", {}).get("contrastive_epochs", 0)
            strategy = "Direct" if con_ep == 0 else "+ SupCon"
            row = [
                label,
                BACKBONE_LABELS.get(backbone, backbone),
                strategy,
                f"{data['accuracy']:.4f}",
                f"{data['macro_f1']:.4f}",
            ] + [f"{data['per_class_f1'].get(c, 0):.4f}" for c in class_names]
            time_min = data.get("training_time_seconds", 0) / 60
            row.append(f"{time_min:.1f}")
            writer.writerow(row)

    print(f"  CSV saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--prefix", type=str, default="exp",
                        help="Load folders starting with this prefix (default: 'exp'). "
                             "Must include ALL ablation experiments (exp15-exp22).")
    args = parser.parse_args()

    print(f"\nLoading experiments from: {args.results_dir}/")
    experiments = load_experiments(args.results_dir, prefix=args.prefix)

    if len(experiments) < 2:
        print("Need at least 2 experiments.")
        return

    groups       = group_by_backbone(experiments)
    ablation_set = build_ablation_set(groups)

    print(f"\nFound {len(experiments)} total experiments, "
          f"{len(ablation_set)} in clean ablation set.\n")

    plot_ablation_bar(
        groups,
        os.path.join(args.results_dir, "comparison_ablation_bar.png")
    )
    plot_delta_bar(
        groups,
        os.path.join(args.results_dir, "comparison_delta_bar.png")
    )
    plot_radar_chart(
        ablation_set,
        os.path.join(args.results_dir, "comparison_radar.png")
    )
    plot_comparison_heatmap(
        ablation_set,
        os.path.join(args.results_dir, "comparison_heatmap.png")
    )
    save_comparison_csv(
        ablation_set,
        os.path.join(args.results_dir, "comparison_summary.csv")
    )

    print("\nAll comparison visuals generated!")


if __name__ == "__main__":
    main()
    