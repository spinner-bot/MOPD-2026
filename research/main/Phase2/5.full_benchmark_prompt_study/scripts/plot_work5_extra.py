#!/usr/bin/env python3
"""Additional figures for the Work5 evaluation report (sections 2, 4, 8, 9)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn")
CONDS = ("S0", "SC", "S1", "S2", "S3")
OKABE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})


def register(font):
    if font and Path(font).is_file():
        matplotlib.font_manager.fontManager.addfont(str(font))
        fam = matplotlib.font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams["font.sans-serif"] = [fam, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def heat(ax, data, rows, cols, title, cbar_label, vmin=None, vmax=None, fmt="{:.1f}"):
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, fmt.format(data[i][j]), ha="center", va="center", fontsize=7,
                    color="#1a1a1a")
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--font", type=str, default=None)
    args = ap.parse_args()
    register(args.font)
    d = json.loads(args.analysis.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- §2 κ between discriminators ---
    k = d["discriminators"]["kappa"]
    names = ("Official native / 官方原生", "Terminal diagnostic / 裁定终行", "Answer extraction / 答案抽取")
    order = ("official", "terminal", "extraction")
    m = [[1.0 if a == b else k.get(f"{a}_vs_{b}", k.get(f"{b}_vs_{a}", float("nan")))
          for b in order] for a in order]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(names)
    ax.set_yticks(range(3)); ax.set_yticklabels(names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i][j]:.3f}", ha="center", va="center",
                    color="white" if m[i][j] > 0.6 else "#111", fontsize=9)
    ax.set_title("Figure 8. Pairwise agreement between discriminators (Cohen kappa)", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("κ", fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_dir / "f8_discriminator_kappa.png", bbox_inches="tight")
    plt.close(fig)

    # --- §4 matrix heatmap ---
    mat = d["matrix"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4), sharey=True)
    for ax, src, lab in zip(axes, ("official", "terminal", "extraction"), names):
        data = [[mat[f"{t}|{c}"][src] for c in CONDS] for t in TRACKS]
        heat(ax, data, TRACKS if ax is axes[0] else [""] * len(TRACKS), CONDS,
             f"{lab}", "Accuracy (%) / 准确率 (%)", vmin=0, vmax=100)
    fig.suptitle("Figure 9. Accuracy matrix: benchmarks x conditions under three score sources",
                 y=1.03, fontsize=10.5)
    fig.tight_layout(); fig.savefig(args.out_dir / "f9_accuracy_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # --- §4 trend lines ---
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for i, t in enumerate(TRACKS):
        ys = [mat[f"{t}|{c}"]["terminal"] for c in CONDS]
        ax.plot(range(5), ys, marker="o", ms=4, lw=1.4, color=OKABE[i], label=t)
    ax.set_xticks(range(5)); ax.set_xticklabels(CONDS)
    ax.set_ylabel("Terminal accuracy (%) / 裁定终行准确率 (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=7.5)
    ax.set_title("Figure 10. The six trend lines are nearly parallel: prompt effects point the same way everywhere", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out_dir / "f10_condition_trend.png", bbox_inches="tight")
    plt.close(fig)

    # --- §8 extraction status composition ---
    st = d["extraction_status"]
    keys = ["ANSWERED", "NO_ANSWER", "AMBIGUOUS"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bottom = [0.0] * 5
    for i, kk in enumerate(keys):
        vals = [st.get(c, {}).get(kk, 0) for c in CONDS]
        tot = [sum(st.get(c, {}).values()) for c in CONDS]
        pct = [100 * v / t for v, t in zip(vals, tot)]
        ax.bar(range(5), pct, bottom=bottom, label=kk, color=OKABE[i], width=0.62)
        bottom = [x + y for x, y in zip(bottom, pct)]
    ax.set_xticks(range(5)); ax.set_xticklabels(CONDS)
    ax.set_ylabel("Share of all records in condition (%) / 占比")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    ax.set_title("Figure 11. Extraction-status composition: ANSWERED dominates every condition (>98%)", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out_dir / "f11_status_composition.png", bbox_inches="tight")
    plt.close(fig)

    # --- §9 redundancy kappa ---
    rk = d["redundancy_kappa"]
    m2 = [[1.0 if a == b else rk.get(f"{a}_vs_{b}", rk.get(f"{b}_vs_{a}", float("nan")))
           for b in CONDS] for a in CONDS]
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(m2, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(CONDS)
    ax.set_yticks(range(5)); ax.set_yticklabels(CONDS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{m2[i][j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if m2[i][j] > 0.75 else "#111")
    ax.set_title("Figure 12. Per-item agreement between conditions (kappa)", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("κ", fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_dir / "f12_redundancy_kappa.png", bbox_inches="tight")
    plt.close(fig)

    print("PASS: 5 张补充图已生成")


if __name__ == "__main__":
    main()
