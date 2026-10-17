#!/usr/bin/env python3
"""Figures for the reference-tier human review analysis (T2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDS = ("S0", "SC", "S1", "S2", "S3")
NAMES = {"official": "Official\n官方原生",
         "terminal": "Terminal\n裁定终行",
         "extraction": "Extraction\n答案抽取"}
OKABE = ("#D55E00", "#0072B2", "#009E73")
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
})


def register(font):
    if font and Path(font).is_file():
        matplotlib.font_manager.fontManager.addfont(str(font))
        fam = matplotlib.font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams["font.sans-serif"] = [fam, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--font", type=str, default=None)
    args = ap.parse_args()
    register(args.font)
    d = json.loads(args.analysis.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    s = d["scorer_vs_human"]
    keys = ("official", "terminal", "extraction")

    # F13 — agreement & kappa
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = range(3)
    ax.bar([i - 0.18 for i in x], [s[k]["agree_pct"] for k in keys], 0.34,
           label="Agreement with human (%) / 与人工一致率", color=OKABE[1])
    ax.bar([i + 0.18 for i in x], [s[k]["kappa"] * 100 for k in keys], 0.34,
           label="Cohen kappa x100 / κ×100", color=OKABE[2])
    for i, k in enumerate(keys):
        ax.annotate(f"{s[k]['agree_pct']:.1f}%", (i - 0.18, s[k]["agree_pct"]),
                    ha="center", va="bottom", fontsize=7.5)
        ax.annotate(f"{s[k]['kappa']:.3f}", (i + 0.18, s[k]["kappa"] * 100),
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(list(x)); ax.set_xticklabels([NAMES[k] for k in keys])
    ax.set_ylim(0, 118); ax.set_ylabel("%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    ax.set_title("Figure 13 [reference]. Agreement between each score source and the human review",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(args.out_dir / "f13_human_agreement.png", bbox_inches="tight")
    plt.close(fig)

    # F14 — by condition
    bc = d["by_condition"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    w = 0.2
    xs = range(len(CONDS))
    ax.bar([i - 1.5 * w for i in xs], [bc[c]["human_yes"] for c in CONDS], w, label="Human / 人工", color="#444444")
    for j, k in enumerate(keys):
        ax.bar([i + (j - 0.5) * w for i in xs], [bc[c][k + "_yes"] for c in CONDS], w, label=NAMES[k], color=OKABE[j])
    ax.set_xticks(list(xs)); ax.set_xticklabels(CONDS)
    ax.set_ylabel("Correct records / 判对条数"); ax.set_xlim(-0.6, 4.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    ax.set_title("Figure 14 [reference]. Correct counts by condition: human vs three sources", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out_dir / "f14_by_condition.png", bbox_inches="tight")
    plt.close(fig)

    # F15 — disagreement direction
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    a = [s[k]["human_yes_auto_no"] for k in keys]
    b = [s[k]["human_no_auto_yes"] for k in keys]
    ax.barh([i + 0.18 for i in range(3)], a, 0.34, label="Human correct / auto wrong / 人工对·自动错", color=OKABE[1])
    ax.barh([i - 0.18 for i in range(3)], b, 0.34, label="Human wrong / auto correct / 人工错·自动对", color=OKABE[0])
    for i, k in enumerate(keys):
        ax.annotate(str(a[i]), (a[i], i + 0.18), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7.5)
        ax.annotate(str(b[i]), (b[i], i - 0.18), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7.5)
    ax.set_yticks(range(3)); ax.set_yticklabels([NAMES[k] for k in keys])
    ax.set_xlabel("Count / 条数")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.set_title("Figure 15 [reference]. Direction of disagreement", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out_dir / "f15_disagreement_direction.png", bbox_inches="tight")
    plt.close(fig)

    print("PASS: 3 张参考级图已生成")


if __name__ == "__main__":
    main()
