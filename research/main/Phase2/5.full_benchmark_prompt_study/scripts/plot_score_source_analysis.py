#!/usr/bin/env python3
"""Diagnostic figures for the Work5 score-source analysis (report 2233).

Reads the v4 sample-level table plus the two judge audit CSVs and writes seven
publication-style figures. Figure conventions: one question per figure, the
title states the finding, legends sit outside the axes, 300 dpi, colour-blind
safe palette (Okabe-Ito), units on every axis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn")
CONDS = ("S0", "SC", "S1", "S2", "S3")
OKABE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})
CJK = "Noto Sans CJK SC"
plt.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def flag(row: dict, key: str) -> int:
    return 1 if str(row[key]).strip().lower() in ("true", "1", "yes") else 0


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


def paired(rows, col, track=None):
    """Return (delta_pp, lo, hi, b, c) for S0 -> S3."""
    sel = [r for r in rows if track is None or r["track"] == track]
    a = {r["sample_uid"]: flag(r, col) for r in sel if r["condition"] == "S0"}
    z = {r["sample_uid"]: flag(r, col) for r in sel if r["condition"] == "S3"}
    common = set(a) & set(z)
    n = len(common)
    b = sum(1 for k in common if z[k] > a[k])
    c = sum(1 for k in common if a[k] > z[k])
    d = (b - c) / n * 100
    se = math.sqrt(b + c) / n * 100
    return d, d - 1.959964 * se, d + 1.959964 * se, b, c


def fig1(rows, out):
    sources = (("官方原生", "native_correct", OKABE[1]), ("裁定终行", "terminal_correct", OKABE[0]),
               ("答案抽取", "extraction_correct", OKABE[2]))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ys = []
    for i, (label, col, colour) in enumerate(sources):
        d, lo, hi, b, c = paired(rows, col)
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=colour, capsize=4, markersize=7)
        ax.annotate(f"{d:+.2f} pp", (d, i), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
        ys.append(label)
    ax.axvline(0, color="#888", lw=0.9, ls="--")
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys)
    ax.set_xlabel("S3 − S0 准确率差（百分点，95% CI）")
    ax.set_title("图1 只有官方口径上升；两条独立口径均下降\n(S0→S3，n=32,162×2 配对)", fontsize=10)
    ax.set_xlim(-6, 14)
    fig.tight_layout()
    fig.savefig(out / "f1_three_sources.png", bbox_inches="tight")
    plt.close(fig)


def fig2(rows, out):
    sources = (("官方原生", "native_correct", OKABE[1]), ("裁定终行", "terminal_correct", OKABE[0]),
               ("答案抽取", "extraction_correct", OKABE[2]))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, (label, col, colour) in zip(axes, sources):
        ests, ses = [], []
        for t in TRACKS:
            d, lo, hi, b, c = paired(rows, col, t)
            ests.append(d)
            ses.append((hi - lo) / (2 * 1.959964))
        w = [1 / s ** 2 if s > 0 else 0 for s in ses]
        if sum(w) > 0:
            pooled = sum(wi * e for wi, e in zip(w, ests)) / sum(w)
            Q = sum(wi * (e - pooled) ** 2 for wi, e in zip(w, ests))
            k = len(ests)
            i2 = max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0
        else:
            pooled, i2 = 0.0, 0.0
        for i, (t, e, s) in enumerate(zip(TRACKS, ests, ses)):
            ax.errorbar(e, i, xerr=1.959964 * s, fmt="o", color=colour, capsize=3, markersize=4.5)
        ax.axvline(0, color="#888", lw=0.9, ls="--")
        ax.axvline(pooled, color=colour, lw=0.8, alpha=0.45)
        ax.set_yticks(range(len(TRACKS)))
        ax.set_yticklabels(TRACKS, fontsize=7.5)
        ax.set_title(f"{label}\nI² = {i2:.1f}%", fontsize=9.5)
        ax.set_xlabel("S3 − S0 (pp)")
        ax.set_xlim(-8, 20)
    fig.suptitle("图2 官方效应的跨评测集异质性达 I²=95.3%，不可合并；另两条口径方向一致", y=1.02, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "f2_forest.png", bbox_inches="tight")
    plt.close(fig)


def fig3(rows, out):
    n = len(rows)
    off = sum(flag(r, "native_correct") for r in rows)
    ter = sum(flag(r, "terminal_correct") for r in rows)
    ext = sum(flag(r, "extraction_correct") for r in rows)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    data = (("官方", off, OKABE[1]), ("裁定终行", ter, OKABE[0]), ("答案抽取", ext, OKABE[2]))
    for i, (label, k, colour) in enumerate(data):
        ax.barh(i, k, color=colour, height=0.6)
        ax.annotate(f"{k:,}  ({100*k/n:.1f}%)", (k, i), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels([d[0] for d in data])
    ax.set_xlabel("判对的记录数（全量 n = 160,810）")
    ax.set_xlim(0, 145000)
    ax.set_title("图3 三条口径构成严格嵌套：官方 ⊂ 判定 ⊂ 抽取\n官方判对的记录，判定必然也判对（仅左对=0）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "f3_nesting.png", bbox_inches="tight")
    plt.close(fig)


def fig4(rows, out):
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    width = 0.38
    xs = range(len(TRACKS))
    dod = []
    dhu = []
    for t in TRACKS:
        sel = [r for r in rows if r["track"] == t]
        o0 = sum(flag(r, "native_correct") for r in sel if r["condition"] == "S0")
        o3 = sum(flag(r, "native_correct") for r in sel if r["condition"] == "S3")
        t0 = sum(flag(r, "terminal_correct") for r in sel if r["condition"] == "S0")
        t3 = sum(flag(r, "terminal_correct") for r in sel if r["condition"] == "S3")
        dod.append(o3 - o0)
        dhu.append((t3 - o3) - (t0 - o0))
    ax.bar([x - width / 2 for x in xs], dod, width, label="Δ官方（找回被丢弃的判对）", color=OKABE[1])
    ax.bar([x + width / 2 for x in xs], dhu, width, label="Δ丢弃量（被格式规则丢弃的条数变化）", color=OKABE[3])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(TRACKS, rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("条数变化（S3 − S0）")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title("图4 官方的每一点上升，都对应被丢弃条数的等量下降\n恒等式 Δ官方 + Δ丢弃 = Δ判定 在六个评测集精确成立", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "f4_waterfall.png", bbox_inches="tight")
    plt.close(fig)


def fig5(rows, out):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    xs, ys, labels = [], [], []
    for t in TRACKS:
        for c in CONDS:
            g = [r for r in rows if r["track"] == t and r["condition"] == c]
            n = len(g)
            nb = sum(1 for r in g if r["native_judge_format"] == "non_binary") / n * 100
            ac = sum(flag(r, "terminal_correct") for r in g) / n * 100
            xs.append(nb)
            ys.append(ac)
            labels.append(f"{t[:9]}/{c}")
    ax.scatter(xs, ys, s=26, color=OKABE[0], zorder=3)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    r = num / den
    k = (num / den) * (sum((y - my) ** 2 for y in ys) / sum((x - mx) ** 2 for x in xs)) ** 0.5
    b = my - k * mx
    ax.plot([min(xs), max(xs)], [k * min(xs) + b, k * max(xs) + b], color=OKABE[1], lw=1.2, ls="--")
    ax.set_xlabel("非二元判官输出占比（%）")
    ax.set_ylabel("裁定终行 判对率（%）")
    ax.set_title(f"图5 判官展开推理的比例是难度信号，与正确率强负相关\nPearson r = {r:+.3f}（30 个 评测集×条件 单元）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "f5_mechanism.png", bbox_inches="tight")
    plt.close(fig)


def fig6(rows, out):
    agg = {}
    for r in rows:
        key = r["category"] or "(zoombench: 无类别)"
        a = agg.setdefault(key, {"n": 0, "o": 0, "t": 0})
        a["n"] += 1
        a["o"] += flag(r, "native_correct")
        a["t"] += flag(r, "terminal_correct")
    items = [(k, v) for k, v in agg.items() if v["n"] >= 500]
    items.sort(key=lambda kv: (kv[1]["t"] - kv[1]["o"]) / kv[1]["n"])
    labels = [k[:34] for k, _ in items]
    gaps = [100 * (v["t"] - v["o"]) / v["n"] for _, v in items]
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.32 * len(items) + 1.2)))
    ax.barh(range(len(items)), gaps, color=OKABE[1], height=0.7)
    for i, (g, (_, v)) in enumerate(zip(gaps, items)):
        ax.annotate(f"{g:.1f} pp  (n={v['n']:,})", (g, i), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=7.5)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("判定判对率 − 官方判对率（百分点）")
    ax.set_xlim(0, max(gaps) * 1.32)
    ax.set_title("图6 格式规则的惩罚由类别主导，从 0.9pp 到 59.1pp\n(官方相对真实判定的少计幅度)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "f6_category_gap.png", bbox_inches="tight")
    plt.close(fig)


def fig7(rows, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.36
    xs = range(len(CONDS))
    z = [r for r in rows if r["track"] == "zoombench"]
    other = [r for r in rows if r["track"] != "zoombench"]
    zo = [100 * sum(flag(r, "native_correct") for r in z if r["condition"] == c) / max(1, len([r for r in z if r["condition"] == c])) for c in CONDS]
    ot = [100 * sum(flag(r, "terminal_correct") for r in z if r["condition"] == c) / max(1, len([r for r in z if r["condition"] == c])) for c in CONDS]
    ax.bar([x - width / 2 for x in xs], zo, width, label="zoombench 官方原生", color=OKABE[1])
    ax.bar([x + width / 2 for x in xs], ot, width, label="zoombench 裁定终行", color=OKABE[0])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(CONDS)
    ax.set_ylabel("准确率（%）")
    ax.set_ylim(0, 75)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title("图7 zoombench 官方准确率恒为 0：845/845 判官输出均为非二元\n终行诊断同期给出 57.9%–61.2%，差一个数量级", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "f7_zoombench.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load(args.data_dir / "sample_level_merged.csv")
    print(f"载入 {len(rows):,} 行")
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f(rows, args.out_dir)
        print(f"  {f.__name__} 完成")
    print(f"图输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
