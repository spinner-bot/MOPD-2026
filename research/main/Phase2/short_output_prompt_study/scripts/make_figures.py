#!/usr/bin/env python3
"""Create publication-style summary figures from summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = data["summary"]
    conditions = [r["condition"] for r in rows]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    args.output.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("accuracy", "Answer accuracy", "Accuracy", "accuracy.png"),
        ("strict_format_rate", "Strict output-format compliance", "Rate", "format_compliance.png"),
        ("reasoning_marker_rate", "Lexical reasoning-marker rate", "Rate", "reasoning_markers.png"),
        ("response_chars_mean", "Visible response length", "Characters", "response_length.png"),
    ]
    for key, title, ylabel, filename in metrics:
        values = np.array([r[key] for r in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
        bars = ax.bar(conditions, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if key.endswith("rate") or key == "accuracy":
            ax.set_ylim(0, 1.05)
            ax.set_yticks(np.linspace(0, 1, 6))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (0.02 if value < 0.95 else 0.005), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output / filename, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    for row, color in zip(rows, colors):
        lo, hi = row["response_chars_ci95"]
        ax.errorbar(row["condition"], row["response_chars_mean"], yerr=[[row["response_chars_mean"] - lo], [hi - row["response_chars_mean"]]], fmt="o", color=color, capsize=4, label=row["condition"])
    ax.set_title("Visible response length with bootstrap 95% CI")
    ax.set_ylabel("Characters")
    fig.tight_layout()
    fig.savefig(args.output / "response_length_ci.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
