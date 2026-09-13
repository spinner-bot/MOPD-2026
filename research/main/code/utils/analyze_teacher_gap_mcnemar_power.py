#!/usr/bin/env python3
"""Exact power audit for the paired-accuracy component of teacher-gap pilots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def binomial_probabilities(n: int, p: float) -> list[float]:
    if n < 0 or not 0 <= p <= 1:
        raise ValueError("invalid binomial parameters")
    if p == 0:
        return [1.0] + [0.0] * n
    if p == 1:
        return [0.0] * n + [1.0]
    values = [math.exp(n * math.log1p(-p))]
    ratio = p / (1 - p)
    for k in range(n):
        values.append(values[-1] * (n - k) / (k + 1) * ratio)
    total = sum(values)
    return [value / total for value in values]


def upper_tail(n: int, p: float, at_least: int) -> float:
    if at_least <= 0:
        return 1.0
    if at_least > n:
        return 0.0
    return sum(binomial_probabilities(n, p)[at_least:])


def rejection_thresholds(max_discordant: int, alpha: float) -> list[int]:
    if max_discordant < 0 or not 0 < alpha < 1:
        raise ValueError("invalid rejection-threshold parameters")
    thresholds = []
    for discordant in range(max_discordant + 1):
        probabilities = binomial_probabilities(discordant, 0.5)
        running = 0.0
        threshold = discordant + 1
        for improved in range(discordant, -1, -1):
            running += probabilities[improved]
            if running <= alpha:
                threshold = improved
            else:
                break
        thresholds.append(threshold)
    return thresholds


def exact_power(
    samples: int,
    discordance_rate: float,
    accuracy_delta: float,
    alpha: float,
) -> float:
    if samples < 1 or not 0 < discordance_rate <= 1:
        raise ValueError("invalid samples or discordance rate")
    if not 0 <= accuracy_delta <= discordance_rate:
        raise ValueError("accuracy delta must be between zero and discordance rate")
    conditional_improvement = (discordance_rate + accuracy_delta) / (2 * discordance_rate)
    discordant_pmf = binomial_probabilities(samples, discordance_rate)
    thresholds = rejection_thresholds(samples, alpha)
    result = 0.0
    for discordant, probability in enumerate(discordant_pmf):
        result += probability * upper_tail(
            discordant,
            conditional_improvement,
            thresholds[discordant],
        )
    return result


def minimum_samples_for_power(
    target_power: float,
    discordance_rate: float,
    accuracy_delta: float,
    alpha: float,
    *,
    start: int,
    maximum: int,
) -> int | None:
    if not 0 < target_power < 1:
        raise ValueError("target power must be between zero and one")
    for samples in range(start, maximum + 1):
        if exact_power(samples, discordance_rate, accuracy_delta, alpha) >= target_power:
            return samples
    return None


def analyze(
    samples: int,
    accuracy_delta: float,
    alpha: float,
    discordance_rates: list[float],
    target_power: float,
    max_samples: int,
) -> dict:
    rows = []
    for discordance in discordance_rates:
        rows.append({
            "discordance_rate": discordance,
            "power_at_locked_n": exact_power(samples, discordance, accuracy_delta, alpha),
            "minimum_n_for_target_power": minimum_samples_for_power(
                target_power,
                discordance,
                accuracy_delta,
                alpha,
                start=samples,
                maximum=max_samples,
            ),
        })
    return {
        "method": "unconditional exact one-sided McNemar power via Binomial(N,d) discordant counts",
        "locked_samples": samples,
        "accuracy_delta": accuracy_delta,
        "one_sided_alpha": alpha,
        "target_power": target_power,
        "rows": rows,
        "scope": "paired binary accuracy only; correct-answer log-probability and Generic-T criteria are not modeled",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--accuracy-delta", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--discordance-rates", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.samples,
        args.accuracy_delta,
        args.alpha,
        args.discordance_rates,
        args.target_power,
        args.max_samples,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
