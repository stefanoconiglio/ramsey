#!/usr/bin/env python3
"""CLI utilities around d_m(|S|) values from Raymond's formula."""

from __future__ import annotations

import argparse
import math
from fractions import Fraction
from typing import Iterable, Tuple

from raymond import raymond


def raymond_formula(s_size: int, m: int) -> int:
    """Return d_m(|S|) computed with the raymond formula."""
    return raymond(s_size, m)


def scan_pairs(max_s: int) -> Iterable[Tuple[int, int]]:
    """Yield (|S|, m) pairs with 2 <= m <= |S| <= max_s."""
    for s_size in range(2, max_s + 1):
        for m in range(2, s_size + 1):
            yield s_size, m


def inequality_holds(s_size: int, m: int) -> bool:
    """Check d_m(|S|+1) < |S|(|S|-1)/2 - 1."""
    lhs = raymond_formula(s_size + 1, m)
    rhs = (s_size * (s_size - 1)) // 2 - 1
    return lhs < rhs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute/plot values based on |S|, m, and a, with optional scanning."
    )
    parser.add_argument("S", type=int, help="|S| >= m and |S| >= 2")
    parser.add_argument("m", type=int, help="m >= 2")
    parser.add_argument("a", type=int, help="a >= 0")
    parser.add_argument(
        "--scan-max",
        type=int,
        default=None,
        metavar="N",
        help="If set, scan all pairs 2 <= m <= |S| <= N and print those where d_m(|S|+1) < |S|(|S|-1)/2 - 1.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Compute values but do not display a plot.",
    )
    args = parser.parse_args()

    if args.m < 2:
        raise ValueError("m must be >= 2")
    if args.S < args.m:
        raise ValueError("|S| must be >= m")
    if args.a < 0:
        raise ValueError("a must be >= 0")

    print(f"|S| = {args.S}")
    print(f"m = {args.m}")
    print(f"a = {args.a}")

    if args.scan_max is not None:
        if args.scan_max < 2:
            raise ValueError("scan-max must be >= 2")
        for s_size, m in scan_pairs(args.scan_max):
            if inequality_holds(s_size, m):
                print(f"|S|={s_size}, m={m}")
        return

    d_m_val_same = raymond_formula(args.S, args.m)
    max_lhs = (args.S * (args.S - 1)) // 2
    max_lhs_prime = ((args.S + args.a) * (args.S + args.a - 1)) // 2
    d_m_val_a = raymond_formula(args.S + args.a, args.m)
    base_step = Fraction((args.m + 1) * (math.comb(args.m, 2) - 1), args.m - 1)
    base_step_val = float(base_step)
    rhs_base = max_lhs - 1
    rhs_base_prime = max_lhs_prime - 1
    print(f"maxLHS = |S|(|S|-1)/2 = {max_lhs}")
    print(f"RHSbase = |S|(|S|-1)/2 - 1 = {rhs_base}")
    print(f"RHSd_m = d_m(|S|) = {d_m_val_same}")
    print(f"base-step = ((m+1)/(m-1))*(C(m,2)-1) = {base_step}")
    print(f"maxLHS' = (|S|+a)(|S|+a-1)/2 = {max_lhs_prime}")
    print(f"RHSbase' = (|S|+a)(|S|+a-1)/2 - 1 = {rhs_base_prime}")
    print(f"RHSd_m' = d_m(|S|+a) = {d_m_val_a}")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting; install it or use --no-plot") from exc

    labels = [
        "maxLHS\n|S|(|S|-1)/2",
        "RHSbase\n|S|(|S|-1)/2 - 1",
        "RHSd_m\nd_m(|S|)",
        "base-step\n((m+1)/(m-1))*(C(m,2)-1)",
        "maxLHS'\n(|S|+a)(|S|+a-1)/2",
        "RHSbase'\n(|S|+a)(|S|+a-1)/2 - 1",
        "RHSd_m'\nd_m(|S|+a)",
    ]
    values = [
        max_lhs,
        rhs_base,
        d_m_val_same,
        base_step_val,
        max_lhs_prime,
        rhs_base_prime,
        d_m_val_a,
    ]

    fig, ax = plt.subplots()
    ax.bar(
        labels,
        values,
        color=["#54A24B", "#F58518", "#72B7B2", "#B279A2", "#A0CBE8", "#E45756", "#4C78A8"],
    )
    ax.set_title(f"|S|={args.S}, m={args.m}, a={args.a}")
    ax.set_ylabel("Value")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
