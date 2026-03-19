#!/usr/bin/env python3
"""Raymond nested-floor formula for d_m(|S|)."""

from __future__ import annotations

import argparse
import math
from fractions import Fraction


def raymond(s_size: int, m: int, verbose: bool = False) -> int:
    """Return d_m(|S|) with Raymond's nested-floor formula."""
    if m < 2:
        raise ValueError("m must be >= 2")
    if s_size < m:
        raise ValueError("|S| must be >= m")

    value = math.comb(m, 2) - 1
    if verbose:
        print(f"start: d_m({m}) = C({m},2)-1 = {value}")
    verbose_rows: list[tuple[str, str, str, str, str, str]] = []
    # Apply nested floors from (m+1)/(m-1) up to |S|/(|S|-2).
    for d in range(m - 1, s_size - 1):
        # d runs: m-1, m, ..., |S|-2; numerator is d+2.
        prev_input = d + 1
        next_input = d + 2
        before_floor = Fraction((d + 2) * value, d)
        before_exact = (
            str(before_floor.numerator)
            if before_floor.denominator == 1
            else f"{before_floor.numerator}/{before_floor.denominator}"
        )
        next_value = ((d + 2) * value) // d
        delta = next_value - value
        before_float_text = f"{float(before_floor):.1f}"
        next_value_text = f"{float(next_value):.1f}"
        delta_text = f"{float(delta):.1f}"
        lhs = f"step {prev_input}: d_m({prev_input}) * {next_input}/{d}"
        rhs = f"d_m({next_input})"
        if verbose:
            verbose_rows.append(
                (lhs, before_exact, before_float_text, rhs, next_value_text, delta_text)
            )
        value = next_value

    if verbose and verbose_rows:
        lhs_width = max(len(row[0]) for row in verbose_rows)
        exact_width = max(len(row[1]) for row in verbose_rows)
        paren_width = max(len(f"(= {row[2]})") for row in verbose_rows)
        rhs_width = max(len(row[3]) for row in verbose_rows)
        floor_width = max(len(f"floor({row[2]})") for row in verbose_rows)
        next_width = max(len(row[4]) for row in verbose_rows)
        delta_width = max(len(row[5]) for row in verbose_rows)
        for lhs, before_exact, before_float_text, rhs, next_value_text, delta_text in verbose_rows:
            paren = f"(= {before_float_text})"
            floor_part = f"floor({before_float_text})"
            print(
                f"{lhs:<{lhs_width}} = {before_exact:>{exact_width}} "
                f"{paren:>{paren_width}}; {rhs:<{rhs_width}} = "
                f"{floor_part:<{floor_width}} = {next_value_text:>{next_width}}; "
                f"delta = {delta_text:>{delta_width}}"
            )
    return value

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute d_m(|S|) with Raymond's nested-floor formula."
    )
    parser.add_argument("s_size", type=int, help="|S| with |S| >= m")
    parser.add_argument("m", type=int, help="m with m >= 2")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each nested multiplication with before/after floor values.",
    )
    args = parser.parse_args()

    print(raymond(args.s_size, args.m, verbose=args.verbose))


if __name__ == "__main__":
    main()
