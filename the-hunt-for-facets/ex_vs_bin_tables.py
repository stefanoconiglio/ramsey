#!/usr/bin/env python3
"""Print LaTeX tables for configurable |S| and q ranges."""

from __future__ import annotations

import argparse
import math

from raymond import raymond


def _tabular_spec(num_columns: int) -> str:
    return "c|" + ("c" * (num_columns - 1))


def _print_latex_table(rows: list[list[str]], caption: str, label: str) -> None:
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{" + _tabular_spec(len(rows[0])) + r"}")
    print(r"\hline")
    for i, row in enumerate(rows):
        print(" & ".join(row) + r" \\")
        if i == 0:
            print(r"\hline")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{" + caption + r"}")
    print(r"\label{" + label + r"}")
    print(r"\end{table}")


def d_q_table_rows(s_min: int, s_max: int, q_min: int, q_max: int) -> list[list[str]]:
    rows: list[list[str]] = []
    header = [r"$q \backslash |S|$"] + [str(s_size) for s_size in range(s_min, s_max + 1)]
    rows.append(header)
    for q in range(q_min, q_max + 1):
        row = [str(q)]
        for s_size in range(s_min, s_max + 1):
            row.append("-" if s_size < q else str(raymond(s_size, q)))
        rows.append(row)
    return rows


def diagonal_rows(s_min: int, s_max: int, q_min: int, q_max: int) -> list[list[str]]:
    rows: list[list[str]] = []
    header = [r"$q = |S|$"] + [str(s_size) for s_size in range(s_min, s_max + 1)]
    rows.append(header)
    for q in range(q_min, q_max + 1):
        row = [str(q)]
        for s_size in range(s_min, s_max + 1):
            row.append(str(math.comb(s_size, 2) - 1) if s_size == q else "")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print LaTeX tables for d_q(|S|) and diagonal binomial values."
    )
    parser.add_argument("s_min", type=int, help="Minimum |S|")
    parser.add_argument("s_max", type=int, help="Maximum |S|")
    parser.add_argument("q_min", type=int, help="Minimum q")
    parser.add_argument("q_max", type=int, help="Maximum q")
    args = parser.parse_args()

    if args.s_min > args.s_max:
        raise ValueError("s-min must be <= s-max")
    if args.q_min > args.q_max:
        raise ValueError("q-min must be <= q-max")
    if args.q_min < 2:
        raise ValueError("q-min must be >= 2")

    _print_latex_table(
        d_q_table_rows(args.s_min, args.s_max, args.q_min, args.q_max),
        r"Values of $d_q(|S|)$ with $q$ as rows and $|S|$ as columns.",
        "tab:dm",
    )
    print()
    _print_latex_table(
        diagonal_rows(args.s_min, args.s_max, args.q_min, args.q_max),
        r"Diagonal values $\binom{|S|}{2} - 1$.",
        "tab:bin",
    )


if __name__ == "__main__":
    main()
