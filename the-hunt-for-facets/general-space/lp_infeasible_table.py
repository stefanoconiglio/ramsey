#!/usr/bin/env python3
"""Scan (|S|, m) and print where the multiplier LP is infeasible."""

from __future__ import annotations

import argparse
import sys

from lp_multipliers_generic import solve_multiplier_lp


def diagonal_pairs(
    s_min: int, s_max: int, m_min: int, m_max: int
) -> list[tuple[int, int]]:
    """Enumerate (|S|, m) by Cantor-like diagonals with alternating direction."""
    order: list[tuple[int, int]] = []
    forward = True
    for diag_sum in range(s_min + m_min, s_max + m_max + 1):
        pairs: list[tuple[int, int]] = []
        m_lo = max(m_min, diag_sum - s_max)
        m_hi = min(m_max, diag_sum - s_min, diag_sum // 2)
        for m in range(m_lo, m_hi + 1):
            s_size = diag_sum - m
            if s_min <= s_size <= s_max and s_size >= m:
                pairs.append((s_size, m))
        if not pairs:
            continue
        if not forward:
            pairs.reverse()
        order.extend(pairs)
        forward = not forward
    return order


def build_grid_lines(
    statuses: dict[tuple[int, int], str],
    s_min: int,
    s_max: int,
    m_min: int,
    m_max: int,
) -> list[str]:
    lines = ["INFEASIBILITY GRID (X = Infeasible, ✓ = Feasible, ? = pending)"]
    header = ["m\\|S|"] + [str(s) for s in range(s_min, s_max + 1)]
    lines.append("\t".join(header))
    for m in range(m_min, m_max + 1):
        row = [str(m)]
        for s_size in range(s_min, s_max + 1):
            if s_size < m:
                row.append("-")
                continue
            status = statuses.get((s_size, m))
            if status is None:
                row.append("?")
            elif status == "Infeasible":
                row.append("X")
            else:
                row.append("✓")
        lines.append("\t".join(row))
    return lines


def render(lines: list[str], previous_lines: int, live_update: bool) -> int:
    if live_update:
        if previous_lines > 0:
            sys.stdout.write(f"\x1b[{previous_lines}A\x1b[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        return len(lines)

    print("\n".join(lines))
    print()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table of infeasible (|S|, m) pairs for |S| >= m >= 3.",
        usage="%(prog)s S_max [--m-max M_MAX] [--strict] [--verbose]",
    )
    parser.add_argument("S_max", type=int, help="Maximum |S| to scan (>= 3)")
    parser.add_argument(
        "--m-max",
        type=int,
        default=None,
        help="Maximum m to scan (default: S_max)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Scan strict variant (same as --strict in lp_multipliers_generic.py).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print solver output for each LP solve.",
    )
    args = parser.parse_args()

    s_min = 3
    m_min = 3
    s_max = args.S_max
    m_max = args.m_max if args.m_max is not None else s_max

    if s_max < s_min:
        raise ValueError("S_max must be >= 3")
    if m_max < m_min:
        raise ValueError("m-max must be >= 3")
    if m_max > s_max:
        raise ValueError("m-max must be <= S_max")

    order = diagonal_pairs(s_min, s_max, m_min, m_max)
    if not order:
        print("No valid (|S|, m) pairs to solve.")
        return

    # In-place refresh works cleanly only on a terminal and with quiet solves.
    live_update = sys.stdout.isatty() and not args.verbose

    statuses: dict[tuple[int, int], str] = {}
    show_license_warning = True
    previous_lines = 0

    for idx, (s_size, m) in enumerate(order, start=1):
        solving_line = f"Solving for |S| = {s_size} and m = {m} ... ({idx}/{len(order)})"
        previous_lines = render(
            [solving_line] + build_grid_lines(statuses, s_min, s_max, m_min, m_max),
            previous_lines,
            live_update,
        )

        result = solve_multiplier_lp(
            s_size,
            m,
            strict=args.strict,
            output_lp=False,
            verbose=args.verbose,
            show_license_warning=show_license_warning,
        )
        if result["used_license_fallback"]:
            show_license_warning = False

        statuses[(s_size, m)] = result["status"]
        label = "Infeasible" if result["status"] == "Infeasible" else "Feasible"
        solved_line = (
            f"Solved for |S| = {s_size} and m = {m}: {label} "
            f"({idx}/{len(order)})"
        )
        previous_lines = render(
            [solved_line] + build_grid_lines(statuses, s_min, s_max, m_min, m_max),
            previous_lines,
            live_update,
        )


if __name__ == "__main__":
    main()
