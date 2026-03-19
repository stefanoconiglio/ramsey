#!/usr/bin/env python3
"""Print a |S'| x |S| table of heuristic-minus-d_m for fixed m."""

from __future__ import annotations

import argparse
from fractions import Fraction

from raymond import raymond
from ex_edge_scaled import edge_scaled_heuristic


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _format_value(value: Fraction, use_fraction: bool) -> str:
    if use_fraction:
        return _format_fraction(value)
    return f"{float(value):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Table of (|E(S')|/|E(S)|)*d_m(|S|) - d_m(|S'|) for fixed m."
        ),
        usage=(
            "%(prog)s m max_size [--frac]\n"
            "       %(prog)s m max_size --unitdiff [--frac]\n"
            "       %(prog)s m --minimal [--frac]\n"
            "       %(prog)s m_min m_max S_prime_min S_prime_max --listdiff [--unitdiff] [--frac] [--m M] [--nofacets]"
        ),
    )
    parser.add_argument(
        "params",
        nargs="+",
        type=int,
        help="Positional parameters depend on mode; see usage.",
    )
    parser.add_argument(
        "--frac",
        action="store_true",
        help="Print exact fractions (default: decimal with 1 digit).",
    )
    parser.add_argument(
        "--unitdiff",
        action="store_true",
        help="Restrict to |S'|=|S|+1; in table mode prints m rows vs |S| columns.",
    )
    parser.add_argument(
        "--listdiff",
        action="store_true",
        help="List triples (m,|S|,|S'|) for |S|>=m and |S'|>=|S|+1.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Only with --listdiff: force m to this fixed value.",
    )
    parser.add_argument(
        "--nofacets",
        action="store_true",
        help="Only with --listdiff: skip rows where facet? is yes.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Only without --listdiff: force |S'| = |S| = m (single-cell table).",
    )
    args = parser.parse_args()

    dm_cache: dict[tuple[int, int], int] = {}

    def d_m_cached(s_prime_val: int, m_val: int) -> int:
        key = (s_prime_val, m_val)
        cached = dm_cache.get(key)
        if cached is None:
            cached = raymond(s_prime_val, m_val)
            dm_cache[key] = cached
        return cached

    if args.listdiff:
        if args.minimal:
            parser.error("--minimal can be used only without --listdiff")
        if len(args.params) != 4:
            parser.error(
                "With --listdiff, provide four positional parameters: "
                "m_min m_max S_prime_min S_prime_max"
            )
        m_min = args.params[0]
        m_max = args.params[1]
        s_prime_min = args.params[2]
        s_prime_max = args.params[3]
        if m_min < 2:
            raise ValueError("m_min must be >= 2")
        if m_max < m_min:
            raise ValueError("m_max must be >= m_min")
        if s_prime_min < 2:
            raise ValueError("S_prime_min must be >= 2")
        if s_prime_max < s_prime_min:
            raise ValueError("S_prime_max must be >= S_prime_min")

        if args.m is None:
            pass
        else:
            if args.m < 2:
                raise ValueError("--m must be >= 2")
            if args.m < m_min or args.m > m_max:
                raise ValueError("--m must satisfy m_min <= --m <= m_max")
            m_min = args.m
            m_max = args.m
            if s_prime_max < args.m + 1:
                raise ValueError("With --listdiff --m M, S_prime_max must be >= M+1")
    else:
        if args.m is not None:
            parser.error("--m can be used only with --listdiff")
        if args.nofacets:
            parser.error("--nofacets can be used only with --listdiff")
        if args.minimal:
            if args.unitdiff:
                parser.error("--minimal cannot be combined with --unitdiff")
            if len(args.params) != 1:
                parser.error("With --minimal, provide one positional parameter: m")
            m = args.params[0]
            if m < 2:
                raise ValueError("m must be >= 2")
            s_prime_min = m
            s_prime_max = m
            s_min = m
            s_max = m
        else:
            if len(args.params) != 2:
                parser.error(
                    "Without --listdiff, provide two positional parameters: m max_size"
                )
            m = args.params[0]
            s_prime_max = args.params[1]
            if m < 2:
                raise ValueError("m must be >= 2")
            if s_prime_max < m:
                raise ValueError("max_size must be >= m")
            s_prime_min = m
            s_min = m
            s_max = s_prime_max

    if args.listdiff:
        print("LIST: (|E(S')|/|E(S)|)*d_m(|S|) - d_m(|S'|)")
        rows: list[list[str]] = []
        rows.append(
            [
                "m",
                "|S|",
                "|S'|",
                "unit",
                "facet?",
                "|E(S')|/|E(S)| (int?)",
                "(|E(S')|/|E(S)|)*d_m(|S|)",
                "d_m(|S'|)",
                "diff",
                "win",
            ]
        )

        # Upside-down order: |S'| from max down, then m from max down, then |S| down.
        for s_prime in range(s_prime_max, s_prime_min - 1, -1):
            for m_row in range(min(m_max, s_prime - 1), m_min - 1, -1):
                dm_s_prime = d_m_cached(s_prime, m_row)
                multiple = "yes" if s_prime % (m_row - 1) != 0 else "no"
                if args.nofacets and multiple == "yes":
                    continue
                if args.unitdiff:
                    s_values = [s_prime - 1]
                else:
                    s_values = range(s_prime - 1, m_row - 1, -1)
                for s in s_values:
                    if s < m_row:
                        continue
                    heuristic = edge_scaled_heuristic(s_prime, s, m_row)
                    ratio = Fraction(s_prime * (s_prime - 1), s * (s - 1))
                    diff = heuristic - Fraction(dm_s_prime, 1)
                    unit = "unit" if s_prime - s == 1 else ""
                    diff_label = "new" if diff > 0 else "dom"
                    win = "win" if multiple == "no" and diff_label == "new" else ""
                    rows.append(
                        [
                            str(m_row),
                            str(s),
                            str(s_prime),
                            unit,
                            multiple,
                            (
                                f"{_format_value(ratio, args.frac)} = int"
                                if ratio.denominator == 1
                                else f"{_format_value(ratio, args.frac)} = frc"
                            ),
                            _format_value(heuristic, args.frac),
                            str(dm_s_prime),
                            f"{_format_value(diff, args.frac)} {diff_label}",
                            win,
                        ]
                    )

        col_count = len(rows[0])
        col_widths = []
        for col in range(col_count):
            width = max(len(row[col]) for row in rows)
            col_widths.append(width)
        for row in rows:
            print("  ".join(cell.rjust(col_widths[idx]) for idx, cell in enumerate(row)))
        return

    table_rows: list[list[str]] = []
    if args.unitdiff:
        if s_prime_max < m + 1:
            raise ValueError("For --unitdiff, max_size must be >= m+1")
        print("TABLE: (|E(S+1)|/|E(S)|)*d_m(|S|) - d_m(|S|+1)")
        s_columns = list(range(m, s_prime_max))
        header = ["m\\|S|"] + [str(s) for s in s_columns]
        table_rows.append(header)

        for m_row in range(m, s_prime_max):
            row = [str(m_row)]
            for s in s_columns:
                if s < m_row:
                    row.append("-")
                    continue
                s_prime_val = s + 1
                heuristic = edge_scaled_heuristic(s_prime_val, s, m_row)
                diff = heuristic - Fraction(d_m_cached(s_prime_val, m_row), 1)
                row.append(_format_value(diff, args.frac))
            table_rows.append(row)
    else:
        print("TABLE: (|E(S')|/|E(S)|)*d_m(|S|) - d_m(|S'|)")
        header = ["|S'|\\|S|"] + [str(s) for s in range(s_min, s_max + 1)]
        table_rows.append(header)

        for s_prime in range(s_prime_min, s_prime_max + 1):
            dm_s_prime = d_m_cached(s_prime, m)
            row = [str(s_prime)]
            for s in range(s_min, s_max + 1):
                if s > s_prime or (s == s_prime and not args.minimal):
                    row.append("-")
                    continue
                heuristic = edge_scaled_heuristic(s_prime, s, m)
                diff = heuristic - Fraction(dm_s_prime, 1)
                row.append(_format_value(diff, args.frac))
            table_rows.append(row)

    col_count = len(table_rows[0])
    col_widths = []
    for col in range(col_count):
        width = max(len(row[col]) for row in table_rows)
        col_widths.append(width)
    col_widths[0] += 2

    for row in table_rows:
        print("  ".join(cell.rjust(col_widths[idx]) for idx, cell in enumerate(row)))


if __name__ == "__main__":
    main()
