#!/usr/bin/env python3
"""Compare d_m(|S|) from three formulas on a range."""

from __future__ import annotations

import argparse

from raymond import raymond
from turan import turan
from wolfram import wolfram


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Loop over m and |S| ranges, compute d_m(|S|) with raymond/turan/wolfram, "
            "and report where they differ."
        ),
        usage="%(prog)s m_min m_max s_min s_max",
    )
    parser.add_argument("m_min", type=int, help="Minimum m (inclusive, m >= 2).")
    parser.add_argument("m_max", type=int, help="Maximum m (inclusive, m >= m_min).")
    parser.add_argument(
        "s_min",
        type=int,
        help="Minimum |S| tested (inclusive; actual loop uses max(s_min,m)).",
    )
    parser.add_argument(
        "s_max",
        type=int,
        help="Maximum |S| tested (inclusive, |S| >= s_min).",
    )
    args = parser.parse_args()

    if args.m_min < 2:
        raise ValueError("m_min must be >= 2")
    if args.m_max < args.m_min:
        raise ValueError("m_max must be >= m_min")
    if args.s_min < 2:
        raise ValueError("s_min must be >= 2")
    if args.s_max < args.s_min:
        raise ValueError("s_max must be >= s_min")

    checked = 0
    mismatches = 0

    print("m  |S|  raymond  turan  wolfram  d_turan  d_wolfram")
    print("-- --- -------- ------- -------- --------- ----------")
    for m in range(args.m_min, args.m_max + 1):
        s_start = max(args.s_min, m)
        for s_size in range(s_start, args.s_max + 1):
            val_raymond = raymond(s_size, m)
            val_turan = turan(s_size, m)
            val_wolfram = wolfram(s_size, m)
            checked += 1
            if val_raymond != val_turan or val_raymond != val_wolfram:
                mismatches += 1
                delta_turan = val_raymond - val_turan
                delta_wolfram = val_raymond - val_wolfram
                print(
                    f"{m:2d} {s_size:3d} {val_raymond:8d} {val_turan:7d} "
                    f"{val_wolfram:8d} {delta_turan:9d} {delta_wolfram:10d}"
                )

    print()
    print(f"Checked pairs: {checked}")
    print(f"Mismatches:    {mismatches}")
    if mismatches == 0:
        print("All tested pairs match.")


if __name__ == "__main__":
    main()
