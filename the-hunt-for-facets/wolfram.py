#!/usr/bin/env python3
"""Wolfram/MathWorld floor formula for d_m(|S|)."""

from __future__ import annotations

import argparse
import math


def wolfram(s_size: int, m: int, verbose: bool = False) -> int:
    """Return the Wolfram floor-form approximation for d_m(|S|)."""
    if m < 2:
        raise ValueError("m must be >= 2")
    if s_size < m:
        raise ValueError("|S| must be >= m")

    # MathWorld uses t(n,k) for K_{k+1}-free graphs:
    # t(n,k) = floor((k-1)n^2/(2k)); set k = m-1.
    return math.floor((m - 2) * s_size * s_size / (2 * (m - 1)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute d_m(|S|) with the Wolfram floor formula."
    )
    parser.add_argument("s_size", type=int, help="|S| with |S| >= m")
    parser.add_argument("m", type=int, help="m with m >= 2")
    args = parser.parse_args()

    print(wolfram(args.s_size, args.m))


if __name__ == "__main__":
    main()
