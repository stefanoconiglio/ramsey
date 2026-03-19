#!/usr/bin/env python3
"""Turan remainder formula for d_m(|S|)."""

from __future__ import annotations

import argparse
import math


def turan(s_size: int, m: int, verbose: bool = False) -> int:
    """Return d_m(|S|) with Turan's remainder formula."""
    if m < 2:
        raise ValueError("m must be >= 2")
    if s_size < m:
        raise ValueError("|S| must be >= m")

    # Turan's formula with remainder r where n ≡ r (mod m-1):
    # d_m(n) = ((m-2)/(2(m-1))) * (n^2 - r^2) + C(r,2)
    r = s_size % (m - 1)
    return ((m - 2) * (s_size * s_size - r * r)) // (2 * (m - 1)) + math.comb(r, 2)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute d_m(|S|) with Turan's remainder formula."
    )
    parser.add_argument("s_size", type=int, help="|S| with |S| >= m")
    parser.add_argument("m", type=int, help="m with m >= 2")
    args = parser.parse_args()

    print(turan(args.s_size, args.m))


if __name__ == "__main__":
    main()
