#!/usr/bin/env python3
"""Solve max-sum-y LP (with optional reversed inequalities)."""

import argparse


def print_system(reverse: bool) -> None:
    if reverse:
        print("System (reversed):")
        print("a1 + a4 <= 1 - y1")
        print("a1 + a3 <= 1 - y2")
        print("a1 + a2 <= 1 - y3")
        print("a2 + a4 <= 1 - y4")
        print("a2 + a3 <= 1 - y5")
        print("a3 + a4 <= 1 - y6")
        print("a1 + a2 + a3 + a4 >= 2 + y7/2")
    else:
        print("System:")
        print("a1 + a4 >= 1 + y1")
        print("a1 + a3 >= 1 + y2")
        print("a1 + a2 >= 1 + y3")
        print("a2 + a4 >= 1 + y4")
        print("a2 + a3 >= 1 + y5")
        print("a3 + a4 >= 1 + y6")
        print("a1 + a2 + a3 + a4 <= 2 - y7/2")


def solve_with_pulp(reverse: bool) -> None:
    import pulp as pl

    a = pl.LpVariable.dicts("a", range(1, 5), lowBound=None)
    y = pl.LpVariable.dicts("y", range(1, 8), lowBound=0)

    prob = pl.LpProblem("strictness_check", pl.LpMaximize)
    prob += pl.lpSum(y[i] for i in range(1, 8))

    pairs = [
        (1, 4, 1),
        (1, 3, 2),
        (1, 2, 3),
        (2, 4, 4),
        (2, 3, 5),
        (3, 4, 6),
    ]
    for i, j, k in pairs:
        if reverse:
            # reverse: >= becomes <= and y sign flips
            prob += a[i] + a[j] <= 1 - y[k]
        else:
            prob += a[i] + a[j] >= 1 + y[k]

    if reverse:
        # reverse: <= becomes >= and y sign flips
        prob += (a[1] + a[2] + a[3] + a[4]) >= 2 + y[7] / 2
    else:
        prob += (a[1] + a[2] + a[3] + a[4]) <= 2 - y[7] / 2

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    print("status:", pl.LpStatus[prob.status])
    print("max sum y =", pl.value(prob.objective))
    print("a:", [pl.value(a[i]) for i in range(1, 5)])
    print("y:", [pl.value(y[i]) for i in range(1, 8)])


def analytic_proof(reverse: bool) -> None:
    # Sum the six pair constraints:
    # (a1+a4) + (a1+a3) + (a1+a2) + (a2+a4) + (a2+a3) + (a3+a4)
    # = 3*(a1+a2+a3+a4) >= 6 + (y1+...+y6)
    #
    # So: a1+a2+a3+a4 >= 2 + (y1+...+y6)/3
    #
    # Last constraint with slack: 2*(sum) <= 4 - y7 -> sum <= 2 - y7/2
    #
    # Combine:
    # 2 + (y1+...+y6)/3 <= 2 - y7/2
    # (y1+...+y6)/3 + y7/2 <= 0
    #
    # With y_i >= 0, this forces all y_i = 0.
    if reverse:
        print("Analytic proof for reversed system not implemented.")
        return
    print("Derived bound: (y1+...+y6)/3 + y7/2 <= 0")
    print("With y_i >= 0, maximum sum y is 0.")
    print("Feasible point: a1=a2=a3=a4=0.5, all y_i=0.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check strictness via max-sum-y LP.")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Flip all inequality signs and flip y signs accordingly.",
    )
    args = parser.parse_args()

    print_system(args.reverse)
    try:
        solve_with_pulp(args.reverse)
    except Exception as exc:
        print("PuLP solve failed; falling back to analytic proof.")
        print("Reason:", exc)
        analytic_proof(args.reverse)


if __name__ == "__main__":
    main()
