#!/usr/bin/env python3
"""Solve a subset-Turan LP and print comparison values.

LP solved (for input |S'|, |S|, m with |S'| >= |S| >= m):

max   sum_{e in E(S')} b_e
s.t.  sum_{e in E(T)} b_e <= d_m(|S|)   for all T subset S', |T| = |S|
      b_e free (no lower bound)
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from fractions import Fraction
from typing import Any

from raymond import raymond


def edge_scaled_heuristic(s_prime: int, s: int, m: int) -> Fraction:
    """Return (|E(S')|/|E(S)|) * d_m(|S|) as an exact fraction."""
    if s_prime < s:
        raise ValueError("Expected |S'| >= |S|")
    if s < m:
        raise ValueError("Expected |S| >= m")
    e_s_prime = s_prime * (s_prime - 1) // 2
    e_s = s * (s - 1) // 2
    return Fraction(e_s_prime * raymond(s, m), e_s)


def _create_model(gp: Any) -> tuple[Any, bool]:
    try:
        return gp.Model("edge_scaled_lp"), False
    except gp.GurobiError:
        original_home = os.environ.get("HOME")
        try:
            # Fall back to Gurobi's bundled restricted license.
            os.environ["HOME"] = "/tmp"
            return gp.Model("edge_scaled_lp"), True
        finally:
            if original_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = original_home


def solve_subset_turan_lp(
    s_prime: int,
    s: int,
    m: int,
    *,
    verbose: bool = False,
    output_lp: bool = False,
    lp_path: str = "edge_scaled_model.lp",
    show_license_warning: bool = True,
) -> dict[str, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gurobipy is required. Install it with: python -m pip install gurobipy"
        ) from exc

    nodes = list(range(1, s_prime + 1))
    edges = list(itertools.combinations(nodes, 2))

    model, used_license_fallback = _create_model(gp)
    if used_license_fallback and show_license_warning:
        print(
            "Warning: local Gurobi license unusable; using bundled restricted license.",
            file=sys.stderr,
        )
    model.Params.OutputFlag = 1 if verbose else 0

    # b_e are free variables in the primal LP.
    b = {
        e: model.addVar(lb=-GRB.INFINITY, name=f"b_{e[0]}_{e[1]}")
        for e in edges
    }
    model.setObjective(gp.quicksum(b[e] for e in edges), GRB.MAXIMIZE)

    rhs = raymond(s, m)
    for idx, subset in enumerate(itertools.combinations(nodes, s), start=1):
        lhs = gp.quicksum(b[e] for e in itertools.combinations(subset, 2))
        model.addConstr(lhs <= rhs, name=f"subset_{idx}")

    if output_lp:
        model.write(lp_path)

    model.optimize()

    status_map = {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.UNBOUNDED: "Unbounded",
        GRB.INF_OR_UNBD: "InfeasibleOrUnbounded",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.INTERRUPTED: "Interrupted",
    }
    status = status_map.get(model.Status, f"StatusCode({model.Status})")
    omega = model.ObjVal if model.SolCount > 0 else None
    return {
        "status": status,
        "omega": omega,
        "lp_path": lp_path if output_lp else None,
        "used_license_fallback": used_license_fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the subset-Turan LP for (|S'|,|S|,m) and print d_m(|S'|) "
            "and |E(S')|/|E(S)|*d_m(|S|)."
        ),
        usage="%(prog)s S_prime S m [--verbose] [--lp] [--tol TOL]",
    )
    parser.add_argument("S_prime", type=int, help="|S'|")
    parser.add_argument("S", type=int, help="|S|")
    parser.add_argument("m", type=int, help="m")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print solver output.",
    )
    parser.add_argument(
        "--lp",
        action="store_true",
        help="Write LP model to file and print it to screen.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for numeric comparisons (default: 1e-6).",
    )
    args = parser.parse_args()

    s_prime = args.S_prime
    s = args.S
    m = args.m

    if m < 2:
        raise ValueError("m must be >= 2")
    if s_prime < 2 or s < 2:
        raise ValueError("|S'| and |S| must be >= 2")
    if s_prime < s:
        raise ValueError("Expected |S'| >= |S|")
    if s_prime < m:
        raise ValueError("Expected |S'| >= m")
    if s < m:
        raise ValueError("Expected |S| >= m")
    if args.tol < 0:
        raise ValueError("--tol must be >= 0")

    result = solve_subset_turan_lp(
        s_prime,
        s,
        m,
        verbose=args.verbose,
        output_lp=args.lp,
        show_license_warning=True,
    )

    if args.lp and result["lp_path"] is not None:
        print("LP model:")
        with open(result["lp_path"], "r", encoding="utf-8") as f:
            print(f.read())

    dm_s_prime = raymond(s_prime, m)
    scaled = edge_scaled_heuristic(s_prime, s, m)
    scaled_float = float(scaled)

    print(f"|E(S')|/|E(S)| * d_m(|S|) (heuristic) = {scaled} ({scaled_float:.10f})")
    if result["omega"] is None:
        print("LP optimum = n/a")
        print(f"heuristic == LP optimum (tol={args.tol:g}): X")
        print(f"d_m(|S'|) = {dm_s_prime}")
        print(f"LP optimum > d_m(|S'|): X")
    else:
        omega = float(result["omega"])
        equal_within_tol = abs(omega - scaled_float) <= args.tol
        more_than_dm = omega > dm_s_prime + args.tol
        print(f"LP optimum = {omega:.10f}")
        print(
            f"heuristic == LP optimum (tol={args.tol:g}): "
            f"{'✓' if equal_within_tol else 'X'}"
        )
        print(f"d_m(|S'|) = {dm_s_prime}")
        print(f"LP optimum > d_m(|S'|): {'✓' if more_than_dm else 'X'}")


if __name__ == "__main__":
    main()
