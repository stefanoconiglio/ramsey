#!/usr/bin/env python3
"""Solve generic multiplier LP for Turán dominance (size |S'| and m)."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Any

from raymond import raymond


def _validate_inputs(s_size: int, m: int) -> None:
    if m < 2:
        raise ValueError("m must be >= 2")
    if s_size < 2:
        raise ValueError("|S'| must be >= 2")
    if s_size < m:
        raise ValueError("|S'| must be >= m")


def _status_map() -> dict[int, str]:
    from gurobipy import GRB

    return {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.UNBOUNDED: "Unbounded",
        GRB.INF_OR_UNBD: "InfeasibleOrUnbounded",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.INTERRUPTED: "Interrupted",
    }


def _create_model(gp: Any) -> tuple[Any, bool]:
    try:
        return gp.Model("multiplier_lp"), False
    except gp.GurobiError as exc:
        original_home = os.environ.get("HOME")
        try:
            # Fall back to Gurobi's bundled restricted license.
            os.environ["HOME"] = "/tmp"
            return gp.Model("multiplier_lp"), True
        except gp.GurobiError as retry_exc:
            raise RuntimeError(
                "Gurobi license expired and fallback to restricted license failed."
            ) from retry_exc
        finally:
            if original_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = original_home


def solve_multiplier_lp(
    s_size: int,
    m: int,
    *,
    strict: bool = False,
    output_lp: bool = False,
    verbose: bool = False,
    lp_path: str = "multiplier_lp_model.lp",
    show_license_warning: bool = True,
) -> dict[str, Any]:
    """Solve the multiplier LP and return status and solution values."""
    _validate_inputs(s_size, m)

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gurobipy is required. Install it with: python -m pip install gurobipy"
        ) from exc

    nodes = list(range(1, s_size + 1))
    m_sets = list(itertools.combinations(nodes, m))
    edges = list(itertools.combinations(nodes, 2))

    model, used_license_fallback = _create_model(gp)
    if used_license_fallback and show_license_warning:
        print(
            "Warning: local Gurobi license unusable; using bundled restricted license.",
            file=sys.stderr,
        )
    model.Params.OutputFlag = 1 if verbose else 0

    alpha = {
        S: model.addVar(lb=0.0, name=f"alpha_{'_'.join(map(str, S))}")
        for S in m_sets
    }
    y: dict[int, Any] = {}
    if strict:
        y = {
            i: model.addVar(lb=0.0, name=f"y_{i}")
            for i in range(1, len(edges) + 2)
        }
        model.setObjective(gp.quicksum(y[i] for i in y), GRB.MAXIMIZE)
    else:
        model.setObjective(0.0, GRB.MINIMIZE)

    # For each edge {i,j}: sum alpha_S over S containing i,j >= 1 (+ y_k if strict)
    for k, (i, j) in enumerate(edges, start=1):
        lhs = gp.quicksum(alpha[S] for S in m_sets if i in S and j in S)
        if strict:
            model.addConstr(lhs >= 1 + y[k], name=f"edge_{k}")
        else:
            model.addConstr(lhs >= 1, name=f"edge_{k}")

    # Sum alpha constraint vs Turán number
    dm_val = raymond(s_size, m)
    coeff = (m * (m - 1)) // 2 - 1
    rhs = dm_val
    total_alpha_expr = gp.quicksum(alpha[S] for S in m_sets)
    if strict:
        model.addConstr(
            coeff * total_alpha_expr <= rhs - y[len(edges) + 1], name="turan_bound"
        )
    else:
        model.addConstr(coeff * total_alpha_expr <= rhs, name="turan_bound")

    if output_lp:
        model.write(lp_path)

    model.optimize()

    status = _status_map().get(model.Status, f"StatusCode({model.Status})")
    max_sum_y = model.ObjVal if strict and model.SolCount > 0 else None
    if status == "Infeasible" or model.SolCount == 0:
        return {
            "status": status,
            "dm_val": dm_val,
            "max_sum_y": max_sum_y,
            "total_alpha": None,
            "alpha_values": None,
            "lp_path": lp_path if output_lp else None,
            "used_license_fallback": used_license_fallback,
        }

    alpha_values = {S: alpha[S].X for S in m_sets}
    return {
        "status": status,
        "dm_val": dm_val,
        "max_sum_y": max_sum_y,
        "total_alpha": sum(alpha_values.values()),
        "alpha_values": alpha_values,
        "lp_path": lp_path if output_lp else None,
        "used_license_fallback": used_license_fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multiplier LP for Turán vs clique inequalities.",
        usage="%(prog)s S m [--strict] [--lp] [--verbose]",
    )
    parser.add_argument("S", type=int, help="|S'| >= m and |S'| >= 2")
    parser.add_argument("m", type=int, help="clique size m >= 2")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Maximize sum y to test strict dominance (y>0).",
    )
    parser.add_argument(
        "--lp",
        action="store_true",
        help="Write the LP model to file and print it to screen.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print solver output.",
    )
    args = parser.parse_args()

    result = solve_multiplier_lp(
        args.S,
        args.m,
        strict=args.strict,
        output_lp=args.lp,
        verbose=args.verbose,
    )

    if args.lp and result["lp_path"] is not None:
        print("LP model:")
        with open(result["lp_path"], "r", encoding="utf-8") as f:
            print(f.read())

    print(f"|S'| = {args.S}, m = {args.m}")
    print(f"d_m(|S'|) = {result['dm_val']}")
    print("status:", result["status"])
    if args.strict:
        if result["max_sum_y"] is not None:
            print("max sum y =", result["max_sum_y"])
        else:
            print("max sum y = n/a")
    if result["status"] == "Infeasible":
        return
    if result["total_alpha"] is None or result["alpha_values"] is None:
        return
    print("total alpha =", result["total_alpha"])
    for S, value in result["alpha_values"].items():
        print(f"alpha{S} = {value}")


if __name__ == "__main__":
    main()
