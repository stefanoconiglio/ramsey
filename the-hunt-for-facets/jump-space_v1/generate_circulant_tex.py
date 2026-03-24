#!/usr/bin/env python3
"""Analyze one or many circulant-graph cases and optionally generate LaTeX output.

Each graph corresponds to a nonzero 0-1 jump vector y over
J={1,...,floor(t/2)}. Therefore the number of possible y-vectors is
2^|J|-1. The script can analyze either the standard inequality
sum_l kappa(l,S) y_l <= d_m(|S|) or its lifted variant.

The CLI accepts either explicit scalar values or inclusive min:max ranges.
When the resolved input expands to a single (t,S,m) case, the script runs the
full detailed analysis. When it expands to multiple cases, the script performs
an internal sweep and prints one summary line per case.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import itertools
import math
import re
import subprocess
import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from turan import turan as turan_bound


@dataclass(frozen=True)
class InequalityData:
    coeffs: list[int]
    raw_coeffs: list[int]
    rhs: int
    pair_count: int
    lift_cap: int | None


@dataclass(frozen=True)
class GraphRecord:
    idx: int
    jumps_l: set[int]
    edges_l: set[tuple[int, int]]
    lhs_value: int
    overlap_count: int
    mark_ok: bool
    is_minimal: bool


@dataclass(frozen=True)
class AnalysisCase:
    t: int
    s_values: tuple[int, ...]
    m: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute summary data for circulant graphs on V={0,...,t-1}, "
            "indexed by nonzero 0-1 jump vectors over J={1,...,floor(t/2)} "
            "(so total vectors = 2^|J|-1). "
            "Use scalars for a single case or min:max ranges for a sweep."
        )
    )
    parser.add_argument(
        "--t",
        type=str,
        required=True,
        help="Number of vertices t, as either an integer or an inclusive min:max range.",
    )
    parser.add_argument(
        "--S",
        type=str,
        default=None,
        help=(
            "Target vertex subset S (subset of V={0,...,t-1}). "
            "Examples: '0,1,2', '{0,3,7}', '5'. "
            "If omitted, use --S-size to sweep all subsets containing 0."
        ),
    )
    parser.add_argument(
        "--S-size",
        "--|S|",
        dest="s_size",
        default=None,
        help=(
            "|S|, as either an integer or an inclusive min:max range. "
            "Used only when --S is omitted; all subsets containing 0 are generated."
        ),
    )
    parser.add_argument(
        "--m",
        type=str,
        default=None,
        help=(
            "Parameter m used in d_m(|S|), as either an integer or an inclusive "
            "min:max range. Default: |S| for each resolved S."
        ),
    )
    lift_group = parser.add_mutually_exclusive_group()
    lift_group.add_argument(
        "--lift",
        "--lifted",
        dest="lift_mode",
        action="store_const",
        const="lifted",
        help=(
            "Analyze the lifted inequality with coefficients "
            "min(kappa(l,S), binom(|S|,2)-d_m(|S|)) instead of the standard one."
        ),
    )
    lift_group.add_argument(
        "--addlifted",
        dest="lift_mode",
        action="store_const",
        const="addlifted",
        help="Run the standard inequality first and then the lifted one.",
    )
    parser.set_defaults(lift_mode="standard")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .tex file path for a single resolved case (used only with --latex). "
            "With --addlifted, this path is used for the standard run and a sibling "
            "_lift file is used for the lifted run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for auto-named .tex files (used only with --latex). "
            "Useful for sweeps; also works for single-case default naming."
        ),
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of graph cells per row (default: 3).",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX/PDF output. Without this flag, only stdout summary is printed.",
    )
    parser.add_argument(
        "--all",
        dest="show_all_graphs",
        action="store_true",
        help=(
            "When used with --latex, include all graphs in the report. "
            "By default, LaTeX includes only graphs that pass the check."
        ),
    )
    parser.add_argument(
        "--fraction",
        action="store_true",
        help=(
            "Use exact Fraction arithmetic for rank/nullspace computations. "
            "Without this flag, computations use float64 for speed."
        ),
    )
    parser.add_argument(
        "--sssp",
        action="store_true",
        help=(
            "Compute exact SSSP via binary subset-sum feasibility check "
            "(whether there exists y in {0,1}^|J| with a'y=rhs). "
            "Without this flag, only fast necessary checks are used."
        ),
    )
    parser.add_argument(
        "--recipe",
        action="store_true",
        help=(
            "Use direct recipe mode: build CORE graphs (all active jumps except one), "
            "then add free jumps (a_j=0) only to the first CORE graph until the set "
            "size reaches floor(t/2) or free jumps run out. "
            "In this mode, Y, Y+, and rank use only the recipe-selected graphs."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="In multi-case mode, stop at the first failing case.",
    )
    return parser.parse_args()


def parse_int_or_closed_range(spec: str, arg_name: str) -> list[int]:
    stripped = spec.strip()
    if not stripped:
        raise ValueError(f"{arg_name} requires an integer or min:max, got an empty value.")
    if ":" not in stripped:
        return [int(stripped)]

    parts = [part.strip() for part in stripped.split(":")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{arg_name} must be in 'min:max' format, got '{spec}'.")
    start = int(parts[0])
    end = int(parts[1])
    if start > end:
        raise ValueError(f"{arg_name} requires min <= max, got '{spec}'.")
    return list(range(start, end + 1))


def parse_explicit_s_values(raw: str) -> tuple[int, ...]:
    values = [int(x) for x in re.findall(r"\d+", raw.strip())]
    if not values:
        raise ValueError(
            f"Could not parse S='{raw}'. Use integers like '0,1,2' or '{{0,3,7}}'."
        )
    return tuple(sorted(set(values)))


def validate_explicit_s_values(values: tuple[int, ...], t: int) -> None:
    invalid = sorted(v for v in values if v < 0 or v >= t)
    if invalid:
        raise ValueError(
            f"Invalid vertices in S: {invalid}. Allowed vertices are in 0..{t - 1}."
        )


def build_s_containing_zero(t: int, s_size: int) -> list[tuple[int, ...]]:
    if s_size < 1 or s_size > t:
        return []
    if s_size == 1:
        return [(0,)]
    return [(0,) + tail for tail in itertools.combinations(range(1, t), s_size - 1)]


def resolve_analysis_cases(args: argparse.Namespace) -> list[AnalysisCase]:
    t_values = parse_int_or_closed_range(args.t, "--t")
    if any(t < 2 for t in t_values):
        raise ValueError("t must be at least 2.")

    if args.S is None and args.s_size is None:
        raise ValueError("Provide either --S or --S-size.")

    explicit_s_values = parse_explicit_s_values(args.S) if args.S is not None else None
    if explicit_s_values is not None and len(t_values) == 1:
        validate_explicit_s_values(explicit_s_values, t_values[0])

    s_size_values: list[int] | None = None
    if explicit_s_values is None:
        assert args.s_size is not None
        s_size_values = parse_int_or_closed_range(args.s_size, "--S-size")
        if any(s_size < 1 for s_size in s_size_values):
            raise ValueError("|S| must be at least 1.")

    m_values = (
        parse_int_or_closed_range(args.m, "--m")
        if args.m is not None
        else None
    )

    cases: list[AnalysisCase] = []
    for t in t_values:
        if explicit_s_values is not None:
            try:
                validate_explicit_s_values(explicit_s_values, t)
            except ValueError:
                continue
            s_candidates = [explicit_s_values]
        else:
            assert s_size_values is not None
            s_candidates = []
            for s_size in s_size_values:
                s_candidates.extend(build_s_containing_zero(t, s_size))

        for s_values in s_candidates:
            current_m_values = m_values if m_values is not None else [len(s_values)]
            for m in current_m_values:
                if m < 2 or m > len(s_values):
                    continue
                cases.append(AnalysisCase(t=t, s_values=s_values, m=m))

    if cases:
        return cases

    if explicit_s_values is not None and len(t_values) == 1:
        validate_explicit_s_values(explicit_s_values, t_values[0])
        if m_values is not None and len(m_values) == 1 and m_values[0] < 2:
            raise ValueError("m must be at least 2. If --m is omitted, this requires |S| >= 2.")
        default_m = m_values[0] if m_values is not None and len(m_values) == 1 else len(explicit_s_values)
        if default_m < 2:
            raise ValueError("m must be at least 2. If --m is omitted, this requires |S| >= 2.")
        if len(explicit_s_values) < default_m:
            raise ValueError("|S| must be at least m so that d_m(|S|) is defined.")

    raise SystemExit(
        "No valid (t,S,m) combinations after filtering 0 <= S < t and 2 <= m <= |S|."
    )


def jump_distance(i: int, j: int, t: int) -> int:
    return min((i - j) % t, (j - i) % t)


def undirected_edges_from_jumps(t: int, jumps: set[int]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(t):
        for j in range(i + 1, t):
            if jump_distance(i, j, t) in jumps:
                edges.add((i, j))
    return edges


def edges_from_vertex_subset(vertices: set[int]) -> set[tuple[int, int]]:
    return {tuple(sorted(edge)) for edge in itertools.combinations(vertices, 2)}


def all_nonempty_jump_sets(j_list: list[int]) -> list[set[int]]:
    """Return all non-empty jump sets (total: 2^|J|-1)."""
    subsets: list[set[int]] = []
    for size in range(1, len(j_list) + 1):
        for combo in itertools.combinations(j_list, size):
            subsets.append(set(combo))
    return subsets


def latex_set(values: list[int]) -> str:
    if not values:
        return r"\varnothing"
    return r"\{" + ",".join(str(v) for v in values) + r"\}"


def latex_set_highlight(values: list[int], highlight_values: set[int]) -> str:
    if not values:
        return r"\varnothing"
    parts: list[str] = []
    for value in values:
        if value in highlight_values:
            parts.append(r"\textcolor{red}{" + str(value) + "}")
        else:
            parts.append(str(value))
    return r"\{" + ",".join(parts) + r"\}"


def latex_y_vector_highlight(
    j_list: list[int],
    jumps_l: set[int],
    highlight_values: set[int],
) -> str:
    y_vector = jump_y_vector(j_list, jumps_l)
    entries: list[str] = []
    for jump, value in zip(j_list, y_vector):
        value_text = str(value)
        if jump in highlight_values:
            entries.append(r"\textcolor{red}{" + value_text + "}")
        else:
            entries.append(value_text)
    return r"\left(" + ",".join(entries) + r"\right)"


def jump_y_vector(j_list: list[int], jumps_l: set[int]) -> list[int]:
    """Return jump-indicator vector y over J, i.e. y_l=1 iff l in L."""
    return [1 if jump in jumps_l else 0 for jump in j_list]


def latex_linear_form(j_list: list[int], coeffs: list[int]) -> str:
    terms: list[str] = []
    for jump, coeff in zip(j_list, coeffs):
        if coeff == 0:
            continue
        if coeff == 1:
            terms.append(rf"y_{{{jump}}}")
        else:
            terms.append(rf"{coeff} y_{{{jump}}}")
    return " + ".join(terms) if terms else "0"


def latex_matrix_from_columns(columns: list[list[int]]) -> str:
    if not columns:
        return r"\varnothing"
    row_count = len(columns[0])
    rows = [
        " & ".join(str(columns[col_idx][row]) for col_idx in range(len(columns)))
        for row in range(row_count)
    ]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def matrix_rank_fraction(rows: list[list[int]]) -> int:
    if not rows or not rows[0]:
        return 0

    mat = [[Fraction(value) for value in row] for row in rows]
    row_count = len(mat)
    col_count = len(mat[0])
    pivot_row = 0
    rank = 0

    for col in range(col_count):
        pivot = None
        for r in range(pivot_row, row_count):
            if mat[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        mat[pivot_row], mat[pivot] = mat[pivot], mat[pivot_row]
        pivot_value = mat[pivot_row][col]
        mat[pivot_row] = [value / pivot_value for value in mat[pivot_row]]

        for r in range(row_count):
            if r == pivot_row or mat[r][col] == 0:
                continue
            factor = mat[r][col]
            mat[r] = [a - factor * b for a, b in zip(mat[r], mat[pivot_row])]

        pivot_row += 1
        rank += 1
        if pivot_row == row_count:
            break

    return rank


def matrix_rank_float(rows: list[list[int]], eps: float = 1e-12) -> int:
    if not rows or not rows[0]:
        return 0

    mat = [[float(value) for value in row] for row in rows]
    row_count = len(mat)
    col_count = len(mat[0])
    pivot_row = 0
    rank = 0

    for col in range(col_count):
        pivot = max(range(pivot_row, row_count), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) <= eps:
            continue

        mat[pivot_row], mat[pivot] = mat[pivot], mat[pivot_row]
        pivot_value = mat[pivot_row][col]
        mat[pivot_row] = [value / pivot_value for value in mat[pivot_row]]

        for r in range(row_count):
            if r == pivot_row or abs(mat[r][col]) <= eps:
                continue
            factor = mat[r][col]
            mat[r] = [a - factor * b for a, b in zip(mat[r], mat[pivot_row])]

        pivot_row += 1
        rank += 1
        if pivot_row == row_count:
            break

    return rank


def matrix_rank(rows: list[list[int]], use_fraction: bool) -> int:
    return matrix_rank_fraction(rows) if use_fraction else matrix_rank_float(rows)


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def nullspace_vector_fraction(rows: list[list[int]]) -> list[int] | None:
    """Return one nonzero integer vector m with A m = 0, if it exists."""
    if not rows or not rows[0]:
        return None

    mat = [[Fraction(value) for value in row] for row in rows]
    row_count = len(mat)
    col_count = len(mat[0])
    pivot_row = 0
    pivot_cols: list[int] = []

    for col in range(col_count):
        pivot = None
        for r in range(pivot_row, row_count):
            if mat[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        mat[pivot_row], mat[pivot] = mat[pivot], mat[pivot_row]
        pivot_value = mat[pivot_row][col]
        mat[pivot_row] = [value / pivot_value for value in mat[pivot_row]]

        for r in range(row_count):
            if r == pivot_row or mat[r][col] == 0:
                continue
            factor = mat[r][col]
            mat[r] = [a - factor * b for a, b in zip(mat[r], mat[pivot_row])]

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break

    free_cols = [col for col in range(col_count) if col not in pivot_cols]
    if not free_cols:
        return None

    x = [Fraction(0) for _ in range(col_count)]
    chosen_free = free_cols[0]
    x[chosen_free] = Fraction(1)

    for pr_idx in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[pr_idx]
        rhs = Fraction(0)
        for free_col in free_cols:
            rhs += mat[pr_idx][free_col] * x[free_col]
        x[pc] = -rhs

    common_den = 1
    for value in x:
        common_den = lcm(common_den, value.denominator)

    int_vec = [int(value * common_den) for value in x]
    gcd_nonzero = 0
    for value in int_vec:
        if value != 0:
            gcd_nonzero = abs(value) if gcd_nonzero == 0 else math.gcd(gcd_nonzero, abs(value))
    if gcd_nonzero > 1:
        int_vec = [value // gcd_nonzero for value in int_vec]

    for value in int_vec:
        if value != 0:
            if value < 0:
                int_vec = [-v for v in int_vec]
            break

    return int_vec if any(value != 0 for value in int_vec) else None


def nullspace_vector_float(rows: list[list[int]], eps: float = 1e-12) -> list[float] | None:
    """Return one nonzero float vector m with A m ~= 0, if it exists."""
    if not rows or not rows[0]:
        return None

    mat = [[float(value) for value in row] for row in rows]
    row_count = len(mat)
    col_count = len(mat[0])
    pivot_row = 0
    pivot_cols: list[int] = []

    for col in range(col_count):
        pivot = max(range(pivot_row, row_count), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) <= eps:
            continue

        mat[pivot_row], mat[pivot] = mat[pivot], mat[pivot_row]
        pivot_value = mat[pivot_row][col]
        mat[pivot_row] = [value / pivot_value for value in mat[pivot_row]]

        for r in range(row_count):
            if r == pivot_row or abs(mat[r][col]) <= eps:
                continue
            factor = mat[r][col]
            mat[r] = [a - factor * b for a, b in zip(mat[r], mat[pivot_row])]

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break

    free_cols = [col for col in range(col_count) if col not in pivot_cols]
    if not free_cols:
        return None

    x = [0.0 for _ in range(col_count)]
    chosen_free = free_cols[0]
    x[chosen_free] = 1.0

    for pr_idx in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[pr_idx]
        rhs = 0.0
        for free_col in free_cols:
            rhs += mat[pr_idx][free_col] * x[free_col]
        x[pc] = -rhs

    return x


def nullspace_vector(rows: list[list[int]], use_fraction: bool) -> list[int] | list[float] | None:
    return nullspace_vector_fraction(rows) if use_fraction else nullspace_vector_float(rows)


def format_scalar(value: int | float) -> str:
    if isinstance(value, float):
        rounded = round(value)
        if abs(value - rounded) <= 1e-9:
            return str(int(rounded))
        return f"{value:.6g}"
    return str(value)


def latex_column_vector(values: list[int] | list[float]) -> str:
    if not values:
        return r"\varnothing"
    return r"\begin{bmatrix}" + r" \\ ".join(format_scalar(v) for v in values) + r"\end{bmatrix}"


def latex_tuple(values: list[int] | list[float]) -> str:
    if not values:
        return r"\varnothing"
    return r"\left(" + ",".join(format_scalar(v) for v in values) + r"\right)"


def matrix_rows_from_columns(columns: list[list[int]]) -> list[list[int]]:
    if not columns:
        return []
    return [
        [columns[col_idx][row_idx] for col_idx in range(len(columns))]
        for row_idx in range(len(columns[0]))
    ]


def plain_set(values: list[int]) -> str:
    if not values:
        return "{}"
    return "{" + ",".join(str(v) for v in values) + "}"


def facet_status(rank_y_plus: int, j_max: int) -> str:
    return "FACET" if rank_y_plus == j_max else "FAILURE"


def a_uniformity_status(a_coeffs: list[int]) -> str:
    nonzero_values = {value for value in a_coeffs if value != 0}
    return "UNIFORM" if len(nonzero_values) <= 1 else "DIFFERT"


def thm2_value(s_size: int, m: int) -> int:
    return 1 if (s_size % (m - 1)) != 0 else 0


def subset_sum_binary_feasible(coeffs: list[int], target: int) -> bool:
    """Return True iff there exists y in {0,1}^n with coeffs'y = target."""
    if target < 0:
        return False
    if target == 0:
        return True

    positive_coeffs = [value for value in coeffs if value > 0]
    if sum(positive_coeffs) < target:
        return False

    reachable = 1  # bit k means sum k is reachable
    mask = (1 << (target + 1)) - 1
    for value in positive_coeffs:
        if value > target:
            continue
        reachable |= reachable << value
        reachable &= mask
        if (reachable >> target) & 1:
            return True
    return ((reachable >> target) & 1) == 1


def sssp_value(a_coeffs: list[int], rhs_value: int, compute_exact: bool) -> str:
    nonzero_values = [value for value in a_coeffs if value != 0]

    # Fast check 1: only one nonzero coefficient with parity mismatch (and rhs > 0) => impossible.
    if len(nonzero_values) == 1 and rhs_value > 0:
        coeff = nonzero_values[0]
        if (coeff % 2) != (rhs_value % 2):
            return "0"

    # Fast check 2: all coefficients even and rhs odd => impossible.
    if all(value % 2 == 0 for value in a_coeffs) and (rhs_value % 2 == 1):
        return "0"

    if not compute_exact:
        return "-"

    return "1" if subset_sum_binary_feasible(a_coeffs, rhs_value) else "0"


def build_inequality_data(
    raw_coeffs: list[int],
    s_size: int,
    d_m_s: int,
    use_lift: bool,
) -> InequalityData:
    pair_count = math.comb(s_size, 2)
    if not use_lift:
        return InequalityData(
            coeffs=list(raw_coeffs),
            raw_coeffs=list(raw_coeffs),
            rhs=d_m_s,
            pair_count=pair_count,
            lift_cap=None,
        )

    lift_cap = pair_count - d_m_s
    coeffs = [min(value, lift_cap) for value in raw_coeffs]
    rhs = d_m_s + sum(coeffs) - pair_count
    return InequalityData(
        coeffs=coeffs,
        raw_coeffs=list(raw_coeffs),
        rhs=rhs,
        pair_count=pair_count,
        lift_cap=lift_cap,
    )


def linear_form_value(
    coeffs: list[int],
    jumps_l: set[int],
    jump_to_idx: dict[int, int],
) -> int:
    return sum(coeffs[jump_to_idx[jump]] for jump in jumps_l)


def build_recipe_jump_sets(
    j_list: list[int],
    a_coeffs: list[int],
    target_count: int,
) -> list[set[int]]:
    """Build recipe jump-sets: CORE + lifts of first CORE by free jumps."""
    active_jumps = [jump for jump, coeff in zip(j_list, a_coeffs) if coeff > 0]
    free_jumps = [jump for jump, coeff in zip(j_list, a_coeffs) if coeff == 0]

    recipe_sets: list[set[int]] = []
    seen: set[frozenset[int]] = set()

    if not active_jumps:
        return recipe_sets

    # CORE: for each active jump, take all active jumps except that one.
    if len(active_jumps) >= 2:
        full_active = set(active_jumps)
        for omitted in active_jumps:
            core_set = set(full_active)
            core_set.discard(omitted)
            frozen = frozenset(core_set)
            if not core_set or frozen in seen:
                continue
            recipe_sets.append(core_set)
            seen.add(frozen)
    else:
        core_set = {active_jumps[0]}
        recipe_sets.append(core_set)
        seen.add(frozenset(core_set))

    if not recipe_sets:
        return recipe_sets

    # Add free jumps only to the first CORE graph until reaching target_count.
    extra_needed = max(0, target_count - len(recipe_sets))
    first_core = set(recipe_sets[0])
    for free_jump in free_jumps[:extra_needed]:
        lifted = set(first_core)
        lifted.add(free_jump)
        frozen = frozenset(lifted)
        if frozen in seen:
            continue
        recipe_sets.append(lifted)
        seen.add(frozen)

    return recipe_sets


def status_line(
    current: int,
    total: int,
    t: int,
    m: int,
    thm2: int,
    s_sorted: list[int],
    j_max: int,
    n_check: int,
) -> str:
    return (
        f"{current}/{total} "
        f"t={t}, m={m}, thm2={thm2}, S={plain_set(s_sorted)}, floor(t/2)={j_max}, "
        f"N={n_check}"
    )


def final_status_line(
    current: int,
    total: int,
    t: int,
    m: int,
    thm2: int,
    s_sorted: list[int],
    j_max: int,
    n_check: int,
    rank_y_plus: int,
    alldiff: int,
    a_coeffs: list[int],
    d_m_s: int,
    rhs_value: int,
    sssp: str,
    use_lift: bool,
) -> str:
    coeffs_str = ",".join(str(value) for value in a_coeffs)
    uniformity = a_uniformity_status(a_coeffs)
    mode = "lifted" if use_lift else "standard"
    return (
        f"{current}/{total} "
        f"t={t}, m={m}, thm2={thm2}, S={plain_set(s_sorted)}, floor(t/2)={j_max}, "
        f"N={n_check}, rank(Y+)={rank_y_plus}, alldiff={alldiff}, d_m(|S|)={d_m_s}, rhs={rhs_value}, "
        f"a'y=({coeffs_str})'y, {uniformity}, SSSP={sssp}, ineq={mode}, "
        f"{facet_status(rank_y_plus, j_max)}"
    )


def print_in_place(line: str, previous_len: int) -> int:
    if previous_len > len(line):
        line = line + (" " * (previous_len - len(line)))
    print(f"\r{line}", end="", flush=True)
    return len(line)


def compute_summary(
    t: int,
    m: int,
    s_set: set[int],
    use_fraction: bool,
    compute_sssp_exact: bool,
    use_recipe: bool,
    use_lift: bool,
    show_progress: bool = True,
) -> dict:
    s_size = len(s_set)
    thm2 = thm2_value(s_size, m)
    d_m_s = turan_bound(s_size, m)
    j_max = t // 2
    j_list = list(range(1, j_max + 1))
    s_sorted = sorted(s_set)
    s_edges = edges_from_vertex_subset(s_set)
    s_jump_set = {jump_distance(i, j, t) for i, j in s_edges}
    alldiff = 1 if len(s_jump_set) == len(s_edges) else 0
    jump_to_idx = {jump: idx for idx, jump in enumerate(j_list)}
    raw_coeffs = [0 for _ in j_list]
    for i, j in s_edges:
        jump = jump_distance(i, j, t)
        raw_coeffs[jump_to_idx[jump]] += 1

    inequality = build_inequality_data(
        raw_coeffs=raw_coeffs,
        s_size=s_size,
        d_m_s=d_m_s,
        use_lift=use_lift,
    )
    a_coeffs = inequality.coeffs
    rhs_value = inequality.rhs
    jump_overlap_count = {jump: a_coeffs[idx] for idx, jump in enumerate(j_list)}
    sssp = sssp_value(a_coeffs, rhs_value, compute_sssp_exact)
    graph_data: list[GraphRecord] = []
    last_status_len = 0
    if use_recipe:
        recipe_subsets = build_recipe_jump_sets(
            j_list=j_list,
            a_coeffs=a_coeffs,
            target_count=j_max,
        )
        total_y_vectors = len(recipe_subsets)

        for graph_idx, jumps_l in enumerate(recipe_subsets, start=1):
            edges_l = undirected_edges_from_jumps(t, jumps_l)
            overlap_count = len(edges_l & s_edges)
            lhs_value = linear_form_value(a_coeffs, jumps_l, jump_to_idx)
            mark_ok = lhs_value == rhs_value
            is_minimal = mark_ok and all(jump_overlap_count.get(jump, 0) > 0 for jump in jumps_l)
            graph_data.append(
                GraphRecord(
                    idx=graph_idx,
                    jumps_l=jumps_l,
                    edges_l=edges_l,
                    lhs_value=lhs_value,
                    overlap_count=overlap_count,
                    mark_ok=mark_ok,
                    is_minimal=is_minimal,
                )
            )

            if show_progress:
                current_status = status_line(
                    current=graph_idx,
                    total=total_y_vectors,
                    t=t,
                    m=m,
                    thm2=thm2,
                    s_sorted=s_sorted,
                    j_max=j_max,
                    n_check=graph_idx,
                )
                last_status_len = print_in_place(current_status, last_status_len)

        selected_graphs = graph_data
    else:
        subsets = all_nonempty_jump_sets(j_list)
        total_y_vectors = len(subsets)
        progress_check_count = 0

        for graph_idx, jumps_l in enumerate(subsets, start=1):
            edges_l = undirected_edges_from_jumps(t, jumps_l)
            overlap_count = len(edges_l & s_edges)
            lhs_value = linear_form_value(a_coeffs, jumps_l, jump_to_idx)
            mark_ok = lhs_value == rhs_value
            is_minimal = mark_ok and all(jump_overlap_count.get(jump, 0) > 0 for jump in jumps_l)
            if mark_ok:
                progress_check_count += 1
            graph_data.append(
                GraphRecord(
                    idx=graph_idx,
                    jumps_l=jumps_l,
                    edges_l=edges_l,
                    lhs_value=lhs_value,
                    overlap_count=overlap_count,
                    mark_ok=mark_ok,
                    is_minimal=is_minimal,
                )
            )

            if show_progress:
                current_status = status_line(
                    current=graph_idx,
                    total=total_y_vectors,
                    t=t,
                    m=m,
                    thm2=thm2,
                    s_sorted=s_sorted,
                    j_max=j_max,
                    n_check=progress_check_count,
                )
                last_status_len = print_in_place(current_status, last_status_len)

        selected_graphs = [entry for entry in graph_data if entry.mark_ok]

    check_indices = [entry.idx for entry in selected_graphs]
    check_y_columns = [jump_y_vector(j_list, entry.jumps_l) for entry in selected_graphs]
    check_count = len(selected_graphs)

    y_plus_columns = [column + [1] for column in check_y_columns]
    y_plus_rows = matrix_rows_from_columns(y_plus_columns)
    y_plus_rank = matrix_rank(y_plus_rows, use_fraction=use_fraction)
    y_plus_max_rank = (
        min(len(y_plus_rows), len(y_plus_columns))
        if y_plus_rows and y_plus_columns
        else 0
    )
    y_plus_multiplier = (
        nullspace_vector(y_plus_rows, use_fraction=use_fraction)
        if y_plus_rank < y_plus_max_rank
        else None
    )
    final_status = final_status_line(
        current=total_y_vectors,
        total=total_y_vectors,
        t=t,
        m=m,
        thm2=thm2,
        s_sorted=s_sorted,
        j_max=j_max,
        n_check=check_count,
        rank_y_plus=y_plus_rank,
        alldiff=alldiff,
        a_coeffs=a_coeffs,
        d_m_s=d_m_s,
        rhs_value=rhs_value,
        sssp=sssp,
        use_lift=use_lift,
    )
    if show_progress:
        if total_y_vectors > 0:
            print_in_place(final_status, last_status_len)
        else:
            print(final_status, end="")
        print()

    return {
        "t": t,
        "m": m,
        "thm2": thm2,
        "d_m_s": d_m_s,
        "rhs_value": rhs_value,
        "sssp": sssp,
        "s_size": s_size,
        "alldiff": alldiff,
        "a_coeffs": a_coeffs,
        "raw_coeffs": raw_coeffs,
        "pair_count": inequality.pair_count,
        "lift_cap": inequality.lift_cap,
        "s_sorted": s_sorted,
        "s_jump_set": s_jump_set,
        "j_max": j_max,
        "j_list": j_list,
        "total_y_vectors": total_y_vectors,
        "graph_data": graph_data,
        "selected_graph_data": selected_graphs,
        "use_recipe": use_recipe,
        "use_lift": use_lift,
        "check_count": check_count,
        "check_indices": check_indices,
        "check_y_columns": check_y_columns,
        "y_rows": matrix_rows_from_columns(check_y_columns),
        "y_plus_columns": y_plus_columns,
        "y_plus_rows": y_plus_rows,
        "y_plus_rank": y_plus_rank,
        "y_plus_multiplier": y_plus_multiplier,
        "status": facet_status(y_plus_rank, j_max),
        "final_status_line": final_status,
    }


def node_positions(t: int, radius: float = 1.35) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for i in range(t):
        angle_deg = 90.0 - (360.0 * i / t)  # clockwise order
        angle_rad = math.radians(angle_deg)
        x = radius * math.cos(angle_rad)
        y = radius * math.sin(angle_rad)
        coords.append((x, y))
    return coords


def complete_graph_highlight_s_tex(
    t: int,
    s_set: set[int],
    m: int,
    d_m_s: int,
    j_list: list[int],
    raw_coeffs: list[int],
    a_coeffs: list[int],
    rhs_value: int,
    use_lift: bool,
    pair_count: int,
    lift_cap: int | None,
    width: str = "0.31\\textwidth",
) -> str:
    coords = node_positions(t)
    s_edges = edges_from_vertex_subset(s_set)
    s_size = len(s_set)
    s_jumps = [jump for jump, coeff in zip(j_list, raw_coeffs) if coeff > 0]
    analysis_terms = latex_linear_form(j_list, a_coeffs)
    lines: list[str] = []
    lines.append(
        r"\smallskip Complete graph on $V$, with vertices in $S$ "
        r"and edges in $E(S)$ highlighted in red. "
        rf"$|E(S)|={len(s_edges)}$"
    )
    lines.append(r"")
    lines.append(rf"\begin{{minipage}}[t]{{{width}}}")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}[")
    lines.append(r"  thick,")
    lines.append(r"  baseline=(current bounding box.north),")
    lines.append(r"  vertex/.style={circle,draw,inner sep=1.2pt,font=\scriptsize},")
    lines.append(r"  svertex/.style={circle,draw=red,text=red,inner sep=1.2pt,font=\scriptsize}")
    lines.append(r"]")

    for i, (x, y) in enumerate(coords):
        style = "svertex" if i in s_set else "vertex"
        lines.append(rf"  \node[{style}] (v{i}) at ({x:.3f},{y:.3f}) {{{i}}};")

    for i in range(t):
        for j in range(i + 1, t):
            if i in s_set and j in s_set:
                lines.append(rf"  \draw[red] (v{i}) -- (v{j});")
            else:
                lines.append(rf"  \draw[black!45,line width=0.35pt] (v{i}) -- (v{j});")

    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{minipage}")
    lines.append(rf"\par\smallskip Jumps in $E(S)$: $J={latex_set(s_jumps)}$")
    if use_lift:
        assert lift_cap is not None
        lines.append(
            r"\par $a_\ell=\bar{\kappa}(\ell,S,m)="
            r"\min\{\kappa(\ell,S),\binom{|S|}{2}-d_m(|S|)\}$"
        )
        lines.append(
            rf"\par $\binom{{|S|}}{{2}}-d_m(|S|)=\binom{{{s_size}}}{{2}}-d_{{{m}}}({s_size})="
            rf"{pair_count}-{d_m_s}={lift_cap}$"
        )
        lines.append(rf"\par $a={latex_tuple(a_coeffs)}$")
        lines.append(
            rf"\par $a^\top y = {analysis_terms} \le "
            rf"d_m(|S|)+\sum_{{\ell\in J}} a_\ell-\binom{{|S|}}{{2}}="
            rf"{d_m_s}+{sum(a_coeffs)}-{pair_count}={rhs_value}$"
        )
    else:
        lines.append(rf"\par $a_\ell=\kappa(\ell,S),\quad a={latex_tuple(a_coeffs)}$")
        lines.append(
            rf"\par $a^\top y = {analysis_terms} \le d_m(|S|)=d_{{{m}}}({s_size})={rhs_value}$"
        )
    return "\n".join(lines)


def graph_cell_tex(
    idx: int,
    t: int,
    s_set: set[int],
    s_jump_set: set[int],
    j_list: list[int],
    jumps_l: set[int],
    edges_l: set[tuple[int, int]],
    lhs_value: int,
    overlap_with_s: int,
    mark_ok: bool,
    is_minimal: bool,
    use_lift: bool,
    width: str = "0.31\\textwidth",
) -> str:
    coords = node_positions(t)
    l_sorted = sorted(jumps_l)
    l_colored = latex_set_highlight(l_sorted, s_jump_set)
    y_vector_colored = latex_y_vector_highlight(j_list, jumps_l, s_jump_set)

    lines: list[str] = []
    lines.append(rf"\begin{{minipage}}[t]{{{width}}}")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}[")
    lines.append(
        r"  thick,"
        r"  vertex/.style={circle,draw,inner sep=1.2pt,font=\scriptsize},"
        r"  svertex/.style={circle,draw=red,text=red,inner sep=1.2pt,font=\scriptsize},"
        r"  baseline=(current bounding box.north)"
    )
    lines.append(r"]")

    for i, (x, y) in enumerate(coords):
        style = "svertex" if i in s_set else "vertex"
        lines.append(
            rf"  \node[{style}] (v{i}) at ({x:.3f},{y:.3f}) {{{i}}};"
        )

    for i, j in sorted(edges_l):
        if i in s_set and j in s_set:
            lines.append(rf"  \draw[red] (v{i}) -- (v{j});")
        else:
            lines.append(rf"  \draw (v{i}) -- (v{j});")

    lines.append(r"\end{tikzpicture}")
    lines.append(rf"\par\smallskip Graph {idx}")
    lines.append(rf"\par $L={l_colored}$")
    lines.append(rf"\par $y={y_vector_colored}$")
    mark_symbol = (
        r"\textcolor{green!60!black}{\checkmark}"
        if mark_ok
        else r"\textcolor{red}{\boldsymbol{\times}}"
    )
    if mark_ok and is_minimal:
        mark_symbol += r"\;\textbf{MIN}"
    if use_lift:
        lines.append(rf"\par $a^\top y={lhs_value}\;\;{mark_symbol}$")
        lines.append(rf"\par $|E(L)\cap E(S)|={overlap_with_s}$")
    else:
        lines.append(rf"\par $|E(L)\cap E(S)|={lhs_value}\;\;{mark_symbol}$")
    lines.append(r"\end{minipage}")
    return "\n".join(lines)


def build_document(summary: dict, cols: int, show_all_graphs: bool) -> str:
    t = summary["t"]
    m = summary["m"]
    d_m_s = summary["d_m_s"]
    rhs_value = summary["rhs_value"]
    s_size = summary["s_size"]
    s_sorted = summary["s_sorted"]
    s_jump_set = set(summary["s_jump_set"])
    j_max = summary["j_max"]
    j_list = summary["j_list"]
    a_coeffs = summary["a_coeffs"]
    raw_coeffs = summary["raw_coeffs"]
    pair_count = summary["pair_count"]
    lift_cap = summary["lift_cap"]
    graph_data = summary["graph_data"]
    selected_graph_data = summary["selected_graph_data"]
    use_recipe = summary["use_recipe"]
    use_lift = summary["use_lift"]
    check_count = summary["check_count"]
    check_indices = summary["check_indices"]
    check_y_columns = summary["check_y_columns"]
    y_plus_columns = summary["y_plus_columns"]
    y_plus_rank = summary["y_plus_rank"]
    y_plus_multiplier = summary["y_plus_multiplier"]
    status = summary["status"]
    s_set = set(s_sorted)

    if use_recipe:
        display_graphs = selected_graph_data
    elif show_all_graphs:
        display_graphs = graph_data
    else:
        display_graphs = [entry for entry in graph_data if entry.mark_ok]
    max_matrix_cols = max(10, len(check_y_columns), len(y_plus_columns))
    mode_label = "lifted" if use_lift else "standard"

    doc: list[str] = []
    doc.append(r"\documentclass[11pt]{article}")
    doc.append(r"\usepackage[a4paper,margin=1.8cm]{geometry}")
    doc.append(r"\usepackage{tikz}")
    doc.append(r"\usepackage{amsmath}")
    doc.append(rf"\setcounter{{MaxMatrixCols}}{{{max_matrix_cols}}}")
    doc.append(r"\usepackage{amssymb}")
    doc.append(r"\usepackage{xcolor}")
    doc.append(r"\pagestyle{empty}")
    doc.append(r"")
    doc.append(r"\begin{document}")
    doc.append(rf"\section*{{$S={latex_set(s_sorted)}$, $m={m}$, $t={t}$}}")
    doc.append(r"\begin{center}")
    doc.append(rf"$\mathbf{{{status}}}$")
    doc.append(r"\end{center}")
    doc.append(rf"$\left\lfloor t/2\right\rfloor={j_max}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$J={latex_set(j_list)}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$N={check_count}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$\operatorname{{rank}}(Y^{{+}})={y_plus_rank}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$\text{{mode}}=\text{{{mode_label}}}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$\text{{rhs}}={rhs_value}$")
    doc.append(r"")

    doc.append(r"\begin{center}")
    doc.append(
        complete_graph_highlight_s_tex(
            t=t,
            s_set=s_set,
            m=m,
            d_m_s=d_m_s,
            j_list=j_list,
            raw_coeffs=raw_coeffs,
            a_coeffs=a_coeffs,
            rhs_value=rhs_value,
            use_lift=use_lift,
            pair_count=pair_count,
            lift_cap=lift_cap,
        )
    )
    doc.append(r"\end{center}")
    doc.append(r"")

    doc.append(
        r"Each drawing below is a circulant graph $G_{L}$ "
        r"for a non-empty $L\subseteq J$ "
        r"with edges $E(L)=\{\{i,j\}\mid i,j\in V,\ "
        r"\min((i-j)\bmod t,\,(j-i)\bmod t)\in L\}$."
    )
    if use_lift:
        doc.append(
            r"For each graph, we show $a^\top y$ for the lifted coefficient vector "
            r"$a_\ell=\bar{\kappa}(\ell,S,m)$ and put "
            r"$\textcolor{green!60!black}{\checkmark}$ iff $a^\top y=\mathrm{rhs}$; "
            r"otherwise $\textcolor{red}{\boldsymbol{\times}}$. "
            r"The raw overlap $|E(L)\cap E(S)|$ is also shown for reference."
        )
    else:
        doc.append(
            r"For each graph, we show $|E(L)\cap E(S)|$ and put "
            r"$\textcolor{green!60!black}{\checkmark}$ iff "
            r"$|E(L)\cap E(S)|=d_m(|S|)$; otherwise "
            r"$\textcolor{red}{\boldsymbol{\times}}$."
        )
    doc.append(
        r"When a checked graph is minimal under single-jump deletion, we append "
        r"$\textbf{MIN}$ right after the checkmark."
    )
    if show_all_graphs:
        doc.append(r"All non-empty jump sets are shown.")
    else:
        doc.append(
            r"Only graphs that pass the check are shown. "
        )
    if use_recipe:
        doc.append(
            r"Recipe mode: shown graphs are built directly as CORE graphs "
            r"(all active jumps except one), plus graphs obtained by adding free jumps "
            r"($a_j=0$) only to the first CORE graph. "
            r"$Y$, $Y^{+}$, and rank are computed only from this shown set."
        )
    doc.append(r"")
    
    if not display_graphs:
        doc.append(r"\noindent No graph satisfies the check for this $(t,S)$.")
    else:
        for start in range(0, len(display_graphs), cols):
            row_subsets = display_graphs[start : start + cols]
            doc.append(r"\noindent")
            row_cells: list[str] = []
            for entry in row_subsets:
                row_cells.append(
                    graph_cell_tex(
                        idx=entry.idx,
                        t=t,
                        s_set=s_set,
                        s_jump_set=s_jump_set,
                        j_list=j_list,
                        jumps_l=entry.jumps_l,
                        edges_l=entry.edges_l,
                        lhs_value=entry.lhs_value,
                        overlap_with_s=entry.overlap_count,
                        mark_ok=entry.mark_ok,
                        is_minimal=entry.is_minimal,
                        use_lift=use_lift,
                    )
                )
            doc.append(r"\hfill".join(row_cells))
            doc.append(r"\par\medskip")

    doc.append(r"")
    doc.append(r"\noindent\textbf{Matrix Data}")
    doc.append(rf"\par $N={check_count}$")
    doc.append(rf"\par $a={latex_tuple(a_coeffs)}$")
    doc.append(rf"\par $\mathrm{{rhs}}={rhs_value}$")
    doc.append(
        rf"\par $I={latex_set(check_indices)}$"
        r"\quad"
        rf"$Y={latex_matrix_from_columns(check_y_columns)}$"
    )
    doc.append(
        rf"\par $Y^{{+}}={latex_matrix_from_columns(y_plus_columns)}$"
        r"\quad"
        rf"$\operatorname{{rank}}(Y^{{+}})={y_plus_rank}$"
    )
    if y_plus_multiplier is not None:
        doc.append(
            r"\par $Y^{+}\lambda=\mathbf{0}$ with "
            rf"$\lambda={latex_column_vector(y_plus_multiplier)}$"
        )

    doc.append(r"\end{document}")
    doc.append("")
    return "\n".join(doc)


def compile_tex(tex_path: Path, verbose: bool = True) -> None:
    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            check=True,
            cwd=tex_path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if verbose:
            print(f"Compiled {tex_path.with_suffix('.pdf')}")
    except FileNotFoundError:
        if verbose:
            print("pdflatex not found; skipping compilation.")
    except subprocess.CalledProcessError:
        if verbose:
            print(f"pdflatex failed for {tex_path.name}; check the .log file.")


def clean_latex_garbage(tex_path: Path, verbose: bool = True) -> None:
    removed = 0
    for ext in (".log", ".aux"):
        file_path = tex_path.with_suffix(ext)
        if file_path.is_file():
            file_path.unlink()
            removed += 1
    if removed > 0 and verbose:
        print(f"Latex garbage cleaned ({removed} files)")


def normalize_output_path(path: Path) -> Path:
    if path.suffix == ".tex" and path.stem.endswith(".lex"):
        return path.with_name(path.stem[:-4] + ".tex")
    return path


def default_output_path(t: int, s_set: set[int], m: int, use_lift: bool) -> Path:
    s_part = ",".join(str(v) for v in sorted(s_set)) if s_set else "empty"
    suffix = "_lift" if use_lift else ""
    return Path(f"S={s_part}_m={m}_t={t}{suffix}.tex")


def append_stem_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(path.name + suffix)


def resolve_output_path(
    args: argparse.Namespace,
    case: AnalysisCase,
    use_lift: bool,
    multi_case: bool,
) -> Path:
    default_path = default_output_path(case.t, set(case.s_values), case.m, use_lift=use_lift)

    if multi_case:
        if args.output_dir is not None:
            return normalize_output_path(args.output_dir / default_path.name)
        return normalize_output_path(default_path)

    if args.output is not None:
        base_output = normalize_output_path(args.output)
        if args.lift_mode == "addlifted" and use_lift:
            return append_stem_suffix(base_output, "_lift")
        return base_output

    if args.output_dir is not None:
        return normalize_output_path(args.output_dir / default_path.name)

    return normalize_output_path(default_path)


def write_latex_output(
    summary: dict,
    output: Path,
    cols: int,
    show_all_graphs: bool,
    verbose: bool,
) -> None:
    tex = build_document(summary, cols, show_all_graphs)
    output.write_text(tex, encoding="utf-8")
    resolved_output = output.resolve()
    if verbose:
        print(f"Wrote {output}")
    compile_tex(resolved_output, verbose=verbose)
    clean_latex_garbage(resolved_output, verbose=verbose)


def run_analysis_case(
    case: AnalysisCase,
    args: argparse.Namespace,
    use_lift: bool,
    show_progress: bool,
    multi_case: bool,
) -> dict:
    summary = compute_summary(
        case.t,
        case.m,
        set(case.s_values),
        use_fraction=args.fraction,
        compute_sssp_exact=args.sssp,
        use_recipe=args.recipe,
        use_lift=use_lift,
        show_progress=show_progress,
    )
    if args.latex:
        output = resolve_output_path(args, case, use_lift=use_lift, multi_case=multi_case)
        write_latex_output(
            summary,
            output,
            cols=args.cols,
            show_all_graphs=args.show_all_graphs,
            verbose=show_progress,
        )
    return summary


def case_error_line(case: AnalysisCase, use_lift: bool, exc: Exception) -> str:
    mode = "lifted" if use_lift else "standard"
    return (
        f"t={case.t}, m={case.m}, S={plain_set(list(case.s_values))}, "
        f"ineq={mode}, ERROR={exc}"
    )


def run_single_mode(case: AnalysisCase, args: argparse.Namespace) -> None:
    if args.lift_mode == "addlifted":
        run_analysis_case(case, args, use_lift=False, show_progress=True, multi_case=False)
        run_analysis_case(case, args, use_lift=True, show_progress=True, multi_case=False)
        return

    run_analysis_case(
        case,
        args,
        use_lift=args.lift_mode == "lifted",
        show_progress=True,
        multi_case=False,
    )


def run_multi_mode(cases: list[AnalysisCase], args: argparse.Namespace) -> int:
    failures = 0

    for case in cases:
        if args.lift_mode == "addlifted":
            standard_summary: dict | None = None
            standard_line: str
            lifted_line: str

            try:
                standard_summary = run_analysis_case(
                    case,
                    args,
                    use_lift=False,
                    show_progress=False,
                    multi_case=True,
                )
                standard_line = standard_summary["final_status_line"]
            except Exception as exc:
                failures += 1
                standard_line = case_error_line(case, use_lift=False, exc=exc)
                if args.stop_on_error:
                    raise

            try:
                lifted_summary = run_analysis_case(
                    case,
                    args,
                    use_lift=True,
                    show_progress=False,
                    multi_case=True,
                )
                lifted_line = lifted_summary["final_status_line"]
                if (
                    standard_summary is not None
                    and standard_summary["status"] != "FACET"
                    and lifted_summary["status"] == "FACET"
                ):
                    lifted_line = replace_terminal_status(lifted_line, "FACET+")
            except Exception as exc:
                failures += 1
                lifted_line = case_error_line(case, use_lift=True, exc=exc)
                if args.stop_on_error:
                    raise

            print(standard_line)
            print(lifted_line)
            continue

        use_lift = args.lift_mode == "lifted"
        try:
            summary = run_analysis_case(
                case,
                args,
                use_lift=use_lift,
                show_progress=False,
                multi_case=True,
            )
            print(summary["final_status_line"])
        except Exception as exc:
            failures += 1
            print(case_error_line(case, use_lift=use_lift, exc=exc))
            if args.stop_on_error:
                raise

    return failures


def main() -> None:
    args = parse_args()
    if args.latex and args.cols < 1:
        raise ValueError("--cols must be at least 1.")

    cases = resolve_analysis_cases(args)
    multi_case = len(cases) > 1

    if not args.latex:
        if args.output is not None:
            print("Ignoring --output because --latex was not set.")
        if args.output_dir is not None:
            print("Ignoring --output-dir because --latex was not set.")
    else:
        if multi_case and args.output is not None:
            raise ValueError(
                "--output requires a single resolved (t,S,m) case; use --output-dir for sweeps."
            )
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)

    if multi_case:
        failures = run_multi_mode(cases, args)
        if failures > 0:
            raise SystemExit(1)
        return

    run_single_mode(cases[0], args)


if __name__ == "__main__":
    main()
