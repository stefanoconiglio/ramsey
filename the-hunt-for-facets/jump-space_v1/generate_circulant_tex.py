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
import csv
from dataclasses import dataclass
from fractions import Fraction
import io
import itertools
import math
import os
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


@dataclass
class GraphRecord:
    idx: int
    jumps_l: set[int]
    lhs_value: int
    overlap_count: int
    mark_ok: bool
    is_feasible: bool | None


@dataclass(frozen=True)
class AnalysisCase:
    t: int
    s_values: tuple[int, ...]
    m: int


class SkipAnalysis(Exception):
    pass


LATEX_GRAPH_COLS = 3


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
        required=True,
        help=(
            "Target vertex subset S or a size specification. "
            "Examples: '0,1,2', '{0,3,7}' for an explicit set, or '5' / '4:6' "
            "to mean |S|=5 or |S| in 4..6 and sweep all subsets containing 0."
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
        "--output-dir",
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory for auto-named output files. Defaults to ./output when "
            "--latex or --csv is enabled."
        ),
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX/PDF output. Without this flag, only stdout summary is printed.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "After LaTeX compilation, open each rendered PDF and wait for "
            "Enter to continue, q to stop, or Ctrl+C to abort. Implies --latex."
        ),
    )
    parser.add_argument(
        "--progress",
        "--show-progress",
        dest="progress",
        action="store_true",
        help="Show live scan progress and prefix summary lines with current/total.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print stdout summaries as a table with one header row and value-only data rows.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write summaries to an auto-named CSV file with SQL-friendly header names.",
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
        "--relax",
        dest="relaxation",
        action="store_true",
        help=(
            "Use the relaxed jump-vector model: do not enforce clique-feasibility "
            "of enumerated graphs. Relaxed facet labels are marked with (REL)."
        ),
    )
    parser.add_argument(
        "--recipe",
        action="store_true",
        help=(
            "Use the proof recipe in relax mode: build Y from all rhs-subsets of "
            "the active jumps (where a_j=1), then extend with one free jump "
            "(a_j=0) at a time from the first such subset. If the recipe does "
            "not apply, raise an error."
        ),
    )
    parser.add_argument(
        "--onlyuniform",
        action="store_true",
        help="Skip any standard/lifted analysis whose coefficient vector is not uniform.",
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


def parse_s_spec(raw: str, arg_name: str) -> tuple[tuple[int, ...] | None, list[int] | None]:
    stripped = raw.strip()
    if any(char in stripped for char in ",{}"):
        return parse_explicit_s_values(raw), None
    return None, parse_int_or_closed_range(stripped, arg_name)


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

    explicit_s_values, s_size_values = parse_s_spec(args.S, "--S")

    if explicit_s_values is not None and len(t_values) == 1:
        validate_explicit_s_values(explicit_s_values, t_values[0])

    if explicit_s_values is None:
        assert s_size_values is not None
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


def adjacency_masks_from_jumps(t: int, jumps: set[int]) -> list[int]:
    masks = [0 for _ in range(t)]
    for vertex in range(t):
        mask = 0
        for jump in jumps:
            plus = (vertex + jump) % t
            minus = (vertex - jump) % t
            if plus != vertex:
                mask |= 1 << plus
            if minus != vertex:
                mask |= 1 << minus
        masks[vertex] = mask
    return masks


def greedy_coloring_order(adj_masks: list[int], candidate_mask: int) -> tuple[list[int], list[int]]:
    order: list[int] = []
    color_bounds: list[int] = []
    uncolored = candidate_mask
    color = 0

    while uncolored:
        color += 1
        color_class = uncolored
        while color_class:
            vertex_bit = color_class & -color_class
            vertex = vertex_bit.bit_length() - 1
            order.append(vertex)
            color_bounds.append(color)
            uncolored &= ~vertex_bit
            color_class &= ~vertex_bit
            color_class &= ~adj_masks[vertex]

    return order, color_bounds


def has_clique_of_size(
    adj_masks: list[int],
    candidate_mask: int,
    target_size: int,
) -> bool:
    if target_size <= 0:
        return True
    if candidate_mask.bit_count() < target_size:
        return False

    def search(vertices: list[int], color_bounds: list[int], chosen_size: int) -> bool:
        local_vertices = list(vertices)
        local_bounds = list(color_bounds)

        while local_vertices:
            if chosen_size + local_bounds[-1] < target_size:
                return False

            vertex = local_vertices.pop()
            local_bounds.pop()
            if chosen_size + 1 >= target_size:
                return True

            next_mask = 0
            for other in local_vertices:
                if (adj_masks[vertex] >> other) & 1:
                    next_mask |= 1 << other

            if next_mask.bit_count() < target_size - chosen_size - 1:
                continue

            next_vertices, next_bounds = greedy_coloring_order(adj_masks, next_mask)
            if search(next_vertices, next_bounds, chosen_size + 1):
                return True

        return False

    vertices, color_bounds = greedy_coloring_order(adj_masks, candidate_mask)
    return search(vertices, color_bounds, 0)


def is_m_clique_free(t: int, jumps: set[int], m: int) -> bool:
    if m <= 1:
        return False
    if m > t:
        return True
    if not jumps:
        return True

    adj_masks = adjacency_masks_from_jumps(t, jumps)
    return not has_clique_of_size(adj_masks, adj_masks[0], m - 1)


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


def matrix_data_from_columns(
    columns: list[list[int]],
    use_fraction: bool,
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[list[int]],
    int | None,
    list[int] | list[float] | None,
]:
    rows = matrix_rows_from_columns(columns)
    y_plus_columns = [column + [1] for column in columns]
    y_plus_rows = matrix_rows_from_columns(y_plus_columns)
    if not y_plus_rows:
        return rows, y_plus_columns, y_plus_rows, None, None

    rank_y_plus = matrix_rank(y_plus_rows, use_fraction=use_fraction)
    y_plus_max_rank = min(len(y_plus_rows), len(y_plus_columns))
    multiplier = (
        nullspace_vector(y_plus_rows, use_fraction=use_fraction)
        if rank_y_plus < y_plus_max_rank
        else None
    )
    return rows, y_plus_columns, y_plus_rows, rank_y_plus, multiplier


def plain_set(values: list[int]) -> str:
    if not values:
        return "{}"
    return "{" + ",".join(str(v) for v in values) + "}"


def facet_status(passed: bool) -> str:
    return "FACET" if passed else "FAILURE"


def terminal_status_label(
    passed: bool,
    use_relaxation: bool,
    facet_label: str | None = None,
) -> str:
    terminal_status = facet_status(passed)
    if facet_label is not None and terminal_status == "FACET":
        terminal_status = facet_label
    if use_relaxation:
        if terminal_status == "FAILURE":
            return "FAILURE"
        return terminal_status.replace("FACET", "FACET(REL)", 1)
    return terminal_status


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


def sssp_value(a_coeffs: list[int], rhs_value: int) -> str:
    nonzero_values = [value for value in a_coeffs if value != 0]

    # Fast check 1: only one nonzero coefficient with parity mismatch (and rhs > 0) => impossible.
    if len(nonzero_values) == 1 and rhs_value > 0:
        coeff = nonzero_values[0]
        if (coeff % 2) != (rhs_value % 2):
            return "0"

    # Fast check 2: all coefficients even and rhs odd => impossible.
    if all(value % 2 == 0 for value in a_coeffs) and (rhs_value % 2 == 1):
        return "0"

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


def raw_coefficients_for_case(case: AnalysisCase) -> tuple[list[int], int, int]:
    j_max = case.t // 2
    j_list = list(range(1, j_max + 1))
    jump_to_idx = {jump: idx for idx, jump in enumerate(j_list)}
    raw_coeffs = [0 for _ in j_list]
    for i, j in edges_from_vertex_subset(set(case.s_values)):
        jump = jump_distance(i, j, case.t)
        raw_coeffs[jump_to_idx[jump]] += 1
    return raw_coeffs, len(case.s_values), turan_bound(len(case.s_values), case.m)


def analysis_uniform_flag(case: AnalysisCase, use_lift: bool) -> int:
    raw_coeffs, s_size, d_m_s = raw_coefficients_for_case(case)
    return uniform_coefficients_flag(
        build_inequality_data(
            raw_coeffs=raw_coeffs,
            s_size=s_size,
            d_m_s=d_m_s,
            use_lift=use_lift,
        ).coeffs
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
    rhs_value: int,
) -> list[set[int]]:
    """Build the proof-recipe jump sets for 0/1 coefficient vectors."""
    if any(coeff not in (0, 1) for coeff in a_coeffs):
        raise ValueError(
            "recipe requires a 0/1 coefficient vector (active jumps with a_j=1, free jumps with a_j=0)"
        )

    active_jumps = [jump for jump, coeff in zip(j_list, a_coeffs) if coeff == 1]
    free_jumps = [jump for jump, coeff in zip(j_list, a_coeffs) if coeff == 0]
    active_count = len(active_jumps)

    if active_count < 2 or rhs_value < 1 or rhs_value >= active_count:
        raise ValueError(
            f"recipe requires 1 <= rhs < number of active jumps and at least two active jumps; "
            f"got rhs={rhs_value}, active_count={active_count}"
        )

    base_sets = [set(combo) for combo in itertools.combinations(active_jumps, rhs_value)]
    if not base_sets:
        raise ValueError("recipe could not build any base rhs-subsets of the active jumps")

    recipe_sets = list(base_sets)
    first_base = set(base_sets[0])
    for free_jump in free_jumps:
        lifted = set(first_base)
        lifted.add(free_jump)
        recipe_sets.append(lifted)

    return recipe_sets


def status_line(
    current: int,
    total: int,
    t: int,
    m: int,
    thm2: int,
    s_sorted: list[int],
    j_max: int,
    tight_count: int,
    feasible_tight_count: int | None,
    use_relaxation: bool,
) -> str:
    feasibility_part = ""
    if not use_relaxation and feasible_tight_count is not None:
        feasibility_part = f", N.feas={feasible_tight_count}"
    return (
        f"{current}/{total} "
        f"t={t}, m={m}, thm2={thm2}, S={plain_set(s_sorted)}, floor(t/2)={j_max}, "
        f"N.tight={tight_count}{feasibility_part}"
    )


def gate_symbol(passed: bool | None) -> str:
    if passed is None:
        return "-"
    return "1" if passed else "0"


def gate_ratio(value: int | None, target: int) -> str:
    if value is None:
        return "-"
    return f"{value}/{target}"


def progress_prefix(current: int, total: int, include_progress_prefix: bool) -> str:
    return f"{current}/{total} " if include_progress_prefix else ""


def format_status_fields(
    fields: list[tuple[str, str]],
    terminal_status: str,
    widths: list[int] | None = None,
    prefix: str = "",
) -> str:
    chunks = [f"{label}={value}" for label, value in fields]
    if widths is not None:
        padded_chunks = [
            f"{chunk},".ljust(widths[idx] + 1)
            for idx, chunk in enumerate(chunks)
        ]
        body = " ".join(padded_chunks)
        return f"{prefix}{body} {terminal_status}"
    body = ", ".join(chunks)
    return f"{prefix}{body}, {terminal_status}"


def status_field_widths(entries: list[list[tuple[str, str]]]) -> list[int]:
    if not entries:
        return []
    field_count = len(entries[0])
    widths = [0 for _ in range(field_count)]
    for fields in entries:
        for idx, (label, value) in enumerate(fields):
            widths[idx] = max(widths[idx], len(f"{label}={value}"))
    return widths


def format_status_table(
    rendered_lines: list[tuple[str, list[tuple[str, str]], str]],
    include_progress_prefix: bool,
) -> str:
    if not rendered_lines:
        return ""

    headers: list[str] = []
    if include_progress_prefix:
        headers.append("scan")
    headers.extend(label for label, _ in rendered_lines[0][1])
    headers.append("status")

    rows: list[list[str]] = []
    for prefix, fields, terminal_status in rendered_lines:
        row: list[str] = []
        if include_progress_prefix:
            row.append(prefix.strip())
        row.extend(value for _, value in fields)
        row.append(terminal_status)
        rows.append(row)

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def csv_header_names() -> list[str]:
    return [
        "t",
        "floor_t_over_2",
        "m",
        "s",
        "thm2",
        "d_m_s",
        "ineq",
        "aTy",
        "rhs",
        "max_lhs",
        "sssp",
        "n_tight",
        "rank_tight",
        "n_feas",
        "rank_feas",
        "chk",
        "uniform",
        "status",
    ]


def format_status_csv(
    rendered_lines: list[tuple[str, list[tuple[str, str]], str]],
) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(csv_header_names())
    for _, fields, terminal_status in rendered_lines:
        writer.writerow([value for _, value in fields] + [terminal_status])
    return buffer.getvalue().rstrip("\n")


def build_status_fields(
    t: int,
    m: int,
    thm2: int,
    s_sorted: list[int],
    j_max: int,
    tight_count: int | None,
    tight_rank: int | None,
    feasible_tight_count: int | None,
    feasible_tight_rank: int | None,
    uniform: int,
    a_coeffs: list[int],
    d_m_s: int,
    rhs_value: int,
    sssp: str,
    tier_results: tuple[bool | None, bool | None, bool | None, bool | None, bool | None],
    use_lift: bool,
    use_relaxation: bool,
    facet_label: str | None = None,
) -> tuple[list[tuple[str, str]], str]:
    coeffs_str = ",".join(str(value) for value in a_coeffs)
    max_lhs = sum(a_coeffs)
    mode = "lifted" if use_lift else "standard"
    final_gate = tier_results[2] if use_relaxation else tier_results[4]
    terminal_status = terminal_status_label(
        passed=(final_gate is True),
        use_relaxation=use_relaxation,
        facet_label=facet_label,
    )
    tiers_str = ",".join(gate_symbol(result) for result in tier_results)
    fields = [
        ("t", str(t)),
        ("floor(t/2)", str(j_max)),
        ("m", str(m)),
        ("S", plain_set(s_sorted)),
        ("thm2", str(thm2)),
        ("d_m(|S|)", str(d_m_s)),
        ("ineq", mode),
        ("a'y", f"({coeffs_str})'y"),
        ("rhs", str(rhs_value)),
        ("maxLHS", str(max_lhs)),
        ("SSSP", sssp),
        ("N.tight", gate_ratio(tight_count, j_max)),
        ("rank.tight", gate_ratio(tight_rank, j_max)),
        ("N.feas", gate_ratio(feasible_tight_count, j_max)),
        ("rank.feas", gate_ratio(feasible_tight_rank, j_max)),
        ("chk", f"({tiers_str})"),
        ("uniform", str(uniform)),
    ]
    return fields, terminal_status


def uniform_coefficients_flag(a_coeffs: list[int]) -> int:
    nonzero_coeffs = {coeff for coeff in a_coeffs if coeff != 0}
    return 1 if len(nonzero_coeffs) <= 1 else 0


def final_status_line(
    current: int,
    total: int,
    t: int,
    m: int,
    thm2: int,
    s_sorted: list[int],
    j_max: int,
    tight_count: int | None,
    tight_rank: int | None,
    feasible_tight_count: int | None,
    feasible_tight_rank: int | None,
    uniform: int,
    a_coeffs: list[int],
    d_m_s: int,
    rhs_value: int,
    sssp: str,
    tier_results: tuple[bool | None, bool | None, bool | None, bool | None, bool | None],
    use_lift: bool,
    use_relaxation: bool,
    include_progress_prefix: bool,
    facet_label: str | None = None,
) -> str:
    fields, terminal_status = build_status_fields(
        t=t,
        m=m,
        thm2=thm2,
        s_sorted=s_sorted,
        j_max=j_max,
        tight_count=tight_count,
        tight_rank=tight_rank,
        feasible_tight_count=feasible_tight_count,
        feasible_tight_rank=feasible_tight_rank,
        uniform=uniform,
        a_coeffs=a_coeffs,
        d_m_s=d_m_s,
        rhs_value=rhs_value,
        sssp=sssp,
        tier_results=tier_results,
        use_lift=use_lift,
        use_relaxation=use_relaxation,
        facet_label=facet_label,
    )
    return format_status_fields(
        fields,
        terminal_status,
        prefix=progress_prefix(current, total, include_progress_prefix),
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
    use_recipe: bool,
    use_relaxation: bool,
    use_lift: bool,
    show_progress: bool = False,
    print_final_status: bool = False,
    include_progress_prefix: bool = False,
    facet_label: str | None = None,
) -> dict:
    s_size = len(s_set)
    thm2 = thm2_value(s_size, m)
    d_m_s = turan_bound(s_size, m)
    j_max = t // 2
    j_list = list(range(1, j_max + 1))
    s_sorted = sorted(s_set)
    s_edges = edges_from_vertex_subset(s_set)
    s_jump_set = {jump_distance(i, j, t) for i, j in s_edges}
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
    uniform = uniform_coefficients_flag(a_coeffs)
    rhs_value = inequality.rhs
    if use_recipe:
        recipe_subsets = build_recipe_jump_sets(
            j_list=j_list,
            a_coeffs=a_coeffs,
            rhs_value=rhs_value,
        )
        total_y_vectors = len(recipe_subsets)
    else:
        total_y_vectors = (1 << len(j_list)) - 1

    sssp = sssp_value(a_coeffs, rhs_value)
    graph_data: list[GraphRecord] = []
    last_status_len = 0
    if sssp == "0":
        tier_results = (False, None, None, None, None)
        status_fields, terminal_status = build_status_fields(
            t=t,
            m=m,
            thm2=thm2,
            s_sorted=s_sorted,
            j_max=j_max,
            tight_count=None,
            tight_rank=None,
            feasible_tight_count=None,
            feasible_tight_rank=None,
            uniform=uniform,
            a_coeffs=a_coeffs,
            d_m_s=d_m_s,
            rhs_value=rhs_value,
            sssp=sssp,
            tier_results=tier_results,
            use_lift=use_lift,
            use_relaxation=use_relaxation,
            facet_label=facet_label,
        )
        final_status = final_status_line(
            current=0,
            total=total_y_vectors,
            t=t,
            m=m,
            thm2=thm2,
            s_sorted=s_sorted,
            j_max=j_max,
            tight_count=None,
            tight_rank=None,
            feasible_tight_count=None,
            feasible_tight_rank=None,
            uniform=uniform,
            a_coeffs=a_coeffs,
            d_m_s=d_m_s,
            rhs_value=rhs_value,
            sssp=sssp,
            tier_results=tier_results,
            use_lift=use_lift,
            use_relaxation=use_relaxation,
            include_progress_prefix=include_progress_prefix,
            facet_label=facet_label,
        )
        if print_final_status:
            print(final_status)
        return {
            "t": t,
            "m": m,
            "thm2": thm2,
            "d_m_s": d_m_s,
            "rhs_value": rhs_value,
            "sssp": sssp,
            "s_size": s_size,
            "uniform": uniform,
            "a_coeffs": a_coeffs,
            "raw_coeffs": raw_coeffs,
            "pair_count": inequality.pair_count,
            "lift_cap": inequality.lift_cap,
            "s_sorted": s_sorted,
            "s_jump_set": s_jump_set,
            "j_max": j_max,
            "j_list": j_list,
            "total_y_vectors": total_y_vectors,
            "graph_data": [],
            "selected_graph_data": [],
            "tight_graph_data": [],
            "feasible_tight_graph_data": [],
            "use_recipe": use_recipe,
            "use_relaxation": use_relaxation,
            "use_lift": use_lift,
            "check_count": 0,
            "tight_count": None,
            "tight_y_columns": [],
            "tight_y_rows": [],
            "tight_y_plus_columns": [],
            "tight_y_plus_rows": [],
            "tight_y_plus_rank": None,
            "tight_y_plus_multiplier": None,
            "feasible_tight_count": None,
            "feasible_tight_y_columns": [],
            "feasible_tight_y_rows": [],
            "feasible_tight_y_plus_columns": [],
            "feasible_tight_y_plus_rows": [],
            "feasible_tight_y_plus_rank": None,
            "feasible_tight_y_plus_multiplier": None,
            "check_y_columns": [],
            "y_rows": [],
            "y_plus_columns": [],
            "y_plus_rows": [],
            "y_plus_rank": None,
            "y_plus_multiplier": None,
            "tier_results": tier_results,
            "status": terminal_status,
            "status_fields": status_fields,
            "terminal_status": terminal_status,
            "final_progress_prefix": progress_prefix(0, total_y_vectors, include_progress_prefix),
            "final_status_line": final_status,
        }

    progress_tight_count = 0
    progress_feasible_tight_count: int | None = 0 if use_relaxation else None
    if use_recipe:
        for graph_idx, jumps_l in enumerate(recipe_subsets, start=1):
            overlap_count = linear_form_value(raw_coeffs, jumps_l, jump_to_idx)
            lhs_value = linear_form_value(a_coeffs, jumps_l, jump_to_idx)
            mark_ok = lhs_value == rhs_value
            if not mark_ok:
                raise ValueError(
                    f"recipe produced a non-tight jump set {plain_set(sorted(jumps_l))}: "
                    f"lhs={lhs_value}, rhs={rhs_value}"
                )
            is_feasible = True if use_relaxation else is_m_clique_free(t, jumps_l, m)
            if not use_relaxation and not is_feasible:
                raise ValueError(
                    f"recipe produced an infeasible jump set {plain_set(sorted(jumps_l))}"
                )
            if mark_ok:
                progress_tight_count += 1
            if use_relaxation and mark_ok and is_feasible and progress_feasible_tight_count is not None:
                progress_feasible_tight_count += 1
            graph_data.append(
                GraphRecord(
                    idx=graph_idx,
                    jumps_l=jumps_l,
                    lhs_value=lhs_value,
                    overlap_count=overlap_count,
                    mark_ok=mark_ok,
                    is_feasible=is_feasible,
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
                    tight_count=progress_tight_count,
                    feasible_tight_count=progress_feasible_tight_count,
                    use_relaxation=use_relaxation,
                )
                last_status_len = print_in_place(current_status, last_status_len)
    else:
        subsets = all_nonempty_jump_sets(j_list)

        for graph_idx, jumps_l in enumerate(subsets, start=1):
            overlap_count = linear_form_value(raw_coeffs, jumps_l, jump_to_idx)
            lhs_value = linear_form_value(a_coeffs, jumps_l, jump_to_idx)
            mark_ok = lhs_value == rhs_value
            is_feasible = True if use_relaxation else None
            if mark_ok:
                progress_tight_count += 1
            if use_relaxation and mark_ok and is_feasible and progress_feasible_tight_count is not None:
                progress_feasible_tight_count += 1
            graph_data.append(
                GraphRecord(
                    idx=graph_idx,
                    jumps_l=jumps_l,
                    lhs_value=lhs_value,
                    overlap_count=overlap_count,
                    mark_ok=mark_ok,
                    is_feasible=is_feasible,
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
                    tight_count=progress_tight_count,
                    feasible_tight_count=progress_feasible_tight_count,
                    use_relaxation=use_relaxation,
                )
                last_status_len = print_in_place(current_status, last_status_len)

    tight_graphs = [entry for entry in graph_data if entry.mark_ok]
    tight_count = len(tight_graphs)
    tight_y_columns = [jump_y_vector(j_list, entry.jumps_l) for entry in tight_graphs]
    tight_y_rows = matrix_rows_from_columns(tight_y_columns)
    tight_y_plus_columns = [column + [1] for column in tight_y_columns]
    tight_y_plus_rows = matrix_rows_from_columns(tight_y_plus_columns)
    tight_y_plus_rank: int | None = None
    tight_y_plus_multiplier: list[int] | list[float] | None = None

    feasible_tight_graphs: list[GraphRecord] = []
    feasible_tight_count: int | None = None
    feasible_tight_y_columns: list[list[int]] = []
    feasible_tight_y_rows: list[list[int]] = []
    feasible_tight_y_plus_columns: list[list[int]] = []
    feasible_tight_y_plus_rows: list[list[int]] = []
    feasible_tight_y_plus_rank: int | None = None
    feasible_tight_y_plus_multiplier: list[int] | list[float] | None = None

    tier_1 = True
    tier_2 = tight_count >= j_max
    tier_3: bool | None = None
    tier_4: bool | None = None
    tier_5: bool | None = None

    if tier_2:
        (
            tight_y_rows,
            tight_y_plus_columns,
            tight_y_plus_rows,
            tight_y_plus_rank,
            tight_y_plus_multiplier,
        ) = matrix_data_from_columns(tight_y_columns, use_fraction=use_fraction)
        tier_3 = tight_y_plus_rank == j_max
        if not use_relaxation and tier_3:
            feasible_tight_graphs = []
            for entry in tight_graphs:
                entry.is_feasible = is_m_clique_free(t, entry.jumps_l, m)
                if entry.is_feasible:
                    feasible_tight_graphs.append(entry)
            feasible_tight_count = len(feasible_tight_graphs)
            feasible_tight_y_columns = [
                jump_y_vector(j_list, entry.jumps_l) for entry in feasible_tight_graphs
            ]
            tier_4 = feasible_tight_count >= j_max
            if tier_4:
                (
                    feasible_tight_y_rows,
                    feasible_tight_y_plus_columns,
                    feasible_tight_y_plus_rows,
                    feasible_tight_y_plus_rank,
                    feasible_tight_y_plus_multiplier,
                ) = matrix_data_from_columns(
                    feasible_tight_y_columns,
                    use_fraction=use_fraction,
                )
                tier_5 = feasible_tight_y_plus_rank == j_max

    if use_relaxation:
        selected_graphs = tight_graphs
        check_count = tight_count
        check_y_columns = tight_y_columns
        y_rows = tight_y_rows
        y_plus_columns = tight_y_plus_columns
        y_plus_rows = tight_y_plus_rows
        y_plus_rank = tight_y_plus_rank
        y_plus_multiplier = tight_y_plus_multiplier
    elif tier_4 is not None:
        selected_graphs = feasible_tight_graphs
        check_count = feasible_tight_count if feasible_tight_count is not None else 0
        check_y_columns = feasible_tight_y_columns
        y_rows = feasible_tight_y_rows
        y_plus_columns = feasible_tight_y_plus_columns
        y_plus_rows = feasible_tight_y_plus_rows
        y_plus_rank = feasible_tight_y_plus_rank
        y_plus_multiplier = feasible_tight_y_plus_multiplier
    else:
        selected_graphs = tight_graphs
        check_count = tight_count
        check_y_columns = tight_y_columns
        y_rows = tight_y_rows
        y_plus_columns = tight_y_plus_columns
        y_plus_rows = tight_y_plus_rows
        y_plus_rank = tight_y_plus_rank
        y_plus_multiplier = tight_y_plus_multiplier

    tier_results = (tier_1, tier_2, tier_3, tier_4, tier_5)
    status_fields, terminal_status = build_status_fields(
        t=t,
        m=m,
        thm2=thm2,
        s_sorted=s_sorted,
        j_max=j_max,
        tight_count=tight_count,
        tight_rank=tight_y_plus_rank,
        feasible_tight_count=feasible_tight_count,
        feasible_tight_rank=feasible_tight_y_plus_rank,
        uniform=uniform,
        a_coeffs=a_coeffs,
        d_m_s=d_m_s,
        rhs_value=rhs_value,
        sssp=sssp,
        tier_results=tier_results,
        use_lift=use_lift,
        use_relaxation=use_relaxation,
        facet_label=facet_label,
    )
    final_status = final_status_line(
        current=total_y_vectors,
        total=total_y_vectors,
        t=t,
        m=m,
        thm2=thm2,
        s_sorted=s_sorted,
        j_max=j_max,
        tight_count=tight_count,
        tight_rank=tight_y_plus_rank,
        feasible_tight_count=feasible_tight_count,
        feasible_tight_rank=feasible_tight_y_plus_rank,
        uniform=uniform,
        a_coeffs=a_coeffs,
        d_m_s=d_m_s,
        rhs_value=rhs_value,
        sssp=sssp,
        tier_results=tier_results,
        use_lift=use_lift,
        use_relaxation=use_relaxation,
        include_progress_prefix=include_progress_prefix,
        facet_label=facet_label,
    )
    if print_final_status:
        if show_progress and total_y_vectors > 0:
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
        "uniform": uniform,
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
        "tight_graph_data": tight_graphs,
        "feasible_tight_graph_data": feasible_tight_graphs,
        "use_recipe": use_recipe,
        "use_relaxation": use_relaxation,
        "use_lift": use_lift,
        "check_count": check_count,
        "tight_count": tight_count,
        "tight_y_columns": tight_y_columns,
        "tight_y_rows": tight_y_rows,
        "tight_y_plus_columns": tight_y_plus_columns,
        "tight_y_plus_rows": tight_y_plus_rows,
        "tight_y_plus_rank": tight_y_plus_rank,
        "tight_y_plus_multiplier": tight_y_plus_multiplier,
        "feasible_tight_count": feasible_tight_count,
        "feasible_tight_y_columns": feasible_tight_y_columns,
        "feasible_tight_y_rows": feasible_tight_y_rows,
        "feasible_tight_y_plus_columns": feasible_tight_y_plus_columns,
        "feasible_tight_y_plus_rows": feasible_tight_y_plus_rows,
        "feasible_tight_y_plus_rank": feasible_tight_y_plus_rank,
        "feasible_tight_y_plus_multiplier": feasible_tight_y_plus_multiplier,
        "check_y_columns": check_y_columns,
        "y_rows": y_rows,
        "y_plus_columns": y_plus_columns,
        "y_plus_rows": y_plus_rows,
        "y_plus_rank": y_plus_rank,
        "y_plus_multiplier": y_plus_multiplier,
        "tier_results": tier_results,
        "status": terminal_status,
        "status_fields": status_fields,
        "terminal_status": terminal_status,
        "final_progress_prefix": progress_prefix(
            total_y_vectors,
            total_y_vectors,
            include_progress_prefix,
        ),
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
    lhs_value: int,
    overlap_with_s: int,
    mark_ok: bool,
    is_feasible: bool | None,
    use_relaxation: bool,
    use_lift: bool,
    width: str = "0.31\\textwidth",
) -> str:
    coords = node_positions(t)
    l_sorted = sorted(jumps_l)
    l_colored = latex_set_highlight(l_sorted, s_jump_set)
    y_vector_colored = latex_y_vector_highlight(j_list, jumps_l, s_jump_set)
    edges_l = undirected_edges_from_jumps(t, jumps_l)

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
    if use_lift:
        lines.append(rf"\par $a^\top y={lhs_value}\;\;{mark_symbol}$")
        lines.append(rf"\par $|E(L)\cap E(S)|={overlap_with_s}$")
    else:
        lines.append(rf"\par $|E(L)\cap E(S)|={lhs_value}\;\;{mark_symbol}$")
    if not use_relaxation and is_feasible is False:
        lines.append(r"\par \textcolor{red}{\textbf{INFEASIBLE}}")
    lines.append(r"\end{minipage}")
    return "\n".join(lines)


def build_document(summary: dict, show_all_graphs: bool) -> str:
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
    tight_graph_data = summary["tight_graph_data"]
    feasible_tight_graph_data = summary["feasible_tight_graph_data"]
    use_recipe = summary["use_recipe"]
    use_relaxation = summary["use_relaxation"]
    use_lift = summary["use_lift"]
    tight_count = summary["tight_count"]
    tight_y_columns = summary["tight_y_columns"]
    tight_y_plus_columns = summary["tight_y_plus_columns"]
    tight_y_plus_rank = summary["tight_y_plus_rank"]
    tight_y_plus_multiplier = summary["tight_y_plus_multiplier"]
    feasible_tight_count = summary["feasible_tight_count"]
    feasible_tight_y_columns = summary["feasible_tight_y_columns"]
    feasible_tight_y_plus_columns = summary["feasible_tight_y_plus_columns"]
    feasible_tight_y_plus_rank = summary["feasible_tight_y_plus_rank"]
    feasible_tight_y_plus_multiplier = summary["feasible_tight_y_plus_multiplier"]
    tier_results = summary["tier_results"]
    status = summary["status"]
    s_set = set(s_sorted)

    if show_all_graphs:
        display_graphs = graph_data
    else:
        display_graphs = [entry for entry in graph_data if entry.mark_ok]
    max_matrix_cols = max(
        10,
        len(tight_y_columns),
        len(tight_y_plus_columns),
        len(feasible_tight_y_columns),
        len(feasible_tight_y_plus_columns),
    )
    mode_label = "lifted" if use_lift else "standard"
    feasibility_label = "relax" if use_relaxation else "strict"
    tiers_str = ",".join(gate_symbol(result) for result in tier_results)
    tight_rank_str = gate_ratio(tight_y_plus_rank, j_max)
    feasible_count_str = gate_ratio(feasible_tight_count, j_max)
    feasible_rank_str = gate_ratio(feasible_tight_y_plus_rank, j_max)

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
    doc.append(
        rf"\section*{{$S={latex_set(s_sorted)}$, $m={m}$, $t={t}$, "
        rf"$\left\lfloor t/2\right\rfloor={j_max}$}}"
    )
    doc.append(r"\begin{center}")
    doc.append(rf"$\mathbf{{{status}}}$")
    doc.append(r"\end{center}")
    doc.append(rf"$J={latex_set(j_list)}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$\text{{chk}}=({tiers_str})$")
    doc.append(r"\quad\quad")
    doc.append(rf"$N_{{\mathrm{{tight}}}}={gate_ratio(tight_count, j_max)}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$\operatorname{{rank}}(Y_{{\mathrm{{tight}}}}^{{+}})={tight_rank_str}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$N_{{\mathrm{{feas}}}}={feasible_count_str}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$\operatorname{{rank}}(Y_{{\mathrm{{feas}}}}^{{+}})={feasible_rank_str}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$\text{{mode}}=\text{{{mode_label}}}$")
    doc.append(r"\quad\quad")
    doc.append(rf"$\text{{feasibility}}=\text{{{feasibility_label}}}$")
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
        r"The gate vector is "
        r"$(\mathrm{SSSP},\ \#\mathrm{tight},\ \operatorname{rank}(Y_{\mathrm{tight}}^+),\ "
        r"\#\mathrm{feas\ tight},\ \operatorname{rank}(Y_{\mathrm{feas}}^+))$, "
        r"and a dash means the gate was not checked because an earlier one failed."
    )
    if not use_relaxation:
        doc.append(
            r"In strict mode, a graph is feasible iff it is $K_m$-free. "
            r"Any displayed infeasible graph is explicitly marked "
            r"$\textcolor{red}{\textbf{INFEASIBLE}}$."
        )
    if show_all_graphs:
        doc.append(r"All non-empty jump sets are shown.")
    else:
        doc.append(
            r"Only graphs that satisfy the inequality at equality are shown. "
        )
    if use_recipe:
        doc.append(
            r"Recipe mode: let the active jumps be those with $a_j=1$ and the free jumps "
            r"be those with $a_j=0$. We build $Y$ from all rhs-subsets of the active jumps, "
            r"then add one free jump at a time to the first such subset to obtain the extra "
            r"columns used for the rank witness."
        )
    doc.append(r"")
    
    if not display_graphs:
        doc.append(r"\noindent No graph satisfies the check for this $(t,S)$.")
    else:
        for start in range(0, len(display_graphs), LATEX_GRAPH_COLS):
            row_subsets = display_graphs[start : start + LATEX_GRAPH_COLS]
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
                        lhs_value=entry.lhs_value,
                        overlap_with_s=entry.overlap_count,
                        mark_ok=entry.mark_ok,
                        is_feasible=entry.is_feasible,
                        use_relaxation=use_relaxation,
                        use_lift=use_lift,
                    )
                )
            doc.append(r"\hfill".join(row_cells))
            doc.append(r"\par\medskip")

    doc.append(r"")
    doc.append(r"\noindent\textbf{Matrix Data}")
    doc.append(rf"\par $Y_{{\mathrm{{tight}}}}={latex_matrix_from_columns(tight_y_columns)}$")
    doc.append(
        rf"\par $Y_{{\mathrm{{tight}}}}^{{+}}={latex_matrix_from_columns(tight_y_plus_columns)}$"
        r"\quad"
        rf"$\operatorname{{rank}}(Y_{{\mathrm{{tight}}}}^{{+}})={tight_rank_str}$"
    )
    if tight_y_plus_multiplier is not None:
        doc.append(
            r"\par $Y_{\mathrm{tight}}^{+}\lambda=\mathbf{0}$ with "
            rf"$\lambda={latex_column_vector(tight_y_plus_multiplier)}$"
        )
    doc.append(
        rf"\par $Y_{{\mathrm{{feas}}}}={latex_matrix_from_columns(feasible_tight_y_columns)}$"
    )
    doc.append(
        rf"\par $Y_{{\mathrm{{feas}}}}^{{+}}={latex_matrix_from_columns(feasible_tight_y_plus_columns)}$"
        r"\quad"
        rf"$\operatorname{{rank}}(Y_{{\mathrm{{feas}}}}^{{+}})={feasible_rank_str}$"
    )
    if feasible_tight_y_plus_multiplier is not None:
        doc.append(
            r"\par $Y_{\mathrm{feas}}^{+}\lambda=\mathbf{0}$ with "
            rf"$\lambda={latex_column_vector(feasible_tight_y_plus_multiplier)}$"
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


def show_pdf(pdf_path: Path, verbose: bool = True) -> bool:
    if not pdf_path.is_file():
        if verbose:
            print(f"PDF {pdf_path} not found; skipping display.")
        return False

    try:
        if sys.platform == "darwin":
            subprocess.Popen(
                ["open", str(pdf_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        elif os.name == "nt":
            os.startfile(str(pdf_path))
        else:
            subprocess.Popen(
                ["xdg-open", str(pdf_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        if verbose:
            print(f"Displayed {pdf_path}")
        return True
    except FileNotFoundError:
        if verbose:
            print("No PDF opener found; skipping display.")
        return False
    except OSError as exc:
        if verbose:
            print(f"Could not display {pdf_path}: {exc}")
        return False


def wait_for_show_command() -> None:
    while True:
        response = input("Press Enter for next PDF, q to stop: ").strip().lower()
        if response == "":
            return
        if response == "q":
            raise SystemExit(0)
        print("Press Enter to continue or q to stop.")


def normalize_output_path(path: Path) -> Path:
    if path.suffix == ".tex" and path.stem.endswith(".lex"):
        return path.with_name(path.stem[:-4] + ".tex")
    return path


def default_output_path(
    t: int,
    s_set: set[int],
    m: int,
    use_lift: bool,
    use_relaxation: bool,
) -> Path:
    s_part = ",".join(str(v) for v in sorted(s_set)) if s_set else "empty"
    suffix = "_relax" if use_relaxation else ""
    if use_lift:
        suffix += "_lift"
    return Path(f"S={s_part}_m={m}_t={t}{suffix}.tex")


def resolve_output_path(
    args: argparse.Namespace,
    case: AnalysisCase,
    use_lift: bool,
) -> Path:
    default_path = default_output_path(
        case.t,
        set(case.s_values),
        case.m,
        use_lift=use_lift,
        use_relaxation=args.relaxation,
    )

    assert args.output_dir is not None
    return normalize_output_path(args.output_dir / default_path.name)


def canonical_range_spec(spec: str, arg_name: str) -> str:
    values = parse_int_or_closed_range(spec, arg_name)
    if len(values) == 1:
        return str(values[0])
    return f"{values[0]}-{values[-1]}"


def canonical_s_spec_for_filename(spec: str) -> str:
    explicit_s_values, s_size_values = parse_s_spec(spec, "--S")
    if explicit_s_values is not None:
        return ",".join(str(value) for value in explicit_s_values)
    assert s_size_values is not None
    if len(s_size_values) == 1:
        return str(s_size_values[0])
    return f"{s_size_values[0]}-{s_size_values[-1]}"


def sanitize_filename_fragment(value: str) -> str:
    sanitized = value.strip().replace(" ", "")
    for source, target in (
        (":", "-"),
        ("/", "_"),
        ("\\", "_"),
        ("{", ""),
        ("}", ""),
        ("'", ""),
        ('"', ""),
    ):
        sanitized = sanitized.replace(source, target)
    return sanitized or "empty"


def resolve_csv_output_path(args: argparse.Namespace) -> Path:
    m_spec = canonical_range_spec(args.m, "--m") if args.m is not None else "auto"
    parts = [
        f"t={canonical_range_spec(args.t, '--t')}",
        f"S={canonical_s_spec_for_filename(args.S)}",
        f"m={m_spec}",
        f"ineq={args.lift_mode}",
    ]
    if args.relaxation:
        parts.append("relax")
    if args.recipe:
        parts.append("recipe")
    if args.onlyuniform:
        parts.append("uniform")
    filename = "_".join(sanitize_filename_fragment(part) for part in parts) + ".csv"
    assert args.output_dir is not None
    return args.output_dir / filename


def write_latex_output(
    summary: dict,
    output: Path,
    show_all_graphs: bool,
    verbose: bool,
    show_pdf_after: bool,
) -> None:
    tex = build_document(summary, show_all_graphs)
    output.write_text(tex, encoding="utf-8")
    resolved_output = output.resolve()
    if verbose:
        print(f"Wrote {output}")
    compile_tex(resolved_output, verbose=verbose)
    clean_latex_garbage(resolved_output, verbose=verbose)
    if show_pdf_after:
        pdf_path = resolved_output.with_suffix(".pdf")
        if show_pdf(pdf_path, verbose=verbose):
            wait_for_show_command()


def write_csv_output(
    rendered_lines: list[tuple[str, list[tuple[str, str]], str]],
    output: Path,
) -> None:
    csv_text = format_status_csv(rendered_lines)
    output.write_text(csv_text + "\n", encoding="utf-8")
    print(f"Results dumped to {output}")


def run_analysis_case(
    case: AnalysisCase,
    args: argparse.Namespace,
    use_lift: bool,
    show_progress: bool,
    print_final_status: bool,
    include_progress_prefix: bool,
    multi_case: bool,
    facet_label: str | None = None,
) -> dict:
    if args.onlyuniform and analysis_uniform_flag(case, use_lift) != 1:
        raise SkipAnalysis()

    summary = compute_summary(
        case.t,
        case.m,
        set(case.s_values),
        use_fraction=args.fraction,
        use_recipe=args.recipe,
        use_relaxation=args.relaxation,
        use_lift=use_lift,
        show_progress=show_progress,
        print_final_status=print_final_status,
        include_progress_prefix=include_progress_prefix,
        facet_label=facet_label,
    )
    if args.latex:
        output = resolve_output_path(args, case, use_lift=use_lift)
        write_latex_output(
            summary,
            output,
            show_all_graphs=args.show_all_graphs,
            verbose=print_final_status,
            show_pdf_after=args.show,
        )
    return summary


def case_error_line(
    case: AnalysisCase,
    use_lift: bool,
    include_progress_prefix: bool,
) -> str:
    fields, terminal_status = case_error_fields(case, use_lift=use_lift)
    return format_status_fields(
        fields,
        terminal_status,
        prefix="-/- " if include_progress_prefix else "",
    )


def case_error_fields(
    case: AnalysisCase,
    use_lift: bool,
) -> tuple[list[tuple[str, str]], str]:
    s_set = set(case.s_values)
    s_size = len(s_set)
    thm2 = thm2_value(s_size, case.m)
    d_m_s = turan_bound(s_size, case.m)
    j_max = case.t // 2
    j_list = list(range(1, j_max + 1))
    s_edges = edges_from_vertex_subset(s_set)
    jump_to_idx = {jump: idx for idx, jump in enumerate(j_list)}
    raw_coeffs = [0 for _ in j_list]
    for i, j in s_edges:
        jump = jump_distance(i, j, case.t)
        raw_coeffs[jump_to_idx[jump]] += 1

    inequality = build_inequality_data(
        raw_coeffs=raw_coeffs,
        s_size=s_size,
        d_m_s=d_m_s,
        use_lift=use_lift,
    )
    a_coeffs = inequality.coeffs
    uniform = uniform_coefficients_flag(a_coeffs)
    rhs_value = inequality.rhs
    sssp = sssp_value(a_coeffs, rhs_value)
    fields, _ = build_status_fields(
        t=case.t,
        m=case.m,
        thm2=thm2,
        s_sorted=list(case.s_values),
        j_max=j_max,
        tight_count=None,
        tight_rank=None,
        feasible_tight_count=None,
        feasible_tight_rank=None,
        uniform=uniform,
        a_coeffs=a_coeffs,
        d_m_s=d_m_s,
        rhs_value=rhs_value,
        sssp=sssp,
        tier_results=(None, None, None, None, None),
        use_lift=use_lift,
        use_relaxation=False,
    )
    return fields, "ERROR"


def uses_structured_stdout(args: argparse.Namespace) -> bool:
    return args.table or args.csv


def run_single_mode(case: AnalysisCase, args: argparse.Namespace) -> int:
    rendered_lines: list[tuple[str, list[tuple[str, str]], str]] = []
    structured_stdout = uses_structured_stdout(args)
    emitted_count = 0

    def emit_structured_output() -> None:
        if not rendered_lines:
            return
        if args.progress and args.table:
            print()
        if args.csv:
            write_csv_output(rendered_lines, resolve_csv_output_path(args))
        elif args.table:
            print(format_status_table(rendered_lines, include_progress_prefix=args.progress))

    if args.lift_mode == "addlifted":
        failures = 0
        try:
            standard_summary = run_analysis_case(
                case,
                args,
                use_lift=False,
                show_progress=args.progress,
                print_final_status=not structured_stdout,
                include_progress_prefix=args.progress,
                multi_case=False,
            )
            if structured_stdout:
                rendered_lines.append(
                    (
                        standard_summary["final_progress_prefix"],
                        standard_summary["status_fields"],
                        standard_summary["terminal_status"],
                    )
                )
            emitted_count += 1
        except SkipAnalysis:
            pass
        except Exception:
            failures += 1
            if structured_stdout:
                error_fields, error_status = case_error_fields(case, use_lift=False)
                rendered_lines.append(
                    (
                        "-/- " if args.progress else "",
                        error_fields,
                        error_status,
                    )
                )
            else:
                print(case_error_line(case, use_lift=False, include_progress_prefix=args.progress))

        try:
            lifted_summary = run_analysis_case(
                case,
                args,
                use_lift=True,
                show_progress=args.progress,
                print_final_status=not structured_stdout,
                include_progress_prefix=args.progress,
                multi_case=False,
            )
            if structured_stdout:
                rendered_lines.append(
                    (
                        lifted_summary["final_progress_prefix"],
                        lifted_summary["status_fields"],
                        lifted_summary["terminal_status"],
                    )
                )
            emitted_count += 1
        except SkipAnalysis:
            pass
        except Exception:
            failures += 1
            if structured_stdout:
                error_fields, error_status = case_error_fields(case, use_lift=True)
                rendered_lines.append(
                    (
                        "-/- " if args.progress else "",
                        error_fields,
                        error_status,
                    )
                )
            else:
                print(case_error_line(case, use_lift=True, include_progress_prefix=args.progress))

        if structured_stdout:
            emit_structured_output()

        if failures > 0:
            raise SystemExit(1)
        if emitted_count == 0:
            raise SystemExit("No analyses remain after applying --onlyuniform.")
        return emitted_count

    use_lift = args.lift_mode == "lifted"
    try:
        summary = run_analysis_case(
            case,
            args,
            use_lift=use_lift,
            show_progress=args.progress,
            print_final_status=not structured_stdout,
            include_progress_prefix=args.progress,
            multi_case=False,
        )
        if structured_stdout:
            rendered_lines.append(
                (
                    summary["final_progress_prefix"],
                    summary["status_fields"],
                    summary["terminal_status"],
                )
            )
            emit_structured_output()
        emitted_count += 1
    except SkipAnalysis:
        raise SystemExit("No analyses remain after applying --onlyuniform.")
    except Exception:
        if structured_stdout:
            error_fields, error_status = case_error_fields(case, use_lift=use_lift)
            rendered_lines.append(
                (
                    "-/- " if args.progress else "",
                    error_fields,
                    error_status,
                )
            )
            emit_structured_output()
        else:
            print(case_error_line(case, use_lift=use_lift, include_progress_prefix=args.progress))
        raise SystemExit(1)
    return emitted_count


def run_multi_mode(cases: list[AnalysisCase], args: argparse.Namespace) -> tuple[int, int]:
    failures = 0
    emitted_count = 0
    rendered_lines: list[tuple[str, list[tuple[str, str]], str]] = []

    for case in cases:
        if args.lift_mode == "addlifted":
            standard_summary: dict | None = None
            standard_fields: list[tuple[str, str]]
            standard_status: str
            standard_prefix: str
            lifted_fields: list[tuple[str, str]]
            lifted_status: str
            lifted_prefix: str

            try:
                standard_summary = run_analysis_case(
                    case,
                    args,
                    use_lift=False,
                    show_progress=False,
                    print_final_status=False,
                    include_progress_prefix=args.progress,
                    multi_case=True,
                )
                standard_fields = standard_summary["status_fields"]
                standard_status = standard_summary["terminal_status"]
                standard_prefix = standard_summary["final_progress_prefix"]
                rendered_lines.append((standard_prefix, standard_fields, standard_status))
                emitted_count += 1
            except SkipAnalysis:
                pass
            except Exception as exc:
                failures += 1
                standard_fields, standard_status = case_error_fields(
                    case,
                    use_lift=False,
                )
                standard_prefix = "-/- " if args.progress else ""
                rendered_lines.append((standard_prefix, standard_fields, standard_status))

            try:
                lifted_summary = run_analysis_case(
                    case,
                    args,
                    use_lift=True,
                    show_progress=False,
                    print_final_status=False,
                    include_progress_prefix=args.progress,
                    multi_case=True,
                )
                lifted_fields = lifted_summary["status_fields"]
                lifted_status = lifted_summary["terminal_status"]
                lifted_prefix = lifted_summary["final_progress_prefix"]
                rendered_lines.append((lifted_prefix, lifted_fields, lifted_status))
                emitted_count += 1
            except SkipAnalysis:
                pass
            except Exception as exc:
                failures += 1
                lifted_fields, lifted_status = case_error_fields(
                    case,
                    use_lift=True,
                )
                lifted_prefix = "-/- " if args.progress else ""
                rendered_lines.append((lifted_prefix, lifted_fields, lifted_status))
            continue

        use_lift = args.lift_mode == "lifted"
        try:
            summary = run_analysis_case(
                case,
                args,
                use_lift=use_lift,
                show_progress=False,
                print_final_status=False,
                include_progress_prefix=args.progress,
                multi_case=True,
            )
            rendered_lines.append(
                (
                    summary["final_progress_prefix"],
                    summary["status_fields"],
                    summary["terminal_status"],
                )
            )
            emitted_count += 1
        except SkipAnalysis:
            continue
        except Exception as exc:
            failures += 1
            error_fields, error_status = case_error_fields(
                case,
                use_lift=use_lift,
            )
            rendered_lines.append(
                (
                    "-/- " if args.progress else "",
                    error_fields,
                    error_status,
                )
            )

    if args.csv:
        write_csv_output(rendered_lines, resolve_csv_output_path(args))
    elif args.table:
        print(format_status_table(rendered_lines, include_progress_prefix=args.progress))
    else:
        widths = status_field_widths([fields for _, fields, _ in rendered_lines])
        for prefix, fields, terminal_status in rendered_lines:
            print(format_status_fields(fields, terminal_status, widths=widths, prefix=prefix))

    return failures, emitted_count


def main() -> None:
    args = parse_args()
    if args.show:
        args.latex = True
    if args.table and args.csv:
        raise ValueError("Use at most one of --table and --csv.")
    if args.csv and args.progress:
        raise ValueError("--csv cannot be combined with --progress.")
    if args.recipe and not args.relaxation:
        raise ValueError("--recipe requires --relax.")

    cases = resolve_analysis_cases(args)
    multi_case = len(cases) > 1

    if args.latex or args.csv:
        if args.output_dir is None:
            args.output_dir = Path("output")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if multi_case:
        failures, emitted_count = run_multi_mode(cases, args)
        if emitted_count == 0 and args.onlyuniform:
            raise SystemExit("No analyses remain after applying --onlyuniform.")
        if failures > 0:
            raise SystemExit(1)
        return

    run_single_mode(cases[0], args)


if __name__ == "__main__":
    main()
