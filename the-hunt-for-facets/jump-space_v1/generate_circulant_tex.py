#!/usr/bin/env python3
"""Analyze circulant graphs on t vertices and optionally generate LaTeX output.

Each graph corresponds to a nonzero 0-1 jump vector y over
J={1,...,floor(t/2)}. Therefore the number of possible y-vectors is
2^|J|-1. The script can analyze either the standard inequality
sum_l kappa(l,S) y_l <= d_m(|S|) or its lifted variant.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute summary data for circulant graphs on V={0,...,t-1}, "
            "indexed by nonzero 0-1 jump vectors over J={1,...,floor(t/2)} "
            "(so total vectors = 2^|J|-1). Analyze the standard inequality by "
            "default, or the lifted inequality with --lifted. "
            "Use --latex to also emit a .tex/.pdf report."
        )
    )
    parser.add_argument(
        "--t",
        type=int,
        required=True,
        help="Number of vertices (t >= 2).",
    )
    parser.add_argument(
        "--S",
        type=str,
        required=True,
        help=(
            "Target vertex subset S (subset of V={0,...,t-1}). "
            "Examples: '0,1,2', '{0,3,7}', '5'."
        ),
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Parameter m used in d_m(|S|). Default: |S|.",
    )
    parser.add_argument(
        "--lift",
        "--lifted",
        dest="lift",
        action="store_true",
        help=(
            "Analyze the lifted inequality with coefficients "
            "min(kappa(l,S), binom(|S|,2)-d_m(|S|)) instead of the standard one."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .tex file path (used only with --latex). "
            "Default: S=<sorted S>_m=<m>_t=<t>[_lift].tex"
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
    return parser.parse_args()


def parse_vertex_set(raw: str, max_vertex: int) -> set[int]:
    stripped = raw.strip()
    if stripped in {"", "{}", "empty", "none", "None"}:
        return set()

    values = [int(x) for x in re.findall(r"\d+", stripped)]
    if not values:
        raise ValueError(
            f"Could not parse S='{raw}'. Use integers like '0,1,2' or '{{0,3,7}}'."
        )

    vertex_set = set(values)
    invalid = sorted(v for v in vertex_set if v < 0 or v > max_vertex)
    if invalid:
        raise ValueError(
            f"Invalid vertices in S: {invalid}. Allowed vertices are in 0..{max_vertex}."
        )
    return vertex_set


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


def compile_tex(tex_path: Path) -> None:
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
        print(f"Compiled {tex_path.with_suffix('.pdf')}")
    except FileNotFoundError:
        print("pdflatex not found; skipping compilation.")
    except subprocess.CalledProcessError:
        print(f"pdflatex failed for {tex_path.name}; check the .log file.")

def clean_latex_garbage(tex_path: Path) -> None:
    removed = 0
    for ext in (".log", ".aux"):
        file_path = tex_path.with_suffix(ext)
        if file_path.is_file():
            file_path.unlink()
            removed += 1
    if removed > 0:
        print(f"Latex garbage cleaned ({removed} files)")

def normalize_output_path(path: Path) -> Path:
    if path.suffix == ".tex" and path.stem.endswith(".lex"):
        return path.with_name(path.stem[:-4] + ".tex")
    return path


def default_output_path(t: int, s_set: set[int], m: int, use_lift: bool) -> Path:
    s_part = ",".join(str(v) for v in sorted(s_set)) if s_set else "empty"
    suffix = "_lift" if use_lift else ""
    return Path(f"S={s_part}_m={m}_t={t}{suffix}.tex")


def main() -> None:
    args = parse_args()
    if args.t < 2:
        raise ValueError("t must be at least 2.")
    if args.latex and args.cols < 1:
        raise ValueError("--cols must be at least 1.")

    s_set = parse_vertex_set(args.S, args.t - 1)
    m = args.m if args.m is not None else len(s_set)
    if m < 2:
        raise ValueError("m must be at least 2. If --m is omitted, this requires |S| >= 2.")
    if len(s_set) < m:
        raise ValueError("|S| must be at least m so that d_m(|S|) is defined.")
    summary = compute_summary(
        args.t,
        m,
        s_set,
        use_fraction=args.fraction,
        compute_sssp_exact=args.sssp,
        use_recipe=args.recipe,
        use_lift=args.lift,
    )

    if not args.latex:
        if args.output is not None:
            print("Ignoring --output because --latex was not set.")
        return

    output = (
        args.output
        if args.output is not None
        else default_output_path(args.t, s_set, m, use_lift=args.lift)
    )
    output = normalize_output_path(output)
    tex = build_document(summary, args.cols, args.show_all_graphs)
    output.write_text(tex, encoding="utf-8")
    print(f"Wrote {output}")
    compile_tex(output.resolve())
    clean_latex_garbage(output.resolve())


if __name__ == "__main__":
    main()
