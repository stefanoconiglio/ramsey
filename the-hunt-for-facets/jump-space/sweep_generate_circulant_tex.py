#!/usr/bin/env python3
"""Sweep t/|S|/m ranges and stream generate_circulant_tex.py output line by line."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def parse_closed_range(spec: str, arg_name: str) -> list[int]:
    """Parse `min:max` into an inclusive integer range."""
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{arg_name} must be in 'min:max' format, got '{spec}'.")
    start = int(parts[0])
    end = int(parts[1])
    if start > end:
        raise ValueError(f"{arg_name} requires min <= max, got '{spec}'.")
    return list(range(start, end + 1))


def build_s_containing_zero(t: int, s_size: int) -> list[str]:
    """Return all S of size s_size in {0,...,t-1} that contain 0, as '0,1,3' strings."""
    if s_size < 1 or s_size > t:
        return []
    if s_size == 1:
        return ["0"]
    results: list[str] = []
    for tail in itertools.combinations(range(1, t), s_size - 1):
        values = (0,) + tail
        results.append(",".join(str(v) for v in values))
    return results


def parse_explicit_s(raw: str) -> list[int]:
    """Parse explicit S like '0,1,2' or '{0,2,4}' into sorted unique ints."""
    stripped = raw.strip().replace("{", "").replace("}", "")
    values = [int(part.strip()) for part in stripped.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Invalid --S value: '{raw}'")
    return sorted(set(values))


def s_list_to_raw(values: list[int]) -> str:
    return ",".join(str(v) for v in values)


def run_and_extract_one_line(cmd: list[str]) -> tuple[int, str]:
    """Run one call and return exactly one representative output line."""
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    merged = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    lines = [line.strip() for line in merged.replace("\r", "\n").splitlines() if line.strip()]

    for line in reversed(lines):
        if "a'y=(" in line and ("FACET" in line or "FAILURE" in line):
            return proc.returncode, line
    if lines:
        return proc.returncode, lines[-1]
    return proc.returncode, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep t, |S|, and m ranges. For each t and |S|, generate all subsets "
            "S containing 0, then run generate_circulant_tex.py for each (t,S,m)."
        )
    )
    parser.add_argument(
        "--t",
        dest="t_range",
        required=True,
        help="t range in min:max format (inclusive).",
    )
    parser.add_argument(
        "--m",
        dest="m_range",
        required=True,
        help="m range in min:max format (inclusive).",
    )
    parser.add_argument(
        "--S-size",
        "--|S|",
        dest="s_size_range",
        required=False,
        help="|S| range in min:max format (inclusive).",
    )
    parser.add_argument(
        "--S",
        dest="explicit_s",
        default=None,
        help=(
            "Explicit S override (e.g. --S 0,1,2). If provided, subset sweeping by "
            "--S-size is disabled and only this S is used."
        ),
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("generate_circulant_tex.py"),
        help="Path to generate_circulant_tex.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument("--latex", action="store_true", help="Forward --latex.")
    parser.add_argument("--all", dest="show_all_graphs", action="store_true", help="Forward --all.")
    parser.add_argument("--fraction", action="store_true", help="Forward --fraction.")
    parser.add_argument("--sssp", action="store_true", help="Forward --sssp.")
    parser.add_argument("--recipe", action="store_true", help="Forward --recipe.")
    parser.add_argument("--cols", type=int, default=None, help="Forward --cols.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, pass --output for each run into this directory.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop sweep at first non-zero return code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_values = parse_closed_range(args.t_range, "--t")
    m_values = parse_closed_range(args.m_range, "--m")
    if args.explicit_s is None and args.s_size_range is None:
        raise ValueError("Provide either --S-size (or --|S|) or --S.")

    explicit_s_values: list[int] | None = None
    s_sizes: list[int] | None = None
    if args.explicit_s is not None:
        explicit_s_values = parse_explicit_s(args.explicit_s)
    else:
        assert args.s_size_range is not None
        s_sizes = parse_closed_range(args.s_size_range, "--S-size")

    script_path = args.script.resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    combos: list[tuple[int, str, int]] = []
    if explicit_s_values is not None:
        s_candidates = [s_list_to_raw(explicit_s_values)]
    else:
        assert s_sizes is not None
        t_max = max(t_values)
        s_candidates = []
        for s_size in s_sizes:
            s_candidates.extend(build_s_containing_zero(t=t_max, s_size=s_size))

    for s_raw in s_candidates:
        s_vals = parse_explicit_s(s_raw)
        s_size = len(s_vals)
        for t in t_values:
            if any(v < 0 or v >= t for v in s_vals):
                continue
            for m in m_values:
                if m < 2 or m > s_size:
                    continue
                combos.append((t, s_raw, m))

    if not combos:
        raise SystemExit("No valid (t,S,m) combinations after filtering m <= |S| and m >= 2.")

    failures = 0
    for t, s_raw, m in combos:
        cmd = [args.python, str(script_path), "--t", str(t), "--S", s_raw]
        cmd.extend(["--m", str(m)])
        if args.latex:
            cmd.append("--latex")
        if args.show_all_graphs:
            cmd.append("--all")
        if args.fraction:
            cmd.append("--fraction")
        if args.sssp:
            cmd.append("--sssp")
        if args.recipe:
            cmd.append("--recipe")
        if args.cols is not None:
            cmd.extend(["--cols", str(args.cols)])
        if args.output_dir is not None:
            out_path = args.output_dir / f"S={s_raw}_m={m}_t={t}.tex"
            cmd.extend(["--output", str(out_path)])

        rc, one_line = run_and_extract_one_line(cmd)
        if one_line:
            print(one_line)
        else:
            print(f"t={t}, S={{{s_raw}}}, m={m}, EXIT={rc}")
        if rc != 0:
            failures += 1
            if args.stop_on_error:
                raise SystemExit(rc)

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
