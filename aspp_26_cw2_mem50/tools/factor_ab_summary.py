#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean

CASE_RE = re.compile(r"^(base|damp_off|zpad_off)_r(\d+)\.json$")


def avg(values):
    return mean(values) if values else float("nan")


def pct_delta(new_value, old_value):
    if old_value == 0.0 or math.isnan(new_value) or math.isnan(old_value):
        return float("nan")
    return 100.0 * (new_value / old_value - 1.0)


def parse_case(path: Path):
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    sups = [float(v) for v in doc.get("GPU", {}).get("sups", [])]
    if not sups:
        return None
    steady = sups[1:] if len(sups) > 1 else sups
    return {
        "all_mean": avg(sups),
        "steady_mean": avg(steady),
        "samples": len(sups),
    }


def collect(directory: Path):
    rows = []
    grouped = {"base": [], "damp_off": [], "zpad_off": []}
    for file in sorted(directory.glob("*_r*.json")):
        m = CASE_RE.match(file.name)
        if not m:
            continue
        case = m.group(1)
        rep = int(m.group(2))
        stats = parse_case(file)
        if not stats:
            continue
        row = {
            "case": case,
            "rep": rep,
            "file": str(file),
            "all_mean": stats["all_mean"],
            "steady_mean": stats["steady_mean"],
            "samples": stats["samples"],
        }
        rows.append(row)
        grouped[case].append(row)
    return rows, grouped


def summarize(grouped):
    ans = {}
    for case, rows in grouped.items():
        ans[case] = {
            "n_runs": len(rows),
            "steady_mean_of_runs": avg([r["steady_mean"] for r in rows]),
            "all_mean_of_runs": avg([r["all_mean"] for r in rows]),
        }
    base = ans["base"]["steady_mean_of_runs"]
    damp_off = ans["damp_off"]["steady_mean_of_runs"]
    zpad_off = ans["zpad_off"]["steady_mean_of_runs"]
    ans["effects_pct"] = {
        "damp_branchless_on_vs_off": pct_delta(base, damp_off),
        "z_padding_on_vs_off": pct_delta(base, zpad_off),
    }
    return ans


def fmt(x):
    return "nan" if math.isnan(x) else f"{x:.2f}"


def write_md(path: Path, rows, summary):
    lines = [
        "# Single-Factor A/B Summary",
        "",
        "| case | rep | steady_sups | all_sups | samples | file |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["case"], r["rep"])):
        lines.append(
            f"| {row['case']} | {row['rep']} | {fmt(row['steady_mean'])} | {fmt(row['all_mean'])} | {row['samples']} | {row['file']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- base steady_mean={fmt(summary['base']['steady_mean_of_runs'])} (runs={summary['base']['n_runs']})",
            f"- damp_off steady_mean={fmt(summary['damp_off']['steady_mean_of_runs'])} (runs={summary['damp_off']['n_runs']})",
            f"- zpad_off steady_mean={fmt(summary['zpad_off']['steady_mean_of_runs'])} (runs={summary['zpad_off']['n_runs']})",
            f"- damp_branchless_on_vs_off={fmt(summary['effects_pct']['damp_branchless_on_vs_off'])}%",
            f"- z_padding_on_vs_off={fmt(summary['effects_pct']['z_padding_on_vs_off'])}%",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Summarize single-factor A/B runs.")
    p.add_argument("--dir", required=True, help="Directory containing *_r*.json run files")
    p.add_argument("--json-out", required=True, help="Output JSON summary path")
    p.add_argument("--md-out", required=True, help="Output markdown summary path")
    args = p.parse_args()

    rows, grouped = collect(Path(args.dir))
    summary = summarize(grouped)
    result = {
        "dir": args.dir,
        "rows": rows,
        "summary": summary,
    }
    Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(Path(args.md_out), rows, summary)


if __name__ == "__main__":
    main()
