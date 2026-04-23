#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


KERNEL_SUM_RE = re.compile(r"nsys_single_(\d+)_stats_cuda_gpu_kern_sum\.csv$")
INTERIOR_KEY = "step_kernel_interior"
BOUNDARY_X_KEY = "step_kernel_boundary_x_faces"
BOUNDARY_Y_KEY = "step_kernel_boundary_y_faces"
BOUNDARY_Z_KEY = "step_kernel_boundary_z_faces"
NAIVE_BYTES_PER_SITE = 88.0
OPTIMISTIC_BYTES_PER_SITE = 40.0


def parse_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text.strip().replace(",", ""))
    except (AttributeError, ValueError):
        return default


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def load_kernel_stats(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    parsed = []
    for row in rows:
        name = row.get("Name", "")
        total_ns = parse_float(row.get("Total Time (ns)", "0"))
        instances = parse_float(row.get("Instances", "0"))
        avg_ns = total_ns / instances if instances > 0 else float("nan")
        parsed.append(
            {
                "name": name,
                "total_ns": total_ns,
                "instances": instances,
                "avg_ns": avg_ns,
            }
        )
    return parsed


def aggregate(rows, key):
    total_ns = 0.0
    instances = 0.0
    for row in rows:
        if key in row["name"]:
            total_ns += row["total_ns"]
            instances += row["instances"]
    avg_ns = total_ns / instances if instances > 0 else float("nan")
    return {
        "total_ns": total_ns,
        "instances": instances,
        "avg_ns": avg_ns,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize NSYS timing into manual roofline-style evidence without NCU counters.")
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--hbm-peak-bytes-per-sec", type=float, default=3.35e12)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    case_rows = []
    boundary_table = []

    for csv_path in sorted(args.dir.glob("*_stats_cuda_gpu_kern_sum.csv")):
        m = KERNEL_SUM_RE.search(csv_path.name)
        if not m:
            continue
        size = int(m.group(1))
        rows = load_kernel_stats(csv_path)

        interior = aggregate(rows, INTERIOR_KEY)
        bx = aggregate(rows, BOUNDARY_X_KEY)
        by = aggregate(rows, BOUNDARY_Y_KEY)
        bz = aggregate(rows, BOUNDARY_Z_KEY)

        if interior["avg_ns"] <= 0 or math.isnan(interior["avg_ns"]):
            continue

        interior_points = float(size - 2) ** 3.0
        sups = interior_points / (interior["avg_ns"] * 1.0e-9)
        inferred_b_per_site = args.hbm_peak_bytes_per_sec / sups if sups > 0 else float("nan")
        naive_util_pct = sups * NAIVE_BYTES_PER_SITE / args.hbm_peak_bytes_per_sec * 100.0
        optimistic_util_pct = sups * OPTIMISTIC_BYTES_PER_SITE / args.hbm_peak_bytes_per_sec * 100.0

        total_kernel_ns = sum(r["total_ns"] for r in rows)
        boundary_total_ns = bx["total_ns"] + by["total_ns"] + bz["total_ns"]
        interior_share_pct = interior["total_ns"] / total_kernel_ns * 100.0 if total_kernel_ns > 0 else float("nan")
        boundary_share_pct = boundary_total_ns / total_kernel_ns * 100.0 if total_kernel_ns > 0 else float("nan")

        case_rows.append(
            {
                "size": size,
                "interior_points": interior_points,
                "interior_avg_ms": interior["avg_ns"] / 1.0e6,
                "sups": sups,
                "inferred_b_per_site": inferred_b_per_site,
                "naive_bw_util_pct": naive_util_pct,
                "optimistic_bw_util_pct": optimistic_util_pct,
                "interior_share_pct": interior_share_pct,
                "boundary_total_share_pct": boundary_share_pct,
                "boundary_x_avg_ms": bx["avg_ns"] / 1.0e6 if not math.isnan(bx["avg_ns"]) else float("nan"),
                "boundary_y_avg_ms": by["avg_ns"] / 1.0e6 if not math.isnan(by["avg_ns"]) else float("nan"),
                "boundary_z_avg_ms": bz["avg_ns"] / 1.0e6 if not math.isnan(bz["avg_ns"]) else float("nan"),
                "kernel_sum_csv": str(csv_path),
            }
        )

        if size == 1000 and total_kernel_ns > 0:
            boundary_table = [
                {
                    "kernel": "step_kernel_interior",
                    "instances": interior["instances"],
                    "avg_ms": interior["avg_ns"] / 1.0e6,
                    "share_pct": interior["total_ns"] / total_kernel_ns * 100.0,
                },
                {
                    "kernel": "step_kernel_boundary_x_faces",
                    "instances": bx["instances"],
                    "avg_ms": bx["avg_ns"] / 1.0e6 if not math.isnan(bx["avg_ns"]) else float("nan"),
                    "share_pct": bx["total_ns"] / total_kernel_ns * 100.0,
                },
                {
                    "kernel": "step_kernel_boundary_y_faces",
                    "instances": by["instances"],
                    "avg_ms": by["avg_ns"] / 1.0e6 if not math.isnan(by["avg_ns"]) else float("nan"),
                    "share_pct": by["total_ns"] / total_kernel_ns * 100.0,
                },
                {
                    "kernel": "step_kernel_boundary_z_faces",
                    "instances": bz["instances"],
                    "avg_ms": bz["avg_ns"] / 1.0e6 if not math.isnan(bz["avg_ns"]) else float("nan"),
                    "share_pct": bz["total_ns"] / total_kernel_ns * 100.0,
                },
            ]

    case_rows.sort(key=lambda r: r["size"])

    out = {
        "rows": case_rows,
        "boundary_table_size_1000": boundary_table,
        "notes": {
            "method": "NSYS kernel timing + manual derivation (NCU removed)",
            "hbm_peak_bytes_per_sec": args.hbm_peak_bytes_per_sec,
            "naive_bytes_per_site": NAIVE_BYTES_PER_SITE,
            "optimistic_bytes_per_site": OPTIMISTIC_BYTES_PER_SITE,
            "formulas": {
                "interior_points": "(size - 2)^3",
                "sups": "interior_points / interior_kernel_time_sec",
                "inferred_b_per_site": "hbm_peak_bytes_per_sec / sups",
                "naive_bw_util_pct": "sups * 88 / hbm_peak_bytes_per_sec * 100",
                "optimistic_bw_util_pct": "sups * 40 / hbm_peak_bytes_per_sec * 100",
            },
        },
    }
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# H100 Roofline Proxy (Nsight Systems + Manual Derivation)",
        "",
        "## 1A) Single-GPU interior kernel manual bandwidth proxy",
        "",
        "| Problem Size | interior kernel time (ms) | SU/s | inferred B/site (B) | naive BW utilization (%) | optimistic BW utilization (%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in case_rows:
        lines.append(
            f"| {r['size']} | {fmt(r['interior_avg_ms'])} | {fmt(r['sups'], 2)} | {fmt(r['inferred_b_per_site'], 2)} | {fmt(r['naive_bw_util_pct'], 2)} | {fmt(r['optimistic_bw_util_pct'], 2)} |"
        )

    lines.extend(
        [
            "",
            "Criterion: if naive utilization > 100%, strong cache reuse is required to explain measured throughput.",
            "",
            "## 1B) Boundary vs interior kernel time share (size=1000)",
            "",
        ]
    )

    if boundary_table:
        lines.extend(
            [
                "| Kernel | Calls | Avg(ms) | Time Share (%) |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in boundary_table:
            lines.append(
                f"| {row['kernel']} | {int(row['instances'])} | {fmt(row['avg_ms'])} | {fmt(row['share_pct'], 2)} |"
            )
    else:
        lines.append("- size=1000 kernel summary CSV not found; boundary comparison table cannot be generated yet.")

    lines.extend(
        [
            "",
            "## 1C) Multi-scale trend (256/512/1000/2000)",
            "",
            "- Track how `SU/s` scales with problem size and whether inferred `B/site` converges.",
            "- If inferred `B/site` approaches the optimistic assumption, cache reuse is stronger.",
            "- If naive utilization remains far above 100% at large sizes, this supports non-pure-HBM behavior.",
            "",
        ]
    )
    lines.append("")
    args.md_out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
