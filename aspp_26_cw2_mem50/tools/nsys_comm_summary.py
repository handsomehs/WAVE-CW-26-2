#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


CASE_RE = re.compile(r"nsys_(a100|h100)_np(\d+)_size(\d+)_stats_cuda_gpu_kern_sum\.csv$")


def parse_float(text: str) -> float:
    return float(text.strip().replace(",", ""))


def load_kernel_sum(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    interior_total = 0.0
    interior_instances = 0.0
    boundary_total = 0.0
    for row in rows:
        name = row.get("Name", "")
        total = parse_float(row.get("Total Time (ns)", "0"))
        inst = parse_float(row.get("Instances", "0"))
        if "step_kernel_interior" in name:
            interior_total += total
            interior_instances += inst
        elif "step_kernel_boundary" in name:
            boundary_total += total
    interior_avg = (interior_total / interior_instances) if interior_instances > 0 else 0.0
    return {
        "interior_total_ns": interior_total,
        "interior_instances": interior_instances,
        "interior_avg_ns": interior_avg,
        "boundary_total_ns": boundary_total,
    }


def load_mpi_sum(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    wait_total = 0.0
    wait_instances = 0.0
    wait_avg = 0.0
    for row in rows:
        name = row.get("Name", "")
        if name != "MPI_Waitall":
            continue
        wait_total += parse_float(row.get("Total Time (ns)", "0"))
        wait_instances += parse_float(row.get("Instances", "0"))
        wait_avg = max(wait_avg, parse_float(row.get("Avg (ns)", "0")))
    if wait_instances > 0 and wait_avg == 0.0:
        wait_avg = wait_total / wait_instances
    return {
        "waitall_total_ns": wait_total,
        "waitall_instances": wait_instances,
        "waitall_avg_ns": wait_avg,
    }


def ns_to_ms(v: float) -> float:
    return v / 1.0e6


def main():
    parser = argparse.ArgumentParser(description="Build communication summary table from Nsight Systems CSV stats.")
    parser.add_argument("--dir", required=True, type=Path, help="Directory with nsys_*_stats_*.csv files")
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    records = []
    for kern_file in sorted(args.dir.glob("*_stats_cuda_gpu_kern_sum.csv")):
        m = CASE_RE.search(kern_file.name)
        if not m:
            continue
        gpu = m.group(1)
        ngpu = int(m.group(2))
        size = int(m.group(3))
        mpi_file = kern_file.with_name(kern_file.name.replace("_stats_cuda_gpu_kern_sum.csv", "_stats_mpi_event_sum.csv"))
        if not mpi_file.exists():
            continue
        k = load_kernel_sum(kern_file)
        mpi = load_mpi_sum(mpi_file)
        compute_total = k["interior_total_ns"] + k["boundary_total_ns"]
        comm_frac = mpi["waitall_total_ns"] / (compute_total + mpi["waitall_total_ns"]) if (compute_total + mpi["waitall_total_ns"]) > 0 else 0.0
        overlap_proxy = max(0.0, 1.0 - (mpi["waitall_total_ns"] / k["interior_total_ns"])) if k["interior_total_ns"] > 0 else 0.0
        records.append(
            {
                "gpu": gpu.upper(),
                "ngpu": ngpu,
                "size": size,
                "interior_avg_ms": ns_to_ms(k["interior_avg_ns"]),
                "mpi_waitall_avg_ms": ns_to_ms(mpi["waitall_avg_ns"]),
                "mpi_waitall_total_ms": ns_to_ms(mpi["waitall_total_ns"]),
                "compute_total_ms": ns_to_ms(compute_total),
                "communication_fraction_pct": 100.0 * comm_frac,
                "overlap_proxy_pct": 100.0 * overlap_proxy,
                "kernel_sum_csv": str(kern_file),
                "mpi_sum_csv": str(mpi_file),
            }
        )

    records.sort(key=lambda r: (r["gpu"], r["ngpu"], r["size"]))
    args.json_out.write_text(json.dumps({"rows": records}, indent=2), encoding="utf-8")

    lines = [
        "# Table 2: Multi-GPU Communication Fraction (Nsight Systems)",
        "",
        "| GPU | GPU数 | 问题规模 | interior kernel (ms) | MPI_Waitall avg (ms) | MPI_Waitall total (ms) | 通信占比 (%) | overlap proxy (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['gpu']} | {r['ngpu']} | {r['size']}³ | {r['interior_avg_ms']:.3f} | {r['mpi_waitall_avg_ms']:.3f} | {r['mpi_waitall_total_ms']:.3f} | {r['communication_fraction_pct']:.2f} | {r['overlap_proxy_pct']:.2f} |"
        )
    lines.append("")
    args.md_out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

