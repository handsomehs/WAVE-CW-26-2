#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_float(text: str) -> float:
    return float(text.strip().replace(",", ""))


def load_kernel_sum(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    interior = {"total_ns": 0.0, "instances": 0.0}
    boundary = {"total_ns": 0.0, "instances": 0.0}

    for row in rows:
        name = row.get("Name", "")
        total = parse_float(row.get("Total Time (ns)", "0"))
        inst = parse_float(row.get("Instances", "0"))
        if "step_kernel_interior" in name:
            interior["total_ns"] += total
            interior["instances"] += inst
        elif "step_kernel_boundary" in name:
            boundary["total_ns"] += total
            boundary["instances"] += inst

    return interior, boundary


def load_mpi_sum(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    result = {
        "waitall_total_ns": 0.0,
        "waitall_instances": 0.0,
        "irecv_total_ns": 0.0,
        "isend_total_ns": 0.0,
    }

    for row in rows:
        name = row.get("Name", "")
        total = parse_float(row.get("Total Time (ns)", "0"))
        inst = parse_float(row.get("Instances", "0"))
        if name == "MPI_Waitall":
            result["waitall_total_ns"] += total
            result["waitall_instances"] += inst
        elif name == "MPI_Irecv":
            result["irecv_total_ns"] += total
        elif name == "MPI_Isend":
            result["isend_total_ns"] += total

    return result


def ns_to_ms(v: float) -> float:
    return v / 1.0e6


def main():
    parser = argparse.ArgumentParser(description="Extract overlap evidence from Nsight Systems CSV summaries")
    parser.add_argument("--kernel-sum", required=True, type=Path)
    parser.add_argument("--mpi-sum", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    interior, boundary = load_kernel_sum(args.kernel_sum)
    mpi = load_mpi_sum(args.mpi_sum)

    compute_total_ns = interior["total_ns"] + boundary["total_ns"]
    waitall_total_ns = mpi["waitall_total_ns"]
    interior_total_ns = interior["total_ns"]

    waitall_vs_interior = (waitall_total_ns / interior_total_ns) if interior_total_ns > 0 else 0.0
    waitall_vs_compute = (waitall_total_ns / compute_total_ns) if compute_total_ns > 0 else 0.0
    overlap_proxy = max(0.0, 1.0 - waitall_vs_interior)

    if waitall_vs_interior < 0.25:
        verdict = "strong-overlap-indication"
    elif waitall_vs_interior < 0.50:
        verdict = "moderate-overlap-indication"
    else:
        verdict = "weak-overlap-indication"

    lines = [
        f"label={args.label}",
        f"interior_total_ms={ns_to_ms(interior_total_ns):.3f}",
        f"interior_instances={int(interior['instances'])}",
        f"boundary_total_ms={ns_to_ms(boundary['total_ns']):.3f}",
        f"boundary_instances={int(boundary['instances'])}",
        f"compute_total_ms={ns_to_ms(compute_total_ns):.3f}",
        f"waitall_total_ms={ns_to_ms(waitall_total_ns):.3f}",
        f"waitall_instances={int(mpi['waitall_instances'])}",
        f"irecv_total_ms={ns_to_ms(mpi['irecv_total_ns']):.3f}",
        f"isend_total_ms={ns_to_ms(mpi['isend_total_ns']):.3f}",
        f"waitall_vs_interior={waitall_vs_interior:.3f}",
        f"waitall_vs_compute={waitall_vs_compute:.3f}",
        f"overlap_proxy={overlap_proxy:.3f}",
        f"verdict={verdict}",
    ]

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
