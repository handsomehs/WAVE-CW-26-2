#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def pstdev(values):
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def read_case(path: Path):
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    gpu = doc.get("GPU") or {}
    cpu = doc.get("CPU") or {}
    gpu_sups = [float(v) for v in gpu.get("sups", [])]
    cpu_sups = [float(v) for v in cpu.get("sups", [])]

    if not gpu_sups:
        raise ValueError(f"No GPU.sups values in {path}")

    steady_values = gpu_sups[1:] if len(gpu_sups) > 1 else gpu_sups
    mean_all = mean(gpu_sups)
    mean_steady = mean(steady_values)
    std_all = pstdev(gpu_sups)
    cpu_mean = mean(cpu_sups)
    have_cpu = len(cpu_sups) > 0

    return {
        "ndiff": int(doc.get("ndiff", -1)),
        "n_samples": len(gpu_sups),
        "gpu_mean_all_sups": mean_all,
        "gpu_mean_steady_sups": mean_steady,
        "gpu_median_sups": float(median(gpu_sups)),
        "gpu_std_sups": std_all,
        "gpu_cv_pct": (std_all / mean_all * 100.0) if mean_all > 0 else float("nan"),
        "gpu_min_sups": min(gpu_sups),
        "gpu_max_sups": max(gpu_sups),
        "warmup_ratio": (mean_steady / mean_all) if mean_all > 0 else float("nan"),
        "cpu_mean_sups": cpu_mean,
        "speedup_all": (mean_all / cpu_mean) if (have_cpu and cpu_mean > 0) else float("nan"),
        "speedup_steady": (mean_steady / cpu_mean) if (have_cpu and cpu_mean > 0) else float("nan"),
    }


def load_gpu_dir(gpu: str, directory: Path):
    out = {}
    for path in sorted(directory.glob("*.json")):
        case = path.stem
        out[case] = read_case(path)
    return out


def write_case_summary(path: Path, data):
    fields = [
        "gpu",
        "case",
        "ndiff",
        "n_samples",
        "gpu_mean_all_sups",
        "gpu_mean_steady_sups",
        "gpu_median_sups",
        "gpu_std_sups",
        "gpu_cv_pct",
        "gpu_min_sups",
        "gpu_max_sups",
        "warmup_ratio",
        "cpu_mean_sups",
        "speedup_all",
        "speedup_steady",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for gpu in sorted(data.keys()):
            for case in sorted(data[gpu].keys()):
                row = {"gpu": gpu, "case": case}
                row.update(data[gpu][case])
                w.writerow(row)


def write_cross_summary(path: Path, data):
    fields = [
        "case",
        "a100_mean_all_sups",
        "h100_mean_all_sups",
        "h100_over_a100_all",
        "a100_mean_steady_sups",
        "h100_mean_steady_sups",
        "h100_over_a100_steady",
    ]

    cases = sorted(set(data["a100"].keys()) & set(data["h100"].keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for case in cases:
            a = data["a100"][case]
            h = data["h100"][case]
            w.writerow(
                {
                    "case": case,
                    "a100_mean_all_sups": a["gpu_mean_all_sups"],
                    "h100_mean_all_sups": h["gpu_mean_all_sups"],
                    "h100_over_a100_all": h["gpu_mean_all_sups"] / a["gpu_mean_all_sups"],
                    "a100_mean_steady_sups": a["gpu_mean_steady_sups"],
                    "h100_mean_steady_sups": h["gpu_mean_steady_sups"],
                    "h100_over_a100_steady": h["gpu_mean_steady_sups"] / a["gpu_mean_steady_sups"],
                }
            )


def write_scaling_summary(path: Path, data):
    fields = [
        "gpu",
        "family",
        "np",
        "gain_all",
        "eff_all_pct",
        "gain_steady",
        "eff_steady_pct",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for gpu in ("a100", "h100"):
            for family in ("strong", "weak"):
                base_key = f"{family}_np1"
                if base_key not in data[gpu]:
                    continue
                base = data[gpu][base_key]
                for np in (2, 4, 8):
                    key = f"{family}_np{np}"
                    if key not in data[gpu]:
                        continue
                    row = data[gpu][key]
                    gain_all = row["gpu_mean_all_sups"] / base["gpu_mean_all_sups"]
                    gain_steady = row["gpu_mean_steady_sups"] / base["gpu_mean_steady_sups"]
                    w.writerow(
                        {
                            "gpu": gpu,
                            "family": family,
                            "np": np,
                            "gain_all": gain_all,
                            "eff_all_pct": gain_all / np * 100.0,
                            "gain_steady": gain_steady,
                            "eff_steady_pct": gain_steady / np * 100.0,
                        }
                    )


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate CW2 JSON metrics with robust statistics")
    parser.add_argument("--a100-dir", required=True, type=Path)
    parser.add_argument("--h100-dir", required=True, type=Path)
    parser.add_argument("--case-out", required=True, type=Path)
    parser.add_argument("--cross-out", required=True, type=Path)
    parser.add_argument("--scaling-out", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    data = {
        "a100": load_gpu_dir("a100", args.a100_dir),
        "h100": load_gpu_dir("h100", args.h100_dir),
    }

    write_case_summary(args.case_out, data)
    write_cross_summary(args.cross_out, data)
    write_scaling_summary(args.scaling_out, data)

    print(f"Wrote case summary: {args.case_out}")
    print(f"Wrote cross summary: {args.cross_out}")
    print(f"Wrote scaling summary: {args.scaling_out}")


if __name__ == "__main__":
    main()
