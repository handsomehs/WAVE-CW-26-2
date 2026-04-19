#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev

DIR_RE = re.compile(r"awave-cw2-mpistack-ab-(a100|h100)-(\d+)g-(pq|uq)(?:-r(\d+))?$")


def safe_mean(values):
    return mean(values) if values else float("nan")


def safe_cv_pct(values):
    if len(values) < 2:
        return float("nan")
    m = safe_mean(values)
    if m <= 0:
        return float("nan")
    return pstdev(values) / m * 100.0


def parse_case_dir(path: Path):
    m = DIR_RE.fullmatch(path.name)
    if not m:
        return None
    gpu, gpus, queue, repeat = m.groups()
    return {
        "gpu": gpu,
        "gpus": int(gpus),
        "queue": queue,
        "repeat": int(repeat) if repeat else None,
    }


def read_case_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    sups = [float(v) for v in doc.get("GPU", {}).get("sups", [])]
    if not sups:
        raise ValueError(f"No GPU.sups in {path}")
    all_mean = safe_mean(sups)
    steady = sups[1:] if len(sups) > 1 else sups
    steady_mean = safe_mean(steady)
    return {
        "all_mean": all_mean,
        "steady_mean": steady_mean,
        "n_chunks": len(sups),
    }


def summarize_case_dir(path: Path):
    meta = parse_case_dir(path)
    if meta is None:
        return None

    default_all = []
    default_steady = []
    ob1_all = []
    ob1_steady = []

    for file in sorted(path.glob("*.json")):
        try:
            stats = read_case_file(file)
        except Exception:
            continue

        name = file.name
        if "stack_default" in name:
            default_all.append(stats["all_mean"])
            default_steady.append(stats["steady_mean"])
        elif "stack_ob1" in name:
            ob1_all.append(stats["all_mean"])
            ob1_steady.append(stats["steady_mean"])

    if not default_steady or not ob1_steady:
        return None

    default_mean_all = safe_mean(default_all)
    ob1_mean_all = safe_mean(ob1_all)
    default_mean_steady = safe_mean(default_steady)
    ob1_mean_steady = safe_mean(ob1_steady)

    gain_all_pct = (ob1_mean_all / default_mean_all - 1.0) * 100.0
    gain_steady_pct = (ob1_mean_steady / default_mean_steady - 1.0) * 100.0

    out = {
        "dir": str(path),
        "gpu": meta["gpu"],
        "gpus": meta["gpus"],
        "queue": meta["queue"],
        "repeat": meta["repeat"],
        "default_n": len(default_steady),
        "ob1_n": len(ob1_steady),
        "default_mean_all": default_mean_all,
        "ob1_mean_all": ob1_mean_all,
        "gain_all_pct": gain_all_pct,
        "default_mean_steady": default_mean_steady,
        "ob1_mean_steady": ob1_mean_steady,
        "gain_steady_pct": gain_steady_pct,
        "default_cv_pct": safe_cv_pct(default_steady),
        "ob1_cv_pct": safe_cv_pct(ob1_steady),
    }
    return out


def build_queue_stats(rows):
    by_queue = {}
    for row in rows:
        by_queue.setdefault(row["queue"], []).append(row["gain_steady_pct"])

    stats = []
    for queue, gains in sorted(by_queue.items()):
        stats.append(
            {
                "queue": queue,
                "repeats": len(gains),
                "gain_steady_pct_mean": safe_mean(gains),
                "gain_steady_pct_min": min(gains),
                "gain_steady_pct_max": max(gains),
            }
        )
    return stats


def decide_group(summary, args):
    gain = summary["gain_steady_pct"]
    pairs = min(summary["default_n"], summary["ob1_n"])
    queue_mean_gains = [q["gain_steady_pct_mean"] for q in summary["queue_stats"]]
    qspread = max(queue_mean_gains) - min(queue_mean_gains) if len(queue_mean_gains) > 1 else 0.0

    if pairs < args.min_pairs:
        if gain >= args.promote_ob1_threshold_pct:
            status = "provisional-ob1"
        elif gain <= args.keep_default_threshold_pct:
            status = "provisional-default"
        else:
            status = "inconclusive"
    else:
        if qspread > args.max_queue_spread_pct:
            if gain >= args.promote_ob1_threshold_pct and min(queue_mean_gains) >= 0.0:
                status = "provisional-ob1"
            elif gain <= args.keep_default_threshold_pct and max(queue_mean_gains) <= 0.0:
                status = "provisional-default"
            else:
                status = "needs-retest"
        elif abs(gain) <= args.neutral_default_band_pct:
            status = "default-neutral"
        elif gain >= args.promote_ob1_threshold_pct:
            status = "ob1"
        elif gain <= args.keep_default_threshold_pct:
            status = "default"
        else:
            status = "needs-retest"

    if status in ("ob1", "provisional-ob1"):
        action = "use-ob1-stack-for-next-stage"
    elif status in ("default", "provisional-default", "default-neutral"):
        action = "keep-default-stack-for-next-stage"
    else:
        action = "run-more-repeats"

    summary["pairs"] = pairs
    summary["queue_spread_pct"] = qspread
    summary["status"] = status
    summary["recommended_action"] = action


def build_group_summaries(case_summaries, args):
    grouped = {}
    for row in case_summaries:
        key = (row["gpu"], row["gpus"])
        grouped.setdefault(key, []).append(row)

    results = []
    for (gpu, gpus), rows in sorted(grouped.items()):
        default_all = []
        ob1_all = []
        for row in rows:
            default_all.extend([row["default_mean_steady"]] * row["default_n"])
            ob1_all.extend([row["ob1_mean_steady"]] * row["ob1_n"])

        repeat_gains = [row["gain_steady_pct"] for row in rows]
        queue_stats = build_queue_stats(rows)

        default_mean = safe_mean(default_all)
        ob1_mean = safe_mean(ob1_all)
        gain = (ob1_mean / default_mean - 1.0) * 100.0

        summary = {
            "gpu": gpu,
            "gpus": gpus,
            "queues": rows,
            "queue_stats": queue_stats,
            "default_n": len(default_all),
            "ob1_n": len(ob1_all),
            "default_mean_steady": default_mean,
            "ob1_mean_steady": ob1_mean,
            "gain_steady_pct": gain,
            "repeat_spread_pct": max(repeat_gains) - min(repeat_gains) if len(repeat_gains) > 1 else 0.0,
        }
        decide_group(summary, args)
        results.append(summary)
    return results


def fmt(x, digits=2):
    if isinstance(x, (int, float)) and not math.isnan(x):
        return f"{x:.{digits}f}"
    return "nan"


def write_markdown(path: Path, case_summaries, group_summaries, args):
    lines = []
    lines.append("# MPI Stack Gate Summary")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append(f"- promote_ob1_threshold_pct: `{args.promote_ob1_threshold_pct}`")
    lines.append(f"- keep_default_threshold_pct: `{args.keep_default_threshold_pct}`")
    lines.append(f"- neutral_default_band_pct: `{args.neutral_default_band_pct}`")
    lines.append(f"- min_pairs: `{args.min_pairs}`")
    lines.append(f"- max_queue_spread_pct: `{args.max_queue_spread_pct}`")
    lines.append("")

    lines.append("## Per Queue Directory")
    lines.append("")
    lines.append("| dir | repeat | ob1_gain_all_pct | ob1_gain_steady_pct | default_n | ob1_n | default_cv_pct | ob1_cv_pct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(case_summaries, key=lambda r: r["dir"]):
        lines.append(
            "| {dir} | {repeat} | {ga} | {gs} | {default_n} | {ob1_n} | {default_cv} | {ob1_cv} |".format(
                dir=row["dir"],
                repeat=(row["repeat"] if row["repeat"] is not None else "-"),
                ga=fmt(row["gain_all_pct"]),
                gs=fmt(row["gain_steady_pct"]),
                default_n=row["default_n"],
                ob1_n=row["ob1_n"],
                default_cv=fmt(row["default_cv_pct"]),
                ob1_cv=fmt(row["ob1_cv_pct"]),
            )
        )
    lines.append("")

    lines.append("## Group Decision")
    lines.append("")
    lines.append("| gpu | gpus | ob1_gain_steady_pct | pairs | queue_spread_pct | status | recommended_action |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for row in sorted(group_summaries, key=lambda r: (r["gpu"], r["gpus"])):
        lines.append(
            "| {gpu} | {gpus} | {gain} | {pairs} | {spread} | {status} | {action} |".format(
                gpu=row["gpu"],
                gpus=row["gpus"],
                gain=fmt(row["gain_steady_pct"]),
                pairs=row["pairs"],
                spread=fmt(row["queue_spread_pct"]),
                status=row["status"],
                action=row["recommended_action"],
            )
        )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Decide MPI stack promote/drop from A/B JSON logs")
    p.add_argument("--logs-root", type=Path, default=Path("logs"))
    p.add_argument("--dir-glob", type=str, default="awave-cw2-mpistack-ab-*")
    p.add_argument("--json-out", type=Path, default=Path("logs/mpistack_gate_summary.json"))
    p.add_argument("--md-out", type=Path, default=Path("logs/mpistack_gate_summary.md"))
    p.add_argument("--promote-ob1-threshold-pct", type=float, default=3.0)
    p.add_argument("--keep-default-threshold-pct", type=float, default=-2.0)
    p.add_argument("--neutral-default-band-pct", type=float, default=0.5)
    p.add_argument("--min-pairs", type=int, default=2)
    p.add_argument("--max-queue-spread-pct", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()

    case_summaries = []
    for path in sorted(args.logs_root.glob(args.dir_glob)):
        if not path.is_dir():
            continue
        row = summarize_case_dir(path)
        if row is not None:
            case_summaries.append(row)

    if not case_summaries:
        raise SystemExit("No mpistack case directories with stack_default/stack_ob1 JSON found")

    group_summaries = build_group_summaries(case_summaries, args)

    payload = {
        "thresholds": {
            "promote_ob1_threshold_pct": args.promote_ob1_threshold_pct,
            "keep_default_threshold_pct": args.keep_default_threshold_pct,
            "neutral_default_band_pct": args.neutral_default_band_pct,
            "min_pairs": args.min_pairs,
            "max_queue_spread_pct": args.max_queue_spread_pct,
        },
        "cases": case_summaries,
        "groups": group_summaries,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.md_out, case_summaries, group_summaries, args)

    print(args.json_out)
    print(args.md_out)


if __name__ == "__main__":
    main()