#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev

DIR_RE = re.compile(r"awave-cw2-waitsome-ab-(a100|h100)-(\d+)g-(pq|uq)(?:-r(\d+))?$")


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

    off_all = []
    off_steady = []
    on_all = []
    on_steady = []

    for file in sorted(path.glob("*.json")):
        try:
            stats = read_case_file(file)
        except Exception:
            continue

        name = file.name
        if "waitsome0" in name:
            off_all.append(stats["all_mean"])
            off_steady.append(stats["steady_mean"])
        elif "waitsome1" in name:
            on_all.append(stats["all_mean"])
            on_steady.append(stats["steady_mean"])

    if not off_steady or not on_steady:
        return None

    off_mean_all = safe_mean(off_all)
    on_mean_all = safe_mean(on_all)
    off_mean_steady = safe_mean(off_steady)
    on_mean_steady = safe_mean(on_steady)

    gain_all_pct = (on_mean_all / off_mean_all - 1.0) * 100.0
    gain_steady_pct = (on_mean_steady / off_mean_steady - 1.0) * 100.0

    out = {
        "dir": str(path),
        "gpu": meta["gpu"],
        "gpus": meta["gpus"],
        "queue": meta["queue"],
        "repeat": meta["repeat"],
        "off_n": len(off_steady),
        "on_n": len(on_steady),
        "off_mean_all": off_mean_all,
        "on_mean_all": on_mean_all,
        "gain_all_pct": gain_all_pct,
        "off_mean_steady": off_mean_steady,
        "on_mean_steady": on_mean_steady,
        "gain_steady_pct": gain_steady_pct,
        "off_cv_pct": safe_cv_pct(off_steady),
        "on_cv_pct": safe_cv_pct(on_steady),
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
    pairs = min(summary["off_n"], summary["on_n"])
    queue_mean_gains = [q["gain_steady_pct_mean"] for q in summary["queue_stats"]]
    qspread = max(queue_mean_gains) - min(queue_mean_gains) if len(queue_mean_gains) > 1 else 0.0

    if pairs < args.min_pairs:
        if gain >= args.promote_threshold_pct:
            status = "provisional-promote"
        elif gain <= args.drop_threshold_pct:
            status = "provisional-drop"
        elif gain <= args.no_promote_threshold_pct:
            status = "provisional-no-promote"
        else:
            status = "inconclusive"
    else:
        if qspread > args.max_queue_spread_pct:
            status = "needs-retest"
        elif gain >= args.promote_threshold_pct:
            status = "promote"
        elif gain <= args.drop_threshold_pct and max(queue_mean_gains) <= 0.0:
            status = "drop"
        elif gain <= args.no_promote_threshold_pct:
            status = "no-promote"
        else:
            status = "needs-retest"

    if summary["gpus"] == 2:
        if status in ("promote", "provisional-promote"):
            action = "enable-waitsome-and-continue-next-optimizations"
        elif status in ("drop", "provisional-drop", "no-promote", "provisional-no-promote"):
            action = "keep-waitsome-off-and-move-next-optimization"
        else:
            action = "run-more-2g-repeats-before-keep-or-drop"
    else:
        if status in ("promote", "provisional-promote"):
            action = "enable-waitsome-for-this-gpu-count"
        elif status in ("drop", "provisional-drop", "no-promote", "provisional-no-promote"):
            action = "keep-waitsome-off-for-this-gpu-count"
        else:
            action = "collect-more-runs"

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
        off_all = []
        on_all = []
        for row in rows:
            off_all.extend([row["off_mean_steady"]] * row["off_n"])
            on_all.extend([row["on_mean_steady"]] * row["on_n"])

        repeat_gains = [row["gain_steady_pct"] for row in rows]
        queue_stats = build_queue_stats(rows)

        off_mean = safe_mean(off_all)
        on_mean = safe_mean(on_all)
        gain = (on_mean / off_mean - 1.0) * 100.0

        summary = {
            "gpu": gpu,
            "gpus": gpus,
            "queues": rows,
            "queue_stats": queue_stats,
            "off_n": len(off_all),
            "on_n": len(on_all),
            "off_mean_steady": off_mean,
            "on_mean_steady": on_mean,
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
    lines.append("# Waitsome Gate Summary")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append(f"- promote_threshold_pct: `{args.promote_threshold_pct}`")
    lines.append(f"- drop_threshold_pct: `{args.drop_threshold_pct}`")
    lines.append(f"- no_promote_threshold_pct: `{args.no_promote_threshold_pct}`")
    lines.append(f"- min_pairs: `{args.min_pairs}`")
    lines.append(f"- max_queue_spread_pct: `{args.max_queue_spread_pct}`")
    lines.append("")

    lines.append("## Per Queue Directory")
    lines.append("")
    lines.append("| dir | repeat | gain_all_pct | gain_steady_pct | off_n | on_n | off_cv_pct | on_cv_pct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(case_summaries, key=lambda r: r["dir"]):
        lines.append(
            "| {dir} | {repeat} | {ga} | {gs} | {off_n} | {on_n} | {off_cv} | {on_cv} |".format(
                dir=row["dir"],
                repeat=(row["repeat"] if row["repeat"] is not None else "-"),
                ga=fmt(row["gain_all_pct"]),
                gs=fmt(row["gain_steady_pct"]),
                off_n=row["off_n"],
                on_n=row["on_n"],
                off_cv=fmt(row["off_cv_pct"]),
                on_cv=fmt(row["on_cv_pct"]),
            )
        )
    lines.append("")

    lines.append("## Group Decision")
    lines.append("")
    lines.append("| gpu | gpus | gain_steady_pct | pairs | queue_spread_pct | status | recommended_action |")
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
    p = argparse.ArgumentParser(description="Decide waitsome promote/drop from A/B JSON logs")
    p.add_argument("--logs-root", type=Path, default=Path("logs"))
    p.add_argument("--dir-glob", type=str, default="awave-cw2-waitsome-ab-*")
    p.add_argument("--json-out", type=Path, default=Path("logs/waitsome_gate_summary.json"))
    p.add_argument("--md-out", type=Path, default=Path("logs/waitsome_gate_summary.md"))
    p.add_argument("--promote-threshold-pct", type=float, default=3.0)
    p.add_argument("--drop-threshold-pct", type=float, default=-2.0)
    p.add_argument(
        "--no-promote-threshold-pct",
        type=float,
        default=2.0,
        help="If gain is <= this value (and other checks pass), stop campaign as low-impact",
    )
    p.add_argument("--min-pairs", type=int, default=3)
    p.add_argument("--max-queue-spread-pct", type=float, default=8.0)
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
        raise SystemExit("No waitsome case directories with waitsome0/waitsome1 JSON found")

    group_summaries = build_group_summaries(case_summaries, args)

    payload = {
        "thresholds": {
            "promote_threshold_pct": args.promote_threshold_pct,
            "drop_threshold_pct": args.drop_threshold_pct,
            "no_promote_threshold_pct": args.no_promote_threshold_pct,
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
