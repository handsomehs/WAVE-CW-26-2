#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean

NAME_RE = re.compile(r"tile([01])_blk(\d+)\.json$")


def safe_mean(values):
    return mean(values) if values else float("nan")


def read_case(path: Path):
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    sups = [float(v) for v in doc.get("GPU", {}).get("sups", [])]
    if not sups:
        return None
    return {
        "all_mean": safe_mean(sups),
        "steady_mean": safe_mean(sups[1:] if len(sups) > 1 else sups),
        "samples": len(sups),
    }


def collect_rows(directory: Path):
    rows = []
    for file in sorted(directory.glob("tile*_blk*.json")):
        m = NAME_RE.match(file.name)
        if not m:
            continue
        stats = read_case(file)
        if not stats:
            continue
        tile = int(m.group(1))
        blk = int(m.group(2))
        rows.append(
            {
                "file": str(file),
                "tile": tile,
                "blk": blk,
                "all_mean": stats["all_mean"],
                "steady_mean": stats["steady_mean"],
                "samples": stats["samples"],
            }
        )
    return rows


def best_rows(rows):
    if not rows:
        return {}
    by_tile = {}
    for tile in (0, 1):
        tile_rows = [r for r in rows if r["tile"] == tile]
        if not tile_rows:
            continue
        by_tile[str(tile)] = max(tile_rows, key=lambda r: r["steady_mean"])
    return by_tile


def fmt(x):
    return "nan" if math.isnan(x) else f"{x:.2f}"


def write_markdown(path: Path, rows, best):
    lines = []
    lines.append("# Block Tuning Summary")
    lines.append("")
    lines.append("| tile | blk | steady_sups | all_sups | samples | file |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda r: (r["tile"], r["blk"])):
        lines.append(
            f"| {row['tile']} | {row['blk']} | {fmt(row['steady_mean'])} | {fmt(row['all_mean'])} | {row['samples']} | {row['file']} |"
        )
    lines.append("")
    lines.append("## Best per tile")
    lines.append("")
    for tile in ("0", "1"):
        item = best.get(tile)
        if not item:
            continue
        lines.append(
            f"- tile{tile}: blk{item['blk']} steady_sups={fmt(item['steady_mean'])} file={item['file']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize tile/block tuning JSON outputs.")
    parser.add_argument("--dir", required=True, help="Directory with tile*_blk*.json files.")
    parser.add_argument("--json-out", required=True, help="Output JSON summary path.")
    parser.add_argument("--md-out", required=True, help="Output markdown summary path.")
    args = parser.parse_args()

    directory = Path(args.dir)
    rows = collect_rows(directory)
    best = best_rows(rows)

    result = {
        "dir": str(directory),
        "cases": rows,
        "best_per_tile": best,
    }
    Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.md_out), rows, best)


if __name__ == "__main__":
    main()
