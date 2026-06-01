#!/usr/bin/env python3
"""Aggregate co-OPSD eval JSONs into a Markdown summary.

Scans `<eval_dir>/<tag>_<dataset>.json`, builds one row per model tag and
one column per dataset, writes `<eval_dir>/SUMMARY.md`, and echoes it to
stdout. Three tables are emitted (Avg@N / Pass@N / Majority@N) so all
metrics surfaced by `evaluate_math.py` are visible at a glance.

Usage:
    python summarize_co_opsd_eval.py <eval_dir>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Canonical benchmark order (small → large; matches run_co_opsd_eval.sh).
DATASET_ORDER = ["amc23", "aime24", "aime25", "hmmt25", "minerva", "math500", "amo-bench"]
DATASET_SET = set(DATASET_ORDER)


def parse_filename(stem: str) -> tuple[str, str] | None:
    """Split `<tag>_<dataset>` by matching the dataset as a suffix."""
    for ds in DATASET_SET:
        if stem.endswith("_" + ds):
            return stem[: -(len(ds) + 1)], ds
    return None


def fmt(v: float | None) -> str:
    return f"{v:5.1f}" if isinstance(v, (int, float)) else "  —  "


def model_sort_key(tag: str) -> tuple:
    """Group base / final / ckpts; within each, order m1 before m2 and ckpt steps numerically."""
    if tag.startswith("base-"):
        bucket = 0
    elif "-final" in tag:
        bucket = 1
    elif "-ckpt" in tag:
        bucket = 2
    else:
        bucket = 3
    # numeric ckpt step for stable ordering
    step = 0
    if "-ckpt" in tag:
        try:
            step = int(tag.split("-ckpt")[1])
        except ValueError:
            pass
    # m1 before m2
    side = 0 if tag.startswith("m1-") else 1 if tag.startswith("m2-") else 2
    return (bucket, side, step, tag)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <eval_dir>")
    eval_dir = Path(sys.argv[1])
    if not eval_dir.is_dir():
        sys.exit(f"not a directory: {eval_dir}")

    # rows[tag][dataset] = dict with the three metrics + val_n
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    skipped: list[str] = []

    for jf in sorted(eval_dir.glob("*.json")):
        parsed = parse_filename(jf.stem)
        if parsed is None:
            skipped.append(jf.name)
            continue
        tag, ds = parsed
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            skipped.append(f"{jf.name} ({e})")
            continue
        rows[tag][ds] = {
            "avg": data.get("average_at_n_pct"),
            "pass": data.get("pass_at_n_pct"),
            "mv": data.get("majority_vote_at_n_pct"),
            "val_n": data.get("val_n"),
            "format_rate": data.get("format_rate"),
        }

    if not rows:
        sys.exit(f"no parseable result JSONs found in {eval_dir}")

    # Only show columns that have at least one value, in canonical order.
    present_ds = [ds for ds in DATASET_ORDER if any(ds in r for r in rows.values())]
    tags_sorted = sorted(rows, key=model_sort_key)

    lines: list[str] = []
    lines.append(f"# co-OPSD eval summary")
    lines.append("")
    lines.append(f"_Directory_: `{eval_dir}`")
    val_ns = {r[ds]["val_n"] for r in rows.values() for ds in r if r[ds].get("val_n") is not None}
    if val_ns:
        lines.append(f"_val_n_: {sorted(val_ns)}")
    lines.append("")

    def render_table(metric_key: str, header_metric: str):
        lines.append(f"## {header_metric}")
        lines.append("")
        lines.append("| Model | " + " | ".join(present_ds) + " |")
        lines.append("|---|" + "---:|" * len(present_ds))
        for tag in tags_sorted:
            cells = [f"`{tag}`"]
            for ds in present_ds:
                v = rows[tag].get(ds)
                cells.append(fmt(v[metric_key]).strip() if v else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    render_table("avg", "Avg@N  (per-problem average accuracy)")
    render_table("pass", "Pass@N  (problem-solved-by-any-sample rate)")
    render_table("mv", "Majority@N  (majority-vote accuracy)")

    # Format rate as a side check — anomalously low values surface boxed-extraction bugs
    lines.append("## Format rate  (% of generations with parseable `\\boxed{}`)")
    lines.append("")
    lines.append("| Model | " + " | ".join(present_ds) + " |")
    lines.append("|---|" + "---:|" * len(present_ds))
    for tag in tags_sorted:
        cells = [f"`{tag}`"]
        for ds in present_ds:
            v = rows[tag].get(ds)
            cells.append(fmt(v["format_rate"]).strip() if v else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    if skipped:
        lines.append("## Skipped files")
        lines.append("")
        for s in skipped:
            lines.append(f"- {s}")
        lines.append("")

    md = "\n".join(lines)
    print(md)

    out_path = eval_dir / "SUMMARY.md"
    out_path.write_text(md)
    print(f"\n[wrote] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
