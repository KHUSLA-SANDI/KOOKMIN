#!/usr/bin/env python3
"""Offline distribution check for raw-vs-compact recorded LaserScans.

This is intentionally an audit tool, not a calibration generator.  With no
camera prediction available it uses a configurable straight reference path;
the live node uses its camera-only CNN path instead.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from track_drive_cnn_gpt.lidar_compact import compact_obstacle_points
from track_drive_cnn_gpt.lidar_geometry import occupied_cell_count, points_to_grid, scan_to_points
from track_drive_cnn_gpt.path_contract import OUT_X


def percentiles(values):
    if not values:
        return None
    return {
        name: round(float(value), 3)
        for name, value in zip(
            ("min", "p25", "p50", "p75", "p95", "max"),
            np.percentile(values, [0, 25, 50, 75, 95, 100]),
        )
    }


def rows_from_jsonl(path: Path, parse_stats: Counter):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                parse_stats["blank"] += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # The 2026-08-18 recorder occasionally interleaved an extra
                # leading ``{`` while flushing JSONL.  Recover only that exact
                # known form and report it; all other damage is skipped.
                if line.startswith("{{"):
                    try:
                        row = json.loads(line[1:])
                    except json.JSONDecodeError:
                        parse_stats["malformed"] += 1
                        continue
                    parse_stats["recovered_extra_open_brace"] += 1
                else:
                    parse_stats["malformed"] += 1
                    continue
            # Recorder schema stores centimetres and uses zero for no return.
            ranges = np.asarray(row["cm"], dtype=np.float32) * 0.01
            ranges[ranges <= 0.0] = np.inf
            yield ranges, float(row["a0"]), float(row["da"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_jsonl", type=Path, nargs="+")
    parser.add_argument("--reference-y", type=float, default=0.0)
    parser.add_argument("--corridor-half-width", type=float, default=0.70)
    parser.add_argument("--range-scale", type=float, default=1.1476)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    reference_y = np.full(OUT_X.shape, args.reference_y, dtype=np.float32)
    reasons = Counter()
    raw_cells = []
    raw_points = []
    compact_cells = []
    compact_points = []
    training_domain = Counter()
    rows = 0
    parse_stats = Counter()
    for path in args.scan_jsonl:
        for ranges, angle_min, angle_increment in rows_from_jsonl(path, parse_stats):
            rows += 1
            points = scan_to_points(
                ranges,
                angle_min,
                angle_increment,
                range_scale=args.range_scale,
                self_x_abs_m=0.25,
                self_y_abs_m=0.15,
            )
            raw_points.append(int(points.shape[0]))
            raw_cells.append(occupied_cell_count(points_to_grid(points)))
            compact = compact_obstacle_points(
                points,
                OUT_X,
                reference_y,
                corridor_half_width_m=args.corridor_half_width,
            )
            reasons[compact.reason] += 1
            if compact.usable:
                compact_points.append(int(compact.points.shape[0]))
                cells = occupied_cell_count(points_to_grid(compact.points))
                compact_cells.append(cells)
                if cells == 0:
                    training_domain["clean_zero"] += 1
                elif 27 <= cells <= 67:
                    training_domain["obstacle_27_to_67"] += 1
                else:
                    training_domain["rejected_compact_cell_count"] += 1
            else:
                training_domain["rejected_ambiguous"] += 1

    report = {
        "schema_version": "lidar_compaction_audit_v1_gpt",
        "note": "straight reference proxy; live mode uses camera-only CNN main path",
        "scan_files": [str(path) for path in args.scan_jsonl],
        "scans": rows,
        "parse_stats": dict(parse_stats),
        "reference_y_m": args.reference_y,
        "corridor_half_width_m": args.corridor_half_width,
        "reason_counts": dict(reasons),
        "post_cell_gate_counts": dict(training_domain),
        "raw_points": percentiles(raw_points),
        "raw_occupied_cells": percentiles(raw_cells),
        "usable_compact_points": percentiles(compact_points),
        "usable_compact_occupied_cells": percentiles(compact_cells),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
