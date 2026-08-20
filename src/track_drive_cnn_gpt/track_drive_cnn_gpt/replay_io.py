"""ROS-independent helpers for deterministic recorded-image replay."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    wall_sec: float
    source_index: int


@dataclass(frozen=True)
class ScanRecord:
    wall_sec: float
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges_m: tuple[float, ...]
    source_index: int


@dataclass(frozen=True)
class ScanLoadStats:
    total_nonblank_lines: int
    loaded_records: int
    skipped_lines: tuple[int, ...]


@dataclass(frozen=True)
class FrameLoadStats:
    total_data_lines: int
    loaded_records: int
    skipped_lines: tuple[int, ...]
    clamped_wall_lines: tuple[int, ...]


def _natural_key(path: Path) -> tuple:
    """Sort numbered frames numerically while retaining deterministic paths."""

    parts = re.split(r"(\d+)", path.as_posix().casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def discover_image_files(source_path: str | Path) -> list[Path]:
    """Return one image or every image below a directory in natural order."""

    source = Path(source_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"replay source does not exist: {source}")
    if source.is_file():
        if source.suffix.casefold() not in IMAGE_SUFFIXES:
            raise ValueError(f"replay source is not a supported image: {source}")
        return [source.resolve()]
    if not source.is_dir():
        raise ValueError(f"replay source is neither an image nor directory: {source}")

    images = [
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    images.sort(key=_natural_key)
    if not images:
        raise ValueError(f"replay source contains no supported images: {source}")
    return images


def load_frame_records_with_stats(
    source_path: str | Path, fallback_fps: float
) -> tuple[list[FrameRecord], str, FrameLoadStats]:
    """Load frame wall times while explicitly reporting damaged metadata rows."""

    if fallback_fps <= 0.0 or not math.isfinite(fallback_fps):
        raise ValueError("fallback_fps must be positive and finite")
    source = Path(source_path).expanduser()
    session_dir = source if source.is_dir() else source.parent
    metadata = session_dir / "frames.csv"
    if not metadata.is_file():
        images = discover_image_files(source)
        records = [
            FrameRecord(path=path, wall_sec=index / fallback_fps, source_index=index)
            for index, path in enumerate(images)
        ]
        return records, "fixed_fps", FrameLoadStats(
            total_data_lines=len(records),
            loaded_records=len(records),
            skipped_lines=(),
            clamped_wall_lines=(),
        )

    records: list[FrameRecord] = []
    skipped_lines: list[int] = []
    clamped_wall_lines: list[int] = []
    total_data_lines = 0
    with metadata.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"idx", "file", "wall_sec"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"frames.csv is missing required fields {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            total_data_lines += 1
            try:
                source_index = int(row["idx"])
                wall_sec = float(row["wall_sec"])
                path = (session_dir / row["file"]).resolve()
            except (TypeError, ValueError):
                skipped_lines.append(row_number)
                continue
            if not math.isfinite(wall_sec):
                skipped_lines.append(row_number)
                continue
            if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
                skipped_lines.append(row_number)
                continue
            # Preserve camera/file order.  The recorder can flush two adjacent
            # wall samples a few milliseconds out of order; clamping is safer
            # than reversing the visual sequence.
            if records and wall_sec < records[-1].wall_sec:
                wall_sec = records[-1].wall_sec
                clamped_wall_lines.append(row_number)
            records.append(FrameRecord(path, wall_sec, source_index))
    if not records:
        raise ValueError(f"frames.csv contains no frames: {metadata}")
    return records, "recorded_wall", FrameLoadStats(
        total_data_lines=total_data_lines,
        loaded_records=len(records),
        skipped_lines=tuple(skipped_lines),
        clamped_wall_lines=tuple(clamped_wall_lines),
    )


def load_frame_records(
    source_path: str | Path, fallback_fps: float
) -> tuple[list[FrameRecord], str]:
    """Compatibility wrapper returning frames and the selected timing mode."""

    records, mode, _stats = load_frame_records_with_stats(source_path, fallback_fps)
    return records, mode


def load_scan_records_with_stats(
    scan_path: str | Path,
) -> tuple[list[ScanRecord], ScanLoadStats]:
    """Load compact scans, explicitly counting and skipping malformed lines.

    The field recorder can leave an interrupted JSON line (or an extra leading
    brace) while it flushes.  Such a line has no unambiguous payload, so replay
    never guesses a repair: it skips the complete line and exposes its number
    in ``ScanLoadStats``.
    """

    path = Path(scan_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"recorded scan does not exist: {path}")
    records: list[ScanRecord] = []
    skipped_lines: list[int] = []
    total_nonblank = 0
    with path.open("r", encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            total_nonblank += 1
            try:
                row = json.loads(line)
                source_index = int(row.get("i", row_number - 1))
                wall_sec = float(row["wall"])
                angle_min = float(row["a0"])
                angle_increment = float(row["da"])
                range_min = float(row["rmin"])
                range_max = float(row["rmax"])
                centimetres = row["cm"]
                numeric = (wall_sec, angle_min, angle_increment, range_min, range_max)
                if not all(math.isfinite(value) for value in numeric):
                    raise ValueError("non-finite metadata")
                if angle_increment == 0.0 or not 0.0 <= range_min < range_max:
                    raise ValueError("invalid scan geometry")
                if not isinstance(centimetres, list) or len(centimetres) < 2:
                    raise ValueError("invalid scan ranges")
                ranges = []
                for value in centimetres:
                    cm = float(value)
                    if not math.isfinite(cm):
                        raise ValueError("non-finite scan range")
                    # The recorder uses zero as no-return.  Recreate
                    # LaserScan's +inf so runtime filtering is identical.
                    ranges.append(math.inf if cm <= 0.0 else cm * 0.01)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                skipped_lines.append(row_number)
                continue
            records.append(
                ScanRecord(
                    wall_sec,
                    angle_min,
                    angle_increment,
                    range_min,
                    range_max,
                    tuple(ranges),
                    source_index,
                )
            )
    if not records:
        raise ValueError(f"scan.jsonl contains no scans: {path}")
    if any(b.wall_sec < a.wall_sec for a, b in zip(records, records[1:])):
        raise ValueError("scan.jsonl wall times must be monotonic")
    return records, ScanLoadStats(
        total_nonblank_lines=total_nonblank,
        loaded_records=len(records),
        skipped_lines=tuple(skipped_lines),
    )


def load_scan_records(scan_path: str | Path) -> list[ScanRecord]:
    """Compatibility wrapper returning only usable recorded scans."""

    records, _stats = load_scan_records_with_stats(scan_path)
    return records


def percentile(values: Iterable[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
