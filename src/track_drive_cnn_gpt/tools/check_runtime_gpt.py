#!/usr/bin/env python3
"""Read-only preflight checks.  This script never starts or stops ROS nodes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> int:
    profile = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "")
    print(f"FASTRTPS_DEFAULT_PROFILES_FILE={profile or '<unset>'}")
    if profile:
        path = Path(profile)
        print(f"DDS profile exists={path.is_file()} path={path}")

    result = subprocess.run(
        ["pgrep", "-af", "usb_cam|yolo|cnn_path|path_planner|motion|dynamic_bridge"],
        check=False,
        text=True,
        capture_output=True,
    )
    print("running relevant processes:")
    print(result.stdout.strip() or "<none>")
    if "--bridge-all-topics" in result.stdout:
        print("WARNING: ros1_bridge uses --bridge-all-topics; raw image topics may be bridged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

