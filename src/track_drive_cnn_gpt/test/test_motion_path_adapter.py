from types import SimpleNamespace

import numpy as np
import pytest

from track_drive_cnn_gpt.motion_path_adapter import (
    prepare_ego_relative_path,
    stamp_to_ns,
    validate_source_stamp_ns,
)


def _stamp(sec, nanosec):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def test_ros_stamp_conversion_is_exact_and_strict():
    assert stamp_to_ns(_stamp(12, 345)) == 12_000_000_345

    for bad in (
        _stamp(-1, 0),
        _stamp(1, -1),
        _stamp(1, 1_000_000_000),
        SimpleNamespace(sec=1),
    ):
        with pytest.raises(ValueError, match="stamp"):
            stamp_to_ns(bad)


def test_source_stamp_freshness_uses_capture_time_not_receipt_time():
    now_ns = 20_000_000_000
    assert validate_source_stamp_ns(
        19_800_000_000,
        now_ns,
        stale_sec=0.25,
    ) == pytest.approx(0.2)

    with pytest.raises(ValueError, match="zero"):
        validate_source_stamp_ns(0, now_ns, stale_sec=0.25)
    with pytest.raises(ValueError, match="stale"):
        validate_source_stamp_ns(
            19_749_999_999,
            now_ns,
            stale_sec=0.25,
        )
    with pytest.raises(ValueError, match="future"):
        validate_source_stamp_ns(
            20_050_000_001,
            now_ns,
            stale_sec=0.25,
            future_tolerance_sec=0.05,
        )


def test_cnn_path_keeps_absolute_forward_x_for_motion_lookahead():
    x = np.linspace(0.3, 3.0, 28, dtype=np.float32)
    y = 0.15 * np.sin(x)

    path = prepare_ego_relative_path(x, y)

    assert path.x[0] == pytest.approx(0.3)
    assert path.x[-1] == pytest.approx(3.0)
    assert np.array_equal(path.focus_x, path.x)
    # Legacy rel_s/reanchor would make the first focus coordinate zero.  The
    # CNN contract deliberately retains metres ahead of the current vehicle.
    assert path.focus_x[0] != pytest.approx(0.0)


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        ([], [], "too few"),
        ([0.3], [0.0], "too few"),
        ([0.3, 0.4], [0.0], "same-length"),
        ([0.3, 0.5, 0.4], [0.0, 0.0, 0.0], "strictly increasing"),
        ([0.2, 0.4], [0.0, 0.0], "outside"),
        ([0.3, 3.1], [0.0, 0.0], "outside"),
        ([0.3, 0.4], [0.0, float("nan")], "NaN"),
    ],
)
def test_invalid_or_empty_cnn_paths_are_rejected_instead_of_reused(x, y, message):
    with pytest.raises(ValueError, match=message):
        prepare_ego_relative_path(x, y)

