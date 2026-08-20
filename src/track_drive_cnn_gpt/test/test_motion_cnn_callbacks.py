from types import SimpleNamespace

import numpy as np

import track_drive_cnn_gpt.motion_cnn_node as motion_module
from track_drive_cnn_gpt.motion_cnn_node import CnnMotionNode


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, **_kwargs):
        self.warnings.append(str(message))


class _Clock:
    def __init__(self, now_ns):
        self._now_ns = int(now_ns)

    def now(self):
        return SimpleNamespace(
            nanoseconds=self._now_ns,
            clock_type="test-clock",
        )


class _FakeTime:
    def __init__(self, *, nanoseconds, clock_type):
        self.nanoseconds = int(nanoseconds)
        self.clock_type = clock_type


class _MotionHarness:
    LOOKAHEAD_X_MIN = 0.3
    LOOKAHEAD_X_MAX = 3.0
    PATH_STALE_SEC = 0.25

    def __init__(self, now_ns):
        self._cnn_path_frame_id = "lidar_frame"
        self._path_future_tolerance_sec = 0.05
        self._cnn_last_source_ns = -1
        self._path_xs = None
        self._path_ys = None
        self._path_stamp = None
        self._clock = _Clock(now_ns)
        self._logger = _Logger()

    def get_clock(self):
        return self._clock

    def get_logger(self):
        return self._logger

    def _clear_cnn_path(self, reason):
        CnnMotionNode._clear_cnn_path(self, reason)


def _message(stamp_ns, *, x=(0.3, 0.4, 0.5), y=(0.0, 0.0, 0.0)):
    poses = [
        SimpleNamespace(position=SimpleNamespace(x=x_value, y=y_value))
        for x_value, y_value in zip(x, y)
    ]
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id="lidar_frame",
            stamp=SimpleNamespace(
                sec=int(stamp_ns) // 1_000_000_000,
                nanosec=int(stamp_ns) % 1_000_000_000,
            ),
        ),
        poses=poses,
    )


def test_motion_callback_keeps_original_header_stamp(monkeypatch):
    monkeypatch.setattr(motion_module, "Time", _FakeTime)
    harness = _MotionHarness(now_ns=10_000_000_000)
    source_ns = 9_900_000_000

    CnnMotionNode._on_path(harness, _message(source_ns))

    assert np.array_equal(harness._path_xs, np.array([0.3, 0.4, 0.5]))
    assert harness._path_stamp.nanoseconds == source_ns
    assert harness._cnn_last_source_ns == source_ns


def test_empty_new_result_clears_motion_path_cache(monkeypatch):
    monkeypatch.setattr(motion_module, "Time", _FakeTime)
    harness = _MotionHarness(now_ns=10_000_000_000)
    CnnMotionNode._on_path(harness, _message(9_850_000_000))
    assert harness._path_xs is not None

    CnnMotionNode._on_path(
        harness,
        _message(9_900_000_000, x=(), y=()),
    )

    assert harness._path_xs is None
    assert harness._path_ys is None
    assert harness._path_stamp is None


def test_stale_new_result_clears_motion_path_cache(monkeypatch):
    monkeypatch.setattr(motion_module, "Time", _FakeTime)
    harness = _MotionHarness(now_ns=10_000_000_000)
    CnnMotionNode._on_path(harness, _message(9_900_000_000))
    assert harness._path_xs is not None

    harness._clock._now_ns = 11_000_000_000
    CnnMotionNode._on_path(harness, _message(10_000_000_000))

    assert harness._path_xs is None
    assert harness._path_stamp is None


def test_delayed_older_empty_does_not_erase_a_newer_valid_path(monkeypatch):
    monkeypatch.setattr(motion_module, "Time", _FakeTime)
    harness = _MotionHarness(now_ns=10_000_000_000)
    CnnMotionNode._on_path(harness, _message(9_900_000_000))
    current_x = harness._path_xs.copy()

    CnnMotionNode._on_path(
        harness,
        _message(9_800_000_000, x=(), y=()),
    )

    assert np.array_equal(harness._path_xs, current_x)
    assert harness._path_stamp.nanoseconds == 9_900_000_000


def test_motion_segment_does_not_reanchor_or_generate_relative_arc_length():
    harness = _MotionHarness(now_ns=10_000_000_000)
    x = np.array([0.3, 0.6, 1.2, 3.0])
    y = np.array([0.2, 0.1, -0.1, -0.2])

    seg_x, seg_y, focus_x = CnnMotionNode._reanchored_segment(harness, x, y)

    assert np.array_equal(seg_x, x)
    assert np.array_equal(seg_y, y)
    assert np.array_equal(focus_x, x)
    assert focus_x[0] == 0.3
