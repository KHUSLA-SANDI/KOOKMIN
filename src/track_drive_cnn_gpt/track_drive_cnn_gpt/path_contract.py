"""CNN 경로 모델의 학습·런타임 공통 계약.

이 모듈은 NumPy만 사용한다. PyTorch가 설치되지 않은 개발 환경에서도 입력,
체크포인트 메타데이터, 모델 출력 경로를 동일한 규칙으로 검증할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


MODEL_SCHEMA_VERSION = "path_cnn_v1_gpt"
BEV_SCHEMA_VERSION = "bev_topology_v1_gpt"
MODEL_KIND_LEGACY = "legacy_single"
MODEL_KIND_DUAL = "canonical_dual"

INPUT_SHAPE: Tuple[int, int, int] = (3, 128, 120)
INPUT_CHANNELS: Tuple[str, str, str] = (
    "mid_yellow",
    "lane_white",
    "lidar_hits",
)
GRID_X_BOUNDS_M: Tuple[float, float] = (-0.20, 3.00)
GRID_Y_BOUNDS_M: Tuple[float, float] = (-1.50, 1.50)
GRID_RESOLUTION_M = 0.025
GRID_ROW_AXIS = "far_to_near"
GRID_COL_AXIS = "left_to_right"
OUT_X = np.round(np.arange(0.3, 3.001, 0.1), 3).astype(np.float32)
OUT_N = int(OUT_X.size)
DUAL_OUTPUT_HEADS: Tuple[str, ...] = (
    "main_y",
    "main_valid",
    "shortcut_y",
    "shortcut_valid",
)
SHORTCUT_POLICY_VALID_USABILITY = "shortcut_valid_usability"


class PathContractError(ValueError):
    """입력 또는 체크포인트가 고정 계약과 맞지 않을 때 발생한다."""


@dataclass(frozen=True)
class PathGuardConfig:
    """모델 출력이 motion으로 전달되기 전 통과해야 할 경계값."""

    valid_threshold: float = 0.5
    shortcut_threshold: float = 0.5
    min_points: int = 6
    min_span_m: float = 0.5
    # Dataset GT reaches |y|=1.50. Across the 8,391 training-eligible samples,
    # shortcut p99 is |dy/dx|=3.078 and |d2y/dx2|=11.187. These limits retain
    # that calibrated domain while still rejecting the extreme label tail.
    # min_points=6 intentionally makes 10 main / 15 shortcut short GT paths
    # unusable at runtime; motion needs at least a 0.5 m planning span.
    max_abs_y_m: float = 1.500001
    max_abs_slope: float = 3.1
    max_abs_curvature: float = 12.0

    def __post_init__(self) -> None:
        if not 0.0 < self.valid_threshold < 1.0:
            raise ValueError("valid_threshold must be between 0 and 1")
        if not 0.0 < self.shortcut_threshold < 1.0:
            raise ValueError("shortcut_threshold must be between 0 and 1")
        if self.min_points < 2 or self.min_points > OUT_N:
            raise ValueError("min_points must be in [2, OUT_N]")
        if self.min_span_m <= 0.0:
            raise ValueError("min_span_m must be positive")
        if min(self.max_abs_y_m, self.max_abs_slope, self.max_abs_curvature) <= 0.0:
            raise ValueError("path guard limits must be positive")


@dataclass(frozen=True)
class SanitizedPath:
    """검증을 통과해 motion에 전달할 수 있는 연속 경로 구간."""

    usable: bool
    reason: str
    x: np.ndarray
    y: np.ndarray
    valid_probability: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.x.size)

    @property
    def span_m(self) -> float:
        if self.x.size < 2:
            return 0.0
        return float(self.x[-1] - self.x[0])


@dataclass(frozen=True)
class SanitizedPathBundle:
    """main/shortcut 두 출력의 검증 결과와 shortcut 가용성."""

    main: SanitizedPath
    shortcut: SanitizedPath
    shortcut_available: bool
    shortcut_probability: float
    shortcut_availability_source: str = "none"


def _empty_path(reason: str, probabilities: Optional[np.ndarray] = None) -> SanitizedPath:
    probs = np.asarray(
        [] if probabilities is None else probabilities,
        dtype=np.float32,
    ).copy()
    return SanitizedPath(
        usable=False,
        reason=reason,
        x=np.empty(0, dtype=np.float32),
        y=np.empty(0, dtype=np.float32),
        valid_probability=probs,
    )


def _as_exact_vector(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PathContractError(f"{name} is not a numeric vector") from exc
    if array.shape != (OUT_N,):
        raise PathContractError(f"{name} shape {array.shape} != {(OUT_N,)}")
    return array


def _metadata_vector(metadata: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in metadata:
        raise PathContractError(f"checkpoint metadata missing {name}")
    return _as_exact_vector(metadata[name], f"metadata.{name}")


def canonical_metadata() -> dict:
    """새 dual 체크포인트에 그대로 저장할 정본 메타데이터를 반환한다."""

    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_kind": MODEL_KIND_DUAL,
        "input_shape": list(INPUT_SHAPE),
        "input_channels": list(INPUT_CHANNELS),
        "input_dtype": "uint8_or_float01",
        "input_range": [0.0, 1.0],
        "bev_schema_version": BEV_SCHEMA_VERSION,
        "bev_x_bounds_m": list(GRID_X_BOUNDS_M),
        "bev_y_bounds_m": list(GRID_Y_BOUNDS_M),
        "bev_resolution_m": GRID_RESOLUTION_M,
        "bev_row_axis": GRID_ROW_AXIS,
        "bev_col_axis": GRID_COL_AXIS,
        "out_x": OUT_X.tolist(),
        "output_heads": list(DUAL_OUTPUT_HEADS),
        "shortcut_availability_policy": SHORTCUT_POLICY_VALID_USABILITY,
    }


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    model_kind: str,
) -> dict:
    """체크포인트 메타데이터를 검증하고 정규화한 복사본을 반환한다.

    legacy 체크포인트는 과거 형식상 전체 메타데이터가 없으므로 ``out_x``만
    필수로 받는다. 다만 input shape/channel/schema가 기록돼 있다면 새 계약과
    정확히 일치해야 한다. canonical dual은 모든 필드를 필수로 요구한다.
    """

    if not isinstance(metadata, Mapping):
        raise PathContractError("checkpoint metadata must be a mapping")
    if model_kind not in (MODEL_KIND_LEGACY, MODEL_KIND_DUAL):
        raise PathContractError(f"unsupported model_kind: {model_kind}")

    normalized = dict(metadata)
    out_x = _metadata_vector(metadata, "out_x")
    if not np.allclose(out_x, OUT_X, rtol=0.0, atol=1e-6):
        raise PathContractError("checkpoint out_x does not match runtime contract")
    normalized["out_x"] = OUT_X.tolist()

    if model_kind == MODEL_KIND_DUAL:
        required = (
            "schema_version",
            "model_kind",
            "input_shape",
            "input_channels",
            "input_dtype",
            "input_range",
            "bev_schema_version",
            "bev_x_bounds_m",
            "bev_y_bounds_m",
            "bev_resolution_m",
            "bev_row_axis",
            "bev_col_axis",
            "output_heads",
            "shortcut_availability_policy",
        )
        missing = [name for name in required if name not in metadata]
        if missing:
            raise PathContractError(
                "canonical checkpoint metadata missing " + ", ".join(missing)
            )

    if "schema_version" in metadata:
        if metadata["schema_version"] != MODEL_SCHEMA_VERSION:
            raise PathContractError("checkpoint schema_version mismatch")
    if "model_kind" in metadata:
        if metadata["model_kind"] != model_kind:
            raise PathContractError("checkpoint model_kind mismatch")
    if "input_shape" in metadata:
        if tuple(int(v) for v in metadata["input_shape"]) != INPUT_SHAPE:
            raise PathContractError("checkpoint input_shape mismatch")
    if "input_channels" in metadata:
        if tuple(str(v) for v in metadata["input_channels"]) != INPUT_CHANNELS:
            raise PathContractError("checkpoint input_channels mismatch")
    if "input_dtype" in metadata:
        if metadata["input_dtype"] != "uint8_or_float01":
            raise PathContractError("checkpoint input_dtype mismatch")
    if "input_range" in metadata:
        try:
            bounds = tuple(float(v) for v in metadata["input_range"])
        except (TypeError, ValueError) as exc:
            raise PathContractError("checkpoint input_range is invalid") from exc
        if bounds != (0.0, 1.0):
            raise PathContractError("checkpoint input_range mismatch")
    if "bev_schema_version" in metadata:
        if metadata["bev_schema_version"] != BEV_SCHEMA_VERSION:
            raise PathContractError("checkpoint bev_schema_version mismatch")
    for name, expected in (
        ("bev_x_bounds_m", GRID_X_BOUNDS_M),
        ("bev_y_bounds_m", GRID_Y_BOUNDS_M),
    ):
        if name in metadata:
            try:
                bounds = tuple(float(v) for v in metadata[name])
            except (TypeError, ValueError) as exc:
                raise PathContractError(f"checkpoint {name} is invalid") from exc
            if len(bounds) != 2 or not np.allclose(
                bounds, expected, rtol=0.0, atol=1e-9
            ):
                raise PathContractError(f"checkpoint {name} mismatch")
    if "bev_resolution_m" in metadata:
        try:
            resolution = float(metadata["bev_resolution_m"])
        except (TypeError, ValueError) as exc:
            raise PathContractError("checkpoint bev_resolution_m is invalid") from exc
        if not np.isclose(resolution, GRID_RESOLUTION_M, rtol=0.0, atol=1e-12):
            raise PathContractError("checkpoint bev_resolution_m mismatch")
    if "bev_row_axis" in metadata:
        if metadata["bev_row_axis"] != GRID_ROW_AXIS:
            raise PathContractError("checkpoint bev_row_axis mismatch")
    if "bev_col_axis" in metadata:
        if metadata["bev_col_axis"] != GRID_COL_AXIS:
            raise PathContractError("checkpoint bev_col_axis mismatch")
    if "output_heads" in metadata:
        heads = tuple(str(v) for v in metadata["output_heads"])
        if model_kind == MODEL_KIND_DUAL and heads != DUAL_OUTPUT_HEADS:
            raise PathContractError("checkpoint output_heads mismatch")
    if "shortcut_availability_policy" in metadata:
        if metadata["shortcut_availability_policy"] != SHORTCUT_POLICY_VALID_USABILITY:
            raise PathContractError("checkpoint shortcut_availability_policy mismatch")

    return normalized


def validate_input_tensor(values: Any, allow_batch: bool = False) -> np.ndarray:
    """CNN 입력을 fail-closed 검증하고 연속 ``float32`` 0~1 배열로 바꾼다."""

    array = np.asarray(values)
    valid_shapes = (INPUT_SHAPE,)
    if allow_batch:
        shape_ok = array.ndim == 4 and tuple(array.shape[1:]) == INPUT_SHAPE
    else:
        shape_ok = tuple(array.shape) in valid_shapes
    if not shape_ok:
        expected = f"(N,{INPUT_SHAPE})" if allow_batch else str(INPUT_SHAPE)
        raise PathContractError(f"input shape {array.shape} != {expected}")

    if array.dtype == np.uint8:
        if array.size and int(array.max()) > 1:
            raise PathContractError("uint8 input values must be binary 0/1")
    elif np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise PathContractError("float input contains NaN or infinity")
        if array.size and (float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6):
            raise PathContractError("float input values must be in [0, 1]")
    else:
        raise PathContractError(f"unsupported input dtype: {array.dtype}")

    return np.ascontiguousarray(array, dtype=np.float32)


def count_camera_evidence(values: Any) -> int:
    """Count occupied yellow/white cells without considering LiDAR channel 2."""

    array = np.asarray(values)
    if tuple(array.shape) != INPUT_SHAPE:
        raise PathContractError(
            f"camera evidence input shape {array.shape} != {INPUT_SHAPE}"
        )
    return int(np.count_nonzero(array[:2]))


def sigmoid(values: Any) -> np.ndarray:
    """overflow 없이 logits를 확률로 변환한다."""

    logits = np.asarray(values, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_value = np.exp(logits[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _true_runs(mask: np.ndarray) -> list[Tuple[int, int]]:
    """Return half-open contiguous true runs in index order."""

    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def sanitize_path_prediction(
    y_values: Any,
    valid_values: Any,
    config: PathGuardConfig = PathGuardConfig(),
    *,
    valid_values_are_logits: bool = True,
) -> SanitizedPath:
    """예측 경로에서 유일한 연속 유효 구간을 좌표와 함께 남긴다.

    모델 출력 오류는 주행 코드까지 예외로 전파하지 않고 ``usable=False``로
    닫는다. 시작점 앞의 false는 허용하지만 분리된 유효 구간이 두 개 이상이면
    모호한 출력으로 거부한다. 메타데이터와 입력 자체의 오류는 별도 검증
    함수에서 예외로 막는다.
    """

    try:
        y = _as_exact_vector(y_values, "path_y")
        valid = _as_exact_vector(valid_values, "path_valid")
    except PathContractError as exc:
        return _empty_path(str(exc))
    if not np.isfinite(y).all() or not np.isfinite(valid).all():
        return _empty_path("prediction contains NaN or infinity")

    if valid_values_are_logits:
        probabilities = sigmoid(valid)
    else:
        if np.any((valid < 0.0) | (valid > 1.0)):
            return _empty_path("valid probabilities are outside [0, 1]")
        probabilities = valid

    runs = _true_runs(probabilities >= config.valid_threshold)
    if not runs:
        return _empty_path("valid mask has no contiguous run", probabilities)
    if len(runs) != 1:
        # Training labels contain one contiguous run (sometimes after leading
        # false values). Multiple predicted islands are ambiguous, so choosing
        # the longest one would silently invent a horizon; fail closed.
        return _empty_path("valid mask contains multiple runs", probabilities)
    start, end = runs[0]
    count = end - start
    if count < config.min_points:
        return _empty_path("valid run has too few points", probabilities)

    x = OUT_X[start:end].astype(np.float64)
    path_y = y[start:end]
    span = float(x[-1] - x[0])
    if span + 1e-9 < config.min_span_m:
        return _empty_path("valid run span is too short", probabilities)
    if np.max(np.abs(path_y)) > config.max_abs_y_m:
        return _empty_path("path exceeds lateral y limit", probabilities)

    dx = np.diff(x)
    slopes = np.diff(path_y) / dx
    if slopes.size and np.max(np.abs(slopes)) > config.max_abs_slope:
        return _empty_path("path exceeds slope limit", probabilities)
    if slopes.size >= 2:
        midpoint_x = 0.5 * (x[:-1] + x[1:])
        curvature = np.diff(slopes) / np.diff(midpoint_x)
        if np.max(np.abs(curvature)) > config.max_abs_curvature:
            return _empty_path("path exceeds curvature limit", probabilities)

    return SanitizedPath(
        usable=True,
        reason="ok",
        x=x.astype(np.float32),
        y=path_y.astype(np.float32),
        valid_probability=probabilities.astype(np.float32),
    )


def sanitize_dual_prediction(
    main_y: Any,
    main_valid: Any,
    shortcut_y: Optional[Any] = None,
    shortcut_valid: Optional[Any] = None,
    shortcut_available: Optional[Any] = None,
    config: PathGuardConfig = PathGuardConfig(),
    *,
    valid_values_are_logits: bool = True,
    allow_shortcut_without_availability: bool = False,
) -> SanitizedPathBundle:
    """단일/dual 출력을 공통 bundle로 변환한다.

    legacy 모델은 shortcut 인자들을 생략한다. canonical 모델은 availability와
    shortcut 경로 검증을 모두 통과해야 shortcut을 선택할 수 있다.
    """

    main = sanitize_path_prediction(
        main_y,
        main_valid,
        config,
        valid_values_are_logits=valid_values_are_logits,
    )
    if shortcut_y is None or shortcut_valid is None:
        return SanitizedPathBundle(
            main=main,
            shortcut=_empty_path("shortcut head is unavailable"),
            shortcut_available=False,
            shortcut_probability=0.0,
            shortcut_availability_source="none",
        )

    shortcut = sanitize_path_prediction(
        shortcut_y,
        shortcut_valid,
        config,
        valid_values_are_logits=valid_values_are_logits,
    )
    if shortcut_available is None:
        # The production four-head model has no fork-presence classifier.  Its
        # manifest proves that the shortcut head is supervised on every sample
        # (non-fork labels are copied from main).  Only an explicitly validated
        # loader may opt into using the sanitized shortcut-valid prefix as the
        # publication gate; this is path usability, not a fork probability.
        if allow_shortcut_without_availability:
            return SanitizedPathBundle(
                main=main,
                shortcut=shortcut,
                shortcut_available=shortcut.usable,
                shortcut_probability=0.0,
                shortcut_availability_source="shortcut_valid_usability",
            )
        return SanitizedPathBundle(
            main=main,
            shortcut=shortcut,
            shortcut_available=False,
            shortcut_probability=0.0,
            shortcut_availability_source="missing_availability_head",
        )

    availability = np.asarray(shortcut_available, dtype=np.float64)
    if availability.size != 1 or not np.isfinite(availability).all():
        probability = 0.0
        available = False
    else:
        scalar = float(availability.reshape(-1)[0])
        probability = float(sigmoid([scalar])[0])
        available = probability >= config.shortcut_threshold and shortcut.usable

    return SanitizedPathBundle(
        main=main,
        shortcut=shortcut,
        shortcut_available=available,
        shortcut_probability=probability,
        shortcut_availability_source="availability_head",
    )
