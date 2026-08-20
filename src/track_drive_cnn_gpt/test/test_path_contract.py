import importlib.util
from pathlib import Path

import numpy as np
import pytest

from track_drive_cnn_gpt.path_contract import (
    BEV_SCHEMA_VERSION,
    DUAL_OUTPUT_HEADS,
    GRID_COL_AXIS,
    GRID_RESOLUTION_M,
    GRID_ROW_AXIS,
    GRID_X_BOUNDS_M,
    GRID_Y_BOUNDS_M,
    INPUT_CHANNELS,
    INPUT_SHAPE,
    MODEL_KIND_DUAL,
    MODEL_KIND_LEGACY,
    OUT_N,
    OUT_X,
    PathContractError,
    PathGuardConfig,
    SHORTCUT_POLICY_VALID_USABILITY,
    canonical_metadata,
    count_camera_evidence,
    sanitize_dual_prediction,
    sanitize_path_prediction,
    validate_checkpoint_metadata,
    validate_input_tensor,
)


def high_prefix(count):
    values = np.full(OUT_N, -10.0, dtype=np.float32)
    values[:count] = 10.0
    return values


def real_training_checkpoint_payload(model, width=8):
    """Exact contract shape written by CNN학습준비_full_gpt/train_cnn_gpt.py."""

    manifest = {
        "schema_version": "cnn_dual_route_full_bev_v1_gpt",
        "input": {
            "dtype": "uint8",
            "shape": [3, 128, 120],
            "values": [0, 1],
            "channels": {
                "0": "mid_yellow",
                "1": "lane_white",
                "2": "lidar_hits",
            },
            "x_min_m": -0.2,
            "x_max_m": 3.0,
            "y_min_m": -1.5,
            "y_max_m": 1.5,
            "resolution_m": 0.025,
            "topology": "full_mask_occupancy_no_x_bin_median",
        },
        "output": {
            "x_m": OUT_X.tolist(),
            "heads": list(DUAL_OUTPUT_HEADS),
            "points_per_route": OUT_N,
            "nonfork_shortcut": "copied_from_main",
        },
    }
    return {
        "schema_version": "dual_route_path_cnn_checkpoint_v1_gpt",
        "model": model.state_dict(),
        "optimizer": {},
        "scheduler": {},
        "epoch": 1,
        "model_config": {
            "input_shape": [3, 128, 120],
            "n_points": OUT_N,
            "width": width,
            "heads": list(DUAL_OUTPUT_HEADS),
        },
        "manifest": manifest,
        "OUT_X": OUT_X.tolist(),
    }


def instantiate_actual_training_model(width=8):
    training_script = (
        Path(__file__).resolve().parents[4]
        / "CNN학습준비_full_gpt"
        / "train_cnn_gpt.py"
    )
    if not training_script.is_file():
        pytest.skip("the external training implementation fixture is not deployed")
    spec = importlib.util.spec_from_file_location(
        "actual_train_cnn_gpt_fixture", training_script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DualRoutePathCNN(width=width)


def test_constants_match_bev_and_path_contract():
    assert INPUT_SHAPE == (3, 128, 120)
    assert INPUT_CHANNELS == ("mid_yellow", "lane_white", "lidar_hits")
    assert OUT_N == 28
    np.testing.assert_allclose(OUT_X, np.arange(0.3, 3.01, 0.1), atol=1e-6)


def test_input_accepts_uint8_and_float_without_torch():
    binary = np.zeros(INPUT_SHAPE, dtype=np.uint8)
    binary[0, 0, 0] = 1
    converted = validate_input_tensor(binary)
    assert converted.dtype == np.float32
    assert converted.flags.c_contiguous

    floating = np.full(INPUT_SHAPE, 0.25, dtype=np.float64)
    assert validate_input_tensor(floating).shape == INPUT_SHAPE


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 120, 120), dtype=np.uint8),
        np.full(INPUT_SHAPE, 2, dtype=np.uint8),
        np.full(INPUT_SHAPE, 1.1, dtype=np.float32),
        np.zeros(INPUT_SHAPE, dtype=np.int16),
    ],
)
def test_input_rejects_shape_range_and_dtype_mismatch(bad):
    with pytest.raises(PathContractError):
        validate_input_tensor(bad)


def test_input_batch_is_explicitly_opt_in():
    batch = np.zeros((2,) + INPUT_SHAPE, dtype=np.uint8)
    with pytest.raises(PathContractError):
        validate_input_tensor(batch)
    assert validate_input_tensor(batch, allow_batch=True).shape == batch.shape


def test_camera_evidence_counts_both_semantic_channels_but_not_lidar():
    values = np.zeros(INPUT_SHAPE, dtype=np.uint8)
    values[2, 10, 10] = 1
    assert count_camera_evidence(values) == 0
    values[1, 10, 11] = 1
    assert count_camera_evidence(values) == 1


def test_canonical_metadata_is_complete_and_valid():
    metadata = canonical_metadata()
    checked = validate_checkpoint_metadata(metadata, MODEL_KIND_DUAL)
    assert tuple(checked["output_heads"]) == DUAL_OUTPUT_HEADS
    assert checked["bev_schema_version"] == BEV_SCHEMA_VERSION
    assert tuple(checked["bev_x_bounds_m"]) == GRID_X_BOUNDS_M
    assert tuple(checked["bev_y_bounds_m"]) == GRID_Y_BOUNDS_M
    assert checked["bev_resolution_m"] == GRID_RESOLUTION_M
    assert checked["bev_row_axis"] == GRID_ROW_AXIS
    assert checked["bev_col_axis"] == GRID_COL_AXIS
    assert (
        checked["shortcut_availability_policy"]
        == SHORTCUT_POLICY_VALID_USABILITY
    )


def test_legacy_metadata_accepts_old_minimal_form():
    checked = validate_checkpoint_metadata(
        {"out_x": OUT_X.tolist()},
        MODEL_KIND_LEGACY,
    )
    assert checked["out_x"] == OUT_X.tolist()


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_shape", [3, 120, 120]),
        ("input_channels", ["lane_white", "mid_yellow", "lidar_hits"]),
        ("out_x", np.linspace(0.0, 2.7, OUT_N).tolist()),
        ("output_heads", list(reversed(DUAL_OUTPUT_HEADS))),
        ("bev_schema_version", "old_bev"),
        ("bev_x_bounds_m", [-0.2, 2.8]),
        ("bev_y_bounds_m", [-1.4, 1.5]),
        ("bev_resolution_m", 0.05),
        ("bev_row_axis", "near_to_far"),
        ("bev_col_axis", "right_to_left"),
        ("shortcut_availability_policy", "unverified"),
    ],
)
def test_canonical_metadata_mismatch_fails_closed(field, value):
    metadata = canonical_metadata()
    metadata[field] = value
    with pytest.raises(PathContractError):
        validate_checkpoint_metadata(metadata, MODEL_KIND_DUAL)


def test_canonical_metadata_missing_bev_contract_fails_closed():
    for field in (
        "bev_schema_version",
        "bev_x_bounds_m",
        "bev_y_bounds_m",
        "bev_resolution_m",
        "bev_row_axis",
        "bev_col_axis",
        "shortcut_availability_policy",
    ):
        metadata = canonical_metadata()
        del metadata[field]
        with pytest.raises(PathContractError, match="missing"):
            validate_checkpoint_metadata(metadata, MODEL_KIND_DUAL)


def test_multiple_valid_runs_fail_closed_instead_of_inventing_horizon():
    y = np.linspace(0.0, 0.3, OUT_N, dtype=np.float32)
    valid = high_prefix(7)
    valid[8:] = 10.0
    path = sanitize_path_prediction(y, valid)
    assert not path.usable
    assert "multiple runs" in path.reason


def test_leading_false_then_single_contiguous_run_preserves_out_x_slice():
    y = np.linspace(0.0, 0.2, OUT_N, dtype=np.float32)
    valid = np.full(OUT_N, -10.0, dtype=np.float32)
    valid[2:9] = 10.0
    path = sanitize_path_prediction(y, valid)
    assert path.usable
    assert path.point_count == 7
    assert path.span_m == pytest.approx(0.6)
    np.testing.assert_allclose(path.x, OUT_X[2:9])
    np.testing.assert_allclose(path.y, y[2:9])


def test_short_valid_prefix_is_unusable():
    path = sanitize_path_prediction(np.zeros(OUT_N), high_prefix(5))
    assert not path.usable
    assert path.point_count == 0
    assert "too few" in path.reason


def test_minimum_span_guard_is_independent_from_point_count():
    config = PathGuardConfig(min_points=6, min_span_m=0.7)
    path = sanitize_path_prediction(np.zeros(OUT_N), high_prefix(7), config)
    assert not path.usable
    assert "span" in path.reason


def test_lateral_y_guard_fails_closed():
    y = np.zeros(OUT_N)
    y[:6] = 1.50001
    path = sanitize_path_prediction(y, high_prefix(6))
    assert not path.usable
    assert "lateral" in path.reason


def test_slope_guard_fails_closed():
    y = np.zeros(OUT_N)
    y[:6] = [0.0, 0.32, 0.0, 0.32, 0.0, 0.32]
    path = sanitize_path_prediction(y, high_prefix(6))
    assert not path.usable
    assert "slope" in path.reason


def test_curvature_guard_fails_closed():
    y = np.zeros(OUT_N)
    y[:6] = [0.00, 0.01, 0.00, 0.01, 0.00, 0.01]
    config = PathGuardConfig(max_abs_slope=10.0, max_abs_curvature=0.5)
    path = sanitize_path_prediction(y, high_prefix(6), config)
    assert not path.usable
    assert "curvature" in path.reason


def test_dual_shortcut_requires_head_and_availability_probability():
    y = np.linspace(0.0, 0.1, OUT_N)
    valid = high_prefix(8)
    bundle = sanitize_dual_prediction(y, valid, y, valid, [10.0])
    assert bundle.main.usable
    assert bundle.shortcut.usable
    assert bundle.shortcut_available
    assert bundle.shortcut_availability_source == "availability_head"

    unavailable = sanitize_dual_prediction(y, valid, y, valid, [-10.0])
    assert unavailable.shortcut.usable
    assert not unavailable.shortcut_available


def test_four_head_shortcut_requires_explicit_validated_policy():
    y = np.linspace(0.0, 0.1, OUT_N)
    valid = high_prefix(8)
    unverified = sanitize_dual_prediction(y, valid, y, valid)
    assert unverified.shortcut.usable
    assert not unverified.shortcut_available
    assert unverified.shortcut_availability_source == "missing_availability_head"

    verified = sanitize_dual_prediction(
        y,
        valid,
        y,
        valid,
        allow_shortcut_without_availability=True,
    )
    assert verified.shortcut_available
    assert verified.shortcut_probability == 0.0
    assert verified.shortcut_availability_source == "shortcut_valid_usability"


def test_legacy_single_prediction_maps_to_main_only():
    y = np.zeros(OUT_N)
    bundle = sanitize_dual_prediction(y, high_prefix(8))
    assert bundle.main.usable
    assert not bundle.shortcut.usable
    assert not bundle.shortcut_available


def test_torch_loaders_support_legacy_and_canonical_dual(tmp_path):
    torch = pytest.importorskip("torch")
    from track_drive_cnn_gpt.path_model import (
        CanonicalDualPathNet,
        LegacySinglePathNet,
        canonical_checkpoint,
        load_path_model,
    )

    legacy = LegacySinglePathNet(width=8)
    with torch.no_grad():
        legacy.head[-1].weight.zero_()
        legacy.head[-1].bias[:OUT_N].zero_()
        legacy.head[-1].bias[OUT_N:].fill_(10.0)
    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": legacy.state_dict(),
            "width": 8,
            "out_x": OUT_X.tolist(),
        },
        legacy_path,
    )
    loaded_legacy = load_path_model(legacy_path)
    legacy_bundle = loaded_legacy.predict(np.zeros(INPUT_SHAPE, dtype=np.uint8))
    assert loaded_legacy.model_kind == MODEL_KIND_LEGACY
    assert legacy_bundle.main.usable
    assert not legacy_bundle.shortcut_available

    dual = CanonicalDualPathNet(width=8)
    with torch.no_grad():
        for head in (dual.main_head, dual.shortcut_head):
            head.weight.zero_()
            head.bias[:OUT_N].zero_()
            head.bias[OUT_N:].fill_(10.0)
    dual_path = tmp_path / "dual.pt"
    torch.save(canonical_checkpoint(dual, width=8), dual_path)
    loaded_dual = load_path_model(dual_path)
    dual_bundle = loaded_dual.predict(np.zeros(INPUT_SHAPE, dtype=np.float32))
    assert loaded_dual.model_kind == MODEL_KIND_DUAL
    assert dual_bundle.main.usable
    assert dual_bundle.shortcut.usable
    assert dual_bundle.shortcut_available
    assert dual_bundle.shortcut_availability_source == "shortcut_valid_usability"


def test_torch_loader_rejects_canonical_metadata_mismatch(tmp_path):
    torch = pytest.importorskip("torch")
    from track_drive_cnn_gpt.path_model import (
        CanonicalDualPathNet,
        PathModelError,
        canonical_checkpoint,
        load_path_model,
    )

    model = CanonicalDualPathNet(width=8)
    payload = canonical_checkpoint(model, width=8)
    payload["metadata"]["input_shape"] = [3, 120, 120]
    path = tmp_path / "bad_metadata.pt"
    torch.save(payload, path)
    with pytest.raises(PathModelError, match="input_shape mismatch"):
        load_path_model(path)


def test_loader_accepts_exact_real_training_checkpoint_format(tmp_path):
    torch = pytest.importorskip("torch")
    from track_drive_cnn_gpt.path_model import (
        TRAINING_CHECKPOINT_SCHEMA,
        load_path_model,
    )

    model = instantiate_actual_training_model(width=8)
    assert any(key.startswith("encoder.0.") for key in model.state_dict())
    assert not any("available" in key for key in model.state_dict())
    with torch.no_grad():
        for head in (model.main_head, model.shortcut_head):
            head.weight.zero_()
            head.bias[:OUT_N].zero_()
            head.bias[OUT_N:].fill_(10.0)

    checkpoint_path = tmp_path / "path_cnn_dual_best_gpt.pt"
    torch.save(real_training_checkpoint_payload(model), checkpoint_path)
    loaded = load_path_model(checkpoint_path)
    assert loaded.metadata["checkpoint_schema_version"] == TRAINING_CHECKPOINT_SCHEMA
    assert loaded.metadata["shortcut_supervision"].startswith("all_samples")
    bundle = loaded.predict(np.zeros(INPUT_SHAPE, dtype=np.uint8))
    assert bundle.main.usable
    assert bundle.shortcut.usable
    assert bundle.shortcut_available
    assert bundle.shortcut_probability == 0.0
    assert bundle.shortcut_availability_source == "shortcut_valid_usability"


def test_loader_verifies_optional_checkpoint_sha256_before_execution(tmp_path):
    import hashlib

    torch = pytest.importorskip("torch")
    from track_drive_cnn_gpt.path_model import (
        CanonicalDualPathNet,
        PathModelError,
        load_path_model,
    )

    checkpoint_path = tmp_path / "verified.pt"
    torch.save(
        real_training_checkpoint_payload(CanonicalDualPathNet(width=8)),
        checkpoint_path,
    )
    expected = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    loaded = load_path_model(checkpoint_path, expected_sha256=expected)
    assert loaded.metadata["checkpoint_sha256"] == expected

    with pytest.raises(PathModelError, match="SHA256 mismatch"):
        load_path_model(checkpoint_path, expected_sha256="0" * 64)
    with pytest.raises(PathModelError, match="64 lowercase hex"):
        load_path_model(checkpoint_path, expected_sha256="not-a-sha")


@pytest.mark.parametrize(
    "mismatch,expected_message",
    [
        ("resolution", "resolution_m mismatch"),
        ("heads", "model_config heads mismatch"),
        ("out_x", "OUT_X mismatch"),
        ("points", "points_per_route mismatch"),
        ("shortcut_supervision", "does not prove all-sample"),
    ],
)
def test_real_training_checkpoint_contract_mismatch_fails_closed(
    tmp_path, mismatch, expected_message
):
    torch = pytest.importorskip("torch")
    from track_drive_cnn_gpt.path_model import (
        CanonicalDualPathNet,
        PathModelError,
        load_path_model,
    )

    payload = real_training_checkpoint_payload(CanonicalDualPathNet(width=8))
    if mismatch == "resolution":
        payload["manifest"]["input"]["resolution_m"] = 0.05
    elif mismatch == "heads":
        payload["model_config"]["heads"] = list(reversed(DUAL_OUTPUT_HEADS))
    elif mismatch == "out_x":
        payload["OUT_X"][0] = 0.2
    elif mismatch == "points":
        payload["manifest"]["output"]["points_per_route"] = 27
    else:
        payload["manifest"]["output"]["nonfork_shortcut"] = "optional"
    checkpoint_path = tmp_path / f"bad_{mismatch}.pt"
    torch.save(payload, checkpoint_path)
    with pytest.raises(PathModelError, match=expected_message):
        load_path_model(checkpoint_path)
