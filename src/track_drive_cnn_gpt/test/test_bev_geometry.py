import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from track_drive_cnn_gpt.bev_geometry import (
    CH_LANE,
    CH_LIDAR,
    CH_MID,
    DEFAULT_EXPECTED_MODEL_SHA256,
    GRID_H,
    GRID_W,
    RES,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    OPENVINO_MANIFEST_SCHEMA,
    BevMaskRemapper,
    class_ids,
    full_mask_to_grid,
    grid_xy_to_rc,
    letterbox_shape,
    lidar_grid,
    load_camera,
    make_bev_image,
    model_class_names,
    native_lane_mid_unions,
    normalize_imgsz,
    optional_native_mask_hw,
    remap_optional_native_masks,
    validate_classifier_model_contract,
    validate_model_contract,
)


def _identity_grid_camera() -> dict[str, object]:
    # Pixel (u, v) maps to ground (x, y) with the exact online grid ordering:
    # x = X_MAX - RES*v and y = Y_MAX - RES*u.
    return {
        "K": np.eye(3, dtype=np.float64),
        "D": np.zeros(5, dtype=np.float64),
        "new_K": np.eye(3, dtype=np.float64),
        "H": np.array(
            [
                [0.0, -RES, X_MAX],
                [-RES, 0.0, Y_MAX],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "car_mask_v": float("inf"),
    }


class _PresenceRemapper:
    """Test double that maps any nonempty native mask to one occupied BEV cell."""

    def apply(self, native_mask: np.ndarray) -> np.ndarray:
        output = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        if np.count_nonzero(native_mask):
            output[10, 20] = 1
        return output


def _fake_yolo_result(class_ids: list[int]) -> SimpleNamespace:
    masks = []
    for index in range(len(class_ids)):
        mask = np.zeros((8, 12), dtype=np.float32)
        mask[1 + index, 2 + index] = 1.0
        masks.append(mask)
    if masks:
        mask_data = np.stack(masks)
    else:
        mask_data = np.zeros((0, 8, 12), dtype=np.float32)
    return SimpleNamespace(
        masks=SimpleNamespace(data=mask_data),
        boxes=SimpleNamespace(cls=np.asarray(class_ids, dtype=np.float32)),
    )


def test_full_topology_contract_is_exact() -> None:
    assert (GRID_H, GRID_W) == (128, 120)
    assert (X_MIN, X_MAX, Y_MIN, Y_MAX, RES) == (-0.20, 3.0, -1.5, 1.5, 0.025)
    assert (CH_MID, CH_LANE, CH_LIDAR) == (0, 1, 2)
    assert GRID_H * RES == pytest.approx(X_MAX - X_MIN)
    assert GRID_W * RES == pytest.approx(Y_MAX - Y_MIN)


def test_ultralytics_auto_letterbox_for_1080p() -> None:
    final_h, final_w, scale, dw, dh = letterbox_shape((1080, 1920), 640)
    assert (final_h, final_w) == (384, 640)
    assert scale == pytest.approx(1.0 / 3.0)
    assert dw == 0.0
    assert dh == 12.0

    rectangular = letterbox_shape((1080, 1920), (384, 640))
    assert rectangular == pytest.approx((384, 640, 1.0 / 3.0, 0.0, 12.0))


def test_imgsz_is_normalized_to_explicit_height_width() -> None:
    assert normalize_imgsz(640) == (640, 640)
    assert normalize_imgsz([384, 640]) == (384, 640)
    assert normalize_imgsz((640,)) == (640, 640)
    with pytest.raises(ValueError, match="one or two"):
        normalize_imgsz([320, 384, 640])


def test_actual_static_openvino_mask_shape_overrides_auto_letterbox() -> None:
    remapper = BevMaskRemapper(
        _identity_grid_camera(),
        (GRID_H, GRID_W),
        640,
        mask_hw=(640, 640),
        supersample=3,
    )
    assert remapper.mask_hw == (640, 640)
    output = remapper.apply(np.ones((640, 640), dtype=np.uint8))
    # Dataset validity is inclusive only through raw pixel index width/height-1;
    # sub-cell samples in the final boundary cells therefore remain outside.
    np.testing.assert_array_equal(
        output[:-1, :-1], np.ones((GRID_H - 1, GRID_W - 1), dtype=np.uint8)
    )
    assert np.count_nonzero(output[-1, :]) == 0
    assert np.count_nonzero(output[:, -1]) == 0


def test_identity_mapping_preserves_full_binary_topology() -> None:
    native = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    native[30:40, 50:60] = 1
    remapper = BevMaskRemapper(
        _identity_grid_camera(),
        (GRID_H, GRID_W),
        640,
        mask_hw=native.shape,
        supersample=3,
    )
    output = remapper.apply(native)
    assert output.shape == (GRID_H, GRID_W)
    assert output.dtype == np.uint8
    assert output[35, 55] == 1
    assert output[0, 0] == 0
    assert output[-1, -1] == 0


def test_one_shot_and_precomputed_remapping_match() -> None:
    native = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    native[:, 58:62] = 1
    camera = _identity_grid_camera()
    expected = BevMaskRemapper(
        camera,
        (GRID_H, GRID_W),
        640,
        mask_hw=native.shape,
        supersample=1,
    ).apply(native)
    actual = full_mask_to_grid(native, camera, supersample=1)
    np.testing.assert_array_equal(actual, expected)


def test_remapper_rejects_wrong_native_shape() -> None:
    remapper = BevMaskRemapper(
        _identity_grid_camera(),
        (GRID_H, GRID_W),
        640,
        mask_hw=(GRID_H, GRID_W),
    )
    with pytest.raises(ValueError, match="does not match"):
        remapper.apply(np.zeros((10, 10), dtype=np.uint8))


def test_bev_image_has_ros_8uc3_channel_order_and_zero_lidar() -> None:
    mid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    lane = np.zeros_like(mid)
    mid[10, 20] = 255
    lane[30, 40] = 7
    packed = make_bev_image(mid, lane)
    assert packed.shape == (GRID_H, GRID_W, 3)
    assert packed.dtype == np.uint8
    assert packed[10, 20, CH_MID] == 1
    assert packed[30, 40, CH_LANE] == 1
    assert np.count_nonzero(packed[..., CH_LIDAR]) == 0


def test_class_ids_are_name_based_not_legacy_numeric_assumptions() -> None:
    names = {0: "GREEN", 1: "LEFT", 2: "RED", 3: "YELLOW", 4: "lane", 5: "mid"}
    assert class_ids(names) == (4, 5)
    with pytest.raises(ValueError, match="mid"):
        class_ids({0: "lane"})


def test_exported_class_names_do_not_initialize_ultralytics_predictor(
    tmp_path: Path,
) -> None:
    classes = {
        "0": "GREEN",
        "1": "LEFT",
        "2": "RED",
        "3": "YELLOW",
        "4": "lane",
        "5": "mid",
    }
    model_dir = tmp_path / "best_openvino_model"
    model_dir.mkdir()
    (model_dir / "export_manifest_gpt.json").write_text(
        json.dumps({"classes": classes}), encoding="utf-8"
    )

    class PredictorTrap:
        @property
        def names(self):
            raise AssertionError("YOLO.names must not be read for exported models")

    assert model_class_names(model_dir, PredictorTrap()) == classes


def test_pytorch_class_names_use_inner_model_without_predictor(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"placeholder")
    names = {4: "lane", 5: "mid"}
    model = SimpleNamespace(model=SimpleNamespace(names=names))
    assert model_class_names(model_path, model) == names


def test_grid_and_lidar_use_forward_left_coordinate_convention() -> None:
    far_row, left_col, valid = grid_xy_to_rc(
        X_MAX - 0.5 * RES, Y_MAX - 0.5 * RES
    )
    assert (int(far_row), int(left_col), bool(valid)) == (0, 0, True)
    center_row, center_col, valid = grid_xy_to_rc(1.0, 0.0)
    assert bool(valid)
    center_row, center_col = int(center_row), int(center_col)
    grid = lidar_grid(
        np.array([[1.0, 0.0], [99.0, 99.0]]), radius_cells=0
    )
    assert grid[center_row, center_col] == 1
    assert np.count_nonzero(grid) == 1


def test_lidar_radius_one_matches_dataset_three_by_three_square() -> None:
    row, col, valid = grid_xy_to_rc(1.0, 0.0)
    assert bool(valid)
    row, col = int(row), int(col)
    grid = lidar_grid(np.array([[1.0, 0.0]]))
    expected = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    expected[row - 1 : row + 2, col - 1 : col + 2] = 1
    np.testing.assert_array_equal(grid, expected)


def test_repository_camera_yaml_loads_and_builds_real_remapper() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "camera.yaml"
    camera = load_camera(config_path)
    assert camera["K"].shape == (3, 3)
    assert camera["D"].shape == (5,)
    assert (camera["image_height"], camera["image_width"]) == (1080, 1920)
    remapper = BevMaskRemapper(
        camera,
        (1080, 1920),
        640,
        mask_hw=(384, 640),
        supersample=1,
    )
    output = remapper.apply(np.zeros((384, 640), dtype=np.uint8))
    assert output.shape == (GRID_H, GRID_W)
    assert np.count_nonzero(output) == 0


@pytest.mark.parametrize(
    ("detected_classes", "expect_lane", "expect_mid"),
    [
        ([4], True, False),
        ([5], False, True),
        ([0], False, False),
    ],
    ids=("lane-only", "mid-only", "neither-lane-nor-mid"),
)
def test_optional_lane_mid_classes_are_processed_independently(
    detected_classes: list[int], expect_lane: bool, expect_mid: bool
) -> None:
    lane_native, mid_native = native_lane_mid_unions(
        _fake_yolo_result(detected_classes), lane_id=4, mid_id=5
    )
    assert (lane_native is not None) is expect_lane
    assert (mid_native is not None) is expect_mid

    mask_hw = optional_native_mask_hw(lane_native, mid_native)
    assert (mask_hw is not None) is (expect_lane or expect_mid)
    if mask_hw is not None:
        assert mask_hw == (8, 12)

    remapper = _PresenceRemapper() if mask_hw is not None else None
    lane_grid, mid_grid = remap_optional_native_masks(
        lane_native, mid_native, remapper  # type: ignore[arg-type]
    )
    assert bool(np.count_nonzero(lane_grid)) is expect_lane
    assert bool(np.count_nonzero(mid_grid)) is expect_mid


def test_pt_model_contract_checks_exact_sha256(tmp_path: Path) -> None:
    model_path = tmp_path / "best_0817.pt"
    model_path.write_bytes(b"known production checkpoint bytes")
    expected = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert validate_model_contract(model_path, expected, 640) == "pytorch"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_model_contract(model_path, "0" * 64, 640)


def _openvino_fixture(tmp_path: Path, **overrides: object) -> tuple[Path, str]:
    model_dir = tmp_path / "best_0817_openvino_model"
    model_dir.mkdir()
    (model_dir / "best_0817.xml").write_text("<xml/>", encoding="utf-8")
    (model_dir / "best_0817.bin").write_bytes(b"openvino weights")
    expected = DEFAULT_EXPECTED_MODEL_SHA256
    manifest: dict[str, object] = {
        "schema_version": OPENVINO_MANIFEST_SCHEMA,
        "source_sha256": expected,
        "imgsz": [384, 640],
        "input_shape": [1, 3, 384, 640],
        "dynamic": False,
        "batch": 1,
        "precision": "FP32",
        "task": "segment",
        "classes": {
            "0": "GREEN",
            "1": "LEFT",
            "2": "RED",
            "3": "YELLOW",
            "4": "lane",
            "5": "mid",
        },
    }
    manifest.update(overrides)
    (model_dir / "export_manifest_gpt.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return model_dir, expected


def test_openvino_contract_accepts_traceable_matching_export(tmp_path: Path) -> None:
    model_dir, expected = _openvino_fixture(tmp_path)
    assert validate_model_contract(model_dir, expected, (384, 640)) == "openvino"


def test_classifier_openvino_contract_accepts_four_signal_classes(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "traffic_classifier_openvino_model"
    model_dir.mkdir()
    (model_dir / "model.xml").write_text("<xml/>", encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"weights")
    expected = "1" * 64
    manifest = {
        "schema_version": OPENVINO_MANIFEST_SCHEMA,
        "source_sha256": expected,
        "imgsz": [224, 224],
        "input_shape": [1, 3, 224, 224],
        "dynamic": False,
        "batch": 1,
        "precision": "FP32",
        "task": "classify",
        "classes": {"0": "GREEN", "1": "LEFT", "2": "RED", "3": "YELLOW"},
    }
    (model_dir / "export_manifest_gpt.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert (
        validate_classifier_model_contract(model_dir, expected, (224, 224))
        == "openvino"
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"source_sha256": "0" * 64}, "source SHA256 mismatch"),
        ({"imgsz": [640, 640], "input_shape": [1, 3, 640, 640]}, "imgsz mismatch"),
        ({"input_shape": [1, 3, 640, 640]}, "input_shape mismatch"),
        ({"dynamic": True}, "dynamic=false"),
        ({"precision": "FP16"}, "precision=FP32"),
        ({"classes": {"4": "lane"}}, "required classes"),
    ],
    ids=(
        "wrong-source",
        "wrong-imgsz",
        "wrong-input-shape",
        "dynamic-export",
        "wrong-precision",
        "missing-mid",
    ),
)
def test_openvino_contract_rejects_wrong_export(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    model_dir, expected = _openvino_fixture(tmp_path, **override)
    with pytest.raises(ValueError, match=message):
        validate_model_contract(model_dir, expected, (384, 640))


def test_openvino_contract_rejects_legacy_square_manifest(tmp_path: Path) -> None:
    model_dir, expected = _openvino_fixture(
        tmp_path, schema_version="yolo_openvino_export_v1_gpt"
    )
    with pytest.raises(ValueError, match="re-export"):
        validate_model_contract(model_dir, expected, (384, 640))


def test_latest_only_buffer_overwrites_without_decoding_old_message() -> None:
    # Node-only utility: skip just this test on non-ROS developer machines;
    # all geometry/model-contract tests above remain fully pure and collected.
    try:
        from sensor_msgs.msg import Image
        from track_drive_cnn_gpt.yolo_bev_node import _LatestOnlyBuffer, _image_to_bgr
    except ModuleNotFoundError as exc:
        pytest.skip(f"ROS/Ultralytics runtime is not installed: {exc}")

    slot = _LatestOnlyBuffer()
    first_message = Image()
    first_message.encoding = "deliberately-invalid-and-must-not-be-decoded"
    newest_message = Image()
    newest_message.height = 1
    newest_message.width = 1
    newest_message.encoding = "rgb8"
    newest_message.step = 3
    newest_message.data = bytes((255, 0, 0))
    assert slot.replace(first_message) is False
    assert slot.replace(newest_message) is True
    selected = slot.take()
    assert selected is newest_message
    # Retained rclpy message owns its buffer after the callback scope; only the
    # selected newest frame is decoded, and RGB red becomes OpenCV BGR red.
    assert _image_to_bgr(selected)[0, 0].tolist() == [0, 0, 255]
    slot.close()
    assert slot.take() is None
    assert slot.replace(object()) is None
