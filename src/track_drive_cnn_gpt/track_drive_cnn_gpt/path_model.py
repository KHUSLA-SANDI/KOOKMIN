"""Legacy single 및 canonical dual CNN 경로 체크포인트 어댑터."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .path_contract import (
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
    SanitizedPathBundle,
    canonical_metadata,
    sanitize_dual_prediction,
    validate_checkpoint_metadata,
    validate_input_tensor,
)


TRAINING_CHECKPOINT_SCHEMA = "dual_route_path_cnn_checkpoint_v1_gpt"
TRAINING_MANIFEST_SCHEMA = "cnn_dual_route_full_bev_v1_gpt"
TRAINING_BEV_TOPOLOGY = "full_mask_occupancy_no_x_bin_median"

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


class PathModelError(RuntimeError):
    """체크포인트 구조 또는 모델 가중치가 계약과 맞지 않을 때 발생한다."""


class TorchUnavailableError(PathModelError):
    """모델 작업을 요청했지만 PyTorch가 설치되지 않은 경우."""


def _require_torch() -> None:
    if torch is None or nn is None:
        raise TorchUnavailableError("PyTorch is required for path model loading/inference")


if nn is not None:

    def _encoder(width: int, *, adaptive_pool: bool = False):
        def block(in_channels: int, out_channels: int):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        layers = [
            block(3, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 4),
            block(width * 4, width * 4),
        ]
        if adaptive_pool:
            layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        return nn.Sequential(*layers)


    class LegacySinglePathNet(nn.Module):
        """기존 ``10_train.py``의 Net과 state_dict key까지 같은 모델."""

        def __init__(self, width: int = 32, output_count: int = OUT_N):
            super().__init__()
            self.enc = _encoder(width)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(width * 4 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, output_count * 2),
            )
            self.output_count = int(output_count)

        def forward(self, values):
            output = self.head(self.enc(values))
            return output[:, : self.output_count], output[:, self.output_count :]


    class CanonicalDualPathNet(nn.Module):
        """실제 ``train_cnn_gpt.py.DualRoutePathCNN``과 동일한 모델.

        모듈 이름(``encoder``, ``shared``, 두 route head)과 forward 반환 형식까지
        학습 코드와 맞춰 실제 state_dict를 strict=True로 불러온다.
        """

        def __init__(self, width: int = 32, output_count: int = OUT_N):
            super().__init__()
            self.n_points = int(output_count)
            self.width = int(width)
            self.encoder = _encoder(width, adaptive_pool=True)
            self.shared = nn.Sequential(
                nn.Flatten(),
                nn.Linear(width * 4 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
            self.main_head = nn.Linear(256, output_count * 2)
            self.shortcut_head = nn.Linear(256, output_count * 2)
            self.output_count = int(output_count)

        def forward(self, values):
            features = self.shared(self.encoder(values))
            main = self.main_head(features)
            shortcut = self.shortcut_head(features)
            count = self.output_count
            return {
                "main_y": main[:, :count],
                "main_valid": main[:, count:],
                "shortcut_y": shortcut[:, :count],
                "shortcut_valid": shortcut[:, count:],
            }

else:

    class LegacySinglePathNet:
        def __init__(self, *args, **kwargs):
            _require_torch()


    class CanonicalDualPathNet:
        def __init__(self, *args, **kwargs):
            _require_torch()


def _state_dict_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    if payload and all(hasattr(value, "shape") for value in payload.values()):
        return payload
    raise PathModelError("checkpoint does not contain a model state_dict")


def _strip_state_prefixes(state_dict: Mapping[str, Any]) -> dict:
    prefixes = ("module.", "_orig_mod.")
    output = {}
    for key, value in state_dict.items():
        normalized = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
        output[normalized] = value
    return output


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PathModelError(f"training checkpoint {name} must be a mapping")
    return value


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise PathModelError(f"training checkpoint {name} is invalid") from exc
    if not np.isfinite(result).all():
        raise PathModelError(f"training checkpoint {name} contains non-finite values")
    return result


def _training_checkpoint_metadata(payload: Mapping[str, Any]) -> dict:
    """Validate and adapt the exact payload written by ``train_cnn_gpt.py``.

    That training script predates ``canonical_checkpoint`` and stores the BEV
    contract in ``model_config`` plus the embedded dataset ``manifest``.  Its
    known manifest schema/topology fixes row=far-to-near and col=left-to-right;
    only after every recorded field matches do we normalize it to the runtime
    metadata contract.
    """

    config = _require_mapping(payload.get("model_config"), "model_config")
    manifest = _require_mapping(payload.get("manifest"), "manifest")
    manifest_input = _require_mapping(manifest.get("input"), "manifest.input")
    manifest_output = _require_mapping(manifest.get("output"), "manifest.output")

    if manifest.get("schema_version") != TRAINING_MANIFEST_SCHEMA:
        raise PathModelError("training checkpoint manifest schema mismatch")
    try:
        input_shape = tuple(int(value) for value in config.get("input_shape", ()))
        n_points = int(config.get("n_points"))
    except (TypeError, ValueError) as exc:
        raise PathModelError("training checkpoint model_config is invalid") from exc
    if input_shape != INPUT_SHAPE:
        raise PathModelError("training checkpoint model_config input_shape mismatch")
    if n_points != OUT_N:
        raise PathModelError("training checkpoint model_config n_points mismatch")
    if tuple(str(value) for value in config.get("heads", ())) != DUAL_OUTPUT_HEADS:
        raise PathModelError("training checkpoint model_config heads mismatch")

    try:
        manifest_shape = tuple(int(value) for value in manifest_input.get("shape", ()))
    except (TypeError, ValueError) as exc:
        raise PathModelError("training checkpoint manifest input shape is invalid") from exc
    if manifest_shape != INPUT_SHAPE:
        raise PathModelError("training checkpoint manifest input shape mismatch")
    if manifest_input.get("dtype") != "uint8":
        raise PathModelError("training checkpoint manifest input dtype mismatch")
    if _float_tuple(manifest_input.get("values", ()), "manifest.input.values") != (
        0.0,
        1.0,
    ):
        raise PathModelError("training checkpoint manifest input values mismatch")
    channels = _require_mapping(manifest_input.get("channels"), "manifest.input.channels")
    try:
        channel_names = tuple(
            str(channels[str(index)] if str(index) in channels else channels[index])
            for index in range(len(INPUT_CHANNELS))
        )
    except (KeyError, TypeError) as exc:
        raise PathModelError("training checkpoint manifest channels are invalid") from exc
    if channel_names != INPUT_CHANNELS:
        raise PathModelError("training checkpoint manifest channels mismatch")

    for key, expected in (
        ("x_min_m", GRID_X_BOUNDS_M[0]),
        ("x_max_m", GRID_X_BOUNDS_M[1]),
        ("y_min_m", GRID_Y_BOUNDS_M[0]),
        ("y_max_m", GRID_Y_BOUNDS_M[1]),
        ("resolution_m", GRID_RESOLUTION_M),
    ):
        values = _float_tuple([manifest_input.get(key)], f"manifest.input.{key}")
        if not np.isclose(values[0], expected, rtol=0.0, atol=1e-9):
            raise PathModelError(f"training checkpoint manifest {key} mismatch")
    if manifest_input.get("topology") != TRAINING_BEV_TOPOLOGY:
        raise PathModelError("training checkpoint manifest topology mismatch")

    manifest_out_x = _float_tuple(
        manifest_output.get("x_m", ()), "manifest.output.x_m"
    )
    checkpoint_out_x = _float_tuple(payload.get("OUT_X", ()), "OUT_X")
    expected_out_x = tuple(float(value) for value in OUT_X)
    for name, values in (
        ("manifest output.x_m", manifest_out_x),
        ("OUT_X", checkpoint_out_x),
    ):
        if len(values) != OUT_N or not np.allclose(
            values, expected_out_x, rtol=0.0, atol=1e-6
        ):
            raise PathModelError(f"training checkpoint {name} mismatch")
    if tuple(str(value) for value in manifest_output.get("heads", ())) != DUAL_OUTPUT_HEADS:
        raise PathModelError("training checkpoint manifest output heads mismatch")
    try:
        manifest_points = int(manifest_output.get("points_per_route"))
    except (TypeError, ValueError) as exc:
        raise PathModelError(
            "training checkpoint manifest points_per_route is invalid"
        ) from exc
    if manifest_points != OUT_N:
        raise PathModelError("training checkpoint manifest points_per_route mismatch")
    if manifest_output.get("nonfork_shortcut") != "copied_from_main":
        raise PathModelError(
            "training checkpoint does not prove all-sample shortcut supervision"
        )

    metadata = canonical_metadata()
    metadata.update(
        {
            "checkpoint_schema_version": TRAINING_CHECKPOINT_SCHEMA,
            "training_manifest_schema_version": TRAINING_MANIFEST_SCHEMA,
            "training_bev_topology": TRAINING_BEV_TOPOLOGY,
            "inferred_row_axis": GRID_ROW_AXIS,
            "inferred_col_axis": GRID_COL_AXIS,
            "shortcut_supervision": "all_samples_nonfork_copied_from_main",
        }
    )
    return metadata


def _metadata_from_payload(payload: Mapping[str, Any]) -> dict:
    if payload.get("schema_version") == TRAINING_CHECKPOINT_SCHEMA:
        return _training_checkpoint_metadata(payload)

    nested = payload.get("metadata")
    metadata = dict(nested) if isinstance(nested, Mapping) else {}
    for key in (
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
        "out_x",
        "output_heads",
        "shortcut_availability_policy",
    ):
        if key not in metadata and key in payload:
            metadata[key] = payload[key]
    return metadata


def _detect_model_kind(payload: Mapping[str, Any], state_dict: Mapping[str, Any]) -> str:
    metadata = _metadata_from_payload(payload)
    explicit = metadata.get("model_kind", payload.get("model_kind"))
    if explicit is not None:
        if explicit not in (MODEL_KIND_LEGACY, MODEL_KIND_DUAL):
            raise PathModelError(f"unsupported checkpoint model_kind: {explicit}")
        return str(explicit)

    keys = tuple(str(key) for key in state_dict)
    dual_markers = ("main_head.", "shortcut_head.")
    if all(any(key.startswith(marker) for key in keys) for marker in dual_markers):
        return MODEL_KIND_DUAL
    if any(key.startswith("head.") for key in keys):
        return MODEL_KIND_LEGACY
    raise PathModelError("cannot determine checkpoint model kind")


def _checkpoint_width(payload: Mapping[str, Any]) -> int:
    metadata = payload.get("metadata")
    nested_width = metadata.get("width") if isinstance(metadata, Mapping) else None
    model_config = payload.get("model_config")
    configured_width = (
        model_config.get("width") if isinstance(model_config, Mapping) else None
    )
    value = (
        configured_width
        if configured_width is not None
        else payload.get("width", nested_width if nested_width is not None else 32)
    )
    try:
        width = int(value)
    except (TypeError, ValueError) as exc:
        raise PathModelError("checkpoint width is not an integer") from exc
    if width <= 0:
        raise PathModelError("checkpoint width must be positive")
    return width


@dataclass
class LoadedPathModel:
    """로드된 torch 모델과 공통 추론·sanitize API."""

    model: Any
    model_kind: str
    metadata: dict
    device: str
    guard: PathGuardConfig

    def predict(self, input_tensor: Any) -> SanitizedPathBundle:
        _require_torch()
        values = validate_input_tensor(input_tensor)
        tensor = torch.from_numpy(values[None]).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)

        if self.model_kind == MODEL_KIND_LEGACY:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise PathModelError("legacy model output must be (y, valid_logits)")
            main_y, main_valid = output
            return sanitize_dual_prediction(
                _first_numpy(main_y),
                _first_numpy(main_valid),
                config=self.guard,
            )

        if isinstance(output, Mapping):
            if set(output) != set(DUAL_OUTPUT_HEADS):
                raise PathModelError(
                    "canonical model output keys do not match four route heads"
                )
            main_y = output["main_y"]
            main_valid = output["main_valid"]
            shortcut_y = output["shortcut_y"]
            shortcut_valid = output["shortcut_valid"]
        elif isinstance(output, (tuple, list)) and len(output) == 4:
            main_y, main_valid, shortcut_y, shortcut_valid = output
        else:
            raise PathModelError(
                "canonical model output must contain four route heads"
            )
        allow_shortcut = (
            self.metadata.get("shortcut_availability_policy")
            == SHORTCUT_POLICY_VALID_USABILITY
        )
        return sanitize_dual_prediction(
            _first_numpy(main_y),
            _first_numpy(main_valid),
            _first_numpy(shortcut_y),
            _first_numpy(shortcut_valid),
            config=self.guard,
            allow_shortcut_without_availability=allow_shortcut,
        )


def _first_numpy(value: Any) -> np.ndarray:
    try:
        array = value.detach().cpu().numpy()
    except AttributeError as exc:
        raise PathModelError("model output is not a torch tensor") from exc
    if array.ndim == 0 or array.shape[0] != 1:
        raise PathModelError(f"model output batch shape is invalid: {array.shape}")
    return np.asarray(array[0])


def _verify_checkpoint_sha256(path: Path, expected_sha256: Optional[str]) -> Optional[str]:
    if expected_sha256 is None or str(expected_sha256).strip() == "":
        return None
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise PathModelError("expected checkpoint SHA256 must be 64 lowercase hex characters")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PathModelError(f"failed to hash checkpoint: {path}") from exc
    actual = digest.hexdigest()
    if actual != expected:
        raise PathModelError(
            f"checkpoint SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_path_model(
    checkpoint_path: Any,
    *,
    device: str = "cpu",
    guard: Optional[PathGuardConfig] = None,
    expected_sha256: Optional[str] = None,
) -> LoadedPathModel:
    """legacy 또는 canonical 체크포인트를 엄격하게 로드한다."""

    _require_torch()
    path = Path(checkpoint_path)
    if not path.is_file():
        raise PathModelError(f"checkpoint not found: {path}")
    verified_sha256 = _verify_checkpoint_sha256(path, expected_sha256)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except Exception as exc:
        raise PathModelError(f"failed to load checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PathModelError("checkpoint root must be a mapping")

    state_dict = _strip_state_prefixes(_state_dict_from_payload(payload))
    model_kind = _detect_model_kind(payload, state_dict)
    metadata = _metadata_from_payload(payload)
    try:
        metadata = validate_checkpoint_metadata(metadata, model_kind)
    except PathContractError as exc:
        raise PathModelError(str(exc)) from exc

    width = _checkpoint_width(payload)
    if model_kind == MODEL_KIND_LEGACY:
        model = LegacySinglePathNet(width=width)
    else:
        model = CanonicalDualPathNet(width=width)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise PathModelError("checkpoint weights do not match declared model") from exc
    model.to(device)
    model.eval()

    metadata = dict(metadata)
    metadata["width"] = width
    if verified_sha256 is not None:
        metadata["checkpoint_sha256"] = verified_sha256
    return LoadedPathModel(
        model=model,
        model_kind=model_kind,
        metadata=metadata,
        device=str(device),
        guard=guard or PathGuardConfig(),
    )


def canonical_checkpoint(model: Any, width: int = 32, **extra: Any) -> dict:
    """학습 코드가 저장할 canonical dual 체크포인트 payload를 만든다."""

    _require_torch()
    if not isinstance(model, CanonicalDualPathNet):
        raise PathModelError("canonical checkpoint requires CanonicalDualPathNet")
    metadata = canonical_metadata()
    metadata["width"] = int(width)
    reserved = set(metadata)
    overlap = reserved.intersection(extra)
    if overlap:
        raise PathModelError(
            "extra checkpoint fields overwrite contract metadata: "
            + ", ".join(sorted(overlap))
        )
    metadata.update(extra)
    return {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
