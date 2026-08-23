from types import SimpleNamespace

import cv2
import numpy as np

from track_drive_cnn_gpt.yolo_bev_node import make_signal_preview


def test_preview_labels_signal_and_start_r_with_source_pixel_coordinates(monkeypatch):
    captured_text: list[str] = []
    original_put_text = cv2.putText

    def recording_put_text(image, text, *args, **kwargs):
        captured_text.append(str(text))
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", recording_put_text)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            cls=np.asarray([2, 3], dtype=np.float32),
            conf=np.asarray([0.91, 0.88], dtype=np.float32),
            xyxy=np.asarray(
                [[100, 200, 300, 400], [300, 200, 500, 400]],
                dtype=np.float32,
            ),
        )
    )

    preview = make_signal_preview(
        image,
        result,
        2,
        {"RED": 0.91},
        traffic_detection={"confidence": 0.91},
        cone_trigger_id=3,
        cone_detection={"confidence": 0.88},
        sequence=17,
        inference_ms=12.5,
        output_width=960,
    )

    assert preview.shape == (540, 960, 3)
    assert any(
        "traffic_light 0.91 C(200,300) B(100,200,300,400)" in text
        for text in captured_text
    )
    assert any(
        "START_R 0.88 C(400,300) B(300,200,500,400)" in text
        for text in captured_text
    )
    assert any("START_R:0.88" in text for text in captured_text)
    assert any("source=1920x1080 px" in text for text in captured_text)


def test_preview_lists_all_target_classes_when_nothing_is_detected(monkeypatch):
    captured_text: list[str] = []
    original_put_text = cv2.putText

    def recording_put_text(image, text, *args, **kwargs):
        captured_text.append(str(text))
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", recording_put_text)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = SimpleNamespace(boxes=None)

    make_signal_preview(
        image,
        result,
        2,
        {},
        traffic_detection=None,
        cone_trigger_id=3,
        cone_detection=None,
        sequence=18,
        inference_ms=10.0,
    )

    status = next(text for text in captured_text if "GREEN:--" in text)
    for name in ("GREEN", "LEFT", "RED", "YELLOW", "START_R"):
        assert f"{name}:--" in status
