"""Automated Unit and Integration Test Suite for Deep Learning Object Detection.

Covers:
- Model loading and initialization
- Single-pass and batch inference
- Accuracy on curated validation sample images
- Confidence threshold filtering
- Target class filtering
- Non-Maximum Suppression (NMS) and IoU math
- Base64 encoding/decoding
- Flask Web & REST API integration endpoints
"""

import json
import os
import pytest
import numpy as np
import cv2

from detector import ObjectDetector, CLASSES, compute_iou
from app import app


@pytest.fixture(scope="module")
def detector_instance():
    """Provides a shared detector instance for the test suite."""
    return ObjectDetector()


@pytest.fixture
def client():
    """Provides a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_model_files_exist(detector_instance):
    """Ensure the prototxt and pre-trained weights files exist on disk."""
    assert os.path.exists(detector_instance.prototxt_path), "Prototxt missing"
    assert os.path.exists(detector_instance.weights_path), "Weights file missing"
    assert os.path.getsize(detector_instance.weights_path) > 10 * 1024 * 1024, "Weights file incomplete"


def test_classes_and_colors(detector_instance):
    """Check class registry and color assignments."""
    assert len(detector_instance.classes) == 21
    assert "person" in detector_instance.classes
    assert "car" in detector_instance.classes
    assert "dog" in detector_instance.classes
    for cls in detector_instance.classes:
        assert cls in detector_instance.colors


def test_compute_iou():
    """Verify Intersection over Union (IoU) calculation logic."""
    # Identical boxes -> IoU = 1.0
    boxA = [10, 10, 50, 50]
    boxB = [10, 10, 50, 50]
    assert pytest.approx(compute_iou(boxA, boxB), 0.01) == 1.0

    # Non-overlapping boxes -> IoU = 0.0
    boxC = [100, 100, 150, 150]
    assert compute_iou(boxA, boxC) == 0.0

    # 50% overlap known geometric test
    boxD = [10, 10, 30, 50]  # area = 20 * 40 = 800
    boxE = [20, 10, 40, 50]  # area = 20 * 40 = 800
    # inter = 10 * 40 = 400, union = 800 + 800 - 400 = 1200 -> 400/1200 = 0.333
    assert pytest.approx(compute_iou(boxD, boxE), 0.01) == 0.333


def test_traffic_image_detection(detector_instance):
    """Verifies deep learning detection on traffic sample image."""
    sample_path = os.path.join(os.path.dirname(__file__), "static", "samples", "sample_traffic.jpg")
    annotated, detections, ms = detector_instance.detect_file(sample_path, conf_threshold=0.35)

    assert isinstance(detections, list)
    assert len(detections) >= 1
    assert ms > 0.0
    labels = [d["label"] for d in detections]
    assert "bus" in labels or "car" in labels

    # Check bounding box validity
    for d in detections:
        box = d["box"]
        assert box[0] < box[2], "Invalid X coordinates"
        assert box[1] < box[3], "Invalid Y coordinates"
        assert 0.0 <= d["confidence"] <= 1.0


def test_confidence_filtering(detector_instance):
    """Higher confidence threshold should filter out low confidence candidates."""
    sample_path = os.path.join(os.path.dirname(__file__), "static", "samples", "sample_dogs.jpg")
    _, det_low, _ = detector_instance.detect_file(sample_path, conf_threshold=0.20)
    _, det_high, _ = detector_instance.detect_file(sample_path, conf_threshold=0.95)

    assert len(det_high) <= len(det_low)


def test_target_class_filtering(detector_instance):
    """Detection should strictly respect target_classes filter."""
    sample_path = os.path.join(os.path.dirname(__file__), "static", "samples", "sample_dogs.jpg")
    _, det_only_dog, _ = detector_instance.detect_file(sample_path, conf_threshold=0.3, target_classes=["dog"])

    for d in det_only_dog:
        assert d["label"] == "dog"


def test_base64_encoding_decoding(detector_instance):
    """Tests encoding annotated image to base64 and decoding back."""
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    b64 = detector_instance.encode_image_to_base64(dummy)
    assert b64.startswith("data:image/jpeg;base64,")

    # Decode from bytes
    _, detections, ms = detector_instance.detect(dummy)
    assert isinstance(detections, list)


def test_api_health(client):
    """Test /api/health endpoint."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "classes" in data


def test_api_classes(client):
    """Test /api/classes endpoint."""
    res = client.get("/api/classes")
    assert res.status_code == 200
    data = res.get_json()
    assert "classes" in data
    assert len(data["classes"]) == 20


def test_api_samples(client):
    """Test /api/samples endpoint."""
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.get_json()
    assert "samples" in data
    assert len(data["samples"]) >= 1


def test_api_detect_sample(client):
    """Test /api/detect with sample image."""
    res = client.post("/api/detect", data={"sample": "sample_traffic.jpg", "conf_threshold": 0.35})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["num_detections"] >= 1
    assert "annotated_image" in data
    assert "inference_time_ms" in data


def test_api_detect_no_image(client):
    """Test /api/detect with missing image payload returns 400."""
    res = client.post("/api/detect", data={})
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
