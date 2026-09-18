import pytest
from pathlib import Path

from config import DEFAULT_GROUND_TRUTH, get_default_recording_dir, get_test_recording_dir
from fusion.training.ground_truth_loader import RadialGroundTruth, compute_box_iou


def test_ground_truth_file_exists_and_loads():
    """Verify that ground_truth.csv exists and contains valid RADIal annotations."""
    assert DEFAULT_GROUND_TRUTH.exists()
    gt = RadialGroundTruth(DEFAULT_GROUND_TRUTH)
    assert len(gt) > 0

    # Must contain our two recording sequences
    train_rec = get_default_recording_dir().name
    test_rec = get_test_recording_dir().name

    train_annos = gt.get_sequence_annotations(train_rec)
    test_annos = gt.get_sequence_annotations(test_rec)

    assert len(train_annos) > 0, f"No annotations found for training recording: {train_rec}"
    assert len(test_annos) > 0, f"No annotations found for test recording: {test_rec}"


def test_frame_annotations_structure():
    """Verify that get_frame_annotations parses coordinate structures correctly."""
    gt = RadialGroundTruth()
    train_rec = get_default_recording_dir().name

    # Frame 0 has labeled vehicles in train_rec
    objs = gt.get_frame_annotations(train_rec, frame_index=0)
    assert len(objs) > 0

    obj = objs[0]
    for key in ("gt_id", "bbox", "forward_m", "lateral_m", "range_m", "azimuth_deg"):
        assert key in obj

    x1, y1, x2, y2 = obj["bbox"]
    assert x2 > x1
    assert y2 > y1
    assert obj["forward_m"] > 0


def test_compute_box_iou():
    """Test IoU computation on synthetic boxes."""
    box_a = [100.0, 100.0, 200.0, 200.0]
    box_b = [100.0, 100.0, 200.0, 200.0]
    assert compute_box_iou(box_a, box_b) == pytest.approx(1.0)

    # Disjoint boxes
    box_c = [300.0, 300.0, 400.0, 400.0]
    assert compute_box_iou(box_a, box_c) == pytest.approx(0.0)

    # 50% overlap horizontally
    box_d = [150.0, 100.0, 250.0, 200.0]
    # intersection: 50 * 100 = 5000; union: 10000 + 10000 - 5000 = 15000; iou = 5000/15000 = 1/3
    assert compute_box_iou(box_a, box_d) == pytest.approx(5000.0 / 15000.0)
