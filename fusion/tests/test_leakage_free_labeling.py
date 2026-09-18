import pytest
from fusion.training.ground_truth_loader import match_ground_truth


def test_leakage_free_matching_same_gt():
    """Verify that candidate pair is labeled 1 when both match the SAME GT object."""
    gt_objects = [
        {
            "gt_id": 0,
            "forward_m": 15.0,
            "lateral_m": 2.0,
            "bbox": [500.0, 400.0, 700.0, 600.0],
        },
        {
            "gt_id": 1,
            "forward_m": 30.0,
            "lateral_m": -3.0,
            "bbox": [1000.0, 450.0, 1200.0, 650.0],
        },
    ]

    # Radar object 0 close to GT 0, Radar object 1 close to GT 1
    radar_objects = [
        {"x_m": 15.2, "y_m": 2.1},   # matches GT 0
        {"x_m": 29.8, "y_m": -2.9},  # matches GT 1
    ]

    # Camera object 0 overlaps GT 0, Camera object 1 overlaps GT 1
    camera_objects = [
        {"bbox": [510.0, 405.0, 690.0, 595.0]},   # matches GT 0
        {"bbox": [1010.0, 455.0, 1195.0, 645.0]},  # matches GT 1
    ]

    labels = match_ground_truth(
        radar_objects,
        camera_objects,
        gt_objects,
        radar_dist_thresh=2.5,
        camera_iou_thresh=0.4,
    )

    # (0, 0) matches same GT 0 -> 1
    assert labels[(0, 0)] == 1
    # (1, 1) matches same GT 1 -> 1
    assert labels[(1, 1)] == 1
    # Cross pairs match different GT objects -> must be 0
    assert labels[(0, 1)] == 0
    assert labels[(1, 0)] == 0


def test_leakage_free_mismatch_different_gt():
    """
    Verify that even if a radar projection happens to be close in pixels,
    if the radar and camera observe different physical GT vehicles, the label is 0.
    """
    gt_objects = [
        {
            "gt_id": 0,
            "forward_m": 10.0,
            "lateral_m": 0.0,
            "bbox": [400.0, 400.0, 600.0, 600.0],
        },
        {
            "gt_id": 1,
            "forward_m": 50.0,
            "lateral_m": 0.0,
            "bbox": [450.0, 450.0, 550.0, 550.0],  # Visually overlapping in 2D projection
        },
    ]

    # Radar detects near vehicle at 10m (GT 0)
    radar_objects = [{"x_m": 10.1, "y_m": 0.1}]

    # Camera detects far vehicle at 50m (GT 1)
    camera_objects = [{"bbox": [448.0, 448.0, 552.0, 552.0]}]

    labels = match_ground_truth(
        radar_objects,
        camera_objects,
        gt_objects,
        radar_dist_thresh=2.5,
        camera_iou_thresh=0.4,
    )

    # They match different GT objects (0 vs 1) -> must be 0!
    assert labels[(0, 0)] == 0


def test_clutter_and_false_alarms_labeled_zero():
    """Verify that radar clutter or false camera detections receive label 0."""
    gt_objects = [
        {
            "gt_id": 0,
            "forward_m": 15.0,
            "lateral_m": 2.0,
            "bbox": [500.0, 400.0, 700.0, 600.0],
        },
    ]

    # Radar clutter at 60m (far from any GT)
    radar_objects = [{"x_m": 60.0, "y_m": 10.0}]

    # Camera detection on GT 0
    camera_objects = [{"bbox": [505.0, 405.0, 695.0, 595.0]}]

    labels = match_ground_truth(
        radar_objects,
        camera_objects,
        gt_objects,
        radar_dist_thresh=2.5,
        camera_iou_thresh=0.4,
    )

    assert labels[(0, 0)] == 0
