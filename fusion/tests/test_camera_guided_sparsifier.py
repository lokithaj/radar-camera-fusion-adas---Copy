"""
Stage 5b Unit Tests: CameraGuidedRadarSparsifier / FusionAwareRadarSparsifier.

Tests cover:
1. Inside-bbox radar object gets higher camera relevance.
2. Outside-FoV radar object is strongly penalized (score = 0.0).
3. No camera detections scenario (neutral inside-FoV baseline).
4. Empty radar input handling.
5. Retention ratio behavior (1.00, 0.75, 0.50, 0.25, 0.10) and min_objects=1 enforcement.
6. Deterministic ranking and stable tie-breaking.
7. Verification of fusion score formulation S_fusion = 0.6 * S_radar + 0.4 * S_camera.
"""

import numpy as np
import pytest
from fusion.src.radar_sparsifier import (
    AdaptiveRadarSparsifier,
    CameraGuidedRadarSparsifier,
    FusionAwareRadarSparsifier,
)


def create_sample_radar_objects():
    """Create sample 3D radar objects mimicking RadarObjectExtractor output."""
    return [
        {
            "cluster_id": 0,
            "x_m": 15.0,
            "y_m": 0.5,
            "z_m": -0.2,
            "range_m": 15.01,
            "velocity_mps": -2.5,
            "azimuth_deg": 1.9,
            "elevation_deg": -0.8,
            "power": 6.5e12,
            "num_points": 12,
        },
        {
            "cluster_id": 1,
            "x_m": 35.0,
            "y_m": -2.0,
            "z_m": 0.0,
            "range_m": 35.06,
            "velocity_mps": 3.0,
            "azimuth_deg": -3.3,
            "elevation_deg": 0.0,
            "power": 7.0e12,
            "num_points": 15,
        },
        {
            "cluster_id": 2,
            "x_m": 5.0,
            "y_m": -5.0,
            "z_m": -0.1,
            "range_m": 7.07,
            "velocity_mps": 7.0,
            "azimuth_deg": -45.0,
            "elevation_deg": -0.8,
            "power": 5.5e12,
            "num_points": 8,
        },
        {
            "cluster_id": 3,
            "x_m": 22.0,
            "y_m": 3.0,
            "z_m": 0.1,
            "range_m": 22.20,
            "velocity_mps": 0.5,
            "azimuth_deg": 7.8,
            "elevation_deg": 0.3,
            "power": 4.0e12,
            "num_points": 6,
        },
    ]


def test_empty_radar_input():
    """Empty radar input must return empty lists without exceptions."""
    sparsifier = CameraGuidedRadarSparsifier(retention_ratio=0.5)
    assert sparsifier.select([]) == []
    assert sparsifier.compute_camera_scores([]) == []
    assert sparsifier.compute_fusion_scores([]) == []


def test_inside_bbox_higher_camera_relevance():
    """Radar point inside a camera bounding box must achieve higher camera relevance."""
    objects = create_sample_radar_objects()[:2]
    # Obj 0 inside bbox [100, 100, 300, 300]
    # Obj 1 far away at pixel (800, 600)
    pixels = np.array([[200.0, 200.0], [800.0, 600.0]], dtype=np.float64)
    valid = np.array([True, True], dtype=bool)
    camera_detections = [
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.85,
            "bbox": [100.0, 100.0, 300.0, 300.0],
        }
    ]

    sparsifier = CameraGuidedRadarSparsifier(retention_ratio=0.5)
    cam_scores = sparsifier.compute_camera_scores(
        objects,
        projected_pixels=pixels,
        valid_in_fov=valid,
        camera_objects=camera_detections,
    )

    assert len(cam_scores) == 2
    assert cam_scores[0] > cam_scores[1]
    assert cam_scores[0] >= 0.75  # Inside bbox receives high relevance (0.75 to 1.00)
    assert cam_scores[1] <= 0.30  # Far point inside FoV sits at baseline (0.25)


def test_outside_fov_penalized():
    """Radar points outside camera FoV must be penalized with camera relevance = 0.0."""
    objects = create_sample_radar_objects()[:2]
    pixels = np.array([[250.0, 300.0], [15000.0, 20000.0]], dtype=np.float64)
    valid = np.array([True, False], dtype=bool)  # Obj 1 is outside FoV
    camera_detections = [
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.90,
            "bbox": [200.0, 200.0, 400.0, 400.0],
        }
    ]

    sparsifier = CameraGuidedRadarSparsifier(retention_ratio=0.50)
    cam_scores = sparsifier.compute_camera_scores(
        objects,
        projected_pixels=pixels,
        valid_in_fov=valid,
        camera_objects=camera_detections,
    )

    assert cam_scores[1] == 0.0
    assert cam_scores[0] > 0.70

    # Sparsifier selection should prioritize Obj 0
    selected = sparsifier.select(
        objects,
        projected_pixels=pixels,
        valid_in_fov=valid,
        camera_objects=camera_detections,
    )
    assert len(selected) == 1
    assert selected[0]["cluster_id"] == 0


def test_no_camera_detections():
    """With no camera detections, points in FoV receive a neutral baseline score."""
    objects = create_sample_radar_objects()[:3]
    pixels = np.array([[200.0, 200.0], [400.0, 300.0], [99999.0, 99999.0]], dtype=np.float64)
    valid = np.array([True, True, False], dtype=bool)

    sparsifier = CameraGuidedRadarSparsifier(retention_ratio=0.50)

    # Empty camera detections list
    cam_scores_empty = sparsifier.compute_camera_scores(
        objects,
        projected_pixels=pixels,
        valid_in_fov=valid,
        camera_objects=[],
    )
    assert cam_scores_empty[0] == 0.50
    assert cam_scores_empty[1] == 0.50
    assert cam_scores_empty[2] == 0.0  # Outside FoV still penalized

    # None camera detections
    cam_scores_none = sparsifier.compute_camera_scores(
        objects,
        projected_pixels=pixels,
        valid_in_fov=valid,
        camera_objects=None,
    )
    assert cam_scores_none == cam_scores_empty


def test_retention_ratio_behavior():
    """All retention ratios must be respected and min_objects=1 strictly enforced."""
    objects = create_sample_radar_objects()  # 4 objects
    pixels = np.array(
        [[200.0, 200.0], [300.0, 300.0], [400.0, 400.0], [500.0, 500.0]],
        dtype=np.float64,
    )
    valid = np.array([True, True, True, True], dtype=bool)

    sparsifier = CameraGuidedRadarSparsifier(retention_ratio=1.0)
    assert len(sparsifier.select(objects, pixels, valid)) == 4

    sparsifier.set_retention_ratio(0.75)
    assert len(sparsifier.select(objects, pixels, valid)) == 3

    sparsifier.set_retention_ratio(0.50)
    assert len(sparsifier.select(objects, pixels, valid)) == 2

    sparsifier.set_retention_ratio(0.25)
    assert len(sparsifier.select(objects, pixels, valid)) == 1

    sparsifier.set_retention_ratio(0.10)
    assert len(sparsifier.select(objects, pixels, valid)) == 1  # min_objects=1 enforced

    # Single object frame
    single = objects[:1]
    for r in (1.00, 0.75, 0.50, 0.25, 0.10):
        sparsifier.set_retention_ratio(r)
        assert len(sparsifier.select(single, pixels[:1], valid[:1])) == 1


def test_deterministic_ranking():
    """Ranking must be deterministic and break ties stably using original index."""
    objects = [
        {"cluster_id": 0, "range_m": 10.0, "velocity_mps": 0.0, "power": 1e6},
        {"cluster_id": 1, "range_m": 10.0, "velocity_mps": 0.0, "power": 1e6},
        {"cluster_id": 2, "range_m": 10.0, "velocity_mps": 0.0, "power": 1e6},
    ]
    pixels = np.array([[200.0, 200.0], [200.0, 200.0], [200.0, 200.0]], dtype=np.float64)
    valid = np.array([True, True, True], dtype=bool)

    sparsifier = CameraGuidedRadarSparsifier(retention_ratio=0.75)
    res1 = sparsifier.select(objects, pixels, valid)
    res2 = sparsifier.select(objects, pixels, valid)

    assert [o["cluster_id"] for o in res1] == [o["cluster_id"] for o in res2]
    # Tie-breaking by index should retain cluster_id 0 and 1
    assert [o["cluster_id"] for o in res1] == [0, 1]


def test_fusion_score_weights():
    """Verification that S_fusion = 0.6 * S_radar + 0.4 * S_camera."""
    objects = create_sample_radar_objects()[:2]
    pixels = np.array([[200.0, 200.0], [800.0, 600.0]], dtype=np.float64)
    valid = np.array([True, True], dtype=bool)
    camera_detections = [
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.80,
            "bbox": [100.0, 100.0, 300.0, 300.0],
        }
    ]

    sparsifier = CameraGuidedRadarSparsifier(retention_ratio=1.0)
    radar_scores = sparsifier.compute_importance_scores(objects)
    camera_scores = sparsifier.compute_camera_scores(
        objects,
        projected_pixels=pixels,
        valid_in_fov=valid,
        camera_objects=camera_detections,
    )
    fusion_scores = sparsifier.compute_fusion_scores(
        objects,
        projected_pixels=pixels,
        valid_in_fov=valid,
        camera_objects=camera_detections,
    )

    for r_s, c_s, f_s in zip(radar_scores, camera_scores, fusion_scores):
        expected = np.clip(0.6 * r_s + 0.4 * c_s, 0.0, 1.0)
        assert np.isclose(f_s, expected, atol=1e-5)


def test_frame15_scenario_resolution():
    """
    Simulate the exact Frame 15 failure case:
    Obj 0: Far, highest power, moderate velocity (inside FoV)
    Obj 1: True car target, lower velocity, inside/adjacent to car bbox
    Obj 2: Close, fastest velocity, OUTSIDE FoV (unmatchable)
    """
    frame15_objects = [
        {"cluster_id": 0, "range_m": 34.45, "velocity_mps": 3.20, "power": 6.99e12},
        {"cluster_id": 1, "range_m": 16.74, "velocity_mps": -2.31, "power": 6.46e12},
        {"cluster_id": 2, "range_m": 6.24, "velocity_mps": 6.72, "power": 5.89e12},
    ]
    pixels = np.array(
        [[1034.7, 594.4], [201.2, 756.1], [119603.7, 30540.7]],
        dtype=np.float64,
    )
    valid = np.array([True, True, False], dtype=bool)
    camera_detections = [
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.837,
            "bbox": [116.84, 578.50, 386.86, 732.35],
        }
    ]

    # Baseline Stage 5 radar-only sparsifier drops Obj 1 at ratio 0.75
    baseline = AdaptiveRadarSparsifier(retention_ratio=0.75)
    baseline_selected = baseline.select(frame15_objects)
    baseline_ids = [o["cluster_id"] for o in baseline_selected]
    assert 1 not in baseline_ids  # Confirms baseline dropped the true target!

    # Stage 5b CameraGuidedRadarSparsifier prioritizes Obj 1 and penalizes Obj 2
    sparsifier_5b = CameraGuidedRadarSparsifier(retention_ratio=0.75)
    selected_5b = sparsifier_5b.select(
        frame15_objects,
        projected_pixels=pixels,
        valid_in_fov=valid,
        camera_objects=camera_detections,
    )
    ids_5b = [o["cluster_id"] for o in selected_5b]
    assert 1 in ids_5b  # True target Obj 1 is successfully retained!
    assert selected_5b[0]["cluster_id"] == 1  # In fact, Obj 1 is ranked #1
    assert 2 not in ids_5b  # Outside-FoV Obj 2 is correctly eliminated
