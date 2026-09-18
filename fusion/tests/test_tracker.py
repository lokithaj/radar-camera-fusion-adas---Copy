"""
Unit tests for MultiObjectTracker, Track, and 2D Kalman Filter state estimation.
"""

import pytest
import numpy as np

from fusion.src.tracker import MultiObjectTracker, Track, KalmanFilter2D


def test_kalman_filter_2d_prediction_and_update():
    """Verify 2D Constant Velocity Kalman Filter predicts and updates correctly."""
    kf = KalmanFilter2D(x=10.0, y=0.0, vx=2.0, vy=0.0)

    # Predict with dt = 0.5s -> expected x = 10 + 2.0*0.5 = 11.0
    pred = kf.predict(dt=0.5)
    assert np.isclose(pred[0], 11.0)
    assert np.isclose(pred[1], 0.0)

    # Update with measurement [11.2, 0.1]
    updated = kf.update(np.array([11.2, 0.1]))
    # Updated position should lie between prediction (11.0) and measurement (11.2)
    assert 11.0 < updated[0] < 11.2
    assert 0.0 < updated[1] < 0.1


def test_track_lifecycle_and_confirmation():
    """Verify track transitions from tentative to confirmed after min_hits."""
    det1 = {
        "forward_m": 15.0,
        "lateral_m": -1.0,
        "velocity_mps": -2.0,
        "class_name": "car",
    }
    track = Track(track_id=1, detection=det1, min_hits=2)
    assert track.track_id == 1
    assert track.hits == 1
    assert not track.confirmed

    # Predict and update second time
    track.predict(dt=0.1)
    det2 = {"forward_m": 14.8, "lateral_m": -1.0}
    track.update(det2)

    assert track.hits == 2
    assert track.confirmed
    assert track.time_since_update == 0


def test_tracker_stable_ids_over_trajectory():
    """Verify that a continuous moving object retains a single, stable track ID."""
    tracker = MultiObjectTracker(max_cost_m=4.0, max_age=3, min_hits=2)

    # Target moving forward from 20m to 16m at 1m/step
    trajectory_x = [20.0, 19.0, 18.0, 17.0, 16.0]

    track_ids_observed = []
    for step, x in enumerate(trajectory_x):
        detections = [{
            "forward_m": x,
            "lateral_m": 0.5,
            "velocity_mps": -10.0,
            "class_name": "car",
            "camera_confidence": 0.9,
        }]
        active_tracks = tracker.update(detections, dt=0.1)
        assert len(active_tracks) == 1
        track_ids_observed.append(active_tracks[0].track_id)

    # Every frame should observe Track ID 1
    assert all(tid == 1 for tid in track_ids_observed)
    assert active_tracks[0].confirmed
    assert active_tracks[0].hits == 5


def test_tracker_coasting_and_deletion():
    """Verify track coasting over temporary missed detections and deletion after max_age."""
    tracker = MultiObjectTracker(max_cost_m=4.0, max_age=2, min_hits=1)

    # Frame 0: detection present
    tracker.update([{"forward_m": 12.0, "lateral_m": 0.0}], dt=0.1)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].time_since_update == 0

    # Frame 1: missed detection (coasting frame 1)
    tracker.update([], dt=0.1)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].time_since_update == 1

    # Frame 2: missed detection (coasting frame 2 <= max_age=2)
    tracker.update([], dt=0.1)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].time_since_update == 2

    # Frame 3: missed detection (time_since_update reaches 3 > max_age=2 -> deleted)
    tracker.update([], dt=0.1)
    assert len(tracker.tracks) == 0


def test_tracker_distance_gating_prevents_false_association():
    """Verify that a distant detection is not associated with an existing track."""
    tracker = MultiObjectTracker(max_cost_m=3.0, max_age=2, min_hits=1)

    # Frame 0: track at 10m
    tracker.update([{"forward_m": 10.0, "lateral_m": 0.0}], dt=0.1)
    assert tracker.tracks[0].track_id == 1

    # Frame 1: detection at 30m (dx=20m > max_cost_m=3.0m)
    # The existing track should coast, and a new track should spawn
    active_tracks = tracker.update([{"forward_m": 30.0, "lateral_m": 0.0}], dt=0.1)
    assert len(active_tracks) == 2

    track_ids = {t.track_id for t in active_tracks}
    assert track_ids == {1, 2}
