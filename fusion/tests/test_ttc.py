"""
Unit tests for Time-To-Collision (TTC) calculations and kinematic conventions.
"""

import pytest
import numpy as np

from fusion.src.ttc import compute_ttc, compute_ttc_from_track, TTCResult
from fusion.src.tracker import Track


def test_ttc_approaching_target():
    """Verify TTC calculation for standard closing trajectory."""
    # Distance = 24.0m, Closing speed = 8.0 m/s -> TTC = 3.0s
    res = compute_ttc(forward_dist_m=24.0, closing_speed_mps=8.0)
    assert res.is_approaching is True
    assert np.isclose(res.ttc_s, 3.0)
    assert res.status == "APPROACHING"


def test_ttc_receding_target():
    """Verify TTC is infinite for receding target."""
    # Distance = 30.0m, Closing speed = -5.0 m/s (moving away)
    res = compute_ttc(forward_dist_m=30.0, closing_speed_mps=-5.0)
    assert res.is_approaching is False
    assert np.isinf(res.ttc_s)
    assert res.status == "RECEDING"


def test_ttc_constant_distance():
    """Verify TTC is infinite when closing speed is near zero."""
    res = compute_ttc(forward_dist_m=15.0, closing_speed_mps=0.0)
    assert res.is_approaching is False
    assert np.isinf(res.ttc_s)
    assert res.status == "CONSTANT_DISTANCE"


def test_ttc_zero_or_negative_distance():
    """Verify edge case where distance has reached or passed 0m."""
    res_zero = compute_ttc(forward_dist_m=0.0, closing_speed_mps=5.0)
    assert res_zero.ttc_s == 0.0
    assert res_zero.status == "IMPACT_OR_PASSED"

    res_neg = compute_ttc(forward_dist_m=-1.5, closing_speed_mps=5.0)
    assert res_neg.ttc_s == 0.0
    assert res_neg.status == "IMPACT_OR_PASSED"


def test_ttc_from_track_kalman_convention():
    """Verify TTC calculation from a Track instance using Kalman state velocity."""
    det = {
        "forward_m": 20.0,
        "lateral_m": 0.0,
        "velocity_mps": -5.0,  # Approaching at 5 m/s
    }
    track = Track(track_id=1, detection=det, min_hits=1)

    # In our coordinate system, approaching means vx < 0, closing_speed = -vx
    # Update track state directly to simulate vx = -4.0 m/s
    track.kf.state[2] = -4.0  # vx = -4.0 m/s

    res = compute_ttc_from_track(track, use_radar_doppler=False)
    assert res.is_approaching is True
    assert np.isclose(res.closing_speed_mps, 4.0)
    assert np.isclose(res.ttc_s, 20.0 / 4.0)  # 5.0 seconds


def test_ttc_from_track_doppler_convention():
    """Verify TTC calculation from Track instance using raw radar Doppler."""
    det = {
        "forward_m": 12.0,
        "lateral_m": 0.0,
        "velocity_mps": -3.0,  # RADIal Doppler negative = closing
    }
    track = Track(track_id=1, detection=det, min_hits=1)

    res = compute_ttc_from_track(track, use_radar_doppler=True)
    assert res.is_approaching is True
    assert np.isclose(res.closing_speed_mps, 3.0)
    assert np.isclose(res.ttc_s, 4.0)  # 12 / 3 = 4.0s
