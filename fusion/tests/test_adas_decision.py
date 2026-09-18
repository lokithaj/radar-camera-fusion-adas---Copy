"""
Unit tests for the ADAS Decision and Collision Warning Module.
"""

import pytest
import numpy as np

from fusion.src.adas_decision import (
    ADASDecisionModule,
    ADASConfig,
    ThreatLevel,
    ADASAlert,
    TrackAssessment,
)
from fusion.src.tracker import Track


def create_mock_track(
    track_id: int,
    x: float,
    y: float,
    vx: float,
    confirmed: bool = True,
) -> Track:
    """Helper to construct a mock Track with precise kinematic state."""
    det = {"forward_m": x, "lateral_m": y, "velocity_mps": 0.0}
    track = Track(track_id=track_id, detection=det, min_hits=1 if confirmed else 5)
    track.kf.state[0] = x
    track.kf.state[1] = y
    track.kf.state[2] = vx
    track.confirmed = confirmed
    return track


def test_threat_level_hierarchy():
    """Verify severity ordering of threat level enum."""
    assert ThreatLevel.CRITICAL > ThreatLevel.WARNING
    assert ThreatLevel.WARNING > ThreatLevel.CAUTION
    assert ThreatLevel.CAUTION > ThreatLevel.SAFE
    assert ThreatLevel.CRITICAL >= ThreatLevel.CRITICAL


def test_adas_critical_warning_in_path():
    """Verify CRITICAL warning when in ego corridor with low TTC or close distance."""
    module = ADASDecisionModule(ADASConfig(ttc_critical_s=1.5, dist_critical_m=5.0))

    # Fast approaching vehicle in ego corridor (y = 0.2m):
    # dist = 10m, vx = -10 m/s (closing speed = 10 m/s) -> TTC = 1.0s <= 1.5s
    trk_crit = create_mock_track(track_id=1, x=10.0, y=0.2, vx=-10.0, confirmed=True)
    assess = module.evaluate_track(trk_crit)
    assert assess.threat_level == ThreatLevel.CRITICAL
    assert assess.in_ego_corridor is True


def test_adas_warning_and_caution_tiers():
    """Verify WARNING and CAUTION transitions for approaching vehicle in path."""
    config = ADASConfig(
        ttc_critical_s=1.5,
        ttc_warning_s=2.7,
        ttc_caution_s=4.0,
        dist_critical_m=5.0,
        dist_warning_m=12.0,
        dist_caution_m=25.0,
    )
    module = ADASDecisionModule(config)

    # Closing at 10 m/s at 20m -> TTC = 2.0s (<= 2.7s) -> WARNING
    trk_warn = create_mock_track(track_id=2, x=20.0, y=0.0, vx=-10.0, confirmed=True)
    assess_warn = module.evaluate_track(trk_warn)
    assert assess_warn.threat_level == ThreatLevel.WARNING

    # Closing at 5 m/s at 18m -> TTC = 3.6s (<= 4.0s) -> CAUTION
    trk_caut = create_mock_track(track_id=3, x=18.0, y=0.0, vx=-5.0, confirmed=True)
    assess_caut = module.evaluate_track(trk_caut)
    assert assess_caut.threat_level == ThreatLevel.CAUTION

    # Far away (40m) moving slowly (-2 m/s -> TTC = 20s) -> SAFE
    trk_safe = create_mock_track(track_id=4, x=40.0, y=0.0, vx=-2.0, confirmed=True)
    assess_safe = module.evaluate_track(trk_safe)
    assert assess_safe.threat_level == ThreatLevel.SAFE


def test_lateral_corridor_gating():
    """Verify that off-path vehicles do not trigger critical in-path warnings."""
    module = ADASDecisionModule()

    # Fast approaching vehicle (closing speed = 10 m/s) at dist=10m (TTC=1.0s),
    # but located at lateral Y = 6.0m (far left shoulder/oncoming lane)
    trk_offpath = create_mock_track(track_id=5, x=10.0, y=6.0, vx=-10.0, confirmed=True)
    assess = module.evaluate_track(trk_offpath)
    assert assess.threat_level == ThreatLevel.SAFE
    assert not assess.in_ego_corridor
    assert not assess.in_adjacent_corridor

    # Target in adjacent lane (Y = 2.5m) is downgraded to at most CAUTION
    trk_adj = create_mock_track(track_id=6, x=10.0, y=2.5, vx=-10.0, confirmed=True)
    assess_adj = module.evaluate_track(trk_adj)
    assert assess_adj.threat_level == ThreatLevel.CAUTION
    assert not assess_adj.in_ego_corridor
    assert assess_adj.in_adjacent_corridor


def test_unconfirmed_track_suppression():
    """Verify that unconfirmed/tentative tracks do not trigger false alarms."""
    config = ADASConfig(require_confirmed_tracks=True)
    module = ADASDecisionModule(config)

    # Imminent hazard kinematics, but track is tentative (confirmed=False)
    trk_tentative = create_mock_track(track_id=7, x=8.0, y=0.0, vx=-10.0, confirmed=False)
    assess = module.evaluate_track(trk_tentative)
    assert assess.threat_level == ThreatLevel.SAFE
    assert "unconfirmed" in assess.reason.lower()


def test_system_level_arbitration():
    """Verify system alert correctly identifies the primary hazard."""
    module = ADASDecisionModule()

    trk1 = create_mock_track(track_id=1, x=35.0, y=0.0, vx=-2.0, confirmed=True)   # SAFE
    trk2 = create_mock_track(track_id=2, x=8.0, y=0.0, vx=-10.0, confirmed=True)   # CRITICAL (TTC=0.8s)
    trk3 = create_mock_track(track_id=3, x=20.0, y=0.0, vx=-8.0, confirmed=True)   # WARNING

    alert: ADASAlert = module.evaluate_frame([trk1, trk2, trk3])
    assert alert.system_threat_level == ThreatLevel.CRITICAL
    assert alert.primary_hazard_track_id == 2
    assert "EMERGENCY BRAKE" in alert.recommended_action
