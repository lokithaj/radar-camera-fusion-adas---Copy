"""
Configurable ADAS Decision and Collision Warning Module for Radar-Camera Fusion.

DISCLAIMER:
This module is a research and educational prototype using prerecorded RADIal
data. The risk levels and warning thresholds implemented here are heuristic
guidelines for prototype demonstration and are NOT certified for production
automotive safety systems.

Provides:
- Configurable ADAS warning thresholds (TTC, forward distance, lateral lane corridors).
- Track-level risk assessment: SAFE, CAUTION, WARNING, CRITICAL.
- Ego vehicle system-level hazard arbitration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from fusion.src.ttc import compute_ttc_from_track, TTCResult


class ThreatLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def severity(self) -> int:
        levels = {
            ThreatLevel.SAFE: 0,
            ThreatLevel.CAUTION: 1,
            ThreatLevel.WARNING: 2,
            ThreatLevel.CRITICAL: 3,
        }
        return levels[self]

    def __ge__(self, other: "ThreatLevel") -> bool:
        if isinstance(other, ThreatLevel):
            return self.severity >= other.severity
        return NotImplemented

    def __gt__(self, other: "ThreatLevel") -> bool:
        if isinstance(other, ThreatLevel):
            return self.severity > other.severity
        return NotImplemented


@dataclass
class ADASConfig:
    """
    Configurable parameters for ADAS collision assessment.
    Thresholds are customizable heuristics.
    """
    # Time-To-Collision thresholds (in seconds)
    ttc_critical_s: float = 1.5
    ttc_warning_s: float = 2.7
    ttc_caution_s: float = 4.0

    # Longitudinal distance thresholds (in meters)
    dist_critical_m: float = 5.0
    dist_warning_m: float = 12.0
    dist_caution_m: float = 25.0

    # Lateral lane corridor (in meters, centered at vehicle Y=0)
    # Standard highway/urban lane width is ~3.5m, so half-lane corridor is 1.75m.
    ego_corridor_half_width_m: float = 1.8
    adjacent_corridor_half_width_m: float = 4.5

    # Minimum track hits before generating high-priority warnings
    require_confirmed_tracks: bool = True

    # Use radar Doppler velocity instead of Kalman state velocity
    use_radar_doppler: bool = False


@dataclass
class TrackAssessment:
    """Detailed risk assessment for a single tracked object."""
    track_id: int
    threat_level: ThreatLevel
    forward_m: float
    lateral_m: float
    range_m: float
    closing_speed_mps: float
    ttc_s: float
    in_ego_corridor: bool
    in_adjacent_corridor: bool
    is_approaching: bool
    reason: str


@dataclass
class ADASAlert:
    """System-level ADAS warning summary for the current frame."""
    system_threat_level: ThreatLevel
    primary_hazard_track_id: Optional[int]
    total_active_tracks: int
    total_confirmed_tracks: int
    track_assessments: List[TrackAssessment]
    recommended_action: str


class ADASDecisionModule:
    """
    Configurable ADAS Forward Collision Warning (FCW) Decision Engine.
    """

    def __init__(self, config: Optional[ADASConfig] = None):
        self.config = config or ADASConfig()

    def evaluate_track(self, track) -> TrackAssessment:
        """
        Evaluate collision hazard for an individual Track object.
        """
        dist_x = track.x
        lat_y = track.y
        in_ego_lane = abs(lat_y) <= self.config.ego_corridor_half_width_m
        in_adjacent_lane = abs(lat_y) <= self.config.adjacent_corridor_half_width_m

        ttc_res: TTCResult = compute_ttc_from_track(
            track,
            use_radar_doppler=self.config.use_radar_doppler,
        )

        ttc = ttc_res.ttc_s
        closing_speed = ttc_res.closing_speed_mps
        is_approaching = ttc_res.is_approaching

        # If track is not confirmed and we require confirmation, suppress to SAFE
        if self.config.require_confirmed_tracks and not track.confirmed:
            return TrackAssessment(
                track_id=track.track_id,
                threat_level=ThreatLevel.SAFE,
                forward_m=dist_x,
                lateral_m=lat_y,
                range_m=track.range_m,
                closing_speed_mps=closing_speed,
                ttc_s=ttc,
                in_ego_corridor=in_ego_lane,
                in_adjacent_corridor=in_adjacent_lane,
                is_approaching=is_approaching,
                reason="Track tentative/unconfirmed",
            )

        # Evaluate risk based on lane corridor and TTC/Distance
        if in_ego_lane:
            # Target directly in our forward path
            if is_approaching:
                if ttc <= self.config.ttc_critical_s or dist_x <= self.config.dist_critical_m:
                    level = ThreatLevel.CRITICAL
                    reason = f"Imminent collision in path (TTC={ttc:.1f}s, Dist={dist_x:.1f}m)"
                elif ttc <= self.config.ttc_warning_s or dist_x <= self.config.dist_warning_m:
                    level = ThreatLevel.WARNING
                    reason = f"Fast closing lead vehicle (TTC={ttc:.1f}s, Dist={dist_x:.1f}m)"
                elif ttc <= self.config.ttc_caution_s or dist_x <= self.config.dist_caution_m:
                    level = ThreatLevel.CAUTION
                    reason = f"Approaching vehicle in lane (TTC={ttc:.1f}s, Dist={dist_x:.1f}m)"
                else:
                    level = ThreatLevel.SAFE
                    reason = "Safe distance in ego corridor"
            else:
                # In lane but not closing (maintaining gap or moving away)
                if dist_x <= self.config.dist_critical_m:
                    level = ThreatLevel.WARNING
                    reason = f"Extremely close vehicle in lane ({dist_x:.1f}m)"
                elif dist_x <= self.config.dist_warning_m:
                    level = ThreatLevel.CAUTION
                    reason = f"Close following distance in lane ({dist_x:.1f}m)"
                else:
                    level = ThreatLevel.SAFE
                    reason = "Stable or opening gap in ego corridor"

        elif in_adjacent_lane:
            # Adjacent lane target: downgrade severity since not in immediate lane
            if is_approaching and (ttc <= self.config.ttc_warning_s or dist_x <= self.config.dist_warning_m):
                level = ThreatLevel.CAUTION
                reason = f"Adjacent vehicle approaching ({dist_x:.1f}m, Lat={lat_y:.1f}m)"
            else:
                level = ThreatLevel.SAFE
                reason = "Adjacent lane vehicle outside collision corridor"
        else:
            # Far lateral off-path target
            level = ThreatLevel.SAFE
            reason = "Off-path lateral target"

        return TrackAssessment(
            track_id=track.track_id,
            threat_level=level,
            forward_m=dist_x,
            lateral_m=lat_y,
            range_m=track.range_m,
            closing_speed_mps=closing_speed,
            ttc_s=ttc,
            in_ego_corridor=in_ego_lane,
            in_adjacent_corridor=in_adjacent_lane,
            is_approaching=is_approaching,
            reason=reason,
        )

    def evaluate_frame(self, tracks) -> ADASAlert:
        """
        Evaluate all active tracks and arbitrate the overall system-level threat.
        """
        assessments: List[TrackAssessment] = [
            self.evaluate_track(track) for track in tracks
        ]

        if not assessments:
            return ADASAlert(
                system_threat_level=ThreatLevel.SAFE,
                primary_hazard_track_id=None,
                total_active_tracks=0,
                total_confirmed_tracks=0,
                track_assessments=[],
                recommended_action="Clear path ahead.",
            )

        # Sort assessments by severity descending, then by distance ascending
        assessments.sort(
            key=lambda a: (a.threat_level.severity, -a.forward_m),
            reverse=True,
        )

        primary = assessments[0]
        sys_level = primary.threat_level

        action_map = {
            ThreatLevel.CRITICAL: "EMERGENCY BRAKE / EVASIVE ACTION REQUIRED",
            ThreatLevel.WARNING: "APPLY BRAKES / DECELERATE IMMEDIATELY",
            ThreatLevel.CAUTION: "MAINTAIN AWARENESS / PREPARE TO BRAKE",
            ThreatLevel.SAFE: "PATH CLEAR",
        }

        confirmed_count = sum(1 for t in tracks if getattr(t, "confirmed", False))

        return ADASAlert(
            system_threat_level=sys_level,
            primary_hazard_track_id=primary.track_id if sys_level != ThreatLevel.SAFE else None,
            total_active_tracks=len(tracks),
            total_confirmed_tracks=confirmed_count,
            track_assessments=assessments,
            recommended_action=action_map[sys_level],
        )
