"""
Time-To-Collision (TTC) Calculation Engine for Radar-Camera Fusion ADAS.

DISCLAIMER:
This module is part of a research and software prototype using prerecorded
RADIal camera + radar dataset recordings. The algorithms and thresholds
implemented herein are NOT certified for production automotive safety systems
(e.g., ISO 26262, Euro NCAP, UN ECE R152).

KINEMATIC & CONVENTION SPECIFICATION:
1. Longitudinal Distance (d_forward):
   - Measured in meters along the vehicle heading (+X boresight axis).
   - In our radar coordinate system, x_m represents forward longitudinal distance.
2. Velocity Sign Convention:
   - In track state coordinates: v_x = dx/dt.
     * v_x < 0 indicates the forward distance is decreasing (target is CLOSING / approaching).
     * v_x > 0 indicates the forward distance is increasing (target is RECEDING / moving away).
   - Therefore, Closing Speed is defined as:
     v_closing = -v_x
     When v_closing > 0, the target is approaching the ego vehicle.
3. Radar Doppler Velocity (velocity_mps):
   - In RADIal Doppler processing: negative velocity indicates approaching objects.
   - Closing speed from Doppler: v_closing_doppler = -velocity_mps.
4. Time-To-Collision (TTC):
   - TTC = d_forward / v_closing, for v_closing > 0.
   - If v_closing <= 0 (target is stationary relative to ego or moving away),
     TTC is infinite (float('inf')), indicating no forward collision trajectory.
   - If d_forward <= 0 (target has reached or passed sensor plane), TTC = 0.0.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass
class TTCResult:
    """
    Result of a Time-To-Collision evaluation for a tracked object.
    """
    forward_dist_m: float
    closing_speed_mps: float
    ttc_s: float
    is_approaching: bool
    status: str

    def __str__(self) -> str:
        if self.is_approaching and np.isfinite(self.ttc_s):
            return f"Dist: {self.forward_dist_m:.1f}m | Vclose: {self.closing_speed_mps:+.1f}m/s | TTC: {self.ttc_s:.1f}s ({self.status})"
        return f"Dist: {self.forward_dist_m:.1f}m | Vclose: {self.closing_speed_mps:+.1f}m/s | TTC: N/A ({self.status})"


def compute_ttc(
    forward_dist_m: float,
    closing_speed_mps: float,
    min_closing_speed_mps: float = 0.1,
) -> TTCResult:
    """
    Calculate Time-To-Collision (TTC) from forward distance and closing speed.

    Parameters
    ----------
    forward_dist_m : float
        Longitudinal forward distance to target in meters (> 0 ahead of ego).
    closing_speed_mps : float
        Relative closing speed in m/s (> 0 means approaching ego vehicle).
    min_closing_speed_mps : float
        Minimum threshold below which closing speed is treated as zero/diverging.

    Returns
    -------
    TTCResult
        Structured calculation result containing TTC in seconds and status.
    """
    d = float(forward_dist_m)
    vc = float(closing_speed_mps)

    if d <= 0.0:
        return TTCResult(
            forward_dist_m=d,
            closing_speed_mps=vc,
            ttc_s=0.0,
            is_approaching=True,
            status="IMPACT_OR_PASSED",
        )

    if vc > min_closing_speed_mps:
        ttc = d / vc
        return TTCResult(
            forward_dist_m=d,
            closing_speed_mps=vc,
            ttc_s=float(ttc),
            is_approaching=True,
            status="APPROACHING",
        )
    elif vc < -min_closing_speed_mps:
        return TTCResult(
            forward_dist_m=d,
            closing_speed_mps=vc,
            ttc_s=float("inf"),
            is_approaching=False,
            status="RECEDING",
        )
    else:
        return TTCResult(
            forward_dist_m=d,
            closing_speed_mps=vc,
            ttc_s=float("inf"),
            is_approaching=False,
            status="CONSTANT_DISTANCE",
        )


def compute_ttc_from_track(
    track,
    use_radar_doppler: bool = False,
    min_closing_speed_mps: float = 0.1,
) -> TTCResult:
    """
    Compute TTC directly from an active Track instance.

    Parameters
    ----------
    track : Track
        Active Track object from multi_object_tracker.
    use_radar_doppler : bool
        If True, use instantaneous radar Doppler velocity (-radar_velocity).
        If False (default), use the Kalman-filtered longitudinal velocity (-vx).
    min_closing_speed_mps : float
        Minimum closing speed threshold.
    """
    d = track.x

    if use_radar_doppler:
        # In RADIal Doppler convention, negative Doppler = closing
        closing_speed = -track.radar_velocity
    else:
        # In Kalman state, vx is dx/dt. Approaching means vx < 0 -> closing_speed = -vx
        closing_speed = -track.vx

    return compute_ttc(d, closing_speed, min_closing_speed_mps=min_closing_speed_mps)
