"""
Multi-Object Tracker (MOT) and Kalman-Filter State Estimation for Radar-Camera Fusion.

Maintains stable track IDs across synchronized frames, smooths positions and velocities,
handles temporary missed detections (coasting), and prevents duplicate tracks.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter2D:
    """
    Constant-Velocity (CV) Kalman Filter for 2D position and velocity estimation.
    State vector: [x, y, vx, vy]^T
    - x: forward distance (m) along boresight (+X)
    - y: lateral distance (m) (+Y left, -Y right)
    - vx: forward velocity (m/s)
    - vy: lateral velocity (m/s)
    """

    def __init__(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        sigma_pos: float = 0.5,
        sigma_vel: float = 1.5,
        q_pos: float = 0.2,
        q_vel: float = 0.8,
    ):
        # State vector [x, y, vx, vy]
        self.state = np.array([x, y, vx, vy], dtype=np.float64)

        # State covariance P
        self.P = np.diag([
            sigma_pos ** 2,
            sigma_pos ** 2,
            sigma_vel ** 2,
            sigma_vel ** 2,
        ]).astype(np.float64)

        # Measurement matrix H (measuring x and y directly)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Measurement noise covariance R
        self.R = np.diag([
            sigma_pos ** 2,
            sigma_pos ** 2,
        ]).astype(np.float64)

        self.q_pos = q_pos
        self.q_vel = q_vel

    def predict(self, dt: float = 0.2) -> np.ndarray:
        """Advance the state by dt seconds under constant velocity assumption (default 0.2s for 5 FPS RADIal)."""
        dt = float(dt)
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Continuous white noise acceleration model for Q
        q = self.q_vel
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0

        Q = np.array([
            [dt4 * q, 0.0, dt3 * q, 0.0],
            [0.0, dt4 * q, 0.0, dt3 * q],
            [dt3 * q, 0.0, dt2 * q, 0.0],
            [0.0, dt3 * q, 0.0, dt2 * q],
        ], dtype=np.float64)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        return self.state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with 2D position measurement [x, y].
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(2,)
        y = z - self.H @ self.state  # Innovation (measurement residual)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Optimal Kalman gain

        self.state = self.state + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return self.state


class Track:
    """
    Represents an individual tracked object over time.
    """

    def __init__(
        self,
        track_id: int,
        detection: Dict[str, Any],
        min_hits: int = 2,
    ):
        self.track_id = int(track_id)
        self.min_hits = int(min_hits)

        # Extract initial coordinates
        x = float(detection.get("forward_m", detection.get("radar_x", 0.0)))
        y = float(detection.get("lateral_m", detection.get("radar_y", 0.0)))
        vx_init = float(detection.get("velocity_mps", 0.0))

        self.kf = KalmanFilter2D(x=x, y=y, vx=vx_init, vy=0.0)

        # Lifecycle metrics
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = (self.hits >= self.min_hits)

        # Attributes from the latest matched detection
        self.class_name = detection.get("class_name", "vehicle")
        self.camera_confidence = float(detection.get("camera_confidence", 0.0))
        self.fusion_confidence = float(detection.get("fusion_confidence", detection.get("probability", 1.0)))
        self.bbox = detection.get("bbox", detection.get("camera_bbox", [0.0, 0.0, 0.0, 0.0]))
        self.radar_velocity = float(detection.get("velocity_mps", 0.0))
        self.radar_power = float(detection.get("radar_power", 0.0))
        self.projected_pixel = detection.get("projected_pixel", None)

    @property
    def x(self) -> float:
        """Longitudinal forward distance in meters."""
        return float(self.kf.state[0])

    @property
    def y(self) -> float:
        """Lateral distance in meters (positive = left, negative = right)."""
        return float(self.kf.state[1])

    @property
    def vx(self) -> float:
        """Forward velocity in m/s (+ moving away, - closing)."""
        return float(self.kf.state[2])

    @property
    def vy(self) -> float:
        """Lateral velocity in m/s."""
        return float(self.kf.state[3])

    @property
    def range_m(self) -> float:
        """Slant range in 2D ground plane."""
        return float(np.hypot(self.x, self.y))

    def predict(self, dt: float = 0.2) -> np.ndarray:
        """Predict forward by dt and increment age (default 0.2s for 5 FPS RADIal)."""
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict(dt)

    def update(self, detection: Dict[str, Any]) -> None:
        """Update track with a newly matched detection."""
        meas_x = float(detection.get("forward_m", detection.get("radar_x", self.x)))
        meas_y = float(detection.get("lateral_m", detection.get("radar_y", self.y)))

        self.kf.update(np.array([meas_x, meas_y]))

        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.min_hits:
            self.confirmed = True

        # Update metadata
        self.class_name = detection.get("class_name", self.class_name)
        self.camera_confidence = float(detection.get("camera_confidence", self.camera_confidence))
        self.fusion_confidence = float(detection.get("fusion_confidence", detection.get("probability", self.fusion_confidence)))
        self.bbox = detection.get("bbox", detection.get("camera_bbox", self.bbox))
        self.radar_velocity = float(detection.get("velocity_mps", self.radar_velocity))
        self.radar_power = float(detection.get("radar_power", self.radar_power))
        if "projected_pixel" in detection:
            self.projected_pixel = detection["projected_pixel"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize track state into dictionary format."""
        return {
            "track_id": self.track_id,
            "forward_m": self.x,
            "lateral_m": self.y,
            "range_m": self.range_m,
            "vx_mps": self.vx,
            "vy_mps": self.vy,
            "radar_velocity_mps": self.radar_velocity,
            "confirmed": self.confirmed,
            "hits": self.hits,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "class_name": self.class_name,
            "camera_confidence": self.camera_confidence,
            "fusion_confidence": self.fusion_confidence,
            "bbox": self.bbox,
            "projected_pixel": self.projected_pixel,
        }


class MultiObjectTracker:
    """
    Multi-Object Tracker using Hungarian frame-to-frame data association.

    Parameters
    ----------
    max_cost_m : float
        Maximum 2D Euclidean distance gate in meters for valid association.
    max_age : int
        Number of consecutive missed frames before a track is deleted (coasting duration).
    min_hits : int
        Number of detections required before a track is considered CONFIRMED.
    """

    def __init__(
        self,
        max_cost_m: float = 4.5,
        max_age: int = 3,
        min_hits: int = 2,
    ):
        self.max_cost_m = float(max_cost_m)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)

        self.tracks: List[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        """Reset tracker state and track IDs."""
        self.tracks.clear()
        self._next_id = 1

    def update(
        self,
        fused_detections: List[Dict[str, Any]],
        dt: float = 0.2,
    ) -> List[Track]:
        """
        Process a new set of fused radar-camera detections for time interval dt.

        Parameters
        ----------
        fused_detections : List[Dict[str, Any]]
            List of fused detections from Stage 3 ML association. Each must have
            at least position coordinates ('forward_m'/'radar_x', 'lateral_m'/'radar_y').
        dt : float
            Elapsed time since last frame in seconds (default 0.2s for 5 FPS RADIal).

        Returns
        -------
        List[Track]
            List of all currently active (coasted and updated) tracks.
        """
        # 1. Predict existing tracks forward
        for track in self.tracks:
            track.predict(dt)

        num_tracks = len(self.tracks)
        num_detections = len(fused_detections)

        if num_tracks == 0:
            # All detections initialize new tracks
            for det in fused_detections:
                new_track = Track(
                    track_id=self._next_id,
                    detection=det,
                    min_hits=self.min_hits,
                )
                self._next_id += 1
                self.tracks.append(new_track)
            return self.tracks

        if num_detections == 0:
            # All tracks are missed this frame
            self.tracks = [
                t for t in self.tracks if t.time_since_update <= self.max_age
            ]
            return self.tracks

        # 2. Build cost matrix (2D Euclidean distance in vehicle ground plane)
        cost_matrix = np.zeros((num_tracks, num_detections), dtype=np.float64)
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(fused_detections):
                dx = track.x - float(det.get("forward_m", det.get("radar_x", 0.0)))
                dy = track.y - float(det.get("lateral_m", det.get("radar_y", 0.0)))
                cost_matrix[i, j] = np.sqrt(dx * dx + dy * dy)

        # 3. Hungarian matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_detections = set()

        for r, c in zip(row_indices, col_indices):
            cost = cost_matrix[r, c]
            if cost <= self.max_cost_m:
                self.tracks[r].update(fused_detections[c])
                matched_tracks.add(r)
                matched_detections.add(c)

        # 4. Handle unmatched existing tracks (already marked missed during predict)
        # Retain tracks that haven't exceeded max_age
        self.tracks = [
            t for i, t in enumerate(self.tracks)
            if (i in matched_tracks) or (t.time_since_update <= self.max_age)
        ]

        # 5. Handle unmatched detections -> initiate new tracks
        for c in range(num_detections):
            if c not in matched_detections:
                new_track = Track(
                    track_id=self._next_id,
                    detection=fused_detections[c],
                    min_hits=self.min_hits,
                )
                self._next_id += 1
                self.tracks.append(new_track)

        return self.tracks

    def get_confirmed_tracks(self) -> List[Track]:
        """Return only tracks that have achieved confirmed status."""
        return [t for t in self.tracks if t.confirmed]
