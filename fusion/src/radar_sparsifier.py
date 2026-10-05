"""
Stage 5: Adaptive Pre-Fusion Radar Sparsification.

Implements adaptive sparsification of radar objects after RadarObjectExtractor
and before RadarCameraProjector / ML association and fusion.

Computes an importance score for each radar object based on:
1. Radar reflected power (signal strength in dB)
2. Absolute radial velocity (kinematic relevance / moving object priority)
3. Target range (proximity / collision urgency)

Preserves the original radar object structure while selecting the highest-scoring
objects according to the user-specified retention ratio.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class AdaptiveRadarSparsifier:
    """
    Adaptive pre-fusion radar sparsifier for computationally efficient radar-camera fusion.

    Operates after RadarObjectExtractor and before RadarCameraProjector / ML fusion.
    Calculates an importance score for each radar object using:
    - radar power (reflected power / SNR)
    - absolute radial velocity (|velocity_mps| / kinematic priority)
    - range (proximity / collision risk)

    Attributes
    ----------
    retention_ratio : float
        Fraction of radar objects to retain, between 0.0 and 1.0.
    w_power : float
        Importance weight for radar power.
    w_velocity : float
        Importance weight for absolute radial velocity.
    w_range : float
        Importance weight for range proximity.
    """

    SUPPORTED_RATIOS = (1.0, 0.75, 0.50, 0.25, 0.10)

    def __init__(
        self,
        retention_ratio: float = 0.5,
        w_power: float = 0.4,
        w_velocity: float = 0.3,
        w_range: float = 0.3,
    ):
        if not (0.0 <= retention_ratio <= 1.0):
            raise ValueError(
                f"retention_ratio must be between 0.0 and 1.0, got {retention_ratio}"
            )

        self.retention_ratio = float(retention_ratio)
        self.w_power = float(w_power)
        self.w_velocity = float(w_velocity)
        self.w_range = float(w_range)

        # Normalize weights so they sum to 1.0
        total_w = self.w_power + self.w_velocity + self.w_range
        if total_w > 0:
            self.w_power /= total_w
            self.w_velocity /= total_w
            self.w_range /= total_w

    def set_retention_ratio(self, ratio: float) -> None:
        """Update retention ratio with validation."""
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(
                f"retention_ratio must be between 0.0 and 1.0, got {ratio}"
            )
        self.retention_ratio = float(ratio)

    def compute_importance_scores(
        self,
        radar_objects: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Calculate an importance score for each radar object using radar power,
        absolute radial velocity, and range.

        Parameters
        ----------
        radar_objects : list of dict
            Radar object dictionaries from RadarObjectExtractor.

        Returns
        -------
        list of float
            Importance scores in [0.0, 1.0] for each radar object.
        """
        if not radar_objects:
            return []

        n_objects = len(radar_objects)
        if n_objects == 1:
            return [1.0]

        powers = []
        velocities = []
        ranges = []

        for obj in radar_objects:
            # 1. Power in dB
            p = float(obj.get("power", 0.0))
            p_db = 10.0 * np.log10(max(p, 1e-9))
            powers.append(p_db)

            # 2. Velocity (handle NaN/inf safely)
            v = float(obj.get("velocity_mps", 0.0))
            if np.isnan(v) or np.isinf(v):
                v = 0.0
            velocities.append(abs(v))

            # 3. Range (non-negative)
            r = max(float(obj.get("range_m", 0.0)), 0.0)
            ranges.append(r)

        powers = np.array(powers, dtype=np.float64)
        velocities = np.array(velocities, dtype=np.float64)
        ranges = np.array(ranges, dtype=np.float64)

        # Power normalization (min-max across current frame)
        p_min, p_max = powers.min(), powers.max()
        if p_max > p_min:
            norm_power = (powers - p_min) / (p_max - p_min)
        else:
            norm_power = np.ones(n_objects, dtype=np.float64)

        # Velocity normalization (min-max across current frame)
        v_min, v_max = velocities.min(), velocities.max()
        if v_max > v_min:
            norm_velocity = (velocities - v_min) / (v_max - v_min)
        else:
            norm_velocity = (
                np.clip(velocities / 15.0, 0.0, 1.0)
                if v_min > 0.0
                else np.full(n_objects, 0.5)
            )

        # Range normalization (proximity: closer objects score higher)
        r_min, r_max = ranges.min(), ranges.max()
        if r_max > r_min:
            norm_range = (r_max - ranges) / (r_max - r_min)
        else:
            norm_range = 1.0 / (1.0 + ranges / 20.0)

        # Weighted combination
        scores = (
            self.w_power * norm_power
            + self.w_velocity * norm_velocity
            + self.w_range * norm_range
        )

        return [float(np.clip(s, 0.0, 1.0)) for s in scores]

    def select(
        self,
        radar_objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Select highest-scoring radar objects based on retention_ratio.

        Parameters
        ----------
        radar_objects : list of dict
            List of radar object dictionaries.

        Returns
        -------
        list of dict
            Selected radar objects, preserving the original dictionary structure.
        """
        if not radar_objects:
            return []

        if self.retention_ratio <= 0.0:
            return []

        n_objects = len(radar_objects)

        # Compute importance scores
        scores = self.compute_importance_scores(radar_objects)

        # Determine target count k
        if self.retention_ratio >= 1.0:
            k = n_objects
        else:
            k = int(round(n_objects * self.retention_ratio + 1e-9))
            k = min(n_objects, max(1, k))

        # Stable sort by descending score (tie-breaking by original index)
        sorted_indices = sorted(
            range(n_objects),
            key=lambda idx: (-scores[idx], idx),
        )
        selected_indices = sorted_indices[:k]

        selected_objects = []
        for idx in selected_indices:
            obj_copy = dict(radar_objects[idx])
            obj_copy["importance_score"] = scores[idx]
            selected_objects.append(obj_copy)

        return selected_objects


class CameraGuidedRadarSparsifier(AdaptiveRadarSparsifier):
    """
    Stage 5b: Camera-Guided / Fusion-Aware Adaptive Radar Sparsifier.

    Combines the Stage 5 radar-only importance score with a cheap geometric
    camera relevance score computed before ML association:
        S_fusion = 0.6 * S_radar + 0.4 * S_camera

    Camera relevance S_camera favors:
    1. Radar points inside a camera bounding box (highest score: 0.70 - 1.00)
    2. Radar points close to a camera bounding box (exponential distance decay)
    3. Radar points inside camera FoV when far from detections / no detections (neutral score: 0.50 / 0.10)
    4. Strongly penalizes radar points outside camera FoV (score: 0.0)

    Attributes
    ----------
    retention_ratio : float
        Fraction of radar objects to retain (1.00, 0.75, 0.50, 0.25, 0.10).
    w_radar : float
        Weight for radar importance score (default 0.6).
    w_camera : float
        Weight for camera relevance score (default 0.4).
    distance_scale : float
        Scale factor in pixels for exponential distance decay (default 100.0).
    """

    def __init__(
        self,
        retention_ratio: float = 0.5,
        w_radar: float = 0.6,
        w_camera: float = 0.4,
        w_power: float = 0.4,
        w_velocity: float = 0.3,
        w_range: float = 0.3,
        distance_scale: float = 100.0,
    ):
        super().__init__(
            retention_ratio=retention_ratio,
            w_power=w_power,
            w_velocity=w_velocity,
            w_range=w_range,
        )
        self.w_radar = float(w_radar)
        self.w_camera = float(w_camera)
        total_w = self.w_radar + self.w_camera
        if total_w > 0:
            self.w_radar /= total_w
            self.w_camera /= total_w
        self.distance_scale = max(float(distance_scale), 1.0)

    def compute_camera_scores(
        self,
        radar_objects: List[Dict[str, Any]],
        projected_pixels: Optional[np.ndarray] = None,
        valid_in_fov: Optional[np.ndarray] = None,
        camera_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """
        Calculate a cheap geometric camera relevance score for each radar object.

        Favors:
        - inside camera bounding box
        - close to camera bounding box
        - inside camera FoV
        Penalizes:
        - outside camera FoV (score = 0.0)
        """
        if not radar_objects:
            return []

        n_objects = len(radar_objects)

        # Fallback extraction from radar object dictionary if projected_pixels not passed explicitly
        if projected_pixels is None:
            extracted_pixels = []
            has_pixels = True
            for obj in radar_objects:
                if "projected_u" in obj and "projected_v" in obj:
                    extracted_pixels.append([obj["projected_u"], obj["projected_v"]])
                elif "pixel" in obj:
                    extracted_pixels.append(obj["pixel"])
                else:
                    has_pixels = False
                    break
            if has_pixels and len(extracted_pixels) == n_objects:
                projected_pixels = np.array(extracted_pixels, dtype=np.float64)

        if valid_in_fov is None:
            if projected_pixels is not None:
                valid_in_fov = np.isfinite(projected_pixels).all(axis=1)
            else:
                valid_in_fov = np.array(
                    [bool(obj.get("valid_in_fov", obj.get("in_fov", True))) for obj in radar_objects],
                    dtype=bool,
                )

        camera_scores = []
        for i in range(n_objects):
            in_fov = bool(valid_in_fov[i]) if valid_in_fov is not None else True

            # Outside FoV or non-projectable -> strong penalty (0.0)
            if not in_fov or projected_pixels is None:
                camera_scores.append(0.0)
                continue

            u, v = float(projected_pixels[i][0]), float(projected_pixels[i][1])
            if not (np.isfinite(u) and np.isfinite(v)):
                camera_scores.append(0.0)
                continue

            # Case: No camera detections in frame
            if not camera_objects:
                # All points inside FoV receive a neutral baseline score
                camera_scores.append(0.50)
                continue

            # Case: Camera detections exist
            best_cam_score = 0.25  # Baseline inside FoV far from detections
            for cam in camera_objects:
                bbox = cam.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(float, bbox)
                conf = float(cam.get("confidence", 1.0))
                conf = np.clip(conf, 0.0, 1.0)

                # Distance to bounding box boundary
                dx = max(x1 - u, 0.0, u - x2)
                dy = max(y1 - v, 0.0, v - y2)
                d = float(np.sqrt(dx * dx + dy * dy))

                if dx == 0.0 and dy == 0.0:
                    # Strictly inside bounding box (0.75 to 1.00)
                    box_score = 0.75 + 0.25 * conf
                else:
                    # Proximity decay (0.25 to ~0.70)
                    box_score = 0.25 + 0.45 * np.exp(-d / self.distance_scale) * conf

                if box_score > best_cam_score:
                    best_cam_score = box_score

            camera_scores.append(float(np.clip(best_cam_score, 0.0, 1.0)))

        return camera_scores

    def compute_fusion_scores(
        self,
        radar_objects: List[Dict[str, Any]],
        projected_pixels: Optional[np.ndarray] = None,
        valid_in_fov: Optional[np.ndarray] = None,
        camera_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """
        Compute fusion-aware score:
            S_fusion = 0.6 * S_radar + 0.4 * S_camera
        """
        if not radar_objects:
            return []

        scores_radar = self.compute_importance_scores(radar_objects)

        if projected_pixels is None and camera_objects is None:
            # Pure radar-only fallback if no camera/projection info is provided
            return scores_radar

        scores_camera = self.compute_camera_scores(
            radar_objects,
            projected_pixels=projected_pixels,
            valid_in_fov=valid_in_fov,
            camera_objects=camera_objects,
        )

        scores_fusion = [
            float(np.clip(self.w_radar * r + self.w_camera * c, 0.0, 1.0))
            for r, c in zip(scores_radar, scores_camera)
        ]
        return scores_fusion

    def select(
        self,
        radar_objects: List[Dict[str, Any]],
        projected_pixels: Optional[np.ndarray] = None,
        valid_in_fov: Optional[np.ndarray] = None,
        camera_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select highest-scoring radar objects based on retention_ratio and fusion score.
        """
        if not radar_objects:
            return []

        if self.retention_ratio <= 0.0:
            return []

        n_objects = len(radar_objects)

        scores_radar = self.compute_importance_scores(radar_objects)
        scores_camera = self.compute_camera_scores(
            radar_objects,
            projected_pixels=projected_pixels,
            valid_in_fov=valid_in_fov,
            camera_objects=camera_objects,
        ) if (projected_pixels is not None or camera_objects is not None) else [0.5] * n_objects

        if projected_pixels is None and camera_objects is None:
            scores_fusion = scores_radar
        else:
            scores_fusion = [
                float(np.clip(self.w_radar * r + self.w_camera * c, 0.0, 1.0))
                for r, c in zip(scores_radar, scores_camera)
            ]

        # Determine target count k
        if self.retention_ratio >= 1.0:
            k = n_objects
        else:
            k = int(round(n_objects * self.retention_ratio + 1e-9))
            k = min(n_objects, max(1, k))

        # Deterministic sort by descending fusion score (tie-breaking by original index)
        sorted_indices = sorted(
            range(n_objects),
            key=lambda idx: (-scores_fusion[idx], idx),
        )
        selected_indices = sorted_indices[:k]

        selected_objects = []
        for idx in selected_indices:
            obj_copy = dict(radar_objects[idx])
            obj_copy["importance_score"] = scores_fusion[idx]
            obj_copy["fusion_score"] = scores_fusion[idx]
            obj_copy["radar_score"] = scores_radar[idx]
            obj_copy["camera_score"] = scores_camera[idx]
            selected_objects.append(obj_copy)

        return selected_objects


# Convenient alias
FusionAwareRadarSparsifier = CameraGuidedRadarSparsifier
