"""
Integrated Stage 4 Pipeline: Stage 3 Ground-Truth ML Fusion + Multi-Object Tracking + TTC + ADAS.

Strictly preserves:
- Existing radar DSP pipeline
- Existing camera YOLO detector
- Stage 3 Ground-Truth ML model (radar_camera_fusion_groundtruth.pkl)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import (
    DEFAULT_GROUNDTRUTH_MODEL_PATH,
    DEFAULT_MODEL_PATH,
)
from fusion.src.fusion_model import FusionMLModel
from fusion.src.tracker import MultiObjectTracker, Track
from fusion.src.ttc import compute_ttc_from_track, TTCResult
from fusion.src.adas_decision import ADASDecisionModule, ADASConfig, ADASAlert, ThreatLevel


def extract_stage3_features(
    radar_object: Dict[str, Any],
    projected_pixel: Tuple[float, float],
    camera_object: Dict[str, Any],
) -> Dict[str, float]:
    """
    Extract the exact 19 leakage-free features required by the Stage 3 Ground-Truth model.
    """
    u, v = projected_pixel
    x1, y1, x2, y2 = map(float, camera_object["bbox"])

    bbox_center_u = (x1 + x2) / 2.0
    bbox_center_v = (y1 + y2) / 2.0
    bbox_width = max(x2 - x1, 1e-3)
    bbox_height = max(y2 - y1, 1e-3)
    bbox_aspect_ratio = bbox_width / bbox_height

    norm_offset_u = (u - bbox_center_u) / bbox_width
    norm_offset_v = (v - bbox_center_v) / bbox_height
    norm_radial_dist = float(np.sqrt(norm_offset_u ** 2 + norm_offset_v ** 2))

    return {
        "range_m": float(radar_object.get("range_m", 0.0)),
        "velocity_mps": float(radar_object.get("velocity_mps", 0.0)),
        "azimuth_deg": float(radar_object.get("azimuth_deg", 0.0)),
        "elevation_deg": float(radar_object.get("elevation_deg", 0.0)),
        "radar_power": float(radar_object.get("power", 0.0)),
        "radar_x": float(radar_object.get("x_m", 0.0)),
        "radar_y": float(radar_object.get("y_m", 0.0)),
        "radar_z": float(radar_object.get("z_m", 0.0)),
        "projected_u": float(u),
        "projected_v": float(v),
        "bbox_center_u": float(bbox_center_u),
        "bbox_center_v": float(bbox_center_v),
        "bbox_width": float(bbox_width),
        "bbox_height": float(bbox_height),
        "bbox_aspect_ratio": float(bbox_aspect_ratio),
        "camera_confidence": float(camera_object.get("confidence", 0.0)),
        "norm_offset_u": float(norm_offset_u),
        "norm_offset_v": float(norm_offset_v),
        "norm_radial_dist": float(norm_radial_dist),
    }


class Stage4ADASPipeline:
    """
    End-to-End Stage 4 Tracking, TTC, and ADAS Decision Pipeline.
    """

    def __init__(
        self,
        fusion_model_path: Optional[Path] = None,
        tracker: Optional[MultiObjectTracker] = None,
        adas_module: Optional[ADASDecisionModule] = None,
        default_dt: float = 0.2,
    ):
        model_path = Path(fusion_model_path) if fusion_model_path else DEFAULT_GROUNDTRUTH_MODEL_PATH
        if not model_path.exists():
            # Fallback to default model if groundtruth model not found
            model_path = DEFAULT_MODEL_PATH

        self.fusion_model = FusionMLModel(model_path)
        self.default_dt = float(default_dt)
        self.tracker = tracker or MultiObjectTracker(max_cost_m=4.5, max_age=3, min_hits=2)
        self.adas_module = adas_module or ADASDecisionModule()

    def associate_stage3(
        self,
        radar_objects: List[Dict[str, Any]],
        projected_pixels: np.ndarray,
        valid_mask: np.ndarray,
        camera_objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run Stage 3 ML association to produce fused radar-camera objects
        using strict 1-to-1 bipartite matching (each camera detection is paired
        with at most one radar detection, and vice-versa).
        """
        fused_objects = []

        if not radar_objects or not camera_objects or len(projected_pixels) == 0:
            return fused_objects

        num_radars = len(radar_objects)
        num_cameras = len(camera_objects)

        # Build candidate cost matrix: cost is (1.0 - probability) for valid matches (pred == 1),
        # and prohibitive penalty (1000.0) for non-matches.
        cost_matrix = np.full((num_radars, num_cameras), 1000.0, dtype=np.float64)
        pair_data = {}

        for r_idx, radar_obj in enumerate(radar_objects):
            if not valid_mask[r_idx]:
                continue

            u, v = projected_pixels[r_idx]
            if not (np.isfinite(u) and np.isfinite(v)):
                continue

            for c_idx, cam_obj in enumerate(camera_objects):
                feat = extract_stage3_features(radar_obj, (u, v), cam_obj)

                pred = self.fusion_model.predict(feat)
                prob = self.fusion_model.predict_probability(feat)

                if pred == 1 and prob > 0.0:
                    cost_matrix[r_idx, c_idx] = 1.0 - float(prob)
                    pair_data[(r_idx, c_idx)] = {
                        "radar_obj": radar_obj,
                        "cam_obj": cam_obj,
                        "probability": float(prob),
                        "pixel": (float(u), float(v)),
                    }

        # If no pairs were predicted as valid matches, return empty
        if not pair_data:
            return fused_objects

        # Bipartite maximum-confidence matching via Hungarian algorithm:
        # Guarantees strictly at most one radar detection per camera detection.
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for r_idx, c_idx in zip(row_indices, col_indices):
            if cost_matrix[r_idx, c_idx] < 1.0:  # Valid matched candidate
                data = pair_data[(r_idx, c_idx)]
                radar_obj = data["radar_obj"]
                cam_obj = data["cam_obj"]
                prob = data["probability"]
                pixel = data["pixel"]

                fused_objects.append({
                    "forward_m": radar_obj["x_m"],
                    "lateral_m": radar_obj["y_m"],
                    "range_m": radar_obj["range_m"],
                    "velocity_mps": radar_obj["velocity_mps"],
                    "azimuth_deg": radar_obj["azimuth_deg"],
                    "elevation_deg": radar_obj["elevation_deg"],
                    "radar_power": radar_obj["power"],
                    "camera_bbox": cam_obj["bbox"],
                    "class_name": cam_obj["class_name"],
                    "camera_confidence": cam_obj["confidence"],
                    "fusion_confidence": prob,
                    "projected_pixel": pixel,
                    "radar_index": r_idx,
                    "camera_index": c_idx,
                })

        return fused_objects

    def process_frame(
        self,
        radar_objects: List[Dict[str, Any]],
        projected_pixels: np.ndarray,
        valid_mask: np.ndarray,
        camera_objects: List[Dict[str, Any]],
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a single synchronized frame through Stage 3 ML fusion -> Tracking -> TTC -> ADAS.
        """
        frame_dt = float(dt) if dt is not None else self.default_dt

        # 1. Spatial Fusion (Stage 3 ML, 1-to-1 matching)
        fused_objects = self.associate_stage3(
            radar_objects,
            projected_pixels,
            valid_mask,
            camera_objects,
        )

        # 2. Multi-Object Tracking & Temporal Fusion
        active_tracks = self.tracker.update(fused_objects, dt=frame_dt)
        confirmed_tracks = self.tracker.get_confirmed_tracks()

        # 3. ADAS Decision & Forward Collision Assessment
        adas_alert: ADASAlert = self.adas_module.evaluate_frame(active_tracks)

        return {
            "fused_objects": fused_objects,
            "tracks": active_tracks,
            "confirmed_tracks": confirmed_tracks,
            "adas_alert": adas_alert,
            "raw_radar_count": len(radar_objects),
            "raw_camera_count": len(camera_objects),
            "dt": frame_dt,
        }
