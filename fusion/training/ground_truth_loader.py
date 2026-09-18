from pathlib import Path
import numpy as np
import pandas as pd

from config import DEFAULT_GROUND_TRUTH


REQUIRED_GT_COLUMNS = [
    "dataset",
    "index",
    "numSample",
    "x1_pix",
    "y1_pix",
    "x2_pix",
    "y2_pix",
    "radar_X_m",
    "radar_Y_m",
    "radar_R_m",
    "radar_A_deg",
    "radar_D_mps",
    "radar_P_db",
]


class RadialGroundTruth:
    """
    Parser and query interface for official RADIal ground-truth annotations.
    """

    def __init__(self, ground_truth_path=None):
        if ground_truth_path is None:
            ground_truth_path = DEFAULT_GROUND_TRUTH

        self.path = Path(ground_truth_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.path}")

        self.df = pd.read_csv(self.path)

        # Validate columns
        missing = [c for c in REQUIRED_GT_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Ground truth file missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def get_sequence_annotations(self, sequence_name):
        """Return DataFrame filtered by dataset / sequence name."""
        return self.df[self.df["dataset"] == sequence_name].copy()

    def get_frame_annotations(self, sequence_name, frame_index):
        """
        Return list of ground truth object dictionaries for a specific sequence frame.

        Coordinate conversion:
        - In RADIal: radar_Y_m is forward (longitudinal) distance in meters.
        - In RADIal: radar_X_m is lateral distance (negative = right, positive = left).
        - In our pipeline: x_m is forward distance, y_m is lateral distance (positive = right, negative = left).
          Hence forward = radar_Y_m, lateral = -radar_X_m.
        """
        sub = self.df[
            (self.df["dataset"] == sequence_name)
            & (self.df["index"] == frame_index)
        ]

        objects = []
        for local_id, (_, row) in enumerate(sub.iterrows()):
            # Discard invalid/placeholder boxes
            x1, y1, x2, y2 = float(row["x1_pix"]), float(row["y1_pix"]), float(row["x2_pix"]), float(row["y2_pix"])
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                continue

            # Forward and lateral coordinates in vehicle frame
            forward_m = float(row["radar_Y_m"])
            lateral_m = -float(row["radar_X_m"])

            objects.append({
                "gt_id": local_id,
                "bbox": [x1, y1, x2, y2],
                "forward_m": forward_m,
                "lateral_m": lateral_m,
                "range_m": float(row["radar_R_m"]),
                "azimuth_deg": float(row["radar_A_deg"]),
                "doppler_mps": float(row["radar_D_mps"]),
                "power_db": float(row["radar_P_db"]),
                "num_sample": int(row["numSample"]),
                "difficult": float(row.get("Difficult", 0.0)),
                "annotation": str(row.get("Annotation", "unknown")),
            })

        return objects


def compute_box_iou(bbox1, bbox2):
    """
    Compute 2D Intersection-over-Union (IoU) between two bounding boxes.

    Format: [x1, y1, x2, y2]
    """
    x1_a, y1_a, x2_a, y2_a = map(float, bbox1)
    x1_b, y1_b, x2_b, y2_b = map(float, bbox2)

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def match_ground_truth(
    radar_objects,
    camera_objects,
    gt_objects,
    radar_dist_thresh=2.5,
    camera_iou_thresh=0.40,
):
    """
    Generate authentic ground-truth labels for candidate radar-camera pairs.

    A candidate pair (radar_i, camera_j) is labeled is_match = 1 IF AND ONLY IF:
    1. Radar object i spatially corresponds to ground-truth object G_k (distance <= radar_dist_thresh).
    2. Camera detection j visual corresponds to ground-truth object G_k (IoU >= camera_iou_thresh).
    3. BOTH correspond to the SAME ground-truth object G_k.

    Otherwise, is_match = 0.

    Parameters
    ----------
    radar_objects : list of dict
        Each dict must contain 'x_m' (forward) and 'y_m' (lateral).
    camera_objects : list of dict
        Each dict must contain 'bbox' ([x1, y1, x2, y2]).
    gt_objects : list of dict
        List of ground-truth objects from get_frame_annotations.
    radar_dist_thresh : float, default 2.5
        Maximum Euclidean distance (m) in radar plane to declare a match with GT.
    camera_iou_thresh : float, default 0.40
        Minimum 2D bounding-box IoU to declare a match with GT.

    Returns
    -------
    dict
        Mapping (radar_index, camera_index) -> int (1 = MATCH, 0 = NO-MATCH)
    """
    pair_labels = {}

    if not gt_objects or not radar_objects or not camera_objects:
        # If no ground truth or no detections, all candidate pairs are negative
        for r_idx in range(len(radar_objects)):
            for c_idx in range(len(camera_objects)):
                pair_labels[(r_idx, c_idx)] = 0
        return pair_labels

    # 1. Match each radar object to nearest GT object
    radar_to_gt = {}
    for r_idx, r_obj in enumerate(radar_objects):
        rx, ry = float(r_obj["x_m"]), float(r_obj["y_m"])
        best_dist = float("inf")
        best_gt_id = None
        for gt in gt_objects:
            gx, gy = gt["forward_m"], gt["lateral_m"]
            dist = float(np.sqrt((rx - gx) ** 2 + (ry - gy) ** 2))
            if dist <= radar_dist_thresh and dist < best_dist:
                best_dist = dist
                best_gt_id = gt["gt_id"]
        radar_to_gt[r_idx] = best_gt_id

    # 2. Match each camera object to best-overlapping GT object
    camera_to_gt = {}
    for c_idx, c_obj in enumerate(camera_objects):
        c_box = c_obj["bbox"]
        best_iou = 0.0
        best_gt_id = None
        for gt in gt_objects:
            iou = compute_box_iou(c_box, gt["bbox"])
            if iou >= camera_iou_thresh and iou > best_iou:
                best_iou = iou
                best_gt_id = gt["gt_id"]
        camera_to_gt[c_idx] = best_gt_id

    # 3. Associate pairs: is_match = 1 iff both match the SAME non-None GT object
    for r_idx in range(len(radar_objects)):
        r_gt = radar_to_gt.get(r_idx)
        for c_idx in range(len(camera_objects)):
            c_gt = camera_to_gt.get(c_idx)

            if r_gt is not None and c_gt is not None and r_gt == c_gt:
                pair_labels[(r_idx, c_idx)] = 1
            else:
                pair_labels[(r_idx, c_idx)] = 0

    return pair_labels
