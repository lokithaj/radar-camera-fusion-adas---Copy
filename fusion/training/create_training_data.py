import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from fusion.src.sync_loader import SynchronizedFusionLoader
from fusion.src.radar_camera_projector import RadarCameraProjector
from fusion.training.ground_truth_loader import RadialGroundTruth, match_ground_truth

from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN
from radar.src.radar_objects import RadarObjectExtractor

from camera.src.camera_detector import CameraDetector

from config import (
    DEFAULT_TRAINING_DATA,
    DEFAULT_GROUND_TRUTH,
    get_default_recording_dir,
    get_dbreader_dir,
    get_radar_calibration_path,
    get_camera_calibration_path,
)

# ============================================================
# Settings
# ============================================================

MAX_RADAR_POINTS = 20
MIN_RANGE_BIN = 10
YOLO_CONFIDENCE = 0.40
RADAR_GT_MATCH_DISTANCE = 2.5  # meters
CAMERA_GT_MATCH_IOU = 0.40     # 2D IoU


def process_radar(data, radar_calibration_path):
    """
    Run the radar processing pipeline for one synchronized sample.
    """
    frame_builder = RadarFrameBuilder()
    complex_frame = frame_builder.build_frame(
        data["radar_ch0"],
        data["radar_ch1"],
        data["radar_ch2"],
        data["radar_ch3"],
    )

    # Static clutter removal
    complex_frame = complex_frame - np.mean(
        complex_frame,
        axis=(0, 1),
        keepdims=True,
    )

    fft_processor = RadarFFTProcessor()
    range_fft = fft_processor.range_fft(complex_frame)
    rd_spectrum = fft_processor.doppler_fft(range_fft)
    rd_power = np.sum(np.abs(rd_spectrum) ** 2, axis=2)

    cfar = CACFAR(
        window=(9, 9),
        guard=(3, 3),
        threshold_db=2.0,
    )
    detections = cfar.detect(rd_power)

    point_generator = RealRadarPointGenerator(
        radar_calibration_path,
        max_points=MAX_RADAR_POINTS,
        min_range_bin=MIN_RANGE_BIN,
    )
    radar_points = point_generator.generate(
        rd_spectrum,
        detections,
    )

    clusterer = RadarDBSCAN(
        eps=1.5,
        min_samples=3,
    )
    labels = clusterer.cluster(radar_points)

    object_extractor = RadarObjectExtractor()
    radar_objects = object_extractor.extract(
        radar_points,
        labels,
    )

    return radar_objects


def generate_dataset_for_sequence(
    recording_dir,
    output_csv,
    ground_truth_path=DEFAULT_GROUND_TRUTH,
    frame_step=1,
    max_frames=None,
):
    """
    Generate sensor fusion dataset for a single recording using independent ground truth.
    """
    recording_dir = Path(recording_dir).resolve()
    output_csv = Path(output_csv).resolve()

    print(f"\n========================================================")
    print(f"Processing sequence: {recording_dir.name}")
    print(f"Output: {output_csv}")
    print(f"Frame step: {frame_step}")
    print(f"========================================================")

    dbreader_dir = get_dbreader_dir(recording_dir)
    radar_calib = get_radar_calibration_path(recording_dir)
    camera_calib = get_camera_calibration_path(recording_dir)

    loader = SynchronizedFusionLoader(recording_dir, dbreader_dir)
    detector = CameraDetector(model_name="yolo11n.pt", confidence=YOLO_CONFIDENCE)
    projector = RadarCameraProjector(camera_calib)
    gt_loader = RadialGroundTruth(ground_truth_path)

    total_synced = len(loader)
    print(f"Total synchronized frames: {total_synced}")

    frame_limit = total_synced if max_frames is None else min(total_synced, max_frames)
    sequence_name = recording_dir.name

    rows = []

    for sample_index in range(0, frame_limit, frame_step):
        data = loader.load(sample_index)
        image = data["camera"]

        # 1. Camera Detection
        try:
            _, camera_objects = detector.detect(image)
        except Exception as error:
            print(f"Camera detection error at frame {sample_index}: {error}")
            continue

        # 2. Radar Processing
        try:
            radar_objects = process_radar(data, radar_calib)
        except Exception as error:
            print(f"Radar processing error at frame {sample_index}: {error}")
            continue

        if not radar_objects or not camera_objects:
            continue

        # 3. Project Radar Centroids to Camera Pixel Coordinates
        radar_xyz = np.array(
            [[obj["x_m"], obj["y_m"], obj["z_m"]] for obj in radar_objects],
            dtype=np.float64,
        )
        pixels = projector.project(radar_xyz)

        # 4. Ingest Independent Ground-Truth Objects for this Frame
        gt_objects = gt_loader.get_frame_annotations(sequence_name, sample_index)

        # 5. Determine Authentic Ground-Truth Matches (No proxy leakage)
        pair_matches = match_ground_truth(
            radar_objects,
            camera_objects,
            gt_objects,
            radar_dist_thresh=RADAR_GT_MATCH_DISTANCE,
            camera_iou_thresh=CAMERA_GT_MATCH_IOU,
        )

        # 6. Build Candidate Pair Feature Rows
        for radar_index, radar_object in enumerate(radar_objects):
            u, v = pixels[radar_index]
            if not (np.isfinite(u) and np.isfinite(v)):
                continue

            for camera_index, camera_object in enumerate(camera_objects):
                x1, y1, x2, y2 = map(float, camera_object["bbox"])
                bbox_center_u = (x1 + x2) / 2.0
                bbox_center_v = (y1 + y2) / 2.0
                bbox_width = max(x2 - x1, 1e-3)
                bbox_height = max(y2 - y1, 1e-3)
                bbox_aspect_ratio = bbox_width / bbox_height

                # Scale-invariant relative alignment features
                norm_offset_u = (u - bbox_center_u) / bbox_width
                norm_offset_v = (v - bbox_center_v) / bbox_height
                norm_radial_dist = float(np.sqrt(norm_offset_u ** 2 + norm_offset_v ** 2))

                # Label derived strictly from independent GT matching
                is_match = pair_matches.get((radar_index, camera_index), 0)

                rows.append({
                    "sample_index": sample_index,
                    "sequence_name": sequence_name,
                    "radar_index": radar_index,
                    "camera_index": camera_index,

                    # Radar Kinematics & Signal
                    "range_m": radar_object["range_m"],
                    "velocity_mps": radar_object["velocity_mps"],
                    "azimuth_deg": radar_object["azimuth_deg"],
                    "elevation_deg": radar_object["elevation_deg"],
                    "radar_power": radar_object["power"],

                    # Radar 3D Physical Position
                    "radar_x": radar_object["x_m"],
                    "radar_y": radar_object["y_m"],
                    "radar_z": radar_object["z_m"],

                    # Projected Geometry & Camera BBox Geometry
                    "projected_u": float(u),
                    "projected_v": float(v),
                    "bbox_center_u": bbox_center_u,
                    "bbox_center_v": bbox_center_v,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_aspect_ratio": bbox_aspect_ratio,
                    "camera_confidence": camera_object["confidence"],

                    # Relative Alignment (Scale-Invariant)
                    "norm_offset_u": norm_offset_u,
                    "norm_offset_v": norm_offset_v,
                    "norm_radial_dist": norm_radial_dist,

                    # Metadata & Target
                    "class_name": camera_object["class_name"],
                    "is_match": is_match,
                })

    if not rows:
        raise RuntimeError(f"No candidate pairs were generated for sequence {sequence_name}.")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    num_matches = int((df["is_match"] == 1).sum())
    num_nomatch = int((df["is_match"] == 0).sum())

    print(f"\n--- DATASET GENERATED FOR {sequence_name} ---")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"MATCH (is_match = 1): {num_matches} ({num_matches / len(df) * 100:.2f}%)")
    print(f"NO-MATCH (is_match = 0): {num_nomatch} ({num_nomatch / len(df) * 100:.2f}%)")
    print(f"Saved to: {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate leakage-free radar-camera fusion dataset.")
    parser.add_argument("--recording", type=str, default=None, help="Recording directory path.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path.")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step.")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process.")
    args = parser.parse_args()

    rec_dir = args.recording if args.recording else get_default_recording_dir()
    out_csv = args.output if args.output else DEFAULT_TRAINING_DATA

    generate_dataset_for_sequence(
        recording_dir=rec_dir,
        output_csv=out_csv,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()