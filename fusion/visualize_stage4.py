"""
Stage 4 Visualization Tool: Multi-Object Tracking, TTC, and ADAS Collision Warning.

Integrates:
- Synchronized RADIal camera & radar streams
- Stage 3 Ground-Truth ML Sensor Fusion
- Multi-Object Tracking (stable IDs across frames)
- TTC calculation & ADAS Decision Banner (SAFE / CAUTION / WARNING / CRITICAL)
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from config import (
    DEFAULT_GROUNDTRUTH_MODEL_PATH,
    DEFAULT_YOLO_MODEL,
    DEFAULT_OUTPUT_DIR,
    get_default_recording_dir,
    get_dbreader_dir,
    get_radar_calibration_path,
    get_camera_calibration_path,
)
from fusion.src.sync_loader import SynchronizedFusionLoader
from fusion.src.radar_camera_projector import RadarCameraProjector
from fusion.src.stage4_pipeline import Stage4ADASPipeline
from fusion.src.adas_decision import ThreatLevel
from camera.src.camera_detector import CameraDetector

from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN
from radar.src.radar_objects import RadarObjectExtractor


def process_radar_sample(data, radar_calibration):
    """Run existing, unmodified radar DSP pipeline."""
    frame_builder = RadarFrameBuilder()
    complex_frame = frame_builder.build_frame(
        data["radar_ch0"],
        data["radar_ch1"],
        data["radar_ch2"],
        data["radar_ch3"],
    )
    complex_frame = complex_frame - np.mean(complex_frame, axis=(0, 1), keepdims=True)

    fft_processor = RadarFFTProcessor()
    range_fft = fft_processor.range_fft(complex_frame)
    rd_spectrum = fft_processor.doppler_fft(range_fft)
    rd_power = np.sum(np.abs(rd_spectrum) ** 2, axis=2)

    cfar = CACFAR(window=(9, 9), guard=(3, 3), threshold_db=2.0)
    detections = cfar.detect(rd_power)

    point_generator = RealRadarPointGenerator(
        radar_calibration,
        max_points=20,
        min_range_bin=10,
    )
    radar_points = point_generator.generate(rd_spectrum, detections)

    clusterer = RadarDBSCAN(eps=1.5, min_samples=3)
    labels = clusterer.cluster(radar_points)

    extractor = RadarObjectExtractor()
    return extractor.extract(radar_points, labels)


def draw_adas_banner(image, adas_alert):
    """Render the prominent ADAS threat warning banner."""
    h, w, _ = image.shape
    level = adas_alert.system_threat_level

    # Color palette (BGR)
    color_map = {
        ThreatLevel.SAFE: (40, 160, 40),        # Forest Green
        ThreatLevel.CAUTION: (0, 190, 240),     # Bright Amber/Yellow
        ThreatLevel.WARNING: (20, 100, 255),    # Vibrant Orange
        ThreatLevel.CRITICAL: (30, 30, 230),    # Strong Red
    }
    banner_color = color_map.get(level, (50, 50, 50))

    # Banner dimensions
    banner_height = 55
    cv2.rectangle(image, (0, 0), (w, banner_height), banner_color, -1)

    # Text formatting
    title_text = f"ADAS: [{level.value}] - {adas_alert.recommended_action}"
    cv2.putText(
        image,
        title_text,
        (25, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_tracks(image, tracks, adas_alert):
    """Draw persistent track overlays and kinematics HUD."""
    # Build a lookup for quick assessment access
    assess_by_id = {
        a.track_id: a for a in adas_alert.track_assessments
    }

    threat_colors = {
        ThreatLevel.SAFE: (0, 255, 0),        # Green
        ThreatLevel.CAUTION: (0, 220, 255),   # Yellow
        ThreatLevel.WARNING: (0, 140, 255),   # Orange
        ThreatLevel.CRITICAL: (0, 0, 255),    # Red
    }

    for track in tracks:
        assess = assess_by_id.get(track.track_id)
        threat = assess.threat_level if assess else ThreatLevel.SAFE
        color = threat_colors.get(threat, (0, 255, 0))

        # 1. Bounding box if available
        bbox = track.bbox
        if bbox and len(bbox) == 4 and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Track ID tag
            tag = f"TRK #{track.track_id} [{track.class_name}]"
            cv2.putText(
                image,
                tag,
                (x1, max(y1 - 10, 70)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

            # Kinematics tag
            ttc_str = f"{assess.ttc_s:.1f}s" if (assess and np.isfinite(assess.ttc_s)) else "N/A"
            vc_val = assess.closing_speed_mps if assess else -track.vx
            kine_tag = f"X:{track.x:.1f}m | Y:{track.y:+.1f}m | Vc:{vc_val:+.1f}m/s | TTC:{ttc_str}"

            cv2.putText(
                image,
                kine_tag,
                (x1, min(y2 + 20, image.shape[0] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 2. Projected radar point if available
        if track.projected_pixel:
            u, v = map(int, track.projected_pixel)
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                cv2.circle(image, (u, v), 8, (255, 255, 0), -1)  # Cyan circle
                cv2.circle(image, (u, v), 14, color, 2)


def run_stage4_evaluation(
    recording_path=None,
    model_path=None,
    max_frames=None,
    frame_step=5,
    frame_interval_s=0.2,
    headless=False,
    output_dir=None,
):
    """
    Run full Stage 4 evaluation on a recording sequence.
    """
    rec_path = Path(recording_path) if recording_path else get_default_recording_dir()
    dbreader = get_dbreader_dir(rec_path)
    radar_calib = get_radar_calibration_path(rec_path)
    cam_calib = get_camera_calibration_path(rec_path)

    loader = SynchronizedFusionLoader(rec_path, dbreader)
    projector = RadarCameraProjector(cam_calib)
    camera_detector = CameraDetector(
        model_name=str(DEFAULT_YOLO_MODEL) if DEFAULT_YOLO_MODEL.exists() else "yolo11n.pt",
        confidence=0.4,
    )

    pipeline = Stage4ADASPipeline(fusion_model_path=model_path, default_dt=frame_interval_s)

    num_samples = len(loader)
    end_idx = min(num_samples, max_frames * frame_step) if max_frames else num_samples

    print(f"\n=======================================================")
    print(f"STAGE 4 ADAS EVALUATION")
    print(f"Recording: {rec_path.name}")
    print(f"Total Available Samples: {num_samples}")
    print(f"Frame Step: {frame_step}")
    print(f"Base Frame Interval: {frame_interval_s:.3f}s (5 FPS nominal)")
    print(f"Headless Mode: {headless}")
    print(f"=======================================================\n")

    frame_count = 0
    prev_timestamp = None
    threat_distribution = {
        ThreatLevel.SAFE: 0,
        ThreatLevel.CAUTION: 0,
        ThreatLevel.WARNING: 0,
        ThreatLevel.CRITICAL: 0,
    }

    for idx in range(0, end_idx, frame_step):
        frame_count += 1
        data = loader.load(idx)
        image = data["camera"].copy()

        # Camera YOLO
        _, camera_objects = camera_detector.detect(image)

        # Radar DSP
        try:
            radar_objects = process_radar_sample(data, radar_calib)
        except Exception as e:
            radar_objects = []

        # Project radar to image
        if radar_objects:
            radar_xyz = np.array(
                [[o["x_m"], o["y_m"], o["z_m"]] for o in radar_objects],
                dtype=np.float64,
            )
            pixels = projector.project(radar_xyz)
            _, valid = projector.project_valid(radar_xyz, image.shape[1], image.shape[0])
        else:
            pixels = np.empty((0, 2), dtype=np.float64)
            valid = np.empty((0,), dtype=bool)

        # Stage 4 Process Frame: compute dt from real sensor timestamps when available
        curr_ts = data.get("radar_timestamp", data.get("camera_timestamp"))
        if prev_timestamp is not None and curr_ts is not None and curr_ts > prev_timestamp:
            measured_dt = (curr_ts - prev_timestamp) / 1e6
            # Sanity check: valid positive delta
            if 0.02 <= measured_dt <= 10.0:
                dt = float(measured_dt)
            else:
                dt = float(frame_step * frame_interval_s)
        else:
            dt = float(frame_step * frame_interval_s)
        prev_timestamp = curr_ts

        result = pipeline.process_frame(
            radar_objects,
            pixels,
            valid,
            camera_objects,
            dt=dt,
        )

        alert = result["adas_alert"]
        threat_distribution[alert.system_threat_level] += 1

        # Logging summary
        trk_count = len(result["tracks"])
        conf_count = len(result["confirmed_tracks"])
        fused_count = len(result["fused_objects"])

        primary_str = f"Trk #{alert.primary_hazard_track_id}" if alert.primary_hazard_track_id else "None"
        print(
            f"Frame {idx:4d} (dt={dt:.2f}s) | Radar: {len(radar_objects):2d} | Cam: {len(camera_objects):2d} | "
            f"Fused: {fused_count:2d} | Tracks: {trk_count:2d} (Conf: {conf_count:2d}) | "
            f"ADAS: {alert.system_threat_level.value:8s} | Hazard: {primary_str}"
        )

        if not headless:
            # Overlays
            draw_tracks(image, result["tracks"], alert)
            draw_adas_banner(image, alert)

            # Footer status
            footer = (
                f"Sample: {idx} | dt: {dt:.2f}s | Tracks Active: {trk_count} (Confirmed: {conf_count}) | "
                f"Fused Pairs: {fused_count}"
            )
            cv2.putText(
                image,
                footer,
                (25, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            display = cv2.resize(image, (1280, 720))
            cv2.imshow("RADIal Stage 4 ADAS", display)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

    if not headless:
        cv2.destroyAllWindows()

    print("\n--- STAGE 4 SUMMARY ---")
    print(f"Processed Frames: {frame_count}")
    print("Threat Distribution across evaluated frames:")
    for level, count in threat_distribution.items():
        pct = (count / max(frame_count, 1)) * 100.0
        print(f"  {level.value:8s}: {count:3d} frames ({pct:5.1f}%)")

    return {
        "frames_processed": frame_count,
        "threat_distribution": threat_distribution,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Stage 4 Tracking & ADAS Evaluation.")
    parser.add_argument("--recording", default=None, help="Path to RADIal recording folder.")
    parser.add_argument("--model", default=None, help="Path to Stage 3 fusion model.")
    parser.add_argument("--max-frames", type=int, default=20, help="Max frames to process.")
    parser.add_argument("--frame-step", type=int, default=5, help="Step between frames.")
    parser.add_argument("--frame-interval", type=float, default=0.2, help="Base frame interval in seconds (default: 0.2s for 5 FPS RADIal).")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")

    args = parser.parse_args()
    run_stage4_evaluation(
        recording_path=args.recording,
        model_path=args.model,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        frame_interval_s=args.frame_interval,
        headless=args.headless,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

