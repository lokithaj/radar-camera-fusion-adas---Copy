import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from fusion.src.sync_loader import SynchronizedFusionLoader
from fusion.src.radar_camera_projector import RadarCameraProjector
from fusion.src.fusion_model import FusionMLModel
from fusion.src.radar_sparsifier import (
    AdaptiveRadarSparsifier,
    CameraGuidedRadarSparsifier,
)

from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN
from radar.src.radar_objects import RadarObjectExtractor

from camera.src.camera_detector import CameraDetector


from config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_YOLO_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RADAR_RETENTION_RATIO,
    get_dbreader_dir,
    get_radar_calibration_path,
    get_camera_calibration_path,
)


DEFAULT_MODEL = DEFAULT_MODEL_PATH
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR


FRAME_STEP = 5
MAX_RADAR_POINTS = 20
MIN_RANGE_BIN = 10
YOLO_CONFIDENCE = 0.4
DISPLAY_DELAY_MS = 100


def process_radar(
    data,
    radar_calibration,
):
    """Process one synchronized radar sample."""

    frame_builder = RadarFrameBuilder()

    complex_frame = frame_builder.build_frame(
        data["radar_ch0"],
        data["radar_ch1"],
        data["radar_ch2"],
        data["radar_ch3"],
    )

    complex_frame = (
        complex_frame
        - np.mean(
            complex_frame,
            axis=(0, 1),
            keepdims=True,
        )
    )

    fft_processor = RadarFFTProcessor()

    range_fft = fft_processor.range_fft(
        complex_frame
    )

    rd_spectrum = fft_processor.doppler_fft(
        range_fft
    )

    rd_power = np.sum(
        np.abs(rd_spectrum) ** 2,
        axis=2,
    )

    cfar = CACFAR(
        window=(9, 9),
        guard=(3, 3),
        threshold_db=2.0,
    )

    detections = cfar.detect(
        rd_power
    )

    point_generator = RealRadarPointGenerator(
        radar_calibration,
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

    labels = clusterer.cluster(
        radar_points
    )

    extractor = RadarObjectExtractor()

    return extractor.extract(
        radar_points,
        labels,
    )


def get_ml_associations(
    radar_objects,
    pixels,
    valid,
    camera_objects,
    fusion_model,
):
    """Use the trained ML model for radar-camera association."""

    associations = []

    for radar_index, pixel in enumerate(pixels):

        if not valid[radar_index]:
            continue

        u, v = pixel

        radar_object = radar_objects[
            radar_index
        ]

        best_camera = None
        best_probability = 0.0

        for camera_index, camera_object in enumerate(
            camera_objects
        ):

            x1, y1, x2, y2 = map(
                float,
                camera_object["bbox"],
            )

            bbox_center_u = (
                x1 + x2
            ) / 2.0

            bbox_center_v = (
                y1 + y2
            ) / 2.0

            bbox_width = x2 - x1
            bbox_height = y2 - y1

            dx = max(
                x1 - u,
                0.0,
                u - x2,
            )

            dy = max(
                y1 - v,
                0.0,
                v - y2,
            )

            pixel_distance = float(
                np.sqrt(
                    dx * dx + dy * dy
                )
            )

            inside_bbox = int(
                x1 <= u <= x2
                and y1 <= v <= y2
            )

            features = {
                "range_m": radar_object["range_m"],
                "velocity_mps": radar_object[
                    "velocity_mps"
                ],
                "azimuth_deg": radar_object[
                    "azimuth_deg"
                ],
                "elevation_deg": radar_object[
                    "elevation_deg"
                ],
                "radar_x": radar_object["x_m"],
                "radar_y": radar_object["y_m"],
                "radar_z": radar_object["z_m"],
                "radar_power": radar_object["power"],
                "projected_u": float(u),
                "projected_v": float(v),
                "bbox_center_u": bbox_center_u,
                "bbox_center_v": bbox_center_v,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "camera_confidence": camera_object[
                    "confidence"
                ],
                "pixel_distance": pixel_distance,
                "inside_bbox": inside_bbox,
            }

            prediction = fusion_model.predict(
                features
            )

            probability = (
                fusion_model.predict_probability(
                    features
                )
            )

            if (
                prediction == 1
                and probability > best_probability
            ):
                best_probability = probability
                best_camera = camera_index

        if best_camera is not None:

            associations.append(
                {
                    "radar_index": radar_index,
                    "camera_index": best_camera,
                    "probability": best_probability,
                }
            )

    return associations


def run(
    recording_path,
    model_path=None,
    dbreader_dir=None,
    radar_calib_path=None,
    camera_calib_path=None,
    output_dir=None,
    retention_ratio=0.5,
    sparsification_mode="radar",
):
    """Run inference on a user-selected recording."""

    recording_path = Path(
        recording_path
    ).resolve()

    if not recording_path.exists():
        raise FileNotFoundError(
            f"Recording not found:\n{recording_path}"
        )

    if not recording_path.is_dir():
        raise NotADirectoryError(
            f"Recording path is not a directory:\n{recording_path}"
        )

    dbreader = get_dbreader_dir(recording_path, dbreader_dir)
    if not dbreader.exists():
        raise FileNotFoundError(
            f"RADIal DBReader directory not found:\n{dbreader}\n"
            "Please specify --dbreader or set the RADIAL_DBREADER_DIR environment variable."
        )

    radar_calibration = get_radar_calibration_path(recording_path, radar_calib_path)
    if not radar_calibration.exists():
        raise FileNotFoundError(
            f"Radar calibration file not found:\n{radar_calibration}\n"
            "Please specify --radar-calib or set the RADIAL_RADAR_CALIB environment variable."
        )

    camera_calibration = get_camera_calibration_path(recording_path, camera_calib_path)
    if not camera_calibration.exists():
        raise FileNotFoundError(
            f"Camera calibration file not found:\n{camera_calibration}\n"
            "Please specify --camera-calib or set the RADIAL_CAMERA_CALIB environment variable."
        )

    model_file = Path(model_path).resolve() if model_path else DEFAULT_MODEL_PATH
    if not model_file.exists():
        raise FileNotFoundError(
            f"Fusion ML model not found:\n{model_file}\n"
            "Please train the model or specify --model."
        )

    print(
        "\n===== RADAR-CAMERA ML INFERENCE ====="
    )

    print(
        "Recording:",
        recording_path,
    )

    loader = SynchronizedFusionLoader(
        recording_path,
        dbreader,
    )

    print(
        "Available samples:",
        len(loader),
    )

    camera_detector = CameraDetector(
        model_name=str(DEFAULT_YOLO_MODEL) if DEFAULT_YOLO_MODEL.exists() else "yolo11n.pt",
        confidence=YOLO_CONFIDENCE,
    )

    projector = RadarCameraProjector(
        camera_calibration
    )

    fusion_model = FusionMLModel(
        model_file
    )

    if sparsification_mode == "radar":
        sparsifier = AdaptiveRadarSparsifier(
            retention_ratio=0.5 if retention_ratio is None else retention_ratio
        )
    elif sparsification_mode == "camera":
        sparsifier = CameraGuidedRadarSparsifier(
            retention_ratio=0.5 if retention_ratio is None else retention_ratio
        )
    elif sparsification_mode == "none":
        sparsifier = None
    else:
        raise ValueError(
            f"Unknown sparsification_mode: '{sparsification_mode}'. Supported: 'none', 'radar', 'camera'"
        )

    print(
        "Trained fusion model loaded."
    )

    for index in range(
        0,
        len(loader),
        FRAME_STEP,
    ):

        print(
            f"Processing sample {index}..."
        )

        data = loader.load(
            index
        )

        image = data["camera"].copy()

        # Camera
        _, camera_objects = camera_detector.detect(
            image
        )

        # Radar
        try:
            radar_objects = process_radar(
                data,
                radar_calibration,
            )

            original_radar_count = len(radar_objects)

            if sparsification_mode == "radar":
                radar_objects = sparsifier.select(radar_objects)
            elif sparsification_mode == "camera":
                if radar_objects:
                    candidate_xyz = np.array(
                        [
                            [obj["x_m"], obj["y_m"], obj["z_m"]]
                            for obj in radar_objects
                        ],
                        dtype=np.float64,
                    )
                    cand_pixels, cand_valid = projector.project_valid(
                        candidate_xyz,
                        image.shape[1],
                        image.shape[0],
                    )
                else:
                    cand_pixels = np.empty((0, 2), dtype=np.float64)
                    cand_valid = np.empty((0,), dtype=bool)

                radar_objects = sparsifier.select(
                    radar_objects,
                    projected_pixels=cand_pixels,
                    valid_in_fov=cand_valid,
                    camera_objects=camera_objects,
                )
            elif sparsification_mode == "none":
                pass

            print(
                f"Radar objects: {original_radar_count} -> {len(radar_objects)}"
            )

        except Exception as error:

            print(
                "Radar processing error:",
                error,
            )

            radar_objects = []

        # Projection
        if radar_objects:

            radar_xyz = np.array(
                [
                    [
                        obj["x_m"],
                        obj["y_m"],
                        obj["z_m"],
                    ]
                    for obj in radar_objects
                ],
                dtype=np.float64,
            )

            pixels = projector.project(
                radar_xyz
            )

            _, valid = projector.project_valid(
                radar_xyz,
                image.shape[1],
                image.shape[0],
            )

        else:

            pixels = np.empty(
                (0, 2),
                dtype=np.float64,
            )

            valid = np.empty(
                (0,),
                dtype=bool,
            )

        # Draw camera detections
        for camera_index, obj in enumerate(
            camera_objects
        ):

            x1, y1, x2, y2 = map(
                int,
                obj["bbox"],
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                image,
                f"{camera_index}: "
                f"{obj['class_name']} "
                f"{obj['confidence']:.2f}",
                (
                    x1,
                    max(y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Draw radar
        for radar_index, pixel in enumerate(
            pixels
        ):

            if not valid[radar_index]:
                continue

            u = int(
                round(pixel[0])
            )

            v = int(
                round(pixel[1])
            )

            radar_object = radar_objects[
                radar_index
            ]

            cv2.circle(
                image,
                (u, v),
                12,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                image,
                (
                    f"R{radar_index} "
                    f"{radar_object['range_m']:.1f}m "
                    f"{radar_object['velocity_mps']:+.1f}m/s"
                ),
                (
                    u + 15,
                    v,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        # ML association
        associations = get_ml_associations(
            radar_objects,
            pixels,
            valid,
            camera_objects,
            fusion_model,
        )

        # Fused information
        text_y = 135

        for association in associations:

            radar_index = association[
                "radar_index"
            ]

            camera_index = association[
                "camera_index"
            ]

            probability = association[
                "probability"
            ]

            radar_object = radar_objects[
                radar_index
            ]

            camera_object = camera_objects[
                camera_index
            ]

            text = (
                f"FUSED R{radar_index} -> "
                f"{camera_object['class_name']} "
                f"{camera_object['confidence']:.2f} | "
                f"Range: "
                f"{radar_object['range_m']:.1f} m | "
                f"Velocity: "
                f"{radar_object['velocity_mps']:+.1f} m/s | "
                f"ML: "
                f"{probability * 100:.1f}%"
            )

            cv2.putText(
                image,
                text,
                (30, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Cyan ring = fused object
            u = int(
                round(
                    pixels[radar_index][0]
                )
            )

            v = int(
                round(
                    pixels[radar_index][1]
                )
            )

            cv2.circle(
                image,
                (u, v),
                18,
                (255, 255, 0),
                2,
            )

            text_y += 35

        # Header
        cv2.putText(
            image,
            "RADAR + CAMERA ML FUSION",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            (
                f"Sample: {index} | "
                f"Radar: {len(radar_objects)} | "
                f"Camera: {len(camera_objects)} | "
                f"Fused: {len(associations)}"
            ),
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            "GREEN=CAMERA | RED=RADAR | CYAN=FUSED",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        display = cv2.resize(
            image,
            (1280, 720),
        )
        print("DISPLAYING FRAME")
        cv2.imshow(
            "RADIal Radar-Camera ML Fusion",
            display,
        )

        key = (
            cv2.waitKey(
                DISPLAY_DELAY_MS
            )
            & 0xFF
        )

        if key == ord("q"):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(
        "\nInference finished."
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run radar-camera fusion on "
            "a synchronized RADIal recording."
        )
    )

    parser.add_argument(
        "--recording",
        required=True,
        help=(
            "Path to the synchronized "
            "RADIal recording folder."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=f"Path to trained fusion ML model (default: {DEFAULT_MODEL_PATH}).",
    )

    parser.add_argument(
        "--dbreader",
        default=None,
        help="Path to RADIal DBReader directory (auto-discovered if omitted).",
    )

    parser.add_argument(
        "--radar-calib",
        default=None,
        help="Path to CalibrationTable.npy (auto-discovered if omitted).",
    )

    parser.add_argument(
        "--camera-calib",
        default=None,
        help="Path to camera_calib.npy (auto-discovered if omitted).",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )

    parser.add_argument(
        "--retention-ratio",
        type=float,
        default=DEFAULT_RADAR_RETENTION_RATIO,
        choices=[1.0, 0.75, 0.50, 0.25, 0.10],
        help="Adaptive radar sparsification retention ratio (default: 0.5). Supported: 1.0, 0.75, 0.50, 0.25, 0.10",
    )

    parser.add_argument(
        "--sparsification-mode",
        type=str,
        default="radar",
        choices=["none", "radar", "camera"],
        help="Sparsification mode to use: 'none' (full baseline), 'radar' (Stage 5 radar-only), 'camera' (Stage 5b camera-guided). Default: radar.",
    )

    args = parser.parse_args()

    run(
        recording_path=args.recording,
        model_path=args.model,
        dbreader_dir=args.dbreader,
        radar_calib_path=args.radar_calib,
        camera_calib_path=args.camera_calib,
        output_dir=args.output_dir,
        retention_ratio=args.retention_ratio,
        sparsification_mode=args.sparsification_mode,
    )


if __name__ == "__main__":
    main()