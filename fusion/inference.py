import argparse
from pathlib import Path

import cv2
import numpy as np

from fusion.src.sync_loader import SynchronizedFusionLoader
from fusion.src.radar_camera_projector import RadarCameraProjector
from fusion.src.fusion_model import FusionMLModel

from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN
from radar.src.radar_objects import RadarObjectExtractor

from camera.src.camera_detector import CameraDetector


DEFAULT_DBREADER = (
    Path.home()
    / "Desktop"
    / "RADIal"
    / "DBReader"
)

DEFAULT_RADAR_CALIBRATION = (
    Path.home()
    / "Desktop"
    / "RADIal"
    / "SignalProcessing"
    / "CalibrationTable.npy"
)

DEFAULT_CAMERA_CALIBRATION = (
    Path.home()
    / "Desktop"
    / "RADIal"
    / "DBReader"
    / "examples"
    / "camera_calib.npy"
)

DEFAULT_MODEL = (
    Path.home()
    / "Desktop"
    / "radar-camera-fusion-adas"
    / "fusion"
    / "models"
    / "radar_camera_fusion.pkl"
)


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


def run(recording_path):
    """Run inference on a user-selected recording."""

    recording_path = Path(
        recording_path
    )

    if not recording_path.exists():
        raise FileNotFoundError(
            f"Recording not found:\n{recording_path}"
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
        DEFAULT_DBREADER,
    )

    print(
        "Available samples:",
        len(loader),
    )

    camera_detector = CameraDetector(
        model_name="yolo11n.pt",
        confidence=YOLO_CONFIDENCE,
    )

    projector = RadarCameraProjector(
        DEFAULT_CAMERA_CALIBRATION
    )

    fusion_model = FusionMLModel(
        DEFAULT_MODEL
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
                DEFAULT_RADAR_CALIBRATION,
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

    args = parser.parse_args()

    run(
        args.recording
    )


if __name__ == "__main__":
    main()