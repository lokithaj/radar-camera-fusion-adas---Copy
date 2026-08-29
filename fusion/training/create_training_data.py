from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fusion.src.sync_loader import SynchronizedFusionLoader
from fusion.src.radar_camera_projector import RadarCameraProjector

from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN
from radar.src.radar_objects import RadarObjectExtractor

from camera.src.camera_detector import CameraDetector


# ============================================================
# Paths
# ============================================================

DATASET = (
    Path.home()
    / "Desktop"
    / "RADIal_data"
    / "RECORD@2020-11-21_13.44.44"
)

DBREADER = (
    Path.home()
    / "Desktop"
    / "RADIal"
    / "DBReader"
)

RADAR_CALIBRATION = (
    Path.home()
    / "Desktop"
    / "RADIal"
    / "SignalProcessing"
    / "CalibrationTable.npy"
)

CAMERA_CALIBRATION = (
    Path.home()
    / "Desktop"
    / "RADIal"
    / "DBReader"
    / "examples"
    / "camera_calib.npy"
)

OUTPUT_CSV = (
    Path.home()
    / "Desktop"
    / "radar-camera-fusion-adas"
    / "fusion"
    / "training"
    / "fusion_training_data.csv"
)


# ============================================================
# Settings
# ============================================================

FRAME_STEP = 5

MAX_RADAR_POINTS = 20

MIN_RANGE_BIN = 10

YOLO_CONFIDENCE = 0.4

# This is only used to create provisional labels.
LABEL_DISTANCE = 30.0


def process_radar(data):
    """
    Run the existing radar pipeline for one synchronized sample.
    """

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
        RADAR_CALIBRATION,
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

    object_extractor = RadarObjectExtractor()

    radar_objects = object_extractor.extract(
        radar_points,
        labels,
    )

    return radar_objects


def point_to_box_distance(
    u,
    v,
    bbox,
):
    """
    Distance from a projected radar pixel to the camera box.

    Zero means the pixel is inside the box.
    """

    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

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

    return float(
        np.sqrt(
            dx * dx + dy * dy
        )
    )


def main():

    loader = SynchronizedFusionLoader(
        DATASET,
        DBREADER,
    )

    detector = CameraDetector(
        model_name="yolo11n.pt",
        confidence=YOLO_CONFIDENCE,
    )

    projector = RadarCameraProjector(
        CAMERA_CALIBRATION
    )

    rows = []

    print(
        "Creating fusion training data..."
    )

    print(
        "Available samples:",
        len(loader),
    )

    for sample_index in range(
        0,
        len(loader),
        FRAME_STEP,
    ):

        print(
            f"Processing sample {sample_index}..."
        )

        data = loader.load(
            sample_index
        )

        image = data["camera"]

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        try:
            _, camera_objects = detector.detect(
                image
            )
        except Exception as error:
            print(
                "Camera error:",
                error,
            )
            continue

        # ----------------------------------------------------
        # Radar
        # ----------------------------------------------------

        try:
            radar_objects = process_radar(
                data
            )
        except Exception as error:
            print(
                "Radar error:",
                error,
            )
            continue

        if not radar_objects:
            continue

        # ----------------------------------------------------
        # Radar projection
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Create radar-camera pairs
        # ----------------------------------------------------

        for radar_index, radar_object in enumerate(
            radar_objects
        ):

            u, v = pixels[
                radar_index
            ]

            if not (
                np.isfinite(u)
                and np.isfinite(v)
            ):
                continue

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

                bbox_width = (
                    x2 - x1
                )

                bbox_height = (
                    y2 - y1
                )

                pixel_distance = (
                    point_to_box_distance(
                        u,
                        v,
                        camera_object["bbox"],
                    )
                )

                inside_bbox = int(
                    x1 <= u <= x2
                    and y1 <= v <= y2
                )

                # ------------------------------------------------
                # Provisional training label
                #
                # For today's proof-of-concept:
                #   1 = candidate match
                #   0 = candidate non-match
                #
                # A robust evaluation should eventually replace
                # these labels with dataset ground truth.
                # ------------------------------------------------

                is_match = int(
                    pixel_distance <= LABEL_DISTANCE
                )

                rows.append(
                    {
                        "sample_index": sample_index,

                        "radar_index": radar_index,
                        "camera_index": camera_index,

                        "range_m": radar_object[
                            "range_m"
                        ],

                        "velocity_mps": radar_object[
                            "velocity_mps"
                        ],

                        "azimuth_deg": radar_object[
                            "azimuth_deg"
                        ],

                        "elevation_deg": radar_object[
                            "elevation_deg"
                        ],

                        "radar_x": radar_object[
                            "x_m"
                        ],

                        "radar_y": radar_object[
                            "y_m"
                        ],

                        "radar_z": radar_object[
                            "z_m"
                        ],

                        "radar_power": radar_object[
                            "power"
                        ],

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

                        "class_name": camera_object[
                            "class_name"
                        ],

                        "is_match": is_match,
                    }
                )

    if not rows:
        raise RuntimeError(
            "No training examples were generated."
        )

    df = pd.DataFrame(
        rows
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "\n===== TRAINING DATA CREATED ====="
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Columns:",
        len(df.columns),
    )

    print(
        "MATCH rows:",
        int(
            (df["is_match"] == 1).sum()
        ),
    )

    print(
        "NO-MATCH rows:",
        int(
            (df["is_match"] == 0).sum()
        ),
    )

    print(
        "Saved to:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()