from pathlib import Path

import cv2
import numpy as np

from fusion.src.sync_loader import (
    SynchronizedFusionLoader,
)

from fusion.src.radar_camera_projector import (
    RadarCameraProjector,
)

from fusion.src.association import (
    RadarCameraAssociator,
)

from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN
from radar.src.radar_objects import RadarObjectExtractor

from camera.src.camera_detector import CameraDetector


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


def test_real_radar_camera_association():

    # ---------------------------------------------------------
    # 1. Load synchronized radar + camera sample
    # ---------------------------------------------------------

    loader = SynchronizedFusionLoader(
        DATASET,
        DBREADER,
    )

    data = loader.load(0)

    camera_frame = data["camera"]

    # ---------------------------------------------------------
    # 2. Radar frame construction
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 3. FFT
    # ---------------------------------------------------------

    fft_processor = RadarFFTProcessor()

    range_fft = fft_processor.range_fft(
        complex_frame
    )

    rd_spectrum = fft_processor.doppler_fft(
        range_fft
    )

    # ---------------------------------------------------------
    # 4. Range-Doppler power
    # ---------------------------------------------------------

    rd_power = np.sum(
        np.abs(rd_spectrum) ** 2,
        axis=2,
    )

    # ---------------------------------------------------------
    # 5. CFAR
    # ---------------------------------------------------------

    cfar = CACFAR(
        window=(9, 9),
        guard=(3, 3),
        threshold_db=2.0,
    )

    detections = cfar.detect(
        rd_power
    )

    # ---------------------------------------------------------
    # 6. Generate radar points
    # ---------------------------------------------------------

    point_generator = RealRadarPointGenerator(
        RADAR_CALIBRATION,
        max_points=100,
        min_range_bin=10,
    )

    radar_points = point_generator.generate(
        rd_spectrum,
        detections,
    )

    # ---------------------------------------------------------
    # 7. Cluster radar points
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 8. Camera detection
    # ---------------------------------------------------------

    detector = CameraDetector(
        model_name="yolo11n.pt",
        confidence=0.4,
    )

    _, camera_objects = detector.detect(
        camera_frame
    )

    # ---------------------------------------------------------
    # 9. Prepare radar XYZ positions
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 10. Project radar objects into camera image
    # ---------------------------------------------------------

    projector = RadarCameraProjector(
        CAMERA_CALIBRATION
    )

    if len(radar_xyz) > 0:

        pixels = projector.project(
            radar_xyz
        )

        _, valid = projector.project_valid(
            radar_xyz,
            image_width=camera_frame.shape[1],
            image_height=camera_frame.shape[0],
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

    # Only keep radar objects whose projected pixel
    # lies inside the camera image.
    valid_indices = np.where(valid)[0]

    valid_radar_objects = [
        radar_objects[i]
        for i in valid_indices
    ]

    valid_pixels = pixels[
        valid_indices
    ]

    # ---------------------------------------------------------
    # 11. Associate radar with camera
    # ---------------------------------------------------------

    associator = RadarCameraAssociator()

    associations = associator.associate(
        valid_radar_objects,
        valid_pixels,
        camera_objects,
    )

    # ---------------------------------------------------------
    # 12. Display results
    # ---------------------------------------------------------

    print(
        "\n===== REAL RADAR-CAMERA ASSOCIATION ====="
    )

    print(
        "Radar objects:",
        len(radar_objects),
    )

    print(
        "Camera objects:",
        len(camera_objects),
    )

    print(
        "Radar objects inside camera image:",
        len(valid_radar_objects),
    )

    print(
        "Associations:",
        len(associations),
    )

    for association in associations:

        radar_obj = association[
            "radar_object"
        ]

        camera_obj = association[
            "camera_object"
        ]

        u, v = association[
            "pixel"
        ]

        print(
            "\nMATCH"
        )

        print(
            f"Radar object: "
            f"range={radar_obj['range_m']:.2f} m, "
            f"x={radar_obj['x_m']:.2f} m, "
            f"y={radar_obj['y_m']:.2f} m"
        )

        print(
            f"Camera object: "
            f"class={camera_obj['class_name']}, "
            f"confidence={camera_obj['confidence']:.2f}"
        )

        print(
            f"Projected radar pixel: "
            f"({u:.1f}, {v:.1f})"
        )

    # ---------------------------------------------------------
    # 13. Basic validation
    # ---------------------------------------------------------

    assert camera_frame.shape == (
        1080,
        1920,
        3,
    )

    assert len(camera_objects) >= 0

    assert len(radar_objects) >= 0

    assert len(valid_radar_objects) <= len(
        radar_objects
    )

    assert len(associations) <= min(
        len(valid_radar_objects),
        len(camera_objects),
    )