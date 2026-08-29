from pathlib import Path

import numpy as np

from fusion.src.sync_loader import SynchronizedFusionLoader
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

CALIBRATION = (
    Path.home()
    / "Desktop"
    / "RADIal"
    / "SignalProcessing"
    / "CalibrationTable.npy"
)


def test_fusion_sample():

    # ---------------------------------------------------------
    # 1. Load one synchronized radar + camera sample
    # ---------------------------------------------------------
    loader = SynchronizedFusionLoader(
        DATASET,
        DBREADER,
    )

    data = loader.load(0)

    camera_frame = data["camera"]

    print("\n===== FUSION SAMPLE =====")

    print(
        "Camera shape:",
        camera_frame.shape,
    )

    print(
        "Radar timestamp:",
        data["radar_timestamp"],
    )

    print(
        "Camera timestamp:",
        data["camera_timestamp"],
    )

    # ---------------------------------------------------------
    # 2. Build radar frame
    # ---------------------------------------------------------
    frame_builder = RadarFrameBuilder()

    complex_frame = frame_builder.build_frame(
        data["radar_ch0"],
        data["radar_ch1"],
        data["radar_ch2"],
        data["radar_ch3"],
    )

    # Remove DC offset.
    complex_frame = (
        complex_frame
        - np.mean(
            complex_frame,
            axis=(0, 1),
            keepdims=True,
        )
    )

    # ---------------------------------------------------------
    # 3. Radar FFT
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
        CALIBRATION,
        max_points=100,
        min_range_bin=10,
    )

    radar_points = point_generator.generate(
        rd_spectrum,
        detections,
    )

    # ---------------------------------------------------------
    # 7. DBSCAN radar clustering
    # ---------------------------------------------------------
    clusterer = RadarDBSCAN(
        eps=1.5,
        min_samples=3,
    )

    labels = clusterer.cluster(
        radar_points
    )

    # ---------------------------------------------------------
    # 8. Radar objects
    # ---------------------------------------------------------
    object_extractor = RadarObjectExtractor()

    radar_objects = object_extractor.extract(
        radar_points,
        labels,
    )

    print(
        "\nRadar objects:",
        len(radar_objects),
    )

    for obj in radar_objects:
        print(
            f"Radar Object {obj['cluster_id']}: "
            f"range={obj['range_m']:.2f} m, "
            f"x={obj['x_m']:.2f} m, "
            f"y={obj['y_m']:.2f} m, "
            f"azimuth={obj['azimuth_deg']:.2f} deg, "
            f"points={obj['num_points']}"
        )

    # ---------------------------------------------------------
    # 9. Camera object detection
    # ---------------------------------------------------------
    camera_detector = CameraDetector(
        model_name="yolo11n.pt",
        confidence=0.4,
    )

    camera_annotated, camera_objects = (
        camera_detector.detect(camera_frame)
    )

    print(
        "\nCamera objects:",
        len(camera_objects),
    )

    for i, obj in enumerate(
        camera_objects
    ):
        print(
            f"Camera Object {i}: "
            f"class={obj['class_name']}, "
            f"confidence={obj['confidence']:.2f}, "
            f"bbox={obj['bbox']}"
        )

    # ---------------------------------------------------------
    # 10. Validation
    # ---------------------------------------------------------
    assert camera_frame.shape == (
        1080,
        1920,
        3,
    )

    assert len(radar_objects) >= 0

    assert len(camera_objects) >= 0

    assert radar_points.shape[1] == 10