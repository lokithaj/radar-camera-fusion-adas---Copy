from pathlib import Path

import cv2
import numpy as np

from fusion.src.sync_loader import (
    SynchronizedFusionLoader,
)

from fusion.src.radar_camera_projector import (
    RadarCameraProjector,
)

from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN
from radar.src.radar_objects import RadarObjectExtractor

from camera.src.camera_detector import CameraDetector


from config import (
    get_default_recording_dir,
    get_dbreader_dir,
    get_radar_calibration_path,
    get_camera_calibration_path,
)


DATASET = get_default_recording_dir()
DBREADER = get_dbreader_dir(DATASET)
RADAR_CALIBRATION = get_radar_calibration_path(DATASET)
CAMERA_CALIBRATION = get_camera_calibration_path(DATASET)


def main():

    # ---------------------------------------------------------
    # 1. Load synchronized radar + camera sample
    # ---------------------------------------------------------

    loader = SynchronizedFusionLoader(
        DATASET,
        DBREADER,
    )

    data = loader.load(0)

    camera_frame = data["camera"].copy()

    print("\n===== VISUAL RADAR-CAMERA ASSOCIATION =====")

    print(
        "Camera resolution:",
        camera_frame.shape[1],
        "x",
        camera_frame.shape[0],
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
    # 6. Generate physical radar points
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
    # 7. DBSCAN
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
        "Radar objects:",
        len(radar_objects),
    )

    # ---------------------------------------------------------
    # 9. Camera YOLO
    # ---------------------------------------------------------

    detector = CameraDetector(
        model_name="yolo11n.pt",
        confidence=0.4,
    )

    annotated_frame, camera_objects = (
        detector.detect(camera_frame)
    )

    print(
        "Camera objects:",
        len(camera_objects),
    )

    # ---------------------------------------------------------
    # 10. Project radar objects into camera
    # ---------------------------------------------------------

    projector = RadarCameraProjector(
        CAMERA_CALIBRATION
    )

    if len(radar_objects) > 0:

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

    # ---------------------------------------------------------
    # 11. Draw YOLO boxes
    # ---------------------------------------------------------

    for index, obj in enumerate(camera_objects):

        x1, y1, x2, y2 = map(
            int,
            obj["bbox"],
        )

        label = (
            f"{obj['class_name']} "
            f"{obj['confidence']:.2f}"
        )

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # ---------------------------------------------------------
    # 12. Draw radar projections
    # ---------------------------------------------------------

    for radar_index, pixel in enumerate(pixels):

        if not valid[radar_index]:
            continue

        u = int(round(pixel[0]))
        v = int(round(pixel[1]))

        cv2.circle(
            annotated_frame,
            (u, v),
            12,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            annotated_frame,
            f"R{radar_index}",
            (u + 15, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        obj = radar_objects[
            radar_index
        ]

        print(
            f"Radar object {radar_index}: "
            f"pixel=({u}, {v}), "
            f"range={obj['range_m']:.2f} m, "
            f"azimuth={obj['azimuth_deg']:.2f} deg, "
            f"valid={valid[radar_index]}"
        )

    # ---------------------------------------------------------
    # 13. Display information
    # ---------------------------------------------------------

    cv2.putText(
        annotated_frame,
        "RADAR + CAMERA ASSOCIATION",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        annotated_frame,
        "GREEN = CAMERA   RED = RADAR",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    print(
        "\nGreen boxes = YOLO camera detections"
    )

    print(
        "Red circles = projected radar objects"
    )

    print(
        "\nPress Q to close."
    )

    cv2.imshow(
        "Radar Camera Association",
        annotated_frame,
    )

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()