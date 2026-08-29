import cv2
import numpy as np

from camera.src.camera_detector import CameraDetector
from radar.src.radial_loader import RADIalLoader
from radar.src.radar_pipeline import RadarPipeline


DATASET = (
    r"C:\Users\lokit\Desktop\RADIal_data"
    r"\RECORD@2020-11-21_13.44.44"
)

DBREADER = (
    r"C:\Users\lokit\Desktop\RADIal"
    r"\DBReader"
)

VIDEO_PATH = (
    r"C:\Users\lokit\Desktop\RADIal_data"
    r"\RECORD@2020-11-21_13.44.44"
    r"\RECORD@2020-11-21_13.44.44_preview.avi"
)


def normalize_radar(rd_power):
    """
    Convert radar power into an image for visualization.
    """

    power_db = 10 * np.log10(
        rd_power + np.finfo(float).eps
    )

    power_db -= power_db.min()

    power_db /= (
        power_db.max()
        + np.finfo(float).eps
    )

    image = (power_db * 255).astype(np.uint8)

    image = cv2.resize(
        image,
        (640, 480),
    )

    image = cv2.applyColorMap(
        image,
        cv2.COLORMAP_JET,
    )

    return image


def main():

    print("Loading radar data...")

    loader = RADIalLoader(
        DATASET,
        dbreader_dir=DBREADER,
    )

    pipeline = RadarPipeline()

    print("Loading first radar frame...")

    radar_frame = loader.load_frame(0)

    radar_result = pipeline.process(
        radar_frame["radar_ch0"],
        radar_frame["radar_ch1"],
        radar_frame["radar_ch2"],
        radar_frame["radar_ch3"],
    )

    radar_image = normalize_radar(
        radar_result["rd_power"]
    )

    radar_points = pipeline.detection_points(
        radar_result["detections"]
    )

    print(
        "Radar detection cells:",
        len(radar_points),
    )

    print("Starting camera...")

    detector = CameraDetector()

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open RADIal camera video."
        )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        camera_image, detections = (
            detector.detect(frame)
        )

        camera_image = cv2.resize(
            camera_image,
            (640, 480),
        )

        radar_display = radar_image.copy()

        cv2.putText(
            radar_display,
            "RADAR - Range Doppler",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            camera_image,
            "CAMERA - YOLO",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        combined = np.hstack(
            [
                camera_image,
                radar_display,
            ]
        )

        cv2.imshow(
            "Radar + Camera Perception",
            combined,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Demo finished.")


if __name__ == "__main__":
    main()