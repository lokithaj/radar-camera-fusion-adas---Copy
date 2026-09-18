from pathlib import Path

import cv2
import numpy as np

from fusion.src.radar_camera_projector import (
    RadarCameraProjector,
)


from config import (
    get_camera_calibration_path,
    get_default_recording_dir,
    find_video_file,
)


CALIBRATION = get_camera_calibration_path()
VIDEO_PATH = find_video_file(get_default_recording_dir())


def main():

    projector = RadarCameraProjector(
        CALIBRATION
    )

    # Radar objects obtained from our
    # previous real RADIal test.
    radar_objects = np.array(
        [
            [11.44, -0.02, -0.31],
            [8.79, -3.29, -0.11],
            [9.34, -10.52, -0.74],
        ],
        dtype=np.float64,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open RADIal camera video."
        )

    ret, frame = cap.read()

    cap.release()

    if not ret:
        raise RuntimeError(
            "Could not read the first video frame."
        )

    height, width = frame.shape[:2]

    pixels, valid = projector.project_valid(
        radar_objects,
        image_width=width,
        image_height=height,
    )

    print("\n===== RADAR → CAMERA VISUALIZATION =====")

    for i, (pixel, is_valid) in enumerate(
        zip(pixels, valid)
    ):

        print(
            f"Radar object {i}: "
            f"pixel=({pixel[0]:.1f}, {pixel[1]:.1f}), "
            f"inside={bool(is_valid)}"
        )

        if is_valid:
            u = int(round(pixel[0]))
            v = int(round(pixel[1]))

            cv2.circle(
                frame,
                (u, v),
                10,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                frame,
                f"Radar {i}",
                (u + 12, v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    cv2.putText(
        frame,
        "RADAR POINTS PROJECTED ON CAMERA",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        "Radar Camera Projection",
        frame,
    )

    print("\nPress Q to close.")

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()