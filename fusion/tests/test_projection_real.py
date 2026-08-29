import numpy as np
from pathlib import Path

from fusion.src.radar_camera_projector import (
    RadarCameraProjector,
)


CALIBRATION = (
    Path.home()
    / "Desktop"
    / "RADIal"
    / "DBReader"
    / "examples"
    / "camera_calib.npy"
)


def test_real_radar_projection():

    projector = RadarCameraProjector(
        CALIBRATION
    )

    # Real radar object from our previous RADIal run.
    radar_object = np.array(
        [
            [
                11.44,
                -0.02,
                -0.31,
            ]
        ],
        dtype=np.float64,
    )

    pixels, valid = projector.project_valid(
        radar_object,
        image_width=1920,
        image_height=1080,
    )

    print("\n===== RADAR → CAMERA PROJECTION =====")

    print(
        "Radar XYZ:",
        radar_object[0],
    )

    print(
        "Projected pixel:",
        pixels[0],
    )

    print(
        "Inside image:",
        bool(valid[0]),
    )

    assert pixels.shape == (1, 2)

    assert np.isfinite(
        pixels
    ).all()