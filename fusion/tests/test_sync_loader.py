from pathlib import Path

import numpy as np

from fusion.src.sync_loader import (
    SynchronizedFusionLoader,
)


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


def test_synchronized_radar_camera():

    loader = SynchronizedFusionLoader(
        DATASET,
        DBREADER,
    )

    data = loader.load(0)

    camera = data["camera"]

    print("\n===== SYNCHRONIZED RADAR + CAMERA =====")

    print(
        "Camera shape:",
        camera.shape,
    )

    print(
        "Camera dtype:",
        camera.dtype,
    )

    print(
        "Radar timestamp:",
        data["radar_timestamp"],
    )

    print(
        "Camera timestamp:",
        data["camera_timestamp"],
    )

    print(
        "Radar sample:",
        data["radar_sample"],
    )

    for channel in (
        "radar_ch0",
        "radar_ch1",
        "radar_ch2",
        "radar_ch3",
    ):
        print(
            channel,
            "shape:",
            data[channel].shape,
        )

    assert camera is not None

    assert isinstance(
        camera,
        np.ndarray,
    )

    assert data["radar_ch0"].dtype == np.int16
    assert data["radar_ch1"].dtype == np.int16
    assert data["radar_ch2"].dtype == np.int16
    assert data["radar_ch3"].dtype == np.int16