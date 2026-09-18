import numpy as np
import pytest
from config import get_default_recording_dir, get_dbreader_dir
from fusion.src.sync_loader import (
    SynchronizedFusionLoader,
)

DATASET = get_default_recording_dir()
DBREADER = get_dbreader_dir(DATASET)


@pytest.mark.skipif(
    not (DATASET.exists() and DBREADER.exists()),
    reason="RADIal dataset or DBReader not found",
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