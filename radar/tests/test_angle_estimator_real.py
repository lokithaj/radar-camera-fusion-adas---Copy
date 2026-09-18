from pathlib import Path

import numpy as np

from radar.src.radial_loader import RADIalLoader
from radar.src.radar_pipeline import RadarPipeline
from radar.src.mimo_reconstructor import MIMOReconstructor
from radar.src.angle_estimator import RadarAngleEstimator


import pytest
from config import (
    get_default_recording_dir,
    get_dbreader_dir,
    get_radar_calibration_path,
)

DATASET = get_default_recording_dir()
DBREADER = get_dbreader_dir(DATASET)
CALIBRATION = get_radar_calibration_path(DATASET)


@pytest.mark.skipif(
    not (DATASET.exists() and DBREADER.exists() and CALIBRATION.exists()),
    reason="RADIal dataset, DBReader, or calibration not found",
)
def test_real_angle_estimation():

    # ---------------------------------------------------------
    # 1. Load one real RADIal radar frame
    # ---------------------------------------------------------
    loader = RADIalLoader(
        DATASET,
        dbreader_dir=DBREADER,
    )

    frame = loader.load_frame(0)

    # ---------------------------------------------------------
    # 2. Generate the real Range-Doppler spectrum
    # ---------------------------------------------------------
    pipeline = RadarPipeline()

    result = pipeline.process(
        frame["radar_ch0"],
        frame["radar_ch1"],
        frame["radar_ch2"],
        frame["radar_ch3"],
    )

    rd_spectrum = result["rd_spectrum"]
    detections = result["detections"]

    # ---------------------------------------------------------
    # 3. Find one usable CFAR detection
    # ---------------------------------------------------------
    detection_indices = np.argwhere(detections)

    valid = detection_indices[
        detection_indices[:, 0] > 10
    ]

    if len(valid) == 0:
        raise RuntimeError(
            "No suitable CFAR detection was found."
        )

    range_bin, doppler_bin = valid[0]

    reduced_doppler_bin = int(
        doppler_bin % 16
    )

    # ---------------------------------------------------------
    # 4. Reconstruct the 192-channel MIMO spectrum
    # ---------------------------------------------------------
    reconstructor = MIMOReconstructor()

    mimo_spectrum = reconstructor.reconstruct(
        rd_spectrum,
        int(range_bin),
        reduced_doppler_bin,
    )

    # ---------------------------------------------------------
    # 5. Estimate azimuth and elevation
    # ---------------------------------------------------------
    estimator = RadarAngleEstimator(
        CALIBRATION
    )

    azimuth_deg, elevation_deg = (
        estimator.estimate(mimo_spectrum)
    )

    print("\n===== REAL ANGLE ESTIMATION =====")
    print(
        "Range bin:",
        int(range_bin),
    )

    print(
        "Doppler bin:",
        int(doppler_bin),
    )

    print(
        "MIMO shape:",
        mimo_spectrum.shape,
    )

    print(
        "Azimuth:",
        azimuth_deg,
        "degrees",
    )

    print(
        "Elevation:",
        elevation_deg,
        "degrees",
    )

    # ---------------------------------------------------------
    # 6. Basic validation
    # ---------------------------------------------------------
    assert mimo_spectrum.shape == (192,)

    assert np.isfinite(azimuth_deg)
    assert np.isfinite(elevation_deg)

    assert -75.0 <= azimuth_deg <= 75.0
    assert -4.0 <= elevation_deg <= 6.0