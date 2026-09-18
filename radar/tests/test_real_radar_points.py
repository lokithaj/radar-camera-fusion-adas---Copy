from pathlib import Path

import numpy as np

from radar.src.radial_loader import RADIalLoader
from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator


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
def test_real_radar_points():

    # ---------------------------------------------------------
    # 1. Load real RADIal radar data
    # ---------------------------------------------------------
    loader = RADIalLoader(
        DATASET,
        dbreader_dir=DBREADER,
    )

    frame = loader.load_frame(0)

    # ---------------------------------------------------------
    # 2. Build the 16-RX complex radar frame
    # ---------------------------------------------------------
    frame_builder = RadarFrameBuilder()

    complex_frame = frame_builder.build_frame(
        frame["radar_ch0"],
        frame["radar_ch1"],
        frame["radar_ch2"],
        frame["radar_ch3"],
    )

    # ---------------------------------------------------------
    # 3. Remove DC offset
    # ---------------------------------------------------------
    complex_frame = (
        complex_frame
        - np.mean(
            complex_frame,
            axis=(0, 1),
            keepdims=True,
        )
    )

    # ---------------------------------------------------------
    # 4. Range FFT
    # ---------------------------------------------------------
    fft_processor = RadarFFTProcessor()

    range_fft = fft_processor.range_fft(
        complex_frame
    )

    # ---------------------------------------------------------
    # 5. Doppler FFT
    # ---------------------------------------------------------
    rd_spectrum = fft_processor.doppler_fft(
        range_fft
    )

    # ---------------------------------------------------------
    # 6. Generate Range-Doppler power map
    # ---------------------------------------------------------
    rd_power = np.sum(
        np.abs(rd_spectrum) ** 2,
        axis=2,
    )

    # ---------------------------------------------------------
    # 7. CFAR
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
    # 8. Generate real RADIal radar points
    # ---------------------------------------------------------
    generator = RealRadarPointGenerator(
        CALIBRATION,
        max_points=20,
        min_range_bin=10,
    )

    points = generator.generate(
        rd_spectrum,
        detections,
    )

    # ---------------------------------------------------------
    # 9. Display results
    # ---------------------------------------------------------
    print("\n===== REAL RADIal RADAR POINTS =====")

    print(
        "Complex frame:",
        complex_frame.shape,
    )

    print(
        "RD spectrum:",
        rd_spectrum.shape,
    )

    print(
        "CFAR detections:",
        int(np.count_nonzero(detections)),
    )

    print(
        "Physical radar points:",
        len(points),
    )

    print(
        "\nColumns:"
    )

    print(
        generator.columns()
    )

    print(
        "\nFirst radar points:"
    )

    for point in points[:10]:

        print(
            f"range={point[2]:.2f} m, "
            f"velocity={point[3]:+.2f} m/s, "
            f"azimuth={point[4]:.2f} deg, "
            f"elevation={point[5]:.2f} deg, "
            f"x={point[6]:.2f} m, "
            f"y={point[7]:.2f} m, "
            f"z={point[8]:.2f} m"
        )

    # ---------------------------------------------------------
    # 10. Validate
    # ---------------------------------------------------------
    assert complex_frame.shape == (
        512,
        256,
        16,
    )

    assert rd_spectrum.shape == (
        512,
        256,
        16,
    )

    assert detections.shape == (
        512,
        256,
    )

    assert points.ndim == 2

    assert points.shape[1] == 10

    assert len(points) > 0

    # Physical values must be finite.
    assert np.all(
        np.isfinite(points[:, 2])
    )

    assert np.all(
        np.isfinite(points[:, 4])
    )

    assert np.all(
        np.isfinite(points[:, 5])
    )

    assert np.all(
        np.isfinite(points[:, 6:9])
    )

    # RADIal calibrated angular range.
    assert np.all(
        (points[:, 4] >= -75.0)
        & (points[:, 4] <= 75.0)
    )

    assert np.all(
        (points[:, 5] >= -4.0)
        & (points[:, 5] <= 6.0)
    )