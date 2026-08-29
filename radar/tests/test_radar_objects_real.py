from pathlib import Path

import numpy as np

from radar.src.radial_loader import RADIalLoader
from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN
from radar.src.radar_objects import RadarObjectExtractor


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


def test_real_radar_objects():

    # ---------------------------------------------------------
    # 1. Load real radar frame
    # ---------------------------------------------------------

    loader = RADIalLoader(
        DATASET,
        dbreader_dir=DBREADER,
    )

    frame = loader.load_frame(0)

    # ---------------------------------------------------------
    # 2. Build complex radar frame
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
    # 4. Range + Doppler FFT
    # ---------------------------------------------------------

    fft_processor = RadarFFTProcessor()

    range_fft = fft_processor.range_fft(
        complex_frame
    )

    rd_spectrum = fft_processor.doppler_fft(
        range_fft
    )

    # ---------------------------------------------------------
    # 5. Range-Doppler power
    # ---------------------------------------------------------

    rd_power = np.sum(
        np.abs(rd_spectrum) ** 2,
        axis=2,
    )

    # ---------------------------------------------------------
    # 6. CFAR
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
    # 7. Generate radar points
    # ---------------------------------------------------------

    point_generator = RealRadarPointGenerator(
        CALIBRATION,
        max_points=100,
        min_range_bin=10,
    )

    points = point_generator.generate(
        rd_spectrum,
        detections,
    )

    # ---------------------------------------------------------
    # 8. DBSCAN
    # ---------------------------------------------------------

    clusterer = RadarDBSCAN(
        eps=1.5,
        min_samples=3,
    )

    labels = clusterer.cluster(
        points
    )

    # ---------------------------------------------------------
    # 9. Convert clusters into objects
    # ---------------------------------------------------------

    extractor = RadarObjectExtractor()

    objects = extractor.extract(
        points,
        labels,
    )

    # ---------------------------------------------------------
    # 10. Display
    # ---------------------------------------------------------

    print(
        "\n===== REAL RADAR OBJECTS ====="
    )

    print(
        "Input radar points:",
        len(points),
    )

    print(
        "Radar objects:",
        len(objects),
    )

    for obj in objects:

        print(
            f"  Range: "
            f"{obj['range_m']:.2f} m"
        )

        print(
            f"  Velocity: "
            f"{obj['velocity_mps']:+.2f} m/s"
        )

        print(
            f"  Position: "
            f"x={obj['x_m']:.2f} m, "
            f"y={obj['y_m']:.2f} m, "
            f"z={obj['z_m']:.2f} m"
        )

        print(
            f"  Azimuth: "
            f"{obj['azimuth_deg']:.2f} deg"
        )

        print(
            f"  Elevation: "
            f"{obj['elevation_deg']:.2f} deg"
        )

        print(
            f"  Points: "
            f"{obj['num_points']}"
        )

    # ---------------------------------------------------------
    # 11. Validation
    # ---------------------------------------------------------

    assert isinstance(
        objects,
        list,
    )

    assert len(objects) > 0

    for obj in objects:

        assert np.isfinite(
            obj["x_m"]
        )

        assert np.isfinite(
            obj["y_m"]
        )

        assert np.isfinite(
            obj["z_m"]
        )

        assert np.isfinite(
            obj["range_m"]
        )

        assert (
            -75.0
            <= obj["azimuth_deg"]
            <= 75.0
        )

        assert (
            -4.0
            <= obj["elevation_deg"]
            <= 6.0
        )