from pathlib import Path

import numpy as np

from radar.src.radial_loader import RADIalLoader
from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN


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
def test_dbscan_on_real_radar():

    # ---------------------------------------------------------
    # 1. Load real RADIal radar frame
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
    # 3. Remove DC
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
    # 7. Generate physical radar points
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

    print("\n===== DBSCAN RADAR CLUSTERING =====")
    print(
        "Physical radar points:",
        len(points),
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

    number_of_clusters = (
        clusterer.cluster_count(labels)
    )

    clusters = clusterer.get_clusters(
        points,
        labels,
    )

    print(
        "Number of clusters:",
        number_of_clusters,
    )

    print(
        "Noise points:",
        int(np.sum(labels == -1)),
    )

    # ---------------------------------------------------------
    # 9. Display cluster information
    # ---------------------------------------------------------
    for cluster_id, cluster_points in clusters.items():

        x_center = np.mean(
            cluster_points[:, 6]
        )

        y_center = np.mean(
            cluster_points[:, 7]
        )

        range_center = np.mean(
            cluster_points[:, 2]
        )

        print(
            f"Cluster {cluster_id}: "
            f"points={len(cluster_points)}, "
            f"x={x_center:.2f} m, "
            f"y={y_center:.2f} m, "
            f"range={range_center:.2f} m"
        )

    # ---------------------------------------------------------
    # 10. Validate
    # ---------------------------------------------------------
    assert points.ndim == 2

    assert points.shape[1] == 10

    assert len(points) > 0

    assert labels.shape == (
        len(points),
    )

    assert labels.dtype == np.int32

    assert number_of_clusters >= 0