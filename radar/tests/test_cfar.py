import numpy as np

from radar.src.cfar import CACFAR


def test_cfar_detects_strong_target():
    rng = np.random.default_rng(42)

    # Create a noisy Range-Doppler map
    rd_map = (
        rng.normal(0, 1, (128, 64))
        + 1j * rng.normal(0, 1, (128, 64))
    )

    # Add a strong synthetic target
    target_range = 64
    target_doppler = 32

    rd_map[target_range, target_doppler] = 100 + 100j

    detector = CACFAR(
        window=(9, 9),
        guard=(3, 3),
        threshold_db=2.0,
    )

    detections = detector.detect(rd_map)

    assert detections.shape == rd_map.shape
    assert detections[target_range, target_doppler]


def test_cfar_output_is_boolean():
    rd_map = np.ones((64, 32), dtype=np.complex64)

    detector = CACFAR()

    detections = detector.detect(rd_map)

    assert detections.dtype == bool