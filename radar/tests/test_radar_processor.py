import numpy as np

from radar.src.radar_processor import RadarProcessor


def test_full_radar_pipeline():
    samples_per_chip = 512 * 4 * 256 * 2

    rng = np.random.default_rng(42)

    adc0 = rng.integers(
        -100,
        100,
        size=samples_per_chip,
        dtype=np.int16,
    )

    adc1 = rng.integers(
        -100,
        100,
        size=samples_per_chip,
        dtype=np.int16,
    )

    adc2 = rng.integers(
        -100,
        100,
        size=samples_per_chip,
        dtype=np.int16,
    )

    adc3 = rng.integers(
        -100,
        100,
        size=samples_per_chip,
        dtype=np.int16,
    )

    processor = RadarProcessor()

    result = processor.process(
        adc0,
        adc1,
        adc2,
        adc3,
    )

    assert "range_doppler" in result
    assert "detections" in result

    assert result["range_doppler"].shape == (512, 256, 16)
    assert result["detections"].shape == (512, 256)

    assert np.iscomplexobj(result["range_doppler"])
    assert result["detections"].dtype == bool