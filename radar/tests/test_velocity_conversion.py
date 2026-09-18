import numpy as np
import pytest

from radar.src.velocity_converter import RadarVelocityConverter


def test_doppler_to_velocity():
    converter = RadarVelocityConverter(
        velocity_resolution=0.1,
        num_doppler_bins=256,
    )

    test_bins = [0, 10, 100, 127, 128, 200, 255]

    print("\n===== DOPPLER → VELOCITY =====")

    for doppler_bin in test_bins:
        velocity = converter.convert(doppler_bin)
        print(
            f"Doppler bin {doppler_bin:3d} "
            f"→ {velocity:+.2f} m/s"
        )

    assert converter.convert(0) == 0.0

    assert np.isclose(
        converter.convert(10),
        1.0,
    )

    assert np.isclose(
        converter.convert(127),
        12.7,
    )

    assert np.isclose(
        converter.convert(128),
        -12.8,
    )

    assert np.isclose(
        converter.convert(255),
        -0.1,
    )


def test_velocity_converter_invalid_bins():
    converter = RadarVelocityConverter()

    with pytest.raises(ValueError):
        converter.convert(-1)

    with pytest.raises(ValueError):
        converter.convert(256)


def test_velocity_converter_invalid_init():
    with pytest.raises(ValueError):
        RadarVelocityConverter(velocity_resolution=0)

    with pytest.raises(ValueError):
        RadarVelocityConverter(num_doppler_bins=-1)