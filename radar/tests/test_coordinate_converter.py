import numpy as np

from radar.src.coordinate_converter import (
    RadarCoordinateConverter,
)


def test_spherical_to_cartesian():

    converter = RadarCoordinateConverter()

    x, y, z = converter.spherical_to_cartesian(
        range_m=30.0,
        azimuth_deg=10.0,
        elevation_deg=2.0,
    )

    print("\n===== COORDINATE CONVERSION =====")
    print("Range: 30.0 m")
    print("Azimuth: 10.0 degrees")
    print("Elevation: 2.0 degrees")

    print("X:", x, "m")
    print("Y:", y, "m")
    print("Z:", z, "m")

    assert np.isfinite(x)
    assert np.isfinite(y)
    assert np.isfinite(z)

    # Distance from origin should remain approximately
    # equal to the original range.
    reconstructed_range = np.sqrt(
        x**2 + y**2 + z**2
    )

    assert np.isclose(
        reconstructed_range,
        30.0,
        atol=1e-5,
    )