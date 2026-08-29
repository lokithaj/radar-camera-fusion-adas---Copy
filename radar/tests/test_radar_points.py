import numpy as np

from radar.src.radar_points import RadarPointExtractor


def test_radial_radar_point():
    extractor = RadarPointExtractor()

    point = extractor.extract_from_detection(
        range_bin=100,
        doppler_bin=40,
        azimuth_deg=10.0,
        elevation_deg=2.0,
        power=5000.0,
    )

    print("\n===== RADIal RADAR POINT =====")

    print("Range bin:", point[0])
    print("Doppler bin:", point[1])
    print("Range:", point[2], "m")
    print("Velocity:", point[3], "m/s")
    print("Azimuth:", point[4], "degrees")
    print("Elevation:", point[5], "degrees")
    print("X:", point[6], "m")
    print("Y:", point[7], "m")
    print("Z:", point[8], "m")
    print("Power:", point[9])

    expected_range = 100 / 512 * 103

    assert np.isclose(
        point[2],
        expected_range,
    )

    assert np.isfinite(point[6])
    assert np.isfinite(point[7])
    assert np.isfinite(point[8])

    reconstructed_range = np.sqrt(
        point[6] ** 2
        + point[7] ** 2
        + point[8] ** 2
    )

    assert np.isclose(
        reconstructed_range,
        point[2],
        atol=1e-5,
    )