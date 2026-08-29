import numpy as np


VELOCITY_RESOLUTION = 0.1
NUM_DOPPLER_BINS = 256


def doppler_bin_to_velocity(doppler_bin):
    """
    Convert a 256-point FFT Doppler bin into signed velocity.

    Assumption:
        velocity resolution = 0.1 m/s per Doppler bin.

    For an unshifted FFT:
        0 ... 127   -> positive bins
        128 ... 255 -> negative bins
    """

    doppler_bin = int(doppler_bin)

    if not 0 <= doppler_bin < NUM_DOPPLER_BINS:
        raise ValueError(
            f"Doppler bin must be in [0, {NUM_DOPPLER_BINS - 1}]"
        )

    if doppler_bin < NUM_DOPPLER_BINS // 2:
        signed_bin = doppler_bin
    else:
        signed_bin = doppler_bin - NUM_DOPPLER_BINS

    return float(
        signed_bin * VELOCITY_RESOLUTION
    )


def test_doppler_to_velocity():

    test_bins = [0, 10, 100, 127, 128, 200, 255]

    print("\n===== DOPPLER → VELOCITY =====")

    for doppler_bin in test_bins:

        velocity = doppler_bin_to_velocity(
            doppler_bin
        )

        print(
            f"Doppler bin {doppler_bin:3d} "
            f"→ {velocity:+.2f} m/s"
        )

    assert doppler_bin_to_velocity(0) == 0.0

    assert np.isclose(
        doppler_bin_to_velocity(10),
        1.0,
    )

    assert np.isclose(
        doppler_bin_to_velocity(127),
        12.7,
    )

    assert np.isclose(
        doppler_bin_to_velocity(128),
        -12.8,
    )

    assert np.isclose(
        doppler_bin_to_velocity(255),
        -0.1,
    )