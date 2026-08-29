import numpy as np


class RadarFFTProcessor:
    """Perform range and Doppler FFT processing on radar data."""

    def __init__(self, samples_per_chirp=512, num_chirps=256):
        self.samples_per_chirp = samples_per_chirp
        self.num_chirps = num_chirps

        # Hann window for range processing
        self.range_window = (
            0.54
            - 0.46
            * np.cos(
                (2 * np.pi * np.arange(self.samples_per_chirp))
                / (self.samples_per_chirp - 1)
            )
        )

        # Hann window for Doppler processing
        self.doppler_window = (
            0.54
            - 0.46
            * np.cos(
                (2 * np.pi * np.arange(self.num_chirps))
                / (self.num_chirps - 1)
            )
        )

    def range_fft(self, radar_frame):
        """Compute the FFT along the range-sample dimension."""

        windowed = (
            radar_frame
            * self.range_window[:, None, None]
        )

        return np.fft.fft(
            windowed,
            n=self.samples_per_chirp,
            axis=0,
        )

    def doppler_fft(self, range_data):
        """Compute the FFT along the chirp dimension."""

        windowed = (
            range_data
            * self.doppler_window[None, :, None]
        )

        return np.fft.fft(
            windowed,
            n=self.num_chirps,
            axis=1,
        )