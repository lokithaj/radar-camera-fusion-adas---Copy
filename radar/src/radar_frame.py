import numpy as np


class RadarFrameBuilder:
    """Build complex radar frames from RADIal ADC data."""

    def __init__(
        self,
        samples_per_chirp=512,
        rx_per_chip=4,
        num_chirps=256,
    ):
        self.samples_per_chirp = samples_per_chirp
        self.rx_per_chip = rx_per_chip
        self.num_chirps = num_chirps

    def _build_chip_frame(self, adc):
        """Convert one chip's interleaved ADC stream to complex samples."""

        complex_adc = adc[0::2] + 1j * adc[1::2]

        return np.reshape(
            complex_adc,
            (
                self.samples_per_chirp,
                self.rx_per_chip,
                self.num_chirps,
            ),
            order="F",
        ).transpose((0, 2, 1))

    def build_frame(self, adc0, adc1, adc2, adc3):
        """Build a complete 16-RX complex radar frame."""

        frame0 = self._build_chip_frame(adc0)
        frame1 = self._build_chip_frame(adc1)
        frame2 = self._build_chip_frame(adc2)
        frame3 = self._build_chip_frame(adc3)

        return np.concatenate(
            [frame3, frame0, frame1, frame2],
            axis=2,
        )