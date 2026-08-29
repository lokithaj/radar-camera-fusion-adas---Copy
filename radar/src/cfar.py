import numpy as np
from scipy import signal


class CACFAR:
    """Cell-Averaging CFAR detector for radar Range-Doppler maps."""

    def __init__(
        self,
        window=(9, 9),
        guard=(3, 3),
        threshold_db=2.0,
    ):
        self.window = window
        self.guard = guard
        self.threshold = 10 ** (threshold_db / 10)

        win_width, win_height = window
        guard_width, guard_height = guard

        self.mask = np.ones(
            (2 * win_height + 1, 2 * win_width + 1),
            dtype=bool,
        )

        self.mask[
            win_height - guard_height : win_height + guard_height + 1,
            win_width - guard_width : win_width + guard_width + 1,
        ] = False

    def detect(self, rd_matrix):
        """
        Detect strong cells in a complex Range-Doppler matrix.

        Returns:
            Boolean matrix indicating detected cells.
        """

        if np.iscomplexobj(rd_matrix):
            power = np.abs(rd_matrix) ** 2
        else:
            power = rd_matrix

        noise_sum = signal.convolve2d(
            power,
            self.mask,
            mode="same",
        )

        valid_cells = signal.convolve2d(
            np.ones_like(power, dtype=float),
            self.mask,
            mode="same",
        )

        noise_power = noise_sum / np.maximum(valid_cells, 1)

        snr = power / np.maximum(noise_power, np.finfo(float).eps)

        return snr > self.threshold