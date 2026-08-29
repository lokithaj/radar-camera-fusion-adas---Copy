import numpy as np


class MIMOReconstructor:
    """
    Reconstruct the 192-channel virtual MIMO spectrum
    used by the RADIal point-cloud processing.

    RADIal configuration:
        16 RX antennas
        12 TX antennas
        192 virtual channels
        256 Doppler bins
        16 reduced-Doppler bins
    """

    def __init__(
        self,
        num_rx=16,
        num_tx=12,
        num_chirps=256,
        num_reduced_doppler=16,
    ):
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_chirps = num_chirps
        self.num_reduced_doppler = num_reduced_doppler

        if self.num_rx * self.num_tx != 192:
            raise ValueError(
                "Expected 16 RX × 12 TX = 192 virtual channels."
            )

        if self.num_chirps % self.num_reduced_doppler != 0:
            raise ValueError(
                "num_chirps must be divisible by "
                "num_reduced_doppler."
            )

    def find_tx0_position(
        self,
        power_spectrum,
        range_bin,
        reduced_doppler_bin,
    ):
        """
        Find the TX-0 Doppler position following the
        RADIal reference implementation.

        Parameters
        ----------
        power_spectrum : np.ndarray
            Power spectrum with shape:
            (range, doppler)

        range_bin : int
            Range-bin index.

        reduced_doppler_bin : int
            Reduced Doppler-bin index [0, 15].

        Returns
        -------
        int
            Estimated full Doppler-bin position for TX-0.
        """

        reduced_doppler_bin = int(reduced_doppler_bin)

        if not 0 <= reduced_doppler_bin < self.num_reduced_doppler:
            raise ValueError(
                "Reduced Doppler bin must be in [0, 15]."
            )

        # Candidate Doppler positions.
        base_positions = np.arange(
            0,
            self.num_chirps,
            self.num_reduced_doppler,
        )

        doppler_idx = (
            base_positions + reduced_doppler_bin
        )

        # Wrap around with the first four positions appended,
        # matching the RADIal implementation.
        doppler_idx = np.concatenate(
            [
                doppler_idx,
                doppler_idx[:4],
            ]
        )

        range_value = int(range_bin)

        if not 0 <= range_value < power_spectrum.shape[0]:
            raise IndexError("Invalid range bin.")

        # Extract power along the candidate Doppler sequence.
        values = power_spectrum[
            range_value,
            doppler_idx,
        ]

        # Four-position moving average.
        N = 4

        cumsum = np.cumsum(
            values
        )

        moving_average = (
            cumsum[N:] - cumsum[:-N]
        ) / N

        section_idx = int(
            np.argmin(moving_average)
        )

        doppler_bin = (
            section_idx * self.num_reduced_doppler
            + reduced_doppler_bin
        )

        return int(doppler_bin)

    def reconstruct(
        self,
        rd_spectrum,
        range_bin,
        reduced_doppler_bin,
    ):
        """
        Extract the RADIal MIMO spectrum for one
        range/reduced-Doppler detection.

        Parameters
        ----------
        rd_spectrum : np.ndarray
            Complex RD spectrum with shape:
            (range, doppler, 16 RX)

        range_bin : int
            Range-bin index.

        reduced_doppler_bin : int
            Reduced Doppler-bin index.

        Returns
        -------
        np.ndarray
            192-channel complex MIMO spectrum.
        """

        rd_spectrum = np.asarray(
            rd_spectrum
        )

        if rd_spectrum.ndim != 3:
            raise ValueError(
                "Expected RD spectrum with shape "
                "(range, doppler, RX)."
            )

        expected_shape = (
            self.num_chirps,
            self.num_rx,
        )

        if rd_spectrum.shape[1:] != expected_shape:
            raise ValueError(
                "Unexpected RD spectrum shape. "
                f"Expected (range, {self.num_chirps}, "
                f"{self.num_rx}), got {rd_spectrum.shape}."
            )

        # Same Doppler-search principle used by RADIal.
        power_spectrum = np.sum(
            np.abs(rd_spectrum),
            axis=2,
        )

        tx0_doppler = self.find_tx0_position(
            power_spectrum,
            range_bin,
            reduced_doppler_bin,
        )

        # Generate the TX-shifted Doppler positions.
        offsets = np.arange(
            0,
            self.num_reduced_doppler
            * self.num_chirps_per_loop,
            self.num_reduced_doppler,
        )

        doppler_sequence = np.remainder(
            tx0_doppler + offsets,
            self.num_chirps,
        )

        # RADIal removes positions 1..4 from the sequence
        # after preserving the first position.
        doppler_sequence = np.concatenate(
            [
                [doppler_sequence[0]],
                doppler_sequence[5:],
            ]
        ).astype(int)

        # Extract the RX data at those Doppler positions.
        selected = rd_spectrum[
            int(range_bin),
            doppler_sequence,
            :,
        ]

        # This should be:
        # 12 TX × 16 RX = 192 values.
        mimo_spectrum = selected.reshape(
            -1
        )

        if mimo_spectrum.size != 192:
            raise RuntimeError(
                "MIMO reconstruction did not produce "
                f"192 values. Got {mimo_spectrum.size}."
            )

        return mimo_spectrum

    @property
    def num_chirps_per_loop(self):
        """RADIal uses 16 chirps per loop."""
        return self.num_reduced_doppler