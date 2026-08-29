class RadarVelocityConverter:
    """
    Convert RADIal Doppler FFT bins into radial velocity.

    RADIal HD Radar specification:
        Velocity resolution = 0.1 m/s

    The radar processing uses a 256-point Doppler FFT.
    """

    def __init__(
        self,
        velocity_resolution=0.1,
        num_doppler_bins=256,
    ):
        self.velocity_resolution = float(
            velocity_resolution
        )

        self.num_doppler_bins = int(
            num_doppler_bins
        )

        if self.velocity_resolution <= 0:
            raise ValueError(
                "Velocity resolution must be positive."
            )

        if self.num_doppler_bins <= 0:
            raise ValueError(
                "Number of Doppler bins must be positive."
            )

    def convert(self, doppler_bin):
        """
        Convert one FFT Doppler-bin index into radial velocity.

        Parameters
        ----------
        doppler_bin : int
            Doppler-bin index in [0, 255].

        Returns
        -------
        float
            Radial velocity in m/s.
        """

        doppler_bin = int(doppler_bin)

        if not 0 <= doppler_bin < self.num_doppler_bins:
            raise ValueError(
                f"Doppler bin must be in "
                f"[0, {self.num_doppler_bins - 1}]"
            )

        # Convert the ordinary FFT index into a signed index.
        half = self.num_doppler_bins // 2

        if doppler_bin < half:
            signed_bin = doppler_bin
        else:
            signed_bin = (
                doppler_bin
                - self.num_doppler_bins
            )

        return float(
            signed_bin
            * self.velocity_resolution
        )