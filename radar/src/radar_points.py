import numpy as np


class RadarPointExtractor:
    """
    Convert radar detections into physical radar points.

    RADIal reference:
        Range = range_bin / 512 * 103 meters

    Output columns:

        0 -> range_bin
        1 -> doppler_bin
        2 -> range_m
        3 -> velocity_mps
        4 -> azimuth_deg
        5 -> elevation_deg
        6 -> x_m
        7 -> y_m
        8 -> z_m
        9 -> power
    """

    def __init__(
        self,
        samples_per_chirp=512,
        max_range_m=103.0,
    ):
        self.samples_per_chirp = samples_per_chirp
        self.max_range_m = max_range_m

    def range_from_bin(self, range_bin):
        """
        Convert a RADIal range-bin index into meters.
        """

        range_bin = float(range_bin)

        if range_bin < 0:
            raise ValueError(
                "Range bin cannot be negative."
            )

        return (
            range_bin
            / self.samples_per_chirp
            * self.max_range_m
        )

    @staticmethod
    def cartesian_from_spherical(
        range_m,
        azimuth_deg,
        elevation_deg,
    ):
        """
        Convert spherical radar coordinates to X/Y/Z.
        """

        azimuth_rad = np.deg2rad(
            azimuth_deg
        )

        elevation_rad = np.deg2rad(
            elevation_deg
        )

        cos_elevation = np.cos(
            elevation_rad
        )

        x_m = (
            range_m
            * cos_elevation
            * np.cos(azimuth_rad)
        )

        y_m = (
            range_m
            * cos_elevation
            * np.sin(azimuth_rad)
        )

        z_m = (
            range_m
            * np.sin(elevation_rad)
        )

        return (
            float(x_m),
            float(y_m),
            float(z_m),
        )

    def create_point(
        self,
        range_bin,
        doppler_bin,
        azimuth_deg,
        elevation_deg,
        power,
        velocity_mps=None,
    ):
        """
        Create one physical radar point.

        velocity_mps is optional because the exact physical
        Doppler-to-velocity conversion requires the radar's
        chirp configuration.
        """

        range_m = self.range_from_bin(
            range_bin
        )

        x_m, y_m, z_m = (
            self.cartesian_from_spherical(
                range_m,
                azimuth_deg,
                elevation_deg,
            )
        )

        if velocity_mps is None:
            velocity_mps = np.nan

        return np.array(
            [
                float(range_bin),
                float(doppler_bin),
                range_m,
                float(velocity_mps),
                float(azimuth_deg),
                float(elevation_deg),
                x_m,
                y_m,
                z_m,
                float(power),
            ],
            dtype=np.float32,
        )

    def extract_from_detection(
        self,
        range_bin,
        doppler_bin,
        azimuth_deg,
        elevation_deg,
        power,
        velocity_mps=None,
    ):
        """
        Convert one detection into a physical radar point.
        """

        return self.create_point(
            range_bin=range_bin,
            doppler_bin=doppler_bin,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            power=power,
            velocity_mps=velocity_mps,
        )

    @staticmethod
    def columns():
        """Return the names of the point columns."""

        return [
            "range_bin",
            "doppler_bin",
            "range_m",
            "velocity_mps",
            "azimuth_deg",
            "elevation_deg",
            "x_m",
            "y_m",
            "z_m",
            "power",
        ]