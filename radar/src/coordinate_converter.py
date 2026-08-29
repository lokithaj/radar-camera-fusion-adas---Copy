import numpy as np


class RadarCoordinateConverter:
    """
    Convert radar spherical coordinates into Cartesian coordinates.

    Input:
        range_m
        azimuth_deg
        elevation_deg

    Output:
        x_m
        y_m
        z_m
    """

    @staticmethod
    def spherical_to_cartesian(
        range_m,
        azimuth_deg,
        elevation_deg,
    ):
        """
        Convert one radar point from spherical coordinates
        to Cartesian coordinates.

        Returns
        -------
        tuple
            (x_m, y_m, z_m)
        """

        range_m = float(range_m)
        azimuth_deg = float(azimuth_deg)
        elevation_deg = float(elevation_deg)

        if range_m < 0:
            raise ValueError(
                "Range cannot be negative."
            )

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