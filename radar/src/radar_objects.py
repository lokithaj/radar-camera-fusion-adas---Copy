import numpy as np


class RadarObjectExtractor:
    """
    Convert DBSCAN radar clusters into object-level measurements.

    Radar point columns:

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

    def extract(self, points, labels):
        """
        Convert DBSCAN clusters into radar objects.
        """

        points = np.asarray(points)
        labels = np.asarray(labels)

        if points.ndim != 2:
            raise ValueError(
                "points must be a 2-D array."
            )

        if points.shape[1] < 10:
            raise ValueError(
                "Expected radar points with 10 columns."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must be a 1-D array."
            )

        if len(points) != len(labels):
            raise ValueError(
                "points and labels must have the same length."
            )

        objects = []

        for cluster_id in np.unique(labels):

            # Ignore DBSCAN noise.
            if cluster_id == -1:
                continue

            cluster_points = points[
                labels == cluster_id
            ]

            if len(cluster_points) == 0:
                continue

            # Position.
            x_m = float(
                np.mean(cluster_points[:, 6])
            )

            y_m = float(
                np.mean(cluster_points[:, 7])
            )

            z_m = float(
                np.mean(cluster_points[:, 8])
            )

            # Range.
            range_m = float(
                np.mean(cluster_points[:, 2])
            )

            # Velocity.
            velocity_values = cluster_points[
                :, 3
            ]

            finite_velocity = velocity_values[
                np.isfinite(velocity_values)
            ]

            if len(finite_velocity) > 0:
                velocity_mps = float(
                    np.mean(finite_velocity)
                )
            else:
                velocity_mps = np.nan

            # Angles.
            azimuth_deg = float(
                np.mean(cluster_points[:, 4])
            )

            elevation_deg = float(
                np.mean(cluster_points[:, 5])
            )

            # Power.
            power = float(
                np.mean(cluster_points[:, 9])
            )

            objects.append(
                {
                    "cluster_id": int(cluster_id),
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": z_m,
                    "range_m": range_m,
                    "velocity_mps": velocity_mps,
                    "azimuth_deg": azimuth_deg,
                    "elevation_deg": elevation_deg,
                    "power": power,
                    "num_points": int(
                        len(cluster_points)
                    ),
                }
            )

        return objects