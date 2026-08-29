import numpy as np

from radar.src.mimo_reconstructor import MIMOReconstructor
from radar.src.angle_estimator import RadarAngleEstimator
from radar.src.radar_points import RadarPointExtractor
from radar.src.velocity_converter import RadarVelocityConverter


class RealRadarPointGenerator:
    """
    Generate physical radar points from a real RADIal frame.

    Pipeline:
        RD spectrum
            -> CFAR detections
            -> select strongest detections
            -> MIMO reconstruction
            -> azimuth/elevation estimation
            -> range conversion
            -> Doppler-to-velocity conversion
            -> X/Y/Z conversion
    """

    def __init__(
        self,
        calibration_path,
        max_points=30,
        min_range_bin=10,
    ):
        self.max_points = int(max_points)
        self.min_range_bin = int(min_range_bin)

        self.mimo = MIMOReconstructor()

        self.angle_estimator = RadarAngleEstimator(
            calibration_path
        )

        self.point_extractor = RadarPointExtractor()

        # RADIal HD Radar specification:
        # velocity resolution = 0.1 m/s
        self.velocity_converter = RadarVelocityConverter(
            velocity_resolution=0.1,
            num_doppler_bins=256,
        )

    def generate(
        self,
        rd_spectrum,
        detections,
    ):
        """
        Generate physical radar points.

        Returns
        -------
        np.ndarray
            Columns:

            0  range_bin
            1  doppler_bin
            2  range_m
            3  velocity_mps
            4  azimuth_deg
            5  elevation_deg
            6  x_m
            7  y_m
            8  z_m
            9  power
        """

        detection_indices = np.argwhere(
            detections
        )

        if len(detection_indices) == 0:
            return np.empty(
                (0, 10),
                dtype=np.float32,
            )

        # Remove detections from the very first range bins.
        detection_indices = detection_indices[
            detection_indices[:, 0]
            >= self.min_range_bin
        ]

        if len(detection_indices) == 0:
            return np.empty(
                (0, 10),
                dtype=np.float32,
            )

        # Calculate total RX power for ranking.
        power_map = np.sum(
            np.abs(rd_spectrum) ** 2,
            axis=2,
        )

        powers = power_map[
            detection_indices[:, 0],
            detection_indices[:, 1],
        ]

        # Keep strongest detections.
        order = np.argsort(
            powers
        )[::-1]

        selected = detection_indices[
            order[: self.max_points]
        ]

        points = []

        for range_bin, doppler_bin in selected:

            range_bin = int(range_bin)
            doppler_bin = int(doppler_bin)

            # RADIal reduces Doppler into 16 bins for the
            # MIMO reconstruction stage.
            reduced_doppler_bin = (
                doppler_bin % 16
            )

            try:
                # -------------------------------------------------
                # 1. Reconstruct the 192-channel MIMO spectrum.
                # -------------------------------------------------
                mimo_spectrum = (
                    self.mimo.reconstruct(
                        rd_spectrum,
                        range_bin,
                        reduced_doppler_bin,
                    )
                )

                # -------------------------------------------------
                # 2. Estimate azimuth and elevation.
                # -------------------------------------------------
                azimuth_deg, elevation_deg = (
                    self.angle_estimator.estimate(
                        mimo_spectrum
                    )
                )

                # -------------------------------------------------
                # 3. Convert range bin to meters.
                # -------------------------------------------------
                range_m = (
                    range_bin
                    / 512.0
                    * 103.0
                )

                # -------------------------------------------------
                # 4. Convert Doppler bin to radial velocity.
                # -------------------------------------------------
                velocity_mps = (
                    self.velocity_converter.convert(
                        doppler_bin
                    )
                )

                # -------------------------------------------------
                # 5. Convert spherical coordinates to XYZ.
                # -------------------------------------------------
                x_m, y_m, z_m = (
                    self.point_extractor.cartesian_from_spherical(
                        range_m,
                        azimuth_deg,
                        elevation_deg,
                    )
                )

                # -------------------------------------------------
                # 6. Radar power.
                # -------------------------------------------------
                power = float(
                    power_map[
                        range_bin,
                        doppler_bin,
                    ]
                )

                point = np.array(
                    [
                        range_bin,
                        doppler_bin,
                        range_m,
                        velocity_mps,
                        azimuth_deg,
                        elevation_deg,
                        x_m,
                        y_m,
                        z_m,
                        power,
                    ],
                    dtype=np.float32,
                )

                points.append(point)

            except (
                ValueError,
                IndexError,
                RuntimeError,
            ):
                # Skip invalid detections without stopping
                # the entire frame.
                continue

        if not points:
            return np.empty(
                (0, 10),
                dtype=np.float32,
            )

        return np.vstack(
            points
        )

    @staticmethod
    def columns():
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