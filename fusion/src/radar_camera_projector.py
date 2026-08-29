from pathlib import Path

import cv2
import numpy as np


class RadarCameraProjector:
    """
    Project radar 3-D points into the RADIal camera image.

    The coordinate preparation follows the convention used
    in the RADIal camera projection example.
    """

    def __init__(self, calibration_path):
        calibration_path = Path(calibration_path)

        if not calibration_path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {calibration_path}"
            )

        calibration = np.load(
            calibration_path,
            allow_pickle=True,
        ).item()

        self.camera_matrix = np.asarray(
            calibration["intrinsic"]["camera_matrix"],
            dtype=np.float64,
        )

        self.distortion_coefficients = np.asarray(
            calibration["intrinsic"][
                "distortion_coefficients"
            ],
            dtype=np.float64,
        )

        self.rotation_vector = np.asarray(
            calibration["extrinsic"]["rotation_vector"],
            dtype=np.float64,
        )

        self.translation_vector = np.asarray(
            calibration["extrinsic"]["translation_vector"],
            dtype=np.float64,
        )

    @staticmethod
    def prepare_radar_points(points):
        """
        Prepare radar XYZ points using the coordinate convention
        shown in the RADIal projection example.

        Input shape:
            (N, 3)

        Input columns:
            X, Y, Z

        Output columns:
            -Y, X, Z
        """

        points = np.asarray(
            points,
            dtype=np.float64,
        )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Expected points with shape (N, 3)."
            )

        transformed = points.copy()

        # RADIal example:
        # pts[:,[0,1,2]] = pts[:,[1,0,2]]
        transformed = transformed[
            :,
            [1, 0, 2],
        ]

        # RADIal example:
        # pts[:,0] *= -1
        transformed[:, 0] *= -1

        return transformed

    def project(self, radar_points_xyz):
        """
        Project radar 3-D points into camera pixel coordinates.

        Parameters
        ----------
        radar_points_xyz : np.ndarray
            Shape (N, 3), columns X/Y/Z in radar coordinates.

        Returns
        -------
        np.ndarray
            Shape (N, 2), pixel coordinates [u, v].
        """

        points = self.prepare_radar_points(
            radar_points_xyz
        )

        image_points, _ = cv2.projectPoints(
            points,
            self.rotation_vector,
            self.translation_vector,
            self.camera_matrix,
            self.distortion_coefficients,
        )

        return image_points.reshape(
            -1,
            2,
        )

    def project_valid(
        self,
        radar_points_xyz,
        image_width,
        image_height,
    ):
        """
        Project points and keep only points that are inside
        the camera image.
        """

        pixels = self.project(
            radar_points_xyz
        )

        valid = (
            np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < image_width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < image_height)
        )

        return (
            pixels,
            valid,
        )