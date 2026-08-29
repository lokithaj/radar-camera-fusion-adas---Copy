import numpy as np


class RadarCameraAssociator:
    """
    Associate projected radar objects with camera detections.

    Association is based on the pixel distance from the
    projected radar point to each camera bounding box.
    """

    def __init__(self, max_pixel_distance=30.0):
        self.max_pixel_distance = float(
            max_pixel_distance
        )

        if self.max_pixel_distance < 0:
            raise ValueError(
                "max_pixel_distance must be non-negative."
            )

    @staticmethod
    def point_to_box_distance(
        u,
        v,
        bbox,
    ):
        """
        Calculate the Euclidean pixel distance from a point
        to a bounding box.

        Distance is zero when the point is inside the box.
        """

        x1, y1, x2, y2 = map(
            float,
            bbox,
        )

        dx = max(
            x1 - u,
            0.0,
            u - x2,
        )

        dy = max(
            y1 - v,
            0.0,
            v - y2,
        )

        return float(
            np.sqrt(
                dx * dx + dy * dy
            )
        )

    def associate(
        self,
        radar_objects,
        radar_pixels,
        camera_objects,
    ):
        """
        Associate radar objects with camera detections.

        A match is accepted when the projected radar point
        is inside a camera bounding box or within the configured
        pixel-distance gate.

        Returns
        -------
        list of dict
            Association results.
        """

        radar_pixels = np.asarray(
            radar_pixels,
            dtype=np.float64,
        )

        if radar_pixels.ndim != 2:
            raise ValueError(
                "radar_pixels must be 2-D."
            )

        if radar_pixels.shape[1] != 2:
            raise ValueError(
                "radar_pixels must have shape (N, 2)."
            )

        if len(radar_objects) != len(radar_pixels):
            raise ValueError(
                "radar_objects and radar_pixels "
                "must have the same length."
            )

        associations = []

        for radar_index, radar_object in enumerate(
            radar_objects
        ):

            u, v = radar_pixels[
                radar_index
            ]

            if not (
                np.isfinite(u)
                and np.isfinite(v)
            ):
                continue

            best_camera_index = None
            best_distance = float("inf")

            for camera_index, camera_object in enumerate(
                camera_objects
            ):

                distance = self.point_to_box_distance(
                    u,
                    v,
                    camera_object["bbox"],
                )

                if (
                    distance
                    <= self.max_pixel_distance
                    and distance < best_distance
                ):
                    best_distance = distance
                    best_camera_index = camera_index

            if best_camera_index is None:
                continue

            associations.append(
                {
                    "radar_index": radar_index,
                    "camera_index": best_camera_index,
                    "radar_object": radar_object,
                    "camera_object": camera_objects[
                        best_camera_index
                    ],
                    "pixel": (
                        float(u),
                        float(v),
                    ),
                    "pixel_distance": float(
                        best_distance
                    ),
                }
            )

        return associations