import numpy as np
from sklearn.cluster import DBSCAN


class RadarDBSCAN:
    """
    Cluster radar points using DBSCAN.

    DBSCAN groups nearby radar detections into clusters
    without requiring the number of objects in advance.
    """

    def __init__(
        self,
        eps=1.5,
        min_samples=3,
    ):
        self.eps = float(eps)
        self.min_samples = int(min_samples)

        if self.eps <= 0:
            raise ValueError("eps must be greater than 0.")

        if self.min_samples < 1:
            raise ValueError(
                "min_samples must be at least 1."
            )

        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
        )

    def cluster(self, points):
        """
        Cluster radar points using X/Y position.

        Parameters
        ----------
        points : np.ndarray
            Radar point array with columns:

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

        Returns
        -------
        labels : np.ndarray
            Cluster label for every input point.

            -1 means noise/outlier.
            0, 1, 2, ... are cluster IDs.
        """

        points = np.asarray(points)

        if points.ndim != 2:
            raise ValueError(
                "points must be a 2-D array."
            )

        if points.shape[1] < 9:
            raise ValueError(
                "Expected radar points with at least "
                "9 columns."
            )

        if len(points) == 0:
            return np.empty(
                (0,),
                dtype=np.int32,
            )

        # Use only horizontal position for clustering.
        xy = points[:, [6, 7]]

        if not np.all(np.isfinite(xy)):
            raise ValueError(
                "X/Y coordinates must be finite."
            )

        labels = self.model.fit_predict(xy)

        return labels.astype(np.int32)

    @staticmethod
    def cluster_count(labels):
        """
        Return the number of actual clusters.

        Noise points labelled -1 are not counted.
        """

        labels = np.asarray(labels)

        valid_labels = labels[
            labels >= 0
        ]

        if len(valid_labels) == 0:
            return 0

        return int(
            len(np.unique(valid_labels))
        )

    @staticmethod
    def get_clusters(points, labels):
        """
        Group radar points by cluster.

        Returns
        -------
        dict
            {
                cluster_id: points_in_cluster
            }
        """

        points = np.asarray(points)
        labels = np.asarray(labels)

        if len(points) != len(labels):
            raise ValueError(
                "points and labels must have the same length."
            )

        clusters = {}

        for cluster_id in np.unique(labels):

            # Ignore DBSCAN noise.
            if cluster_id == -1:
                continue

            clusters[int(cluster_id)] = points[
                labels == cluster_id
            ]

        return clusters