import numpy as np


class RadarAngleEstimator:
    """
    Estimate azimuth and elevation using the RADIal
    calibration table.
    """

    def __init__(self, calibration_path):
        calibration_path = str(calibration_path)

        calibration = np.load(
            calibration_path,
            allow_pickle=True,
        ).item()

        self.azimuth_table = np.asarray(
            calibration["Azimuth_table"]
        )

        self.elevation_table = np.asarray(
            calibration["Elevation_table"]
        )

        self.window = np.asarray(
            calibration["H"][0]
        )

        signal = np.asarray(
            calibration["Signal"]
        )

        # Original RADIal shape:
        # (azimuth, virtual_antennas, elevation)
        if signal.shape != (
            len(self.azimuth_table),
            192,
            len(self.elevation_table),
        ):
            raise ValueError(
                "Unexpected calibration signal shape: "
                f"{signal.shape}"
            )

        # RADIal rearranges:
        #
        # (751, 192, 11)
        #     ↓
        # (751, 11, 192)
        #
        # and then flattens azimuth/elevation:
        #
        # (751 * 11, 192)
        #
        self.calib_matrix = np.rollaxis(
            signal,
            2,
            1,
        ).reshape(
            len(self.azimuth_table)
            * len(self.elevation_table),
            192,
        )

        self.num_virtual_antennas = 192

    def estimate(self, mimo_spectrum):
        """
        Estimate azimuth and elevation for one radar target.

        Parameters
        ----------
        mimo_spectrum : np.ndarray
            Complex 192-channel MIMO spectrum.

        Returns
        -------
        tuple
            azimuth_deg, elevation_deg
        """

        mimo_spectrum = np.asarray(
            mimo_spectrum,
            dtype=np.complex64,
        ).reshape(-1)

        if mimo_spectrum.size != self.num_virtual_antennas:
            raise ValueError(
                "Expected 192 virtual antenna values, "
                f"got {mimo_spectrum.size}"
            )

        # Apply the RADIal calibration/window weighting.
        weighted = (
            mimo_spectrum * self.window
        )

        # Compare the measured 192-channel spectrum
        # against all calibrated azimuth/elevation
        # responses.
        angle_response = np.abs(
            self.calib_matrix @ weighted
        )

        # Restore:
        #
        # 751 azimuth values ×
        # 11 elevation values
        #
        angle_response = angle_response.reshape(
            len(self.azimuth_table),
            len(self.elevation_table),
        )

        # Find strongest calibration response.
        azimuth_index, elevation_index = np.unravel_index(
            np.argmax(angle_response),
            angle_response.shape,
        )

        azimuth_deg = float(
            self.azimuth_table[azimuth_index]
        )

        elevation_deg = float(
            self.elevation_table[elevation_index]
        )

        return azimuth_deg, elevation_deg