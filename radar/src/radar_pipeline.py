import numpy as np

from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.radar_points import RadarPointExtractor
from radar.src.velocity_converter import RadarVelocityConverter


class RadarPipeline:
    """
    End-to-end radar perception baseline.

    RADIal ADC
        -> complex radar frame
        -> range FFT
        -> Doppler FFT
        -> RX power combination
        -> CA-CFAR
        -> radar point representation
    """

    def __init__(
        self,
        samples_per_chirp=512,
        rx_per_chip=4,
        num_chirps=256,
        range_resolution=0.2,
        velocity_resolution=0.5,
    ):
        # 1. Radar frame construction
        self.frame_builder = RadarFrameBuilder(
            samples_per_chirp=samples_per_chirp,
            rx_per_chip=rx_per_chip,
            num_chirps=num_chirps,
        )

        # 2. FFT processing
        self.fft_processor = RadarFFTProcessor(
            samples_per_chirp=samples_per_chirp,
            num_chirps=num_chirps,
        )

        # 3. CFAR detection
        self.cfar = CACFAR(
            window=(9, 9),
            guard=(3, 3),
            threshold_db=2.0,
        )

        # 4. Convert detection bins into radar points
        max_range_m = 103.0
        if range_resolution is not None and range_resolution != 0.2:
            max_range_m = float(samples_per_chirp * range_resolution)

        self.point_extractor = RadarPointExtractor(
            samples_per_chirp=samples_per_chirp,
            max_range_m=max_range_m,
        )

        self.velocity_converter = RadarVelocityConverter(
            velocity_resolution=velocity_resolution,
            num_doppler_bins=num_chirps,
        )

    def process(self, adc0, adc1, adc2, adc3):
        """
        Process one real RADIal radar frame.

        Parameters
        ----------
        adc0, adc1, adc2, adc3 : np.ndarray
            Raw ADC data from the four radar chips.

        Returns
        -------
        dict
            complex_frame
            range_fft
            rd_spectrum
            rd_power
            detections
            points
        """

        # ---------------------------------------------------------
        # 1. Build 16-RX complex radar frame
        # ---------------------------------------------------------
        complex_frame = self.frame_builder.build_frame(
            adc0,
            adc1,
            adc2,
            adc3,
        )

        # ---------------------------------------------------------
        # 2. Remove DC offset
        # ---------------------------------------------------------
        complex_frame = (
            complex_frame
            - np.mean(
                complex_frame,
                axis=(0, 1),
                keepdims=True,
            )
        )

        # ---------------------------------------------------------
        # 3. Range FFT
        # ---------------------------------------------------------
        range_fft = self.fft_processor.range_fft(
            complex_frame
        )

        # ---------------------------------------------------------
        # 4. Doppler FFT
        # ---------------------------------------------------------
        rd_spectrum = self.fft_processor.doppler_fft(
            range_fft
        )

        # ---------------------------------------------------------
        # 5. Combine the 16 RX channels
        # ---------------------------------------------------------
        #
        # Calculate power for every RX channel and sum across
        # all 16 channels to obtain a single 2-D
        # Range-Doppler power map.
        #
        rd_power = np.sum(
            np.abs(rd_spectrum) ** 2,
            axis=2,
        )

        # ---------------------------------------------------------
        # 6. CA-CFAR detection
        # ---------------------------------------------------------
        detections = self.cfar.detect(
            rd_power
        )

        # ---------------------------------------------------------
        # 7. Convert detections into radar points
        # ---------------------------------------------------------
        detection_indices = self.detection_points(detections)

        if len(detection_indices) == 0:
            points = np.empty((0, 4), dtype=np.float32)
        else:
            range_bins = detection_indices[:, 0]
            doppler_bins = detection_indices[:, 1]
            ranges_m = np.array(
                [self.point_extractor.range_from_bin(rb) for rb in range_bins],
                dtype=np.float32,
            )
            velocities_mps = np.array(
                [self.velocity_converter.convert(db) for db in doppler_bins],
                dtype=np.float32,
            )
            points = np.column_stack([
                range_bins.astype(np.float32),
                doppler_bins.astype(np.float32),
                ranges_m,
                velocities_mps,
            ]).astype(np.float32)

        return {
            "complex_frame": complex_frame,
            "range_fft": range_fft,
            "rd_spectrum": rd_spectrum,
            "rd_power": rd_power,
            "detections": detections,
            "points": points,
        }

    @staticmethod
    def detection_points(detection_matrix):
        """
        Convert a CFAR boolean matrix into
        (range_bin, doppler_bin) coordinates.
        """

        return np.argwhere(
            detection_matrix
        )