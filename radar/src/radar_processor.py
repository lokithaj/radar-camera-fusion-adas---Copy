import numpy as np
from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR


class RadarProcessor:
    """End-to-end radar signal-processing pipeline."""

    def __init__(self):
        self.frame_builder = RadarFrameBuilder()

        self.fft_processor = RadarFFTProcessor(
            samples_per_chirp=512,
            num_chirps=256,
        )

        self.cfar = CACFAR(
            window=(9, 9),
            guard=(3, 3),
            threshold_db=2.0,
        )

    def process(self, adc0, adc1, adc2, adc3):
        """
        Process one radar frame from four radar chips.

        Returns:
            Dictionary containing Range-Doppler data and detections.
        """

        radar_frame = self.frame_builder.build_frame(
            adc0,
            adc1,
            adc2,
            adc3,
        )

        range_data = self.fft_processor.range_fft(radar_frame)

        rd_data = self.fft_processor.doppler_fft(range_data)

        # Combine the 16 RX channels into a 2-D power map
        rd_power = np.sum(
            np.abs(rd_data) ** 2,
            axis=2,
        )

        # Detect targets using CFAR
        detections = self.cfar.detect(rd_power)

        return {
            "range_doppler": rd_data,
            "detections": detections,
        }