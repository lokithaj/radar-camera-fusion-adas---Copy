import numpy as np

from radar.src.radar_frame import RadarFrameBuilder


def test_build_frame_shape():
    builder = RadarFrameBuilder()

    samples_per_chip = 512 * 4 * 256 * 2

    adc0 = np.zeros(samples_per_chip, dtype=np.int16)
    adc1 = np.zeros(samples_per_chip, dtype=np.int16)
    adc2 = np.zeros(samples_per_chip, dtype=np.int16)
    adc3 = np.zeros(samples_per_chip, dtype=np.int16)

    frame = builder.build_frame(adc0, adc1, adc2, adc3)

    assert frame.shape == (512, 256, 16)
    assert np.iscomplexobj(frame)