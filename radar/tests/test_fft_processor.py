import numpy as np

from radar.src.fft_processor import RadarFFTProcessor


def test_range_fft_shape():
    processor = RadarFFTProcessor()

    radar_frame = np.random.randn(512, 256, 16) + 1j * np.random.randn(
        512, 256, 16
    )

    result = processor.range_fft(radar_frame)

    assert result.shape == (512, 256, 16)
    assert np.iscomplexobj(result)


def test_doppler_fft_shape():
    processor = RadarFFTProcessor()

    range_data = np.random.randn(512, 256, 16) + 1j * np.random.randn(
        512, 256, 16
    )

    result = processor.doppler_fft(range_data)

    assert result.shape == (512, 256, 16)
    assert np.iscomplexobj(result)