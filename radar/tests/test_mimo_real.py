import numpy as np
import pytest
from config import get_default_recording_dir, get_dbreader_dir
from radar.src.radial_loader import RADIalLoader
from radar.src.radar_pipeline import RadarPipeline
from radar.src.mimo_reconstructor import MIMOReconstructor

DATASET = get_default_recording_dir()
DBREADER = get_dbreader_dir(DATASET)


@pytest.mark.skipif(
    not (DATASET.exists() and DBREADER.exists()),
    reason="RADIal dataset or DBReader not found",
)
def test_real_mimo_reconstruction():
    # Load real RADIal radar data.
    loader = RADIalLoader(
        DATASET,
        dbreader_dir=DBREADER,
    )

    frame = loader.load_frame(0)

    # Generate the real Range-Doppler spectrum.
    pipeline = RadarPipeline()

    result = pipeline.process(
        frame["radar_ch0"],
        frame["radar_ch1"],
        frame["radar_ch2"],
        frame["radar_ch3"],
    )

    rd_spectrum = result["rd_spectrum"]
    detections = result["detections"]

    # Find one CFAR detection away from the first row
    # so that it is less likely to be a DC/edge artifact.
    detection_indices = np.argwhere(detections)

    valid = detection_indices[
        detection_indices[:, 0] > 10
    ]

    if len(valid) == 0:
        raise RuntimeError(
            "No suitable CFAR detection was found."
        )

    range_bin, doppler_bin = valid[0]

    # The current CFAR matrix uses the full 256-Doppler map.
    # RADIal's MIMO reconstruction works from a reduced
    # Doppler bin in the range 0..15.
    reduced_doppler_bin = int(
        doppler_bin % 16
    )

    reconstructor = MIMOReconstructor()

    mimo = reconstructor.reconstruct(
        rd_spectrum,
        int(range_bin),
        reduced_doppler_bin,
    )

    print("\n===== REAL MIMO TEST =====")
    print("Selected range bin:", int(range_bin))
    print("Selected Doppler bin:", int(doppler_bin))
    print(
        "Reduced Doppler bin:",
        reduced_doppler_bin,
    )
    print(
        "MIMO shape:",
        mimo.shape,
    )
    print(
        "MIMO dtype:",
        mimo.dtype,
    )
    print(
        "MIMO values:",
        mimo[:10],
    )

    assert mimo.shape == (192,)
    assert np.iscomplexobj(mimo)