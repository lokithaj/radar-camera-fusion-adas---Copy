import pytest
from config import get_default_recording_dir, get_dbreader_dir
from radar.src.radial_loader import RADIalLoader
from radar.src.radar_pipeline import RadarPipeline

DATASET = get_default_recording_dir()
DBREADER = get_dbreader_dir(DATASET)


@pytest.mark.skipif(
    not (DATASET.exists() and DBREADER.exists()),
    reason="RADIal dataset or DBReader not found",
)
def test_real_radar_pipeline():
    # ---------------------------------------------------------
    # 1. Load real RADIal data
    # ---------------------------------------------------------
    loader = RADIalLoader(
        DATASET,
        dbreader_dir=DBREADER,
    )

    assert len(loader) > 0

    # ---------------------------------------------------------
    # 2. Create radar processing pipeline
    # ---------------------------------------------------------
    pipeline = RadarPipeline()

    # ---------------------------------------------------------
    # 3. Load first synchronized real radar frame
    # ---------------------------------------------------------
    frame = loader.load_frame(0)

    # ---------------------------------------------------------
    # 4. Process the real radar frame
    # ---------------------------------------------------------
    result = pipeline.process(
        frame["radar_ch0"],
        frame["radar_ch1"],
        frame["radar_ch2"],
        frame["radar_ch3"],
    )

    # ---------------------------------------------------------
    # 5. Display processing results
    # ---------------------------------------------------------
    print("\n========================================")
    print("       REAL RADAR PIPELINE")
    print("========================================")

    print(
        "Complex frame:",
        result["complex_frame"].shape,
        result["complex_frame"].dtype,
    )

    print(
        "Range FFT:",
        result["range_fft"].shape,
        result["range_fft"].dtype,
    )

    print(
        "RD spectrum:",
        result["rd_spectrum"].shape,
        result["rd_spectrum"].dtype,
    )

    print(
        "RD power:",
        result["rd_power"].shape,
        result["rd_power"].dtype,
    )

    print(
        "CFAR matrix:",
        result["detections"].shape,
        result["detections"].dtype,
    )

    # ---------------------------------------------------------
    # 6. Get CFAR detection coordinates
    # ---------------------------------------------------------
    detection_points = pipeline.detection_points(
        result["detections"]
    )

    print(
        "Number of CFAR detections:",
        len(detection_points),
    )

    print(
        "First detections:",
        detection_points[:10],
    )

    # ---------------------------------------------------------
    # 7. Get physical radar points
    #
    # Format:
    # [range_bin, doppler_bin, range_m, velocity_mps]
    # ---------------------------------------------------------
    radar_points = result["points"]

    print(
        "Radar points:",
        radar_points.shape,
    )

    print(
        "First radar points:"
    )

    print(
        radar_points[:10]
    )

    # ---------------------------------------------------------
    # 8. Verify radar frame dimensions
    # ---------------------------------------------------------
    assert result["complex_frame"].shape == (
        512,
        256,
        16,
    )

    # ---------------------------------------------------------
    # 9. Verify Range FFT dimensions
    # ---------------------------------------------------------
    assert result["range_fft"].shape == (
        512,
        256,
        16,
    )

    # ---------------------------------------------------------
    # 10. Verify Range-Doppler spectrum dimensions
    # ---------------------------------------------------------
    assert result["rd_spectrum"].shape == (
        512,
        256,
        16,
    )

    # ---------------------------------------------------------
    # 11. Verify Range-Doppler power map
    # ---------------------------------------------------------
    assert result["rd_power"].shape == (
        512,
        256,
    )

    # ---------------------------------------------------------
    # 12. Verify CFAR detection matrix
    # ---------------------------------------------------------
    assert result["detections"].shape == (
        512,
        256,
    )

    assert result["detections"].dtype == bool

    # ---------------------------------------------------------
    # 13. Verify radar point representation
    # ---------------------------------------------------------
    assert radar_points.ndim == 2

    assert radar_points.shape[1] == 4

    # The point representation should contain:
    #
    # [range_bin, doppler_bin, range_m, velocity_mps]

    assert radar_points.dtype == "float32"