"""
End-to-End Integration Tests for Stage 4: Tracking, TTC, and ADAS Decision Module
on real synchronized RADIal recordings (training and unseen test sequences).
"""

from pathlib import Path
import pytest
import numpy as np

from config import (
    DEFAULT_GROUNDTRUTH_MODEL_PATH,
    DEFAULT_YOLO_MODEL,
    get_default_recording_dir,
    get_test_recording_dir,
    get_dbreader_dir,
    get_radar_calibration_path,
    get_camera_calibration_path,
)
from fusion.src.sync_loader import SynchronizedFusionLoader
from fusion.src.radar_camera_projector import RadarCameraProjector
from fusion.src.stage4_pipeline import Stage4ADASPipeline
from fusion.src.adas_decision import ThreatLevel
from camera.src.camera_detector import CameraDetector

from radar.src.radar_frame import RadarFrameBuilder
from radar.src.fft_processor import RadarFFTProcessor
from radar.src.cfar import CACFAR
from radar.src.real_point_generator import RealRadarPointGenerator
from radar.src.dbscan_clusterer import RadarDBSCAN
from radar.src.radar_objects import RadarObjectExtractor


def run_dsp_and_project(data, radar_calib, projector, image_w, image_h):
    """Helper to run existing unmodified radar pipeline and projection."""
    frame_builder = RadarFrameBuilder()
    complex_frame = frame_builder.build_frame(
        data["radar_ch0"],
        data["radar_ch1"],
        data["radar_ch2"],
        data["radar_ch3"],
    )
    complex_frame = complex_frame - np.mean(complex_frame, axis=(0, 1), keepdims=True)

    fft_processor = RadarFFTProcessor()
    range_fft = fft_processor.range_fft(complex_frame)
    rd_spectrum = fft_processor.doppler_fft(range_fft)
    rd_power = np.sum(np.abs(rd_spectrum) ** 2, axis=2)

    cfar = CACFAR(window=(9, 9), guard=(3, 3), threshold_db=2.0)
    detections = cfar.detect(rd_power)

    point_generator = RealRadarPointGenerator(
        radar_calib,
        max_points=20,
        min_range_bin=10,
    )
    radar_points = point_generator.generate(rd_spectrum, detections)

    clusterer = RadarDBSCAN(eps=1.5, min_samples=3)
    labels = clusterer.cluster(radar_points)

    extractor = RadarObjectExtractor()
    radar_objects = extractor.extract(radar_points, labels)

    if radar_objects:
        radar_xyz = np.array(
            [[o["x_m"], o["y_m"], o["z_m"]] for o in radar_objects],
            dtype=np.float64,
        )
        pixels = projector.project(radar_xyz)
        _, valid = projector.project_valid(radar_xyz, image_w, image_h)
    else:
        pixels = np.empty((0, 2), dtype=np.float64)
        valid = np.empty((0,), dtype=bool)

    return radar_objects, pixels, valid


@pytest.mark.skipif(
    not DEFAULT_GROUNDTRUTH_MODEL_PATH.exists(),
    reason="Stage 3 ground-truth ML model not found",
)
def test_stage4_pipeline_training_recording():
    """Verify Stage 4 pipeline executes on default training recording sequence."""
    rec_path = get_default_recording_dir()
    dbreader = get_dbreader_dir(rec_path)
    radar_calib = get_radar_calibration_path(rec_path)
    cam_calib = get_camera_calibration_path(rec_path)

    if not (rec_path.exists() and dbreader.exists() and radar_calib.exists() and cam_calib.exists()):
        pytest.skip("Dataset or calibration files missing on machine")

    loader = SynchronizedFusionLoader(rec_path, dbreader)
    projector = RadarCameraProjector(cam_calib)
    detector = CameraDetector(
        model_name=str(DEFAULT_YOLO_MODEL) if DEFAULT_YOLO_MODEL.exists() else "yolo11n.pt",
        confidence=0.4,
    )
    pipeline = Stage4ADASPipeline(fusion_model_path=DEFAULT_GROUNDTRUTH_MODEL_PATH)

    # Process 3 sequential frames (0, 5, 10)
    prev_track_ids = set()
    for frame_idx in [0, 5, 10]:
        data = loader.load(frame_idx)
        image = data["camera"]
        h, w, _ = image.shape

        _, camera_objects = detector.detect(image)
        radar_objects, pixels, valid = run_dsp_and_project(data, radar_calib, projector, w, h)

        res = pipeline.process_frame(
            radar_objects,
            pixels,
            valid,
            camera_objects,
            dt=0.5,
        )

        assert "fused_objects" in res
        assert "tracks" in res
        assert "confirmed_tracks" in res
        assert "adas_alert" in res

        alert = res["adas_alert"]
        assert isinstance(alert.system_threat_level, ThreatLevel)
        assert alert.total_active_tracks == len(res["tracks"])

    # At least some tracks should be active
    assert len(res["tracks"]) > 0


@pytest.mark.skipif(
    not DEFAULT_GROUNDTRUTH_MODEL_PATH.exists(),
    reason="Stage 3 ground-truth ML model not found",
)
def test_stage4_pipeline_unseen_test_recording():
    """Verify Stage 4 pipeline executes on unseen RADIal test recording sequence."""
    test_rec = get_test_recording_dir()
    dbreader = get_dbreader_dir(test_rec)
    radar_calib = get_radar_calibration_path(test_rec)
    cam_calib = get_camera_calibration_path(test_rec)

    if not (test_rec.exists() and dbreader.exists() and radar_calib.exists() and cam_calib.exists()):
        pytest.skip("Test recording or calibration files missing on machine")

    loader = SynchronizedFusionLoader(test_rec, dbreader)
    projector = RadarCameraProjector(cam_calib)
    detector = CameraDetector(
        model_name=str(DEFAULT_YOLO_MODEL) if DEFAULT_YOLO_MODEL.exists() else "yolo11n.pt",
        confidence=0.4,
    )
    pipeline = Stage4ADASPipeline(fusion_model_path=DEFAULT_GROUNDTRUTH_MODEL_PATH)

    # Run frame 0 on unseen sequence
    data = loader.load(0)
    image = data["camera"]
    h, w, _ = image.shape

    _, camera_objects = detector.detect(image)
    radar_objects, pixels, valid = run_dsp_and_project(data, radar_calib, projector, w, h)

    res = pipeline.process_frame(
        radar_objects,
        pixels,
        valid,
        camera_objects,
        dt=0.2,
    )

    assert "tracks" in res
    assert "adas_alert" in res
    assert isinstance(res["adas_alert"].system_threat_level, ThreatLevel)


@pytest.mark.skipif(
    not DEFAULT_GROUNDTRUTH_MODEL_PATH.exists(),
    reason="Stage 3 ground-truth ML model not found",
)
def test_stage4_one_to_one_association():
    """
    Verify that associate_stage3 strictly enforces one-to-one matching:
    even if multiple radar detections produce MATCH predictions for the same camera detection,
    at most one radar detection is assigned to that camera detection.
    """
    pipeline = Stage4ADASPipeline(fusion_model_path=DEFAULT_GROUNDTRUTH_MODEL_PATH)

    # 1 camera detection
    camera_objects = [
        {
            "bbox": [900.0, 600.0, 1100.0, 800.0],
            "class_name": "car",
            "confidence": 0.90,
        }
    ]

    # 2 radar objects at similar forward distance
    radar_objects = [
        {
            "range_m": 11.4,
            "velocity_mps": -2.5,
            "azimuth_deg": 0.0,
            "elevation_deg": -1.0,
            "power": 120000.0,
            "x_m": 11.4,
            "y_m": 0.0,
            "z_m": -0.2,
        },
        {
            "range_m": 11.6,
            "velocity_mps": -2.4,
            "azimuth_deg": 0.2,
            "elevation_deg": -1.0,
            "power": 110000.0,
            "x_m": 11.6,
            "y_m": 0.1,
            "z_m": -0.2,
        },
    ]

    # Projected pixels near the camera bbox center (1000, 700)
    projected_pixels = np.array([
        [1005.0, 705.0],
        [995.0, 695.0],
    ], dtype=np.float64)
    valid_mask = np.array([True, True], dtype=bool)

    fused = pipeline.associate_stage3(
        radar_objects,
        projected_pixels,
        valid_mask,
        camera_objects,
    )

    # Camera object index 0 must appear at most ONCE in fused output
    matched_cam_indices = [f["camera_index"] for f in fused]
    assert len(matched_cam_indices) == len(set(matched_cam_indices))
    assert len(matched_cam_indices) <= 1


def test_stage4_dt_configuration():
    """Verify that Stage4ADASPipeline uses 0.2s default dt and can be configured."""
    pipeline = Stage4ADASPipeline(default_dt=0.2)
    assert pipeline.default_dt == 0.2

    # Process empty frame without dt argument uses default_dt
    res = pipeline.process_frame([], np.empty((0, 2)), np.empty((0,), dtype=bool), [])
    assert res["dt"] == 0.2

    # Providing explicit dt
    res_custom = pipeline.process_frame([], np.empty((0, 2)), np.empty((0,), dtype=bool), [], dt=0.5)
    assert res_custom["dt"] == 0.5
