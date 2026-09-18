import os
import pytest
from pathlib import Path

from config import (
    PROJECT_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_GROUNDTRUTH_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAINING_DATA,
    DEFAULT_TEST_DATA,
    DEFAULT_GROUND_TRUTH,
    get_radial_root,
    get_dbreader_dir,
    get_radar_calibration_path,
    get_camera_calibration_path,
    get_default_recording_dir,
    get_test_recording_dir,
    find_video_file,
)
from fusion.inference import run as run_inference


def test_project_root_and_defaults():
    """Verify that project paths are rooted within the repository."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "config.py").exists()

    # Model and data paths must be inside PROJECT_ROOT
    assert PROJECT_ROOT in DEFAULT_MODEL_PATH.parents
    assert PROJECT_ROOT in DEFAULT_GROUNDTRUTH_MODEL_PATH.parents
    assert PROJECT_ROOT in DEFAULT_OUTPUT_DIR.parents or DEFAULT_OUTPUT_DIR.parent == PROJECT_ROOT
    assert PROJECT_ROOT in DEFAULT_TRAINING_DATA.parents
    assert PROJECT_ROOT in DEFAULT_TEST_DATA.parents
    assert PROJECT_ROOT in DEFAULT_GROUND_TRUTH.parents


def test_env_overrides(tmp_path, monkeypatch):
    """Verify that environment variables take precedence in path resolution."""
    fake_radial = tmp_path / "custom_radial"
    fake_radial.mkdir()
    monkeypatch.setenv("RADIAL_ROOT", str(fake_radial))
    assert get_radial_root() == fake_radial.resolve()

    fake_dbreader = tmp_path / "custom_dbreader"
    fake_dbreader.mkdir()
    monkeypatch.setenv("RADIAL_DBREADER_DIR", str(fake_dbreader))
    assert get_dbreader_dir() == fake_dbreader.resolve()

    fake_radar_calib = tmp_path / "custom_calib.npy"
    fake_radar_calib.touch()
    monkeypatch.setenv("RADIAL_RADAR_CALIB", str(fake_radar_calib))
    assert get_radar_calibration_path() == fake_radar_calib.resolve()

    fake_cam_calib = tmp_path / "custom_cam.npy"
    fake_cam_calib.touch()
    monkeypatch.setenv("RADIAL_CAMERA_CALIB", str(fake_cam_calib))
    assert get_camera_calibration_path() == fake_cam_calib.resolve()

    fake_rec = tmp_path / "custom_rec"
    fake_rec.mkdir()
    monkeypatch.setenv("RADIAL_RECORDING", str(fake_rec))
    assert get_default_recording_dir() == fake_rec.resolve()

    fake_test_rec = tmp_path / "custom_test_rec"
    fake_test_rec.mkdir()
    monkeypatch.setenv("RADIAL_TEST_RECORDING", str(fake_test_rec))
    assert get_test_recording_dir() == fake_test_rec.resolve()


def test_find_video_file(tmp_path):
    """Verify video discovery logic."""
    rec_dir = tmp_path / "test_recording"
    rec_dir.mkdir()

    # Missing video should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        find_video_file(rec_dir)

    # Adding an AVI video should be discovered
    video_file = rec_dir / "preview_video.avi"
    video_file.touch()
    assert find_video_file(rec_dir) == video_file.resolve()


def test_inference_invalid_recording_error(tmp_path):
    """Verify that missing/invalid recording paths produce useful errors."""
    non_existent = tmp_path / "non_existent_recording_path"
    with pytest.raises(FileNotFoundError, match="Recording not found"):
        run_inference(non_existent)

    # Path exists but is a file instead of directory
    file_path = tmp_path / "not_a_dir.txt"
    file_path.touch()
    with pytest.raises(NotADirectoryError, match="Recording path is not a directory"):
        run_inference(file_path)
