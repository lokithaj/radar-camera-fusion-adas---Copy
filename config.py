"""
Central configuration and path resolution for the Radar-Camera Fusion ADAS project.

Provides:
- Project root and project-relative model/data/output paths.
- Configurable environment variable overrides.
- Dynamic discovery for RADIal toolkit (DBReader, calibration files) and dataset sequences.
"""

from pathlib import Path
import os

# Project root (directory containing this config.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# Project-relative assets
DEFAULT_MODEL_PATH = PROJECT_ROOT / "fusion" / "models" / "radar_camera_fusion.pkl"
DEFAULT_GROUNDTRUTH_MODEL_PATH = PROJECT_ROOT / "fusion" / "models" / "radar_camera_fusion_groundtruth.pkl"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "yolo11n.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_TRAINING_DATA = PROJECT_ROOT / "fusion" / "training" / "fusion_training_data.csv"
DEFAULT_TEST_DATA = PROJECT_ROOT / "fusion" / "training" / "fusion_test_data.csv"
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "fusion" / "training" / "ground_truth.csv"


def get_radial_root():
    """Locate RADIal repository root directory."""
    env_val = os.environ.get("RADIAL_ROOT")
    if env_val and Path(env_val).exists():
        return Path(env_val).resolve()

    candidates = [
        PROJECT_ROOT.parent / "RADIal",
        Path.home() / "Desktop" / "RADIal",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[1]


def get_dbreader_dir(recording_dir=None, dbreader_dir=None):
    """
    Locate official RADIal DBReader directory.

    Order of precedence:
    1. Explicit dbreader_dir argument
    2. RADIAL_DBREADER_DIR environment variable
    3. Proximity to the provided recording_dir
    4. RADIAL_ROOT / "DBReader"
    5. Fallback location
    """
    if dbreader_dir:
        path = Path(dbreader_dir).resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified DBReader directory not found: {path}")

    env_val = os.environ.get("RADIAL_DBREADER_DIR")
    if env_val and Path(env_val).exists():
        return Path(env_val).resolve()

    if recording_dir:
        rec = Path(recording_dir).resolve()
        candidates = [
            rec / "DBReader",
            rec.parent / "DBReader",
            rec.parent / "RADIal" / "DBReader",
        ]
        if len(rec.parents) > 1:
            candidates.append(rec.parents[1] / "RADIal" / "DBReader")

        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate.resolve()

    radial_root = get_radial_root()
    dbreader = radial_root / "DBReader"
    if dbreader.exists():
        return dbreader.resolve()

    return Path.home() / "Desktop" / "RADIal" / "DBReader"


def get_radar_calibration_path(recording_dir=None, calib_path=None):
    """
    Locate RADIal radar calibration table (CalibrationTable.npy).

    Order of precedence:
    1. Explicit calib_path argument
    2. RADIAL_RADAR_CALIB environment variable
    3. Proximity to recording_dir or RADIal root
    4. Fallback location
    """
    if calib_path:
        path = Path(calib_path).resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified radar calibration file not found: {path}")

    env_val = os.environ.get("RADIAL_RADAR_CALIB")
    if env_val and Path(env_val).exists():
        return Path(env_val).resolve()

    if recording_dir:
        rec = Path(recording_dir).resolve()
        candidates = [
            rec / "CalibrationTable.npy",
            rec.parent / "CalibrationTable.npy",
            rec.parent / "SignalProcessing" / "CalibrationTable.npy",
        ]
        if len(rec.parents) > 1:
            candidates.append(rec.parents[1] / "RADIal" / "SignalProcessing" / "CalibrationTable.npy")

        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate.resolve()

    radial_root = get_radial_root()
    calib = radial_root / "SignalProcessing" / "CalibrationTable.npy"
    if calib.exists():
        return calib.resolve()

    return Path.home() / "Desktop" / "RADIal" / "SignalProcessing" / "CalibrationTable.npy"


def get_camera_calibration_path(recording_dir=None, calib_path=None):
    """
    Locate RADIal camera calibration (camera_calib.npy).

    Order of precedence:
    1. Explicit calib_path argument
    2. RADIAL_CAMERA_CALIB environment variable
    3. Proximity to recording_dir or RADIal root
    4. Fallback location
    """
    if calib_path:
        path = Path(calib_path).resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified camera calibration file not found: {path}")

    env_val = os.environ.get("RADIAL_CAMERA_CALIB")
    if env_val and Path(env_val).exists():
        return Path(env_val).resolve()

    if recording_dir:
        rec = Path(recording_dir).resolve()
        candidates = [
            rec / "camera_calib.npy",
            rec.parent / "camera_calib.npy",
            rec.parent / "DBReader" / "examples" / "camera_calib.npy",
        ]
        if len(rec.parents) > 1:
            candidates.append(rec.parents[1] / "RADIal" / "DBReader" / "examples" / "camera_calib.npy")

        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate.resolve()

    radial_root = get_radial_root()
    calib = radial_root / "DBReader" / "examples" / "camera_calib.npy"
    if calib.exists():
        return calib.resolve()

    return Path.home() / "Desktop" / "RADIal" / "DBReader" / "examples" / "camera_calib.npy"


def get_default_recording_dir():
    """Locate default recording sequence for demos/tests."""
    env_val = os.environ.get("RADIAL_RECORDING")
    if env_val and Path(env_val).exists():
        return Path(env_val).resolve()

    candidates = [
        PROJECT_ROOT.parent / "RADIal_data" / "RECORD@2020-11-21_13.44.44",
        Path.home() / "Desktop" / "RADIal_data" / "RECORD@2020-11-21_13.44.44",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[1]


def get_test_recording_dir():
    """Locate unseen test recording sequence for Stage 3 evaluation."""
    env_val = os.environ.get("RADIAL_TEST_RECORDING")
    if env_val and Path(env_val).exists():
        return Path(env_val).resolve()

    candidates = [
        PROJECT_ROOT.parent / "RADIal_data" / "RECORD@2020-11-22_12.24.44",
        Path.home() / "Desktop" / "RADIal_data" / "RECORD@2020-11-22_12.24.44",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[1]


def find_video_file(recording_dir):
    """Automatically find preview or camera video inside a recording folder."""
    recording_dir = Path(recording_dir)
    if not recording_dir.exists():
        raise FileNotFoundError(f"Recording directory does not exist: {recording_dir}")

    # Look for preview or avi/mp4 video files
    preview_files = sorted(recording_dir.glob("*preview.avi"))
    if preview_files:
        return preview_files[0].resolve()

    avi_files = sorted(recording_dir.glob("*.avi"))
    if avi_files:
        return avi_files[0].resolve()

    mp4_files = sorted(recording_dir.glob("*.mp4"))
    if mp4_files:
        return mp4_files[0].resolve()

    raise FileNotFoundError(
        f"No video file (*preview.avi, *.avi, *.mp4) found in recording directory: {recording_dir}"
    )
