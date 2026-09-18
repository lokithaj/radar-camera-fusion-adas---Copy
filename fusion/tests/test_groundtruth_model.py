from pathlib import Path
import pandas as pd
import pytest

from fusion.src.fusion_model import FusionMLModel
from config import (
    DEFAULT_GROUNDTRUTH_MODEL_PATH,
    DEFAULT_TRAINING_DATA,
    DEFAULT_TEST_DATA,
)


def test_groundtruth_model_file_exists():
    """Verify that the Stage 3 ground-truth model was created without overwriting the old model."""
    assert DEFAULT_GROUNDTRUTH_MODEL_PATH.exists()
    # Check old model also remains intact
    old_model_path = DEFAULT_GROUNDTRUTH_MODEL_PATH.parent / "radar_camera_fusion.pkl"
    assert old_model_path.exists()


def test_groundtruth_model_features_and_loading():
    """Verify model loading and confirm exact 19 approved features without leakage."""
    model = FusionMLModel(DEFAULT_GROUNDTRUTH_MODEL_PATH)

    expected_features = [
        "range_m",
        "velocity_mps",
        "azimuth_deg",
        "elevation_deg",
        "radar_power",
        "radar_x",
        "radar_y",
        "radar_z",
        "projected_u",
        "projected_v",
        "bbox_center_u",
        "bbox_center_v",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "camera_confidence",
        "norm_offset_u",
        "norm_offset_v",
        "norm_radial_dist",
    ]

    assert len(model.features) == 19
    assert list(model.features) == expected_features
    assert "pixel_distance" not in model.features
    assert "inside_bbox" not in model.features


def test_groundtruth_model_prediction():
    """Verify inference output format and range."""
    model = FusionMLModel(DEFAULT_GROUNDTRUTH_MODEL_PATH)

    # Typical well-aligned candidate pair features
    sample_features = {
        "range_m": 11.41,
        "velocity_mps": -0.26,
        "azimuth_deg": 0.1,
        "elevation_deg": -1.5,
        "radar_power": 1200.0,
        "radar_x": 11.41,
        "radar_y": 0.02,
        "radar_z": -0.3,
        "projected_u": 942.6,
        "projected_v": 767.3,
        "bbox_center_u": 976.5,
        "bbox_center_v": 632.5,
        "bbox_width": 257.0,
        "bbox_height": 224.0,
        "bbox_aspect_ratio": 1.15,
        "camera_confidence": 0.89,
        "norm_offset_u": -0.13,
        "norm_offset_v": 0.60,
        "norm_radial_dist": 0.61,
    }

    pred = model.predict(sample_features)
    prob = model.predict_probability(sample_features)

    assert pred in (0, 1)
    assert 0.0 <= prob <= 1.0


def test_groundtruth_model_scoring_test_dataset():
    """Verify scoring on unseen test recording dataset."""
    assert DEFAULT_TEST_DATA.exists()
    df_test = pd.read_csv(DEFAULT_TEST_DATA)

    model = FusionMLModel(DEFAULT_GROUNDTRUTH_MODEL_PATH)
    assert len(df_test) == 38

    predictions = []
    probabilities = []
    for _, row in df_test.iterrows():
        feat = {col: row[col] for col in model.features}
        pred = model.predict(feat)
        prob = model.predict_probability(feat)
        predictions.append(pred)
        probabilities.append(prob)

    assert len(predictions) == 38
    assert all(p in (0, 1) for p in predictions)
    assert all(0.0 <= pr <= 1.0 for pr in probabilities)
