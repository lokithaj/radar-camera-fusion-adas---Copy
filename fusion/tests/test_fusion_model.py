from pathlib import Path

from fusion.src.fusion_model import FusionMLModel


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "radar_camera_fusion.pkl"
)


def test_trained_fusion_model():

    model = FusionMLModel(
        MODEL_PATH
    )

    features = {
        "range_m": 11.44,
        "velocity_mps": -1.61,
        "azimuth_deg": -0.08,
        "elevation_deg": -1.56,
        "radar_x": 11.44,
        "radar_y": -0.02,
        "radar_z": -0.31,
        "radar_power": 1000.0,
        "projected_u": 948.0,
        "projected_v": 767.7,
        "bbox_center_u": 976.0,
        "bbox_center_v": 632.0,
        "bbox_width": 257.0,
        "bbox_height": 224.0,
        "camera_confidence": 0.89,
        "pixel_distance": 24.0,
        "inside_bbox": 0,
    }

    prediction = model.predict(
        features
    )

    probability = model.predict_probability(
        features
    )

    print(
        "\n===== TRAINED FUSION MODEL ====="
    )

    print(
        "Prediction:",
        "MATCH" if prediction == 1 else "NO MATCH",
    )

    print(
        f"MATCH probability: {probability:.3f}"
    )

    assert prediction in (
        0,
        1,
    )

    assert 0.0 <= probability <= 1.0