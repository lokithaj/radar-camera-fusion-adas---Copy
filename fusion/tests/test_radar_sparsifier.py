"""
Tests for Stage 5 AdaptiveRadarSparsifier.
"""

import numpy as np
import pytest
from fusion.src.radar_sparsifier import AdaptiveRadarSparsifier


def create_sample_radar_objects():
    """Create sample radar objects mimicking RadarObjectExtractor output."""
    return [
        {
            "cluster_id": 0,
            "x_m": 11.44,
            "y_m": -0.02,
            "z_m": 0.1,
            "range_m": 11.44,
            "velocity_mps": -5.2,
            "azimuth_deg": -0.08,
            "elevation_deg": 0.5,
            "power": 1.5e7,
            "num_points": 57,
        },
        {
            "cluster_id": 1,
            "x_m": 8.79,
            "y_m": -3.29,
            "z_m": 0.05,
            "range_m": 9.39,
            "velocity_mps": 1.1,
            "azimuth_deg": -20.53,
            "elevation_deg": 0.3,
            "power": 8.0e6,
            "num_points": 35,
        },
        {
            "cluster_id": 2,
            "x_m": 9.34,
            "y_m": -10.52,
            "z_m": -0.2,
            "range_m": 14.08,
            "velocity_mps": 0.0,
            "azimuth_deg": -48.40,
            "elevation_deg": -0.8,
            "power": 2.1e5,
            "num_points": 5,
        },
        {
            "cluster_id": 3,
            "x_m": 25.0,
            "y_m": 2.0,
            "z_m": 0.0,
            "range_m": 25.08,
            "velocity_mps": -12.5,
            "azimuth_deg": 4.58,
            "elevation_deg": 0.0,
            "power": 5.0e6,
            "num_points": 20,
        },
    ]


def test_empty_objects():
    sparsifier = AdaptiveRadarSparsifier(retention_ratio=0.5)
    selected = sparsifier.select([])
    assert selected == []
    scores = sparsifier.compute_importance_scores([])
    assert scores == []


def test_single_object():
    sparsifier = AdaptiveRadarSparsifier(retention_ratio=0.5)
    obj = create_sample_radar_objects()[:1]
    selected = sparsifier.select(obj)
    assert len(selected) == 1
    assert selected[0]["cluster_id"] == 0


def test_retention_ratios():
    objects = create_sample_radar_objects()  # 4 objects
    assert len(objects) == 4

    # 1.0 -> 4
    sparsifier_100 = AdaptiveRadarSparsifier(retention_ratio=1.0)
    res_100 = sparsifier_100.select(objects)
    assert len(res_100) == 4

    # 0.75 -> 3
    sparsifier_75 = AdaptiveRadarSparsifier(retention_ratio=0.75)
    res_75 = sparsifier_75.select(objects)
    assert len(res_75) == 3

    # 0.50 -> 2
    sparsifier_50 = AdaptiveRadarSparsifier(retention_ratio=0.50)
    res_50 = sparsifier_50.select(objects)
    assert len(res_50) == 2

    # 0.25 -> 1
    sparsifier_25 = AdaptiveRadarSparsifier(retention_ratio=0.25)
    res_25 = sparsifier_25.select(objects)
    assert len(res_25) == 1

    # 0.10 -> 1 (at least 1 when ratio > 0 and objects exist)
    sparsifier_10 = AdaptiveRadarSparsifier(retention_ratio=0.10)
    res_10 = sparsifier_10.select(objects)
    assert len(res_10) == 1


def test_preserves_dictionary_structure():
    objects = create_sample_radar_objects()
    sparsifier = AdaptiveRadarSparsifier(retention_ratio=0.5)
    selected = sparsifier.select(objects)

    required_keys = {
        "cluster_id",
        "x_m",
        "y_m",
        "z_m",
        "range_m",
        "velocity_mps",
        "azimuth_deg",
        "elevation_deg",
        "power",
        "num_points",
    }
    for obj in selected:
        for k in required_keys:
            assert k in obj, f"Missing key {k}"
        assert "importance_score" in obj
        assert 0.0 <= obj["importance_score"] <= 1.0


def test_handles_nan_velocity():
    objects = [
        {
            "cluster_id": 0,
            "x_m": 10.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "range_m": 10.0,
            "velocity_mps": np.nan,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "power": 1e6,
            "num_points": 10,
        },
        {
            "cluster_id": 1,
            "x_m": 20.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "range_m": 20.0,
            "velocity_mps": 5.0,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "power": 1e6,
            "num_points": 10,
        },
    ]
    sparsifier = AdaptiveRadarSparsifier(retention_ratio=0.5)
    selected = sparsifier.select(objects)
    assert len(selected) == 1
    assert not np.isnan(selected[0]["importance_score"])


def test_invalid_retention_ratio():
    with pytest.raises(ValueError):
        AdaptiveRadarSparsifier(retention_ratio=-0.1)

    with pytest.raises(ValueError):
        AdaptiveRadarSparsifier(retention_ratio=1.5)
