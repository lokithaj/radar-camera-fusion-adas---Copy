"""
Test end-to-end radar sparsification on real RADIal data across all research retention ratios:
1.0, 0.75, 0.50, 0.25, 0.10.
"""

import pytest
import numpy as np
from pathlib import Path

from config import (
    get_default_recording_dir,
    get_dbreader_dir,
    get_radar_calibration_path,
)
from fusion.src.sync_loader import SynchronizedFusionLoader
from fusion.src.radar_sparsifier import AdaptiveRadarSparsifier
from fusion.inference import process_radar

DATASET = get_default_recording_dir()
DBREADER = get_dbreader_dir(DATASET)
CALIBRATION = get_radar_calibration_path(DATASET)


@pytest.mark.skipif(
    not (DATASET.exists() and DBREADER.exists() and CALIBRATION.exists()),
    reason="RADIal dataset, DBReader, or calibration not found",
)
def test_sparsification_pipeline_real_sample():
    loader = SynchronizedFusionLoader(DATASET, DBREADER)
    data = loader.load(0)

    # 1. Baseline radar processing
    radar_objects = process_radar(data, CALIBRATION)
    original_radar_count = len(radar_objects)

    print(f"\nOriginal radar objects count: {original_radar_count}")
    assert original_radar_count > 0, "Expected non-zero radar objects from sample 0"

    # Test all 5 retention ratios
    test_ratios = [1.0, 0.75, 0.50, 0.25, 0.10]
    results = {}

    for ratio in test_ratios:
        sparsifier = AdaptiveRadarSparsifier(retention_ratio=ratio)
        selected_objects = sparsifier.select(radar_objects)
        selected_count = len(selected_objects)

        print(f"Ratio {ratio:4.2f} -> Radar objects: {original_radar_count} -> {selected_count}")

        # Structure checks
        if selected_count > 0:
            for obj in selected_objects:
                assert "cluster_id" in obj
                assert "x_m" in obj
                assert "y_m" in obj
                assert "z_m" in obj
                assert "range_m" in obj
                assert "velocity_mps" in obj
                assert "power" in obj
                assert "importance_score" in obj

        results[ratio] = selected_count

    # Baseline (1.0) must retain 100% of objects
    assert results[1.0] == original_radar_count
    # Smaller ratios must not exceed larger ratios
    assert results[0.10] <= results[0.25] <= results[0.50] <= results[0.75] <= results[1.0]
