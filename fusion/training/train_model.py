from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "fusion"
    / "training"
    / "fusion_training_data.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "fusion"
    / "models"
    / "radar_camera_fusion.pkl"
)


# ============================================================
# Feature columns
# ============================================================

FEATURE_COLUMNS = [
    "range_m",
    "velocity_mps",
    "azimuth_deg",
    "elevation_deg",
    "radar_x",
    "radar_y",
    "radar_z",
    "radar_power",
    "projected_u",
    "projected_v",
    "bbox_center_u",
    "bbox_center_v",
    "bbox_width",
    "bbox_height",
    "camera_confidence",
    "pixel_distance",
    "inside_bbox",
]


def main():

    print("Loading training data...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "Total rows:",
        len(df)
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [
            "is_match",
            "sample_index",
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Remove invalid feature values
    # --------------------------------------------------------

    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    before = len(df)

    df = df.dropna(
        subset=FEATURE_COLUMNS
        + ["is_match"]
    )

    print(
        "Rows after cleaning:",
        len(df),
        "(removed",
        before - len(df),
        ")"
    )

    if df["is_match"].nunique() < 2:
        raise ValueError(
            "Training data contains only one class."
        )

    # --------------------------------------------------------
    # Features and labels
    # --------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "is_match"
    ].astype(int)

    groups = df[
        "sample_index"
    ]

    # --------------------------------------------------------
    # Split by synchronized sample
    #
    # This prevents pairs from the same frame appearing
    # in both train and test sets.
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]

    print(
        "\nTraining rows:",
        len(X_train)
    )

    print(
        "Testing rows:",
        len(X_test)
    )

    print(
        "Training MATCH:",
        int((y_train == 1).sum())
    )

    print(
        "Training NO-MATCH:",
        int((y_train == 0).sum())
    )

    print(
        "Testing MATCH:",
        int((y_test == 1).sum())
    )

    print(
        "Testing NO-MATCH:",
        int((y_test == 0).sum())
    )

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(
        "\n===== FUSION MODEL RESULTS ====="
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "NO-MATCH",
                "MATCH",
                ],
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(
        ascending=False
    )

    print(
        "\nFeature importance:"
    )

    print(
        importance.to_string()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
        },
        MODEL_FILE,
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_FILE
    )


if __name__ == "__main__":
    main()