from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from config import (
    DEFAULT_TRAINING_DATA,
    DEFAULT_TEST_DATA,
    DEFAULT_GROUNDTRUTH_MODEL_PATH,
)

TRAIN_DATA_FILE = DEFAULT_TRAINING_DATA
TEST_DATA_FILE = DEFAULT_TEST_DATA
MODEL_FILE = DEFAULT_GROUNDTRUTH_MODEL_PATH

# 19 Approved ML Features (Leakage-free, scale-invariant relative alignment)
FEATURE_COLUMNS = [
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


def train_and_evaluate():
    print("================================================================")
    print("STAGE 3: GROUND-TRUTH RADAR-CAMERA FUSION MODEL TRAINING")
    print("================================================================")

    if not TRAIN_DATA_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA_FILE}")

    df_train = pd.read_csv(TRAIN_DATA_FILE)
    print(f"Loaded training data: {TRAIN_DATA_FILE.name}")
    print(f"Total training rows: {len(df_train)}")
    print(f"Training sequence(s): {df_train['sequence_name'].unique().tolist()}")

    # Check required columns
    required_cols = FEATURE_COLUMNS + ["is_match", "sample_index"]
    missing = [c for c in required_cols if c not in df_train.columns]
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")

    # Verify absence of leaked proxy features
    assert "pixel_distance" not in df_train.columns, "Data leakage: pixel_distance found in training data!"
    assert "inside_bbox" not in df_train.columns, "Data leakage: inside_bbox found in training data!"

    # Clean NaNs if any
    df_train = df_train.replace([float("inf"), float("-inf")], pd.NA)
    df_train = df_train.dropna(subset=FEATURE_COLUMNS + ["is_match"])

    X_train_full = df_train[FEATURE_COLUMNS]
    y_train_full = df_train["is_match"].astype(int)
    groups = df_train["sample_index"]

    n_match = int((y_train_full == 1).sum())
    n_nomatch = int((y_train_full == 0).sum())
    print(f"Class distribution: {n_match} MATCH ({n_match/len(df_train)*100:.2f}%), {n_nomatch} NO-MATCH ({n_nomatch/len(df_train)*100:.2f}%)")

    # ------------------------------------------------------------
    # Frame-Grouped Cross-Validation on Training Recording
    # ------------------------------------------------------------
    print("\n----------------------------------------------------------------")
    print("FRAME-GROUPED 5-FOLD CROSS-VALIDATION (Within Training Sequence)")
    print("Note: Validates frame-to-frame generalization; NOT a cross-sequence test.")
    print("----------------------------------------------------------------")

    gkf = GroupKFold(n_splits=5)
    cv_precisions, cv_recalls, cv_f1s, cv_accuracies, cv_specificities = [], [], [], [], []

    for fold, (t_idx, v_idx) in enumerate(gkf.split(X_train_full, y_train_full, groups=groups), 1):
        X_tr, y_tr = X_train_full.iloc[t_idx], y_train_full.iloc[t_idx]
        X_val, y_val = X_train_full.iloc[v_idx], y_train_full.iloc[v_idx]

        fold_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        fold_model.fit(X_tr, y_tr)
        val_pred = fold_model.predict(X_val)

        acc = accuracy_score(y_val, val_pred)
        prec = precision_score(y_val, val_pred, zero_division=0)
        rec = recall_score(y_val, val_pred, zero_division=0)
        f1 = f1_score(y_val, val_pred, zero_division=0)

        cm = confusion_matrix(y_val, val_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        cv_accuracies.append(acc)
        cv_precisions.append(prec)
        cv_recalls.append(rec)
        cv_f1s.append(f1)
        cv_specificities.append(spec)

        print(f"Fold {fold}: Val Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, Spec={spec:.4f} (TP={tp}, FP={fp}, FN={fn}, TN={tn})")

    print(f"\nMean CV Accuracy:    {np.mean(cv_accuracies):.4f} +/- {np.std(cv_accuracies):.4f}")
    print(f"Mean CV Precision:   {np.mean(cv_precisions):.4f} +/- {np.std(cv_precisions):.4f}")
    print(f"Mean CV Recall:      {np.mean(cv_recalls):.4f} +/- {np.std(cv_recalls):.4f}")
    print(f"Mean CV F1:          {np.mean(cv_f1s):.4f} +/- {np.std(cv_f1s):.4f}")
    print(f"Mean CV Specificity: {np.mean(cv_specificities):.4f} +/- {np.std(cv_specificities):.4f}")

    # ------------------------------------------------------------
    # Final Model Training (on full Training Recording)
    # ------------------------------------------------------------
    print("\n----------------------------------------------------------------")
    print("TRAINING FINAL MODEL ON ALL TRAINING FRAMES")
    print("----------------------------------------------------------------")

    rf_config = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    print("Random Forest Hyperparameters:")
    for k, v in rf_config.items():
        print(f"  - {k}: {v}")

    final_model = RandomForestClassifier(**rf_config)
    final_model.fit(X_train_full, y_train_full)

    # Feature Importance
    importances = pd.Series(
        final_model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    print("\nFeature Importances:")
    for feat, imp in importances.items():
        print(f"  {feat:<22} : {imp:.4f}")

    # ------------------------------------------------------------
    # Save Final Model
    # ------------------------------------------------------------
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "features": FEATURE_COLUMNS,
            "config": rf_config,
        },
        MODEL_FILE,
    )
    print(f"\nModel bundle saved separately to:\n  {MODEL_FILE}")

    # ------------------------------------------------------------
    # Evaluation on Completely Unseen Held-Out Test Recording
    # ------------------------------------------------------------
    print("\n================================================================")
    print("EVALUATION ON HELD-OUT TEST RECORDING (RECORD@2020-11-22_12.24.44)")
    print("================================================================")

    if not TEST_DATA_FILE.exists():
        print(f"Test data file not found: {TEST_DATA_FILE}. Skipping test recording evaluation.")
        return

    df_test = pd.read_csv(TEST_DATA_FILE)
    print(f"Loaded test data: {TEST_DATA_FILE.name}")
    print(f"Test recording: {df_test['sequence_name'].unique().tolist()}")
    print(f"Total test rows: {len(df_test)}")

    df_test = df_test.replace([float("inf"), float("-inf")], pd.NA)
    df_test = df_test.dropna(subset=FEATURE_COLUMNS + ["is_match"])

    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test["is_match"].astype(int)

    n_test_pos = int((y_test == 1).sum())
    n_test_neg = int((y_test == 0).sum())
    print(f"Ground-truth positive pairs in test set: {n_test_pos}")
    print(f"Ground-truth negative pairs in test set: {n_test_neg}")

    test_preds = final_model.predict(X_test)
    n_pred_match = int((test_preds == 1).sum())
    n_pred_nomatch = int((test_preds == 0).sum())

    test_acc = accuracy_score(y_test, test_preds)
    cm_test = confusion_matrix(y_test, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm_test.ravel()

    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\nPredicted MATCH:    {n_pred_match}")
    print(f"Predicted NO-MATCH: {n_pred_nomatch}")
    print(f"\nConfusion Matrix (labels: [0: NO-MATCH, 1: MATCH]):")
    print(f"  [[TN={tn}, FP={fp}],")
    print(f"   [FN={fn}, TP={tp}]]")

    print(f"\nAccuracy:    {test_acc:.4f}")
    print(f"Specificity: {test_spec:.4f} (TN / (TN + FP) = {tn}/{tn + fp})")

    if n_test_pos == 0:
        print("\nIMPORTANT NOTE ON TEST EVALUATION METRICS:")
        print("  - The held-out test recording contains 0 positive ground-truth matches.")
        print("  - Recall and F1 for the positive class are UNDEFINED (0/0).")
        print("  - Precision for positive predictions is 0.0 if FP > 0, or undefined if no positives predicted.")
        print("  - Accuracy (correct negative rejections / total negatives) equals Specificity.")
        print("  - This test demonstrates FALSE-ALARM REJECTION (specificity) under an unseen sequence,")
        print("    but cannot establish true-positive detection performance.")
    else:
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        print(f"Precision:   {test_prec:.4f}")
        print(f"Recall:      {test_rec:.4f}")
        print(f"F1 Score:    {test_f1:.4f}")


if __name__ == "__main__":
    train_and_evaluate()