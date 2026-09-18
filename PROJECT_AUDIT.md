# Project Audit: Radar-Camera Fusion ADAS

**Audit Date:** September 16, 2026  
**Repository Baseline:** `radar-camera-fusion-adas` (Personal Development Copy)  
**Dataset Grounding:** RADIal Dataset (Valeo HD Radar + Camera)

---

## Executive Summary

The project provides an operational baseline for multi-modal radar-camera perception utilizing real recordings from the RADIal dataset. It integrates:
- Low-level FMCW radar ADC signal processing (I/Q frame construction, range/Doppler FFTs, 2D CA-CFAR, 192-virtual antenna MIMO angular super-resolution, and DBSCAN clustering).
- YOLOv11-based camera 2D object detection.
- Camera-radar spatial projection via calibrated intrinsic/extrinsic transformation matrices.
- Spatial bounding-box distance gating and a trained Random Forest binary classifier for radar-camera object association.
- Live visualization playback scripts.

However, the audit identified **critical architectural flaws, active runtime test failures, severe circular data leakage in ML training, and hardcoded user-specific path dependencies** that undermine reproducibility, evaluation validity, and deployment safety.

---

## 1. System Architecture Overview

```
                          ┌──────────────────────────┐
                          │  RADIal Dataset Storage  │
                          │  (Raw ADC, Camera Video) │
                          └─────────────┬────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌────────────────────────┐                   ┌────────────────────────┐
     │  Radar Sensor Branch   │                   │  Camera Sensor Branch  │
     └───────────┬────────────┘                   └───────────┬────────────┘
                 │ Raw 4-chip ADC (int16)                     │ Camera Frames
                 ▼                                             ▼
     ┌────────────────────────┐                   ┌────────────────────────┐
     │   RadarFrameBuilder    │                   │     CameraDetector     │
     │   (16 RX, 512x256)     │                   │       (YOLO11n)        │
     └───────────┬────────────┘                   └───────────┬────────────┘
                 │ Complex Frame                              │ 2D Bounding Boxes
                 ▼                                             │ [x1, y1, x2, y2]
     ┌────────────────────────┐                                │
     │   RadarFFTProcessor    │                                │
     │  (Range & Doppler FFT) │                                │
     └───────────┬────────────┘                                │
                 │ 2D Power Map                                │
                 ▼                                             │
     ┌────────────────────────┐                                │
     │     CACFAR (2D)        │                                │
     │ (Peak Power Detection) │                                │
     └───────────┬────────────┘                                │
                 │ Peak Detections                             │
                 ▼                                             │
     ┌────────────────────────┐                                │
     │ RealRadarPointGenerator│                                │
     │  (MIMO 192-Ch + AoA)   │                                │
     └───────────┬────────────┘                                │
                 │ 3D Points [X, Y, Z, Vr, P]                  │
                 ▼                                             │
     ┌────────────────────────┐                                │
     │ RadarDBSCAN + Extractor│                                │
     │   (Radar 3D Objects)   │                                │
     └───────────┬────────────┘                                │
                 │ 3D Radar Object Centroids                   │
                 ▼                                             │
     ┌─────────────────────────────────────────────────────────┴──────────┐
     │                      RadarCameraProjector                          │
     │      (Coordinate Swap [-Y, X, Z] + cv2.projectPoints + K, R, T)    │
     └────────────────────────────────────┬───────────────────────────────┘
                                          │ Projected 2D Points (u, v)
                                          ▼
     ┌────────────────────────────────────────────────────────────────────┐
     │                     Association & ML Fusion Layer                  │
     │  1. Spatial BBox Distance Gating (RadarCameraAssociator)           │
     │  2. Random Forest Pair Classifier (FusionMLModel)                  │
     └────────────────────────────────────┬───────────────────────────────┘
                                          │ Fused Radar-Camera Tracks
                                          ▼
     ┌────────────────────────────────────────────────────────────────────┐
     │                       Visualization Layer                          │
     │       (cv2.imshow overlay: Green=Camera, Red=Radar, Cyan=Fused)    │
     └────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component-by-Component Audit

### 2.1 Directory Structure
- **Layout:**
  - `camera/`: Contains detector implementation (`camera_detector.py`), manual video playback runner, and an **empty** `tests/` directory.
  - `radar/`: Contains radar signal processing modules and 14 unit/integration tests in `radar/tests/`.
  - `fusion/`: Contains spatial projection, association, ML model loader, training scripts, CSV datasets, trained model weights (`radar_camera_fusion.pkl`), and 5 tests in `fusion/tests/`.
  - Root: `demo.py`, `requirements.txt`, `yolo11n.pt`, and `README.md`.
- **Gaps:**
  - Missing top-level standard directories: `config/`, `docs/`, `scripts/`.
  - Incomplete separation between library code (`src/`) and executables/runners (standalone scripts placed directly under package directories).

### 2.2 Radar Processing Pipeline
- **Modules:** `radial_loader.py`, `radar_frame.py`, `fft_processor.py`, `cfar.py`, `mimo_reconstructor.py`, `angle_estimator.py`, `velocity_converter.py`, `coordinate_converter.py`, `radar_points.py`, `real_point_generator.py`, `dbscan_clusterer.py`, `radar_objects.py`, `radar_pipeline.py`, `radar_processor.py`.
- **What is Working:**
  - Raw 16-RX complex frame assembly (`(512, 256, 16)`) is mathematically correct and matches RADIal hardware specifications.
  - Range and Doppler FFTs produce clean Range-Doppler spectra.
  - MIMO reconstruction correctly extracts virtual antenna Doppler offsets for 192 virtual channels.
  - Angle-of-Arrival (AoA) estimation via the RADIal calibration table generates plausible azimuth and elevation angles.
  - DBSCAN effectively clusters dispersed radar returns into centroid objects with physical coordinates ($X, Y, Z$, Range, Radial Velocity).
- **Flaws & Inconsistencies:**
  - **Critical Constructor Bug:** `RadarPipeline` in `radar/src/radar_pipeline.py` attempts to initialize `RadarPointExtractor(range_resolution=range_resolution, velocity_resolution=velocity_resolution)`. However, `RadarPointExtractor.__init__` only accepts `samples_per_chirp` and `max_range_m`. This causes immediate `TypeError` exceptions.
  - **Window Function Misnomer:** `fft_processor.py` comments state "Hann window", but implements a Hamming window (`0.54 - 0.46 * cos(...)`).
  - **Performance Bottleneck:** `CACFAR` uses `scipy.signal.convolve2d(power, mask, mode='same')` with a $19 \times 19$ kernel across $512 \times 256$ matrices, taking 50–150 ms per frame on CPU.
  - **Code Duplication:** The complete radar pipeline logic is duplicated across `create_training_data.py`, `inference.py`, `visualize_fusion.py`, `visualize_association.py`, and `test_fusion_sample.py` instead of utilizing a unified, reusable pipeline class.

### 2.3 Camera Detection
- **Modules:** `camera/src/camera_detector.py`.
- **What is Working:**
  - Integrates Ultralytics YOLOv11 (`yolo11n.pt`) for real-time 2D bounding box detection on camera frames ($1920 \times 1080$).
  - Extracts class IDs, class names, confidence scores, and bounding box coordinates.
- **Flaws & Inconsistencies:**
  - Bounding box filtering is limited to confidence thresholding; no class-specific filtering (e.g., ADAS target classes such as vehicles, pedestrians, cyclists).
  - Zero unit tests exist in `camera/tests/`.

### 2.4 Calibration and Projection
- **Modules:** `fusion/src/radar_camera_projector.py`.
- **What is Working:**
  - Loads camera intrinsics ($3 \times 3$ matrix and distortion coefficients) and radar-to-camera extrinsics ($R_{vec}, T_{vec}$) from RADIal calibration files.
  - Correctly transforms radar coordinates to the camera coordinate convention ($[-Y, X, Z]$) and projects 3D points to 2D image pixels via `cv2.projectPoints`.
  - Rejects points lying outside image boundaries (`project_valid`).
- **Flaws & Inconsistencies:**
  - Assumes positive depth without explicit forward-plane filtering ($Z_{cam} > 0$) prior to projection, which can lead to inverted projections for targets behind the camera sensor plane.

### 2.5 Radar-Camera Association
- **Modules:** `fusion/src/association.py`.
- **What is Working:**
  - Computes exact geometric Euclidean distance from projected pixel coordinates to 2D bounding boxes.
  - Associates radar objects within a pixel-distance threshold (`max_pixel_distance = 30.0` px).
- **Flaws & Inconsistencies:**
  - **Greedy One-Way Association:** Iterates over radar objects and picks the closest camera detection. If multiple radar returns fall near or inside the same box, all can link to that box with no resolution of competition.
  - **No Global Optimization:** Lacks Hungarian/Munkres bipartite matching.
  - **Missing Depth/Kinematic Gating:** Association is purely 2D pixel-based. It does not check whether radar range matches estimated monocular depth or whether camera motion matches radar radial Doppler velocity.
  - **No Temporal Tracking:** Does not implement a multi-object tracker (e.g., Kalman Filter, EKF, SORT).

### 2.6 ML Fusion
- **Modules:** `fusion/src/fusion_model.py`.
- **What is Working:**
  - Serializes and deserializes a trained scikit-learn `RandomForestClassifier` bundle (`radar_camera_fusion.pkl`) with associated feature names.
  - Provides `predict()` and `predict_probability()` interfaces.
- **Flaws & Inconsistencies:**
  - Treats sensor fusion solely as a binary classification problem (predicting whether a candidate radar-camera pair is a match) rather than state estimation (fused 3D position, velocity, covariance, orientation).
  - The model decision threshold is hardcoded (`prediction == 1` and `probability > best_probability`).

### 2.7 Ground-Truth Data
- **Modules:** `fusion/training/ground_truth.csv`.
- **Finding:**
  - **`ground_truth.csv` is completely empty (0 bytes).**
  - The official RADIal dataset contains ground-truth annotations (bounding boxes, 3D locations, and sensor timestamps), but these are not ingested or parsed.
  - In lieu of actual ground truth, pseudo-labels were generated heuristically using spatial distance.

### 2.8 Training and Evaluation (Data Leakage & Evaluation Concerns)
- **Modules:** `fusion/training/create_training_data.py`, `fusion/training/train_model.py`, `fusion/training/fusion_training_data.csv`.
- **CRITICAL DEFECT — Severe Target Leakage & Circular Logic:**
  1. **Label Generation:** In `create_training_data.py` (lines 357–360), labels are created via:
     $$\text{is\_match} = \mathbb{I}(\text{pixel\_distance} \le 30.0)$$
  2. **Feature Set:** In `train_model.py` (lines 40–58), the feature list explicitly includes `pixel_distance` and `inside_bbox`.
  3. **Result:** The Random Forest is tasked with predicting a label that is an exact deterministic threshold of one of its input features. This is a severe form of target leakage. The reported high accuracy (>98%) is mathematically trivial and invalid as an assessment of real sensor fusion quality.
- **Additional Evaluation Risks:**
  - `fusion_training_data.csv` contains only 132 rows derived from just 13 frames (sampled every 5 frames between index 0 and 60 of a single sequence).
  - The training dataset contains only 6 positive (`is_match = 1`) instances and 126 negative instances (~4.5% positive class imbalance).
  - No evaluation is performed across diverse sequences, lighting conditions, or road topologies.

### 2.9 Inference Pipeline
- **Modules:** `fusion/inference.py`.
- **What is Working:**
  - CLI accepting `--recording <path>`.
  - Runs end-to-end processing across frames with live visualization of detections and fusion overlays.
- **Flaws & Inconsistencies:**
  - Processes every 5th frame (`FRAME_STEP = 5`), skipping 80% of data and making temporal filtering impossible.
  - Requires GUI display (`cv2.imshow`); cannot run headlessly or output evaluation metrics / recorded video files.
  - Hardcoded fallback paths to external user desktop folders.

### 2.10 Visualization
- **Modules:** `demo.py`, `fusion/visualize_projection.py`, `fusion/visualize_association.py`, `fusion/visualize_fusion.py`.
- **What is Working:**
  - Generates clear, high-contrast OpenCV visualizations (color-coded bounding boxes, radar points, and text tags).
- **Flaws & Inconsistencies:**
  - `demo.py` loads radar frame 0 once and repeats it indefinitely while camera video plays continuously.
  - No Bird's-Eye-View (BEV) radar map rendered alongside the perspective camera image.
  - Interactive GUI blocks execution; missing export functions (saving annotated frames/videos to disk).

### 2.11 Test Suite
- **Current Pytest Status:**
  - **Total Tests Discovered:** 21
  - **Passed:** 18
  - **Failed:** 3
- **Failure Details:**
  1. `radar/tests/test_angle_estimator_real.py::test_real_angle_estimation`: `TypeError: RadarPointExtractor.__init__() got an unexpected keyword argument 'range_resolution'`
  2. `radar/tests/test_mimo_real.py::test_real_mimo_reconstruction`: `TypeError: RadarPointExtractor.__init__() got an unexpected keyword argument 'range_resolution'`
  3. `radar/tests/test_radar_pipeline_real.py::test_real_radar_pipeline`: `TypeError: RadarPointExtractor.__init__() got an unexpected keyword argument 'range_resolution'`
- **Test Design Flaws:**
  - Unit tests in `radar/tests/` and `fusion/tests/` depend on physical dataset files existing at `Path.home() / "Desktop" / "RADIal_data"`. On any other machine or CI runner, tests crash with `FileNotFoundError`.
  - Lack of synthetic/mocked tests for real pipelines.
  - `camera/tests/` is completely empty.
  - `test_velocity_conversion.py` tests a duplicate local function rather than the actual `RadarVelocityConverter` class.

### 2.12 Configuration
- **Status:** No configuration files (`yaml`, `json`, or `.env`) exist.
- **Risk:** Critical ADAS and signal processing parameters (carrier frequency, chirp slope, CFAR guard/training cells, DBSCAN $\epsilon$ and `min_samples`, YOLO confidence, gate thresholds) are scattered as magic numbers across 10+ python scripts.

### 2.13 Hardcoded Paths
- **Audit Findings:** The repository contains numerous hardcoded absolute paths pointing to `C:\Users\lokit\...` and `Path.home() / "Desktop" / ...`.
  - Specific files with hardcoded paths:
    - `demo.py`: `C:\Users\lokit\Desktop\RADIal_data\...`, `C:\Users\lokit\Desktop\RADIal\DBReader`
    - `camera/run_camera.py` and `camera/test_video.py`: Hardcoded video path
    - `fusion/inference.py`: Hardcoded paths to `DBReader`, `CalibrationTable.npy`, `camera_calib.npy`, and `radar-camera-fusion-adas`
    - `fusion/training/create_training_data.py`: Hardcoded output path pointing to `radar-camera-fusion-adas` (which points outside the current folder `radar-camera-fusion-adas - Copy`)
    - `radar/tests/*` and `fusion/tests/*`: Hardcoded paths to user desktop

### 2.14 Dependencies
- **`requirements.txt`:**
  - `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `joblib`, `mkl_fft`, `pytest`, `ultralytics`.
- **Missing / Undeclared Dependencies:**
  - `scipy`: Used directly in `radar/src/cfar.py` (`from scipy import signal`), but omitted from `requirements.txt`.
  - `RADIal DBReader`: Imported via dynamic `sys.path.insert`, requiring an external repository on disk.
  - No package version pins.

### 2.15 README Documentation
- **Status:** Incomplete (2 lines total).
- **Missing Information:** Setup instructions, environment prerequisites, dataset layout, CLI execution guides, calibration requirements, architecture summary.

---

## 3. Detailed Summary Table of Findings

| Dimension | Current Implementation Status | Operational Health | Primary Risks / Deficiencies |
| :--- | :--- | :--- | :--- |
| **1. Directory Structure** | Standard Python package folders (`camera`, `radar`, `fusion`) | Partially Organized | Empty `camera/tests`; no `config/` or `docs/`. |
| **2. Radar Pipeline** | 16-RX ADC, Range/Doppler FFT, 2D CA-CFAR, MIMO AoA, DBSCAN | Working with Bugs | `RadarPipeline` constructor raises `TypeError`; duplicated pipeline code. |
| **3. Camera Detection** | YOLO11n wrapper extracting 2D boxes and classes | Working | Zero automated tests; no ADAS class filtering. |
| **4. Calibration & Projection** | $R_{vec}, T_{vec}, K$ matrix projection ($[-Y, X, Z]$ format) | Working | No forward-plane check ($Z > 0$); external calibration dependencies. |
| **5. Association** | 2D Euclidean distance to bounding box edge | Working (Basic) | Greedy association; no Hungarian matching; no kinematics or depth checks. |
| **6. ML Fusion** | Random Forest binary classifier for pair matching | Functional | Pair-classifier only; no 3D state/covariance estimation; no tracking. |
| **7. Ground Truth** | `ground_truth.csv` present in tree | Broken / Empty | 0 bytes; actual RADIal labels never ingested. |
| **8. Training & Evaluation** | Random Forest training script and CSV generator | Invalid / Leaked | Target label is `pixel_distance <= 30`, while `pixel_distance` is an input feature. |
| **9. Inference** | Interactive playback CLI (`inference.py`) | Working (Interactive) | Frame skipping ($5\times$); GUI only; hardcoded file locations. |
| **10. Visualization** | Multi-frame OpenCV visualizer with text & color overlays | Working | `demo.py` radar display is static; lacks Bird's-Eye-View (BEV) radar map. |
| **11. Tests** | 21 pytest test cases across radar and fusion | 3 Failing | 3 tests fail with `TypeError`; tests fail without external dataset on Desktop. |
| **12. Configuration** | None | Absent | Magic numbers hardcoded across files. |
| **13. Hardcoded Paths** | Hardcoded Windows paths in 12+ files | High Risk | Relies on local desktop directories; breaks on other machines. |
| **14. Dependencies** | Basic `requirements.txt` | Missing Dependencies | Missing `scipy`; unpinned versions; unmanaged `DBReader` dependency. |
| **15. README** | 2-line placeholder | Absent | Zero setup, usage, or architecture documentation. |

---

## 4. Recommended Incremental Development Roadmap

To transition this system into a robust, publishable ADAS radar-camera fusion platform without breaking existing working code, the following staged development order is recommended:

### Phase 1: Stability, Bug Fixes & Self-Contained Testing (Immediate Baseline)
1. **Fix `RadarPipeline` Constructor:** Align `RadarPipeline` with `RadarPointExtractor` arguments so that all 21 existing tests pass cleanly.
2. **Add Missing Dependency:** Add `scipy` and version constraints to `requirements.txt`.
3. **Resolve Path Management:** Create a centralized path and configuration module (`config.py` / `config.yaml`) with environment variable or parameter fallbacks, preventing path hardcoding.
4. **Mocked Unit Tests:** Add standalone unit tests that run with synthetic numpy arrays so tests pass in any environment without requiring external dataset downloads.
5. **Camera Unit Tests:** Add unit tests for `CameraDetector` in `camera/tests/`.

### Phase 2: Radar Pipeline Consolidation & Performance
1. **Unified Radar Perception Pipeline:** Consolidate the duplicated processing steps (ADC $\to$ FFT $\to$ CFAR $\to$ MIMO $\to$ DBSCAN $\to$ Objects) into a single, clean pipeline class.
2. **CFAR Optimization:** Accelerate CA-CFAR by replacing full 2D convolution with separable 1D passes or integral image summation, reducing per-frame latency.
3. **Range-Velocity Calibration:** Document and formalize radial velocity sign conventions and range bins.

### Phase 3: Legitimate Ground Truth & ML Fusion De-Biasing
1. **RADIal Ground Truth Parser:** Ingest the actual RADIal annotation files (labels, 3D object centers, boxes) into `ground_truth.csv`.
2. **Eliminate Data Leakage:**
   - Redefine association matching against genuine spatial/3D ground-truth overlaps (e.g., 3D IoU or 2D camera-box IoU).
   - Remove leaked proxy features (`pixel_distance`, `inside_bbox`) or reformulate the model to predict physical fusion attributes (e.g., fused position, velocity, class confidence).
3. **Multi-Sequence Dataset Split:** Generate training and test splits across distinct recording sequences to prevent spatial-temporal memorization.

### Phase 4: Tracking & Multi-Object Fusion (ADAS Quality)
1. **Global Bipartite Association:** Implement Hungarian/Munkres algorithm for optimal 1-to-1 association with distance and Mahalanobis gating.
2. **Multi-Object Tracking (MOT):** Integrate an Extended Kalman Filter (EKF) or Constant Velocity Kalman Filter to maintain persistent track IDs, smooth velocities, and bridge temporary sensor dropouts.
3. **ADAS Decision Logic:** Add basic ADAS warning logic: Forward Collision Warning (FCW) based on Time-to-Collision (TTC) using fused distance and radar radial velocity.

### Phase 5: Visualization, Headless Inference & Documentation
1. **Dual-View Visualization:** Add a synchronized Bird's-Eye-View (BEV) radar display alongside the perspective camera image.
2. **Headless / Batch Mode:** Enable `inference.py` to write video files or evaluation JSONs without requiring interactive GUI windows.
3. **Comprehensive README:** Document architecture, data setup, training workflow, and inference commands.
