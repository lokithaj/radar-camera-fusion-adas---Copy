# RADIal Ground Truth & Sensor Fusion ML Design

**Date:** September 2026  
**Module:** `fusion/training`  
**Purpose:** Formulate a scientifically valid, leakage-free ground truth association and evaluation methodology for radar-camera sensor fusion using the RADIal dataset.

---

## 1. RADIal Ground Truth Interface & Data Specification

The RADIal dataset (*Raw High-Definition Radar for Multi-Task Learning*, CVPR 2022) provides synchronized multimodal recordings collected with automotive-grade sensors (5 Mpix camera, 16-RX/12-TX 192-virtual channel FMCW radar, 16-layer LiDAR, CAN bus, and GPS).

Ground-truth vehicle annotations are provided in `labels_CVPR.csv` (ingested as `fusion/training/ground_truth.csv`).

### 1.1 Camera Bounding-Box Annotations
Each labeled vehicle entry provides exact 2D bounding box coordinates on the $1920 \times 1080$ camera image plane:
- `x1_pix`: Left boundary pixel coordinate.
- `y1_pix`: Top boundary pixel coordinate.
- `x2_pix`: Right boundary pixel coordinate.
- `y2_pix`: Bottom boundary pixel coordinate.
- `Annotation`: Annotation classification (`"weak"` for semi-automatically tracked/propagated boxes, `"strong"` for verified annotations).
- `Difficult`: Binary flag (`0.0` for clear vehicle views, `1.0` for heavily occluded or distant targets).

### 1.2 Radar Annotations
For each labeled vehicle, RADIal provides physical 2D spatial positions, polar coordinates, and kinematic reflections in the radar coordinate frame:
- `radar_X_m`: Lateral distance in meters (in RADIal convention, $X$ represents lateral offset; negative indicates right, positive indicates left).
- `radar_Y_m`: Longitudinal forward distance in meters ($Y$ represents forward distance along vehicle heading).
- `radar_R_m`: Radial distance / slant range in meters ($R = \sqrt{X^2 + Y^2}$).
- `radar_A_deg`: Azimuth angle in degrees ($0^\circ$ = boresight, negative = right, positive = left).
- `radar_D_mps`: Doppler radial velocity (in m/s or Doppler FFT bins).
- `radar_P_db`: Reflected radar signal power in dB.
- *Cross-Modal Reference:* `laser_X_m` and `laser_Y_m` provide 3D LiDAR spatial coordinates from the 16-layer LiDAR scanner mounted on the vehicle front grille.

### 1.3 Object and Category Information
- **Category:** All annotated bounding boxes and radar targets represent on-road **vehicles** (passenger cars, vans, trucks).
- **Format:** Labeled rows with valid positive coordinates represent target vehicles. Rows with `-1` indicate frames where no vehicle was present or labeled.

### 1.4 Frame and Sample Identifiers
- `dataset`: The sequence/recording name (e.g., `RECORD@2020-11-21_13.44.44` or `RECORD@2020-11-22_12.24.44`).
- `index`: The zero-based frame index within that specific recording sequence. This index maps directly to `SyncReader.GetSensorData(index)`.
- `numSample`: The global dataset-wide synchronized sample number across all 91 recordings (range: 0 to 25,000+).

### 1.5 Correspondence to Synchronized Samples
In the official `DBReader.SyncReader`, each recording sequence is indexed by frame number `index`. When calling `SyncReader.GetSensorData(index)`:
- The camera frame, raw 4-chip radar ADC buffers (`radar_ch0`–`radar_ch3`), LiDAR point clouds, and CAN odometry are synchronized using event log timestamps (`*_events_log.rec`).
- A ground truth entry with `(dataset == sequence_name, index == frame_idx)` corresponds exactly to the sensor data returned at `SyncReader.GetSensorData(frame_idx)`.
- When multiple vehicles appear in a single frame, multiple rows exist sharing the same `dataset` and `index`.

### 1.6 Cross-Sensor Association Through RADIal Ground Truth
**Crucial Architectural Finding:** Radar and camera annotations **are already unified and paired** in the RADIal ground truth table:
- Every row in `ground_truth.csv` represents a single physical vehicle entity that was simultaneously observed by the camera (`[x1_pix, y1_pix, x2_pix, y2_pix]`) and the radar (`[radar_X_m, radar_Y_m, radar_R_m, radar_A_deg]`).
- Projecting the radar coordinates into the camera coordinate system using the calibrated extrinsic and intrinsic transformation ($X_{forward} = radar\_Y\_m$, $Y_{lateral} = -radar\_X\_m$, $Z_{vertical} = 0$) confirms that the ground-truth radar return projects directly into or immediately adjacent to the ground-truth visual bounding box of that same vehicle.

---

## 2. Audit of the Current ML Dataset & Training Pipeline

An inspection of `fusion/training/create_training_data.py`, `fusion/training/train_model.py`, and `fusion/training/fusion_training_data.csv` reveals critical defects in the Stage 1/2 baseline:

### 2.1 The Circular Target Leakage Flaw
1. **Heuristic Label Creation (`create_training_data.py`, lines 307–334):**
   ```python
   pixel_distance = point_to_box_distance(u, v, camera_object["bbox"])
   inside_bbox = int(x1 <= u <= x2 and y1 <= v <= y2)
   is_match = int(pixel_distance <= 30.0)  # LABEL_DISTANCE = 30.0
   ```
   The target variable `is_match` was defined purely by whether projected pixel distance was $\le 30.0$ px. No ground-truth annotations were consulted.
2. **Model Training Features (`train_model.py`, lines 29–47):**
   ```python
   FEATURE_COLUMNS = [
       ...
       "pixel_distance",
       "inside_bbox",
   ]
   ```
3. **Circular Target Leakage:**
   The Random Forest classifier was trained on `pixel_distance` to predict `pixel_distance <= 30.0`. This is a textbook circular dependency: the classifier merely learns an identity threshold on its own input feature. The reported >98% accuracy was a trivial artifact of this circularity and had zero scientific validity regarding real-world sensor fusion.

### 2.2 Dataset Size and Imbalance Deficiencies
- `fusion_training_data.csv` contained only 132 rows generated from only 13 frames (`FRAME_STEP = 5` up to frame 60) of a single recording.
- Only 6 rows had `is_match = 1` (95.5% negative class imbalance).
- `ground_truth.csv` in the repository was an empty 0-byte placeholder.

---

## 3. Leakage-Free Ground-Truth Labeling Strategy

To achieve a scientifically rigorous evaluation, `is_match` must be derived **strictly from independent ground truth annotations**, completely separated from any candidate pair feature.

```
       Candidate Pair (Radar Object R_i, Camera Detection B_j)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 Does R_i match GT object G_k?         Does B_j match GT object G_k?
 (Radar distance in 2D sensor frame)   (Camera 2D IoU with GT box)
  d_radar(R_i, G_k) <= 2.5 m            IoU(B_j, G_k.bbox) >= 0.4
            │                                     │
            └──────────────────┬──────────────────┘
                               │
             Do BOTH match the SAME GT object G_k?
                               │
              ┌────────────────┴────────────────┐
             YES                                NO
              │                                 │
              ▼                                 ▼
      is_match = 1                      is_match = 0
  (Valid Physical Match)            (Non-Match / Clutter / Mismatch)
```

### Labeling Rules:
1. **Radar-to-GT Spatial Matching:**
   A detected radar cluster centroid $R_i = (x_m, y_m)$ matches a ground-truth object $G_k$ if its Euclidean distance in the radar coordinate frame is within a gating radius $\tau_r = 2.5$ meters:
   $$d(R_i, G_k) = \sqrt{(x_{m, i} - G_{k, \text{forward}})^2 + (y_{m, i} - G_{k, \text{lateral}})^2} \le 2.5\text{ m}$$
   where $G_{k, \text{forward}} = G_k.radar\_Y\_m$ and $G_{k, \text{lateral}} = -G_k.radar\_X\_m$.
2. **Camera-to-GT Bounding-Box Matching:**
   A detected YOLO camera bounding box $B_j$ matches ground-truth object $G_k$ if their 2D Intersection-over-Union meets standard visual detection thresholds:
   $$\text{IoU}(B_j, G_k.\text{bbox}) \ge 0.40$$
3. **Fusion Match Label Definition:**
   - Candidate pair $(R_i, B_j)$ receives label `is_match = 1` **if and only if** $R_i$ matches $G_k$ AND $B_j$ matches the same $G_k$.
   - If either object is unassociated with a ground truth vehicle, or if they associate with two different ground truth vehicles, `is_match = 0`.
4. **Target Leakage Prohibition:**
   Neither `pixel_distance <= threshold` nor any model input feature is used to define `is_match`. The label reflects whether both sensors detected the **same physical ground-truth vehicle**.

---

## 4. Recording-Level Train/Test Split

To test genuine generalization and prevent spatial-temporal data leakage, splitting must occur strictly at the **recording sequence level**, never by randomly splitting rows or frames from the same sequence.

### Available Physical Sequences on Disk:
1. **Training Sequence:** `RECORD@2020-11-21_13.44.44`
   - Total synchronized frames: 65 (sampled at step 1 across all frames).
   - Labeled frames in ground truth: 37 frames (indices 0..52).
   - Ground truth vehicle annotations: 155 vehicle instances.
2. **Unseen Test Sequence:** `RECORD@2020-11-22_12.24.44`
   - Total synchronized frames: 22 (sampled across all frames).
   - Labeled frames in ground truth: 6 frames (indices 8..13).
   - Ground truth vehicle annotations: 6 vehicle instances.
   - **Zero overlap:** Held out completely during model training and feature normalization.

---

## 5. Feature Engineering: Retained vs. Removed Features

Features must represent legitimate, physically meaningful multimodal fusion properties without leaking proxy thresholds:

| Feature Name | Type | Status | Justification |
| :--- | :--- | :--- | :--- |
| `range_m` | Radar Kinematic | **KEPT** | Radial distance measured directly by FMCW range FFT. |
| `velocity_mps` | Radar Kinematic | **KEPT** | Radial Doppler velocity measured directly by FMCW Doppler FFT. |
| `azimuth_deg` | Radar Angular | **KEPT** | Azimuth angle estimated via 192-channel MIMO super-resolution. |
| `elevation_deg` | Radar Angular | **KEPT** | Elevation angle estimated via MIMO super-resolution. |
| `radar_power` | Radar Signal | **KEPT** | Reflected signal power/cross-section from CFAR peak. |
| `radar_x` | Radar Position | **KEPT** | Cartesian longitudinal distance in meters ($X_{forward}$). |
| `radar_y` | Radar Position | **KEPT** | Cartesian lateral distance in meters ($Y_{lateral}$). |
| `radar_z` | Radar Position | **KEPT** | Cartesian vertical distance in meters ($Z_{vertical}$). |
| `projected_u` | Projection Geometry| **KEPT** | Image horizontal coordinate of projected radar centroid. |
| `projected_v` | Projection Geometry| **KEPT** | Image vertical coordinate of projected radar centroid. |
| `bbox_center_u` | Camera Geometry | **KEPT** | Center horizontal coordinate of 2D bounding box. |
| `bbox_center_v` | Camera Geometry | **KEPT** | Center vertical coordinate of 2D bounding box. |
| `bbox_width` | Camera Geometry | **KEPT** | Bounding box width (pixels). |
| `bbox_height` | Camera Geometry | **KEPT** | Bounding box height (pixels). |
| `bbox_aspect_ratio`| Camera Geometry | **KEPT** | Ratio $W / H$, informative of vehicle orientation. |
| `camera_confidence`| Camera Signal | **KEPT** | YOLO objectness confidence score. |
| `norm_offset_u` | Relative Geometry | **KEPT** | Normalized offset: $(u - \text{center\_u}) / W$. Scale-invariant. |
| `norm_offset_v` | Relative Geometry | **KEPT** | Normalized offset: $(v - \text{center\_v}) / H$. Scale-invariant. |
| `norm_radial_dist` | Relative Geometry | **KEPT** | Dimensionless offset: $\sqrt{(\Delta u / W)^2 + (\Delta v / H)^2}$. |
| `pixel_distance` | Raw Distance | **REMOVED** | Removed to eliminate any residue of the old proxy threshold rule. |
| `inside_bbox` | Binary Indicator | **REMOVED** | Direct correlate of the flawed proxy label. |

---

## 6. Training, Evaluation & Test Protocol

1. **Dataset Generation:**
   - Execute `create_training_data.py` on both sequences separately using the leakage-free ground truth matching algorithm.
   - Output `fusion_training_data.csv` for `RECORD@2020-11-21_13.44.44` and `fusion_test_data.csv` for `RECORD@2020-11-22_12.24.44`.
2. **Model Training:**
   - Train `RandomForestClassifier` on the training dataset only.
   - Save model bundle to `fusion/models/radar_camera_fusion_groundtruth.pkl` (preserving the original model intact).
3. **Rigorous Evaluation on Unseen Test Sequence:**
   - Compute and report Precision, Recall, F1-score, and Confusion Matrix on `RECORD@2020-11-22_12.24.44`.
   - Provide an honest appraisal of sample size constraints (6 GT annotations across 22 frames in the test sequence).
4. **Automated Test Suite:**
   - Add unit tests validating ground truth ingestion, label generation, train/test sequence isolation, and model inference.
