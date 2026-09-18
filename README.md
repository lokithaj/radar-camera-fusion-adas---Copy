# radar-camera-fusion-adas
Modular Radar-Camera Fusion ADAS system with radar perception, computer vision, multi-object tracking, sensor fusion, and ADAS decision logic.

## Installation

```bash
pip install -r requirements.txt
```

## Running Inference

Run inference on any synchronized RADIal recording directory:

```bash
python -m fusion.inference --recording "PATH_TO_RECORDING"
```

Optional CLI arguments:
- `--model`: Path to trained fusion ML model (defaults to project-relative `fusion/models/radar_camera_fusion.pkl`)
- `--dbreader`: Path to RADIal DBReader folder (auto-discovered if omitted)
- `--radar-calib`: Path to `CalibrationTable.npy` (auto-discovered if omitted)
- `--camera-calib`: Path to `camera_calib.npy` (auto-discovered if omitted)
- `--output-dir`: Output directory for exports and predictions (defaults to `outputs/`)

## Environment Variables (Optional)

Configure custom dataset or toolkit paths globally:
- `RADIAL_ROOT`: Root path to RADIal toolkit repository
- `RADIAL_DBREADER_DIR`: Path to the `DBReader` directory
- `RADIAL_RADAR_CALIB`: Path to `CalibrationTable.npy`
- `RADIAL_CAMERA_CALIB`: Path to `camera_calib.npy`
- `RADIAL_RECORDING`: Default sequence path for tests and demos
