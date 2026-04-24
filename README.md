# Multi-Person Detection and Re-Identification Dashboard

A real-time computer vision system for person detection, tracking, cross-camera identity association, and live analytics visualization.

This project combines YOLO-based person detection/tracking, appearance-based ReID, trajectory mapping, and activity heatmaps inside a Streamlit dashboard.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Scenarios](#scenarios)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Models and Data](#models-and-data)
- [Outputs](#outputs)
- [Demo Videos](#demo-videos)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Overview

The system supports three operational modes:

1. Single camera multi-person tracking.
2. Four virtual-camera (2x2 grid) single-person ReID tracking.
3. Multi-camera multi-person ReID tracking on EPFL-RLC frame sequences.

For each frame, the pipeline generates:

- Annotated video frame
- 2D occupancy/motion map
- Activity heatmap
- Live metrics (active IDs, total IDs, frame index)

These analytics are rendered in a Streamlit dashboard.

## Key Features

- YOLO-based person detection and track persistence
- ReID-based global ID matching across views/cameras
- Real-time trajectory rendering on a 2D map
- Temporal-decay heatmap generation
- Streamlit UI with scenario switching and live metrics
- Standalone OpenCV visualization for script-level execution

## System Architecture

High-level flow:

1. Input loader reads frames from either a video stream or synchronized camera folders.
2. YOLO detects/tracks person class instances.
3. ReID manager extracts appearance embeddings and resolves global IDs.
4. Analytics module updates trajectories and heatmap accumulators.
5. Pipeline returns `(frame, stats_dict)` to the UI.
6. Dashboard renders live feed + analytics panels + metrics.

Core modules:

- `src/input/video_input.py`: Unified frame loaders (`VideoLoader`, `MultiCameraLoader`)
- `src/reid/reid_manager.py`: Feature extraction and global ID matching
- `src/analytics/heatmap.py`: Heatmap update/render/overlay
- `src/utils/visualization.py`: Color management and drawing utilities
- `src/ui/dashboard.py`: Streamlit application and scenario routing
- `scripts/*.py`: Scenario-specific runtime pipelines

## Repository Structure

```text
multiperson_detection/
├── config.yaml
├── requirements.txt
├── README.md
├── models/
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   └── yolov8m.pt
├── data/
│   └── input.mp4
├── EPFL-RLC_dataset/
│   ├── calibration/
│   ├── frames/
│   │   ├── cam0/
│   │   ├── cam1/
│   │   └── cam2/
│   └── mv_examples/
├── output/
├── scripts/
│   ├── single_cam.py
│   ├── multi_cam_single_person.py
│   └── multi_cam_multi_person.py
└── src/
    ├── analytics/
    │   └── heatmap.py
    ├── input/
    │   └── video_input.py
    ├── reid/
    │   └── reid_manager.py
    ├── ui/
    │   └── dashboard.py
    └── utils/
        └── visualization.py
```

## Scenarios

### Scenario 1: Single Camera Multi-Person

- Script: `scripts/single_cam.py`
- Purpose: Track multiple people in one camera feed.
- Output: Annotated frame + 2D map + heatmap + live metrics.

### Scenario 2: 4-Cam Single Person ReID

- Script: `scripts/multi_cam_single_person.py`
- Purpose: Split one video into a 2x2 layout and maintain consistent IDs using ReID logic.
- Output: 2x2 annotated grid + cross-view map + heatmap + metrics.

### Scenario 3: Multi-Cam Multi-Person

- Script: `scripts/multi_cam_multi_person.py`
- Purpose: Process synchronized frames from `EPFL-RLC_dataset/frames/cam0..cam2` and assign global IDs across physical cameras.
- Output: Multi-view grid + global trajectory map + heatmap + metrics.

## Setup and Installation

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Git
- Optional: CUDA-capable GPU for faster inference/ReID

### 1) Clone repository

```bash
git clone https://github.com/<your-username>/multiperson_detection.git
cd multiperson_detection
```

### 2) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Verify weights and data

Ensure these exist before running:

- YOLO weights in `models/` (at least `yolov8n.pt`)
- Input video in `data/input.mp4` (for single-camera mode)
- EPFL frames under `EPFL-RLC_dataset/frames/cam0..cam2` (for multi-cam mode)

## How to Run

### Run Streamlit dashboard

```bash
streamlit run src/ui/dashboard.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

Use the sidebar to switch scenarios:

- Scenario 1: Single Camera Multi-Person
- Scenario 2: 4-Cam Single Person ReID
- Scenario 3: Multi-Cam Multi-Person

### Run scenario scripts directly (OpenCV window mode)

```bash
python scripts/single_cam.py
python scripts/multi_cam_single_person.py
python scripts/multi_cam_multi_person.py
```

## Configuration

Project-level config file: `config.yaml`

Current sections:

- `input`: source type, grid shape, default video path
- `processing`: resolution and GPU intent
- `tracking`: algorithm parameters
- `reid`: model and enable flag
- `output`: UI/video/heatmap toggles
- `ui`: visualization toggles

Note: Several runtime constants are currently defined directly inside scenario scripts (for example thresholds and default paths). If you need centralized control, move those constants into `config.yaml` and load them at runtime.

## Models and Data

### Models

- `models/yolov8n.pt` (default in scripts)
- `models/yolov8s.pt`
- `models/yolov8m.pt`

### Input Sources

- Single video mode: `data/input.mp4` (fallback paths may exist in scripts)
- Multi-camera mode: folder sequences under `EPFL-RLC_dataset/frames/`

### ReID Backend

- Uses `torchreid` `FeatureExtractor` (default model `osnet_x1_0`)
- Features are normalized and matched using cosine similarity and spatial/temporal constraints

## Outputs

- Live dashboard analytics panels:
  - Heatmap
  - 2D Map
  - Active IDs
  - Total IDs
  - Frame index
- Standalone scripts can write output videos to `output/` (for supported script flows)

## Demo Videos

Add your recordings in a dedicated folder, for example:

```text
docs/
└── demos/
    ├── scenario1_single_cam.mp4
    ├── scenario2_grid_reid.mp4
    └── scenario3_multi_cam_multi_person.mp4
```

### Embedded Demo Placeholders

Replace each `src` with your real file path or hosted URL.

#### Scenario 1 Demo

```html
<video controls width="900" src="docs/demos/scenario1_single_cam.mp4">
  Your browser does not support the video tag.
</video>
```

#### Scenario 2 Demo

```html
<video controls width="900" src="docs/demos/scenario2_grid_reid.mp4">
  Your browser does not support the video tag.
</video>
```

#### Scenario 3 Demo

```html
<video controls width="900" src="docs/demos/scenario3_multi_cam_multi_person.mp4">
  Your browser does not support the video tag.
</video>
```

### Demo Checklist

Before publishing demos:

- Record one full run per scenario
- Show sidebar scenario selection and live metrics
- Include at least one segment with stable ID continuity
- Include a short clip showing heatmap/map update behavior

## Troubleshooting

### Heatmap or 2D Map not rendering

- Confirm `stats_dict` contains `heatmap` and `map` arrays from pipeline scripts.
- Ensure arrays are `uint8` and 3-channel before returning to dashboard.
- Restart Streamlit after pipeline/UI changes.

### Model loading errors

- Verify YOLO weights exist in `models/`.
- Ensure `torch`, `ultralytics`, and `torchreid` installed in the active environment.

### Dataset path issues (Scenario 3)

- Confirm path exists: `EPFL-RLC_dataset/frames/cam0..cam2`
- Ensure each camera folder has readable image frames.

### Slow performance

- Use smaller YOLO model (`yolov8n.pt`)
- Lower input resolution
- Enable GPU-backed environment for Torch
