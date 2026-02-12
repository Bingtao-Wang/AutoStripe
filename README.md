# AutoStripe

**Vision-Driven Automated Highway Lane Marking System**

[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-blue)](https://carla.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-green)](https://www.python.org/)

---

## Overview

AutoStripe is an automated highway lane marking simulation system built on CARLA 0.9.15. The vehicle acts as a marking machine, using vision-based perception to detect road edges, plan a driving path, and paint lane markings at a fixed offset from the road curb.

The project implements a complete **Perception - Planning - Control** pipeline:

- **Perception**: VLLiNet AI road segmentation + depth camera -> road edge extraction (G key toggles AI/GT)
- **Planning**: Right road edge offset + polynomial extrapolation -> driving path + nozzle path
- **Control**: PD controller + Pure Pursuit path tracking + auto-paint state machine
- **Painting**: Nozzle trajectory with solid/dashed modes, auto-convergence to 3.0m target
- **Evaluation**: Map API ground truth comparison, per-frame logging, 10 visualization plots

### Current Status (V4.2)

- VLLiNet AI perception (MaxF=98.33%, IoU=96.72%)
- PD controller with adaptive steer filter and dynamic driving offset
- Auto-paint state machine (CONVERGING → STABILIZED → PAINTING)
- Dashed/solid line modes (D key toggle)
- Evaluation pipeline: E key recording → GT comparison → CSV export
- Per-frame 31-column framelog with perception accuracy metrics
- 7 map-based spatial visualizations + 8-panel timeseries (Nature style, PDF/SVG)

---

## System Architecture

```
CARLA 0.9.15 Server
       |
       v
 Scene Setup (setup_scene_v2.py)
   - RGB camera (1248x384, FOV=90)
   - Depth camera (same position)
   - Semantic camera (for GT mode)
   - Overhead camera (bird's eye)
       |
       v
 Perception (perception/)
   road_segmentor_ai.py  -- VLLiNet AI road segmentation (G key: AI/GT toggle)
   road_segmentor.py     -- GT CityScapes segmentor (fallback)
   edge_extractor.py     -- road mask -> left/right edge pixels
   depth_projector.py    -- pixel + depth -> 3D world coords
   perception_pipeline.py -- AI/GT dual mode, 8-tuple return
       |
       v
 Planning (planning/)
   vision_path_planner.py -- right edge -> driving path + nozzle path
                             + polynomial extrapolation for blind spot
       |
       v
 Control (control/)
   marker_vehicle_v2.py -- Pure Pursuit + PD offset controller
                           + adaptive steer filter
       |
       v
 Painting (manual_painting_control_v4.py)
   - Auto-paint state machine (CONVERGING/STABILIZED/PAINTING)
   - Solid/dashed line modes (D key)
   - Evaluation recording (E key) -> framelog + GT comparison
```

---

## Project Structure

```
AutoStripe/
├── manual_painting_control_v4.py  # V4 main entry (AI/GT + PD control + eval)
├── manual_painting_control.py     # V3 main entry (GT only)
├── main_v2.py                     # V2 entry (auto only)
├── main_v1.py                     # V1 entry (Map API based)
├── diag_vllinet.py                # VLLiNet model verification
├── carla_env/
│   ├── setup_scene.py             # V1 scene setup
│   └── setup_scene_v2.py          # V2-V4 scene (1248x384 cameras)
├── perception/
│   ├── road_segmentor_ai.py       # VLLiNet AI segmentation
│   ├── road_segmentor.py          # GT CityScapes segmentation
│   ├── edge_extractor.py          # Road mask -> edge pixels
│   ├── depth_projector.py         # Pixel + depth -> world coords
│   └── perception_pipeline.py     # AI/GT dual mode pipeline
├── planning/
│   ├── lane_planner.py            # V1 Map API planner + road geometry
│   └── vision_path_planner.py     # Vision planner + polynomial extrapolation
├── control/
│   ├── marker_vehicle.py          # V1 Pure Pursuit
│   └── marker_vehicle_v2.py       # V2-V4 Pure Pursuit + PD controller
├── evaluation/
│   ├── trajectory_evaluator.py    # GT comparison + metrics + CSV export
│   ├── perception_metrics.py      # Mask IoU + edge deviation
│   ├── frame_logger.py            # Per-frame 31-column CSV recorder
│   ├── visualize_eval.py          # Eval plots + 8-panel timeseries
│   └── visualize_map.py           # 7 map-based spatial visualizations
├── ros_interface/                 # ROS integration (optional)
├── VLLiNet_models/                # VLLiNet model + checkpoint
├── docs/                          # Design documents
└── TEST/                          # Reference test scripts
```

---

## Prerequisites

- CARLA 0.9.15 (compiled version)
- Python 3.7+ (conda env recommended)
- PyTorch 1.13+
- OpenCV 4.2+
- pygame 2.6+
- numpy

---

## Quick Start

### 1. Launch CARLA Server

```bash
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh
```

### 2. Run V4 (Recommended)

```bash
cd AutoStripe
python manual_painting_control_v4.py
```

### 3. Alternative: Run V3 (GT perception only)

```bash
python manual_painting_control.py
```

### 4. Alternative: Run V1 (Map API baseline)

```bash
python main_v1.py
```

---

## V4 Keyboard Controls

| Key | Function |
|-----|----------|
| SPACE | Toggle painting ON/OFF |
| TAB | Toggle Auto/Manual drive mode |
| G | Toggle AI/GT perception mode |
| D | Toggle dashed/solid line mode |
| E | Toggle eval recording (start/stop + GT evaluation) |
| R | Toggle video recording |
| WASD/Arrows | Manual drive (throttle/steer/brake) |
| Q | Toggle reverse mode |
| V | Toggle spectator follow/free camera |
| X | Handbrake |
| ESC | Quit |

---

## Key Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Perception | Model | VLLiNet_Lite (MaxF=98.33%, IoU=96.72%) |
| Perception | Camera | x=1.5, z=2.4, pitch=-15, 1248x384, FOV=90 |
| Perception | Intrinsics | fx=fy=624, cx=624, cy=192 |
| Planning | Polynomial fit | Quadratic (deg=2), edge points 3-20m range |
| Planning | Max range | 20m (truncates far depth noise) |
| Control | PD Controller | Kp=0.5, Kd=0.3, OFFSET_SMOOTH=0.12 |
| Control | Steer filter | Adaptive 0.15 (smooth) - 0.50 (aggressive) |
| Control | Target speed | 3.0 m/s |
| Painting | Target distance | 3.0m nozzle-to-edge |
| Painting | Auto-paint | Tolerance enter=0.3m, exit=0.45m, grace=15 frames |
| Painting | Dashed line | 3.0m paint / 3.0m gap |
| Evaluation | Coverage threshold | 2.0m |

---

## Version History

| Version | Description | Entry Point |
|---------|-------------|-------------|
| V1 | Map API path planning + Pure Pursuit + nozzle painting | `main_v1.py` |
| V2 | Vision perception replaces Map API (standalone, auto only) | `main_v2.py` |
| V3 | Manual/auto control + CityScapes segmentation + enhanced visualization | `manual_painting_control.py` |
| V4 | VLLiNet AI perception + polynomial extrapolation + AI/GT toggle | `manual_painting_control_v4.py` |
| V4.1 | Adaptive steer filter + dynamic driving offset + auto-paint state machine | `manual_painting_control_v4.py` |
| V4.2 | PD controller + evaluation pipeline + dashed lines + perception metrics + map visualization | `manual_painting_control_v4.py` |

## Future Work

- Replace VLLiNet with real-world trained model (transfer from CARLA to real camera)
- Center lines, multi-lane support
- All-weather support (rain/fog/night)
- Obstacle avoidance

---

## References

- [CARLA Simulator](https://carla.org/)
- GB 5768-2009: Road Traffic Signs and Markings

---

**Project Status**: V4.2 Active Development
**Last Updated**: 2026-02-11
