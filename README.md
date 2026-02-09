# AutoStripe

**Vision-Driven Automated Highway Lane Marking System**

[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-blue)](https://carla.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-green)](https://www.python.org/)

---

## Overview

AutoStripe is an automated highway lane marking simulation system built on CARLA 0.9.15. The vehicle acts as a marking machine, using vision-based perception to detect road edges, plan a driving path, and paint lane markings at a fixed offset from the road curb.

The project implements a complete **Perception - Planning - Control** pipeline:

- **Perception**: CityScapes semantic segmentation + depth camera -> road edge extraction in world coordinates
- **Planning**: Right road edge offset -> driving path + nozzle path generation (per-frame, no Map API)
- **Control**: Pure Pursuit path tracking + manual/auto drive mode switching
- **Painting**: Nozzle trajectory drawing with pause/resume support

### Current Status (V3)

- Vision-based road edge detection (CityScapes color matching)
- Slope-aware 3D visualization (adapts to bridges/ramps)
- Manual painting control with pygame (drive + paint toggle)
- Right road edge visualization (red dots with filtering)
- Nozzle-to-edge and tracking-point-to-edge distance display
- Short-term memory fallback when perception fails
- 20m distance truncation for depth noise reduction

---

## System Architecture

```
CARLA 0.9.15 Server
       |
       v
 Scene Setup (setup_scene_v2.py)
   - Semantic camera (CityScapes)
   - Depth camera (same position)
   - RGB front camera
   - Overhead camera (bird's eye)
       |
       v
 Perception (perception/)
   road_segmentor.py    -- CityScapes color match -> road mask
   edge_extractor.py    -- road mask -> left/right edge pixels
   depth_projector.py   -- pixel + depth -> 3D world coords
   perception_pipeline.py -- combines above three steps
       |
       v
 Planning (planning/)
   vision_path_planner.py -- right edge -> driving path + nozzle path
                            (1m bins, smoothing, left offset, 20m range)
       |
       v
 Control (control/)
   marker_vehicle_v2.py -- Pure Pursuit tracking (dynamic path update)
       |
       v
 Painting (manual_painting_control.py)
   - Nozzle position = vehicle + 2m right offset
   - debug.draw_line for yellow paint trail
   - Pause/resume with None gap markers
```

---

## Project Structure

```
AutoStripe/
├── manual_painting_control.py  # V3 main entry (pygame + manual/auto control)
├── main_v2.py                  # V2 entry (auto only, no manual control)
├── main_v1.py                  # V1 entry (Map API based)
├── carla_env/
│   ├── setup_scene.py          # V1 scene setup
│   └── setup_scene_v2.py       # V2/V3 scene (semantic + depth + RGB + overhead)
├── perception/
│   ├── road_segmentor.py       # CityScapes color matching -> road mask
│   ├── edge_extractor.py       # Road mask -> left/right edge pixels
│   ├── depth_projector.py      # Pixel + depth -> world coordinates
│   └── perception_pipeline.py  # Combines above three steps
├── planning/
│   ├── lane_planner.py         # V1 Map API planner
│   └── vision_path_planner.py  # V2/V3 vision-based planner (with z, memory)
├── control/
│   ├── marker_vehicle.py       # V1 Pure Pursuit
│   └── marker_vehicle_v2.py    # V2/V3 Pure Pursuit + dynamic path update
├── ros_interface/
│   └── autostripe_node.py      # ROS node (optional)
├── configs/                    # Configuration files
├── launch/                     # ROS launch files
├── docs/
│   └── Project_Design.md       # Full design document
└── TEST/                       # Reference test scripts
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

### 2. Run V3 (Recommended)

```bash
cd AutoStripe
python manual_painting_control.py
```

### 3. Alternative: Run V1 (Map API baseline)

```bash
python main_v1.py
```

---

## V3 Keyboard Controls

| Key | Function |
|-----|----------|
| SPACE | Toggle painting ON/OFF |
| TAB | Toggle Auto/Manual drive mode |
| WASD/Arrows | Manual drive (throttle/steer/brake) |
| Q | Toggle reverse mode |
| V | Toggle spectator follow/free camera |
| X | Handbrake |
| ESC | Quit |

---

## Key Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Perception | Road color (CityScapes BGR) | (128, 64, 128), tolerance=10 |
| Perception | Front cameras | x=2.5, z=2.8, pitch=-15, 800x600, FOV=90 |
| Perception | Depth decode | (R + G*256 + B*65536) / (256^3-1) * 1000 |
| Planning | Max range | 20m (truncates far depth noise) |
| Planning | Memory frames | 10 (fallback when perception fails) |
| Planning | Smooth window | 5, 1m longitudinal bins |
| Control | Pure Pursuit | wheelbase=2.875, Kdd=3.0, lookahead=8 waypoints |
| Control | Target speed | 3.0 m/s |
| Painting | Nozzle offset | 2.0m right of vehicle center |
| Visualization | Z offset | +0.3m above slope-projected road surface |

---

## Version History

| Version | Description | Entry Point |
|---------|-------------|-------------|
| V1 | Map API path planning + Pure Pursuit + nozzle painting | `main_v1.py` |
| V2 | Vision perception replaces Map API (standalone, auto only) | `main_v2.py` |
| V3 | Manual/auto control + CityScapes segmentation + enhanced visualization | `manual_painting_control.py` |

## Future Work

- Replace CARLA semantic camera with LUNA-Net for real-world perception
- Dashed lane dividers, center lines, multi-lane support
- Evaluation pipeline: compare vision trajectory vs Map API ground truth
- All-weather support (rain/fog/night)

---

## References

- [CARLA Simulator](https://carla.org/)
- GB 5768-2009: Road Traffic Signs and Markings

---

**Project Status**: V3 Active Development
**Last Updated**: 2026-02-09
