# AutoStripe

**Vision-Driven Automated Highway Lane Marking System**

[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-blue)](https://carla.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

AutoStripe is an automated highway lane marking system powered by AI-driven road perception. It combines:

- **CARLA Simulator** for high-fidelity highway scenario testing
- **LUNA-Net** for all-weather road segmentation and edge detection
- **Automated Path Planning** for precise lane marking generation
- **Multi-Modal Sensing** (RGB + Depth + LiDAR)

### Key Features

- ✅ Autonomous road surface detection
- ✅ All-weather operation (day/night/fog)
- ✅ High-precision marking (±3cm lateral error)
- ✅ Full perception-planning-control pipeline
- ✅ CARLA-based validation platform

---

## System Architecture

```
┌──────────────────────────────────────────────────┐
│              AutoStripe Pipeline                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  Perception → Planning → Control                 │
│  (LUNA-Net)   (Path Gen)  (Paint Exec)          │
│      ↓            ↓            ↓                 │
│   Road Mask   Lane Paths   Spray Control        │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
AutoStripe/
├── carla_env/          # CARLA simulation environment
├── perception/         # LUNA-Net road perception
├── planning/           # Lane marking path planning
├── control/            # Marking execution control
├── evaluation/         # Metrics & visualization
├── datasets/           # CARLA-collected datasets
├── configs/            # Configuration files
└── docs/               # Documentation
```

---

## Installation

### Prerequisites

- Python 3.8+
- CARLA 0.9.15
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone repository
git clone <your-repo-url>
cd AutoStripe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set CARLA Python API path
export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg
```

---

## Quick Start

### 1. Launch CARLA Server

```bash
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh
```

### 2. Collect Data

```bash
python carla_env/data_collector.py --map Town04 --weather ClearDay
```

### 3. Run Road Perception

```bash
python perception/road_segmentation.py --input datasets/carla_highway/
```

### 4. Generate Marking Paths

```bash
python planning/lane_calculator.py --config configs/marking_standard.yaml
```

---

## Lane Marking Standards

Based on **GB 5768-2009** (China Road Traffic Signs and Markings):

| Type | Color | Width | Pattern |
|------|-------|-------|---------|
| Lane Divider | White Dashed | 15cm | 6m line / 9m gap |
| Edge Line | White Solid | 15cm | Continuous |
| Center Line | Yellow Solid | 15cm | Double lines |

---

## Evaluation Metrics

| Category | Metric | Target |
|----------|--------|--------|
| Perception | Road IoU | >95% |
| Planning | Lateral Error | <±3cm |
| Execution | Line Width Error | <±1cm |
| System | FPS | >10 |

---

## Roadmap

- [ ] Phase 1: CARLA environment setup + data collection
- [ ] Phase 2: LUNA-Net adaptation & training
- [ ] Phase 3: Lane marking path planning algorithm
- [ ] Phase 4: Closed-loop simulation & evaluation
- [ ] Phase 5: Real-world deployment preparation

---

## References

- [CARLA Simulator](https://carla.org/)
- [LUNA-Net Paper](https://github.com/your-luna-net-repo)
- [SNE-RoadSegV2 (IEEE TIM 2025)](https://ieeexplore.ieee.org/)
- GB 5768-2009: Road Traffic Signs and Markings

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Contact

**Project Status**: Planning Phase
**Last Updated**: 2026-02-08

For questions or collaboration: [Your Contact]

---

🚀 **Powered by CARLA + LUNA-Net**
