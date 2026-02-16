# AutoStripe: 新建高速公路自动标线系统

## CARLA仿真平台 + 多模态道路感知（GT / VLLiNet / LUNA-Net）

## 1. 项目背景

### 1.1 现状问题

| 问题 | 描述 |
|------|------|
| **人工依赖** | 当前高速公路标线主要依赖人工手推式划线机，效率低 |
| **精度不足** | 人工操作导致线型偏移、宽度不均、间距不一致 |
| **安全隐患** | 施工人员长时间暴露在高速公路环境中 |
| **效率瓶颈** | 单台手推机日均标线量有限，新建高速通车周期受限 |
| **夜间施工** | 为避免交通影响，标线施工常在夜间进行，人工操作更加困难 |

### 1.2 创新目标

设计一种**自动标线系统**，能够：
- 自主感知新建高速公路路面区域
- 自动规划标线路径（车道线、边缘线、导流线等）
- 精确控制喷涂执行
- 适应昼夜及不同天气条件施工

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                    AutoStripe V5 系统架构                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │    感知模块       │──→│  规划模块     │──→│  执行模块     │  │
│  │ (3-Mode Switch) │   │(Path Planner)│   │(Paint Control)│  │
│  └─────────────────┘   └──────────────┘   └──────────────┘  │
│       │                      │                    │          │
│       ▼                      ▼                    ▼          │
│  ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ G键循环切换:      │   │ 视觉路径规划  │   │ 自动喷涂控制  │  │
│  │  GT (CityScapes)│   │ 多项式外推    │   │ PD控制器      │  │
│  │  VLLiNet (AI)   │   │ 曲率前馈      │   │ 虚实线切换    │  │
│  │  LUNA-Net (SNE) │   │ 喷嘴距离估计  │   │ 迟滞状态机    │  │
│  └─────────────────┘   └──────────────┘   └──────────────┘  │
│                                                              │
│  ┌─────────────────┐   ┌──────────────────────────────────┐  │
│  │  评估模块 (E键)  │   │  数据采集 (collect_night_dataset) │  │
│  │ Map API GT对比   │   │  KITTI格式 + SNE预计算            │  │
│  │ 感知精度指标     │   │  3地图 × 4天气 = 3600帧           │  │
│  │ 31列帧日志CSV   │   │  训练/验证 按地图划分              │  │
│  └─────────────────┘   └──────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. LUNA-Net 在本项目中的作用

### 3.1 为什么用 LUNA-Net？

| 需求 | LUNA-Net 能力 | 适配度 |
|------|--------------|--------|
| 识别可行驶路面区域 | 道路分割 (Road Segmentation) | ★★★★★ |
| 夜间施工场景 | 低光照增强 (LLEM模块) | ★★★★★ |
| 检测路面边缘/路沿 | 边缘检测 (Edge Head) | ★★★★☆ |
| 恶劣天气施工 | 光照自适应融合 (IAF模块) | ★★★★☆ |
| 深度感知路面平整度 | 表面法线估计 (R-SNE) | ★★★☆☆ |

### 3.2 LUNA-Net 感知输出（V5 实际实现）

```
LUNA-Net 输入:
  RGB图像 [0,1] (1248×384) + SNE表面法线 (3, 384, 1248)
  SNE由深度相机计算: depth(H,W) + cam_param(3,4) → normal(3,H,W)

LUNA-Net 输出:
  └── 2-class logits → argmax → road_mask (0/255 二值掩码)

VLLiNet 输入 (对比):
  RGB (ImageNet归一化) + Depth (min-max归一化), 均为 1248×384

VLLiNet 输出:
  └── sigmoid > 0.5 → road_mask (624×192, 上采样至1248×384)
```

### 3.3 感知 → 标线规划 流程

```
Step 1: LUNA-Net 分割出路面区域
         ↓
Step 2: 从road_mask提取路面边界线
         ↓
Step 3: 根据道路宽度和标准，计算车道线位置
         ↓
Step 4: 生成标线路径 (直线段、虚线间距、导流线弧线)
         ↓
Step 5: 控制喷涂执行
```

---

## 4. CARLA 仿真方案

### 4.1 为什么用 CARLA？

- 提供高保真3D高速公路场景
- 支持多天气/光照条件仿真
- 可获取精确的Ground Truth（路面语义、深度、车道线位置）
- 支持自定义车辆（模拟划线机）
- 无需实际施工即可验证算法

### 4.2 仿真场景设计

| 场景 | CARLA地图 | 天气条件 | 目的 |
|------|----------|----------|------|
| 场景1 | Town04 (高速公路) | ClearDay | 基准测试 |
| 场景2 | Town04 | ClearNight | 夜间施工验证 |
| 场景3 | Town04 | HeavyFog | 雾天施工验证 |
| 场景4 | Town06 (多车道) | ClearDay | 多车道标线 |
| 场景5 | 自定义地图 | Mixed | 新建公路 (无标线) |

### 4.3 CARLA 仿真流程（实际执行）

```
Phase 1: 数据采集 ✓ (已完成)
  - 3地图 (Town04/05/06) × 4天气 × 300帧 = 3600帧
  - 采集 RGB + Depth(u16+float32) + SNE Normal + 语义分割GT + 标定
  - KITTI格式，按地图划分训练/验证集
  - 数据集: CARLA_Unified_Dataset (28GB, 压缩后1.2GB)

Phase 2: 感知模型训练 ✓ (已完成)
  - VLLiNet: MobileNetV3 backbone, MaxF 98.33%, IoU 96.72%
  - LUNA-Net: Swin Transformer + SNE, ClearNight F1=97.18%, IoU=94.52%

Phase 3: 闭环仿真验证 ✓ (已完成)
  - V1: Map API 验证控制逻辑
  - V2: 视觉感知闭环 (语义相机GT)
  - V3: 手动控制 + CityScapes感知
  - V4: VLLiNet AI感知 + 多项式外推 + PD控制 + 评估系统
  - V5: LUNA-Net三模态感知 + 夜间模式

Phase 4: 定量评估 ✓ (已完成)
  - Map API GT对比 (trajectory_evaluator)
  - 感知精度: mask IoU + edge deviation (perception_metrics)
  - 31列帧日志CSV + 8面板时序图 + 地图可视化
```

### 4.4 关键传感器配置（V5 当前）

```python
# CARLA 传感器配置 (V4/V5, 匹配 setup_scene_v2.py)
sensors = {
    'front_rgb': {
        'type': 'sensor.camera.rgb',
        'x': 2.5, 'y': 0.0, 'z': 3.5,
        'pitch': -15,
        'width': 1248, 'height': 384,
        'fov': 90
    },
    'front_depth': {
        'type': 'sensor.camera.depth',
        # 与RGB相机同位置同参数
    },
    'front_semantic': {
        'type': 'sensor.camera.semantic_segmentation',
        # 与RGB相机同位置同参数 (GT模式使用)
    },
    'overhead_rgb': {
        'type': 'sensor.camera.rgb',
        'z': 25, 'pitch': -90,
        'width': 1800, 'height': 1600,
        'fov': 90
    }
}

# 相机内参 (FOV=90, 1248x384)
# fx = fy = 1248 / (2 * tan(45°)) = 624
# cx = 624, cy = 192
```

**注**：V5 不使用 LiDAR，深度信息完全来自 CARLA 深度相机。

---

## 5. 标线类型与规范

### 5.1 高速公路标线类型

根据 GB 5768-2009《道路交通标志和标线》：

| 标线类型 | 颜色 | 宽度(cm) | 说明 |
|----------|------|----------|------|
| 车道分界线 | 白色虚线 | 15 | 长600cm，间距900cm |
| 车道边缘线 | 白色实线 | 15 | 连续实线 |
| 路缘带标线 | 白色实线 | 20 | 路肩边缘 |
| 中央分隔线 | 黄色实线 | 15 | 双黄线 |
| 导流线 | 白色斜线 | 45 | 分流/合流区域 |
| 减速标线 | 白色菱形 | - | 收费站/匝道前 |

### 5.2 标线精度要求

| 指标 | 要求 |
|------|------|
| 横向偏移 | ≤ ±3cm |
| 线宽误差 | ≤ ±1cm |
| 虚线间距误差 | ≤ ±5cm |
| 直线度 | ≤ 5mm/m |

---

## 6. 技术路线

### 6.1 整体技术路线（实际执行）

```
阶段一: CARLA环境搭建 + 数据采集 ✓
  │
  ├── Town04/05/06 高速公路场景
  ├── 1248×384 相机, x=2.5, z=3.5, pitch=-15, FOV=90
  ├── 4种天气: ClearNight/ClearDay/HeavyFoggyNight/HeavyRainFoggyNight
  └── KITTI格式数据集 (RGB + Depth + SNE Normal + GT Mask + Calib)
  │
阶段二: 感知模型训练 ✓
  │
  ├── VLLiNet (MobileNetV3): MaxF 98.33%, IoU 96.72%
  ├── LUNA-Net (Swin-T + SNE): ClearNight F1=97.18%, IoU=94.52%
  └── 三模态切换: GT / VLLiNet / LUNA-Net (G键循环)
  │
阶段三: 闭环控制算法 ✓
  │
  ├── V1: Map API + Pure Pursuit 验证
  ├── V2: 视觉感知闭环 (语义相机)
  ├── V3: 手动控制 + CityScapes感知
  ├── V4: AI感知 + PD控制 + 曲率前馈 + 自动喷涂状态机
  └── V5: LUNA-Net三模态 + 夜间模式
  │
阶段四: 定量评估系统 ✓
  │
  ├── Map API GT对比 (覆盖率、横向误差)
  ├── 感知精度 (mask IoU, edge deviation)
  ├── 31列帧日志 + 8面板时序图
  └── 7种地图可视化 (Nature风格, PDF/SVG矢量输出)
```

### 6.2 评估指标（V5 实际使用）

| 类别 | 指标 | 说明 | 实现状态 |
|------|------|------|----------|
| **感知** | Road F1 / IoU | 道路分割精度 | ✓ LUNA-Net F1=97.18% |
| **感知** | mask_iou | AI vs GT 掩码交并比 | ✓ 逐帧计算 |
| **感知** | edge_dev (px) | AI vs GT 右边缘偏差 | ✓ mean/median/max |
| **感知** | inference_ms | 模型推理耗时 | ✓ CUDA同步计时 |
| **规划** | Lateral Error (m) | 喷嘴到路沿横向误差 | ✓ Map API GT对比 |
| **规划** | Coverage Rate (%) | GT路径覆盖率 | ✓ 2m阈值 |
| **控制** | nozzle_dist (m) | 喷嘴到路沿实时距离 | ✓ 多项式外推 |
| **控制** | driving_offset (m) | 动态驾驶偏移量 | ✓ PD+前馈控制 |
| **系统** | FPS | 系统帧率 | ✓ HUD实时显示 |
| **系统** | sne_time_ms | SNE计算耗时 | ✓ LUNA-Net模式 |

---

## 7. 项目结构（V5 实际）

```
AutoStripe/
├── manual_painting_control_v4.py  # V5 主入口 (3模态感知 + N键夜间)
├── manual_painting_control.py     # V3 主入口 (GT感知 + 手动控制)
├── main_v2.py                     # V2 独立入口 (自动模式)
├── main_v1.py                     # V1 入口 (Map API)
├── diag_luna.py                   # LUNA-Net 独立验证脚本
├── diag_vllinet.py                # VLLiNet 独立验证脚本
├── carla_env/
│   ├── setup_scene.py             # V1 场景
│   └── setup_scene_v2.py          # V2-V5 场景: 1248×384, (2.5, 3.5, -15)
├── perception/
│   ├── road_segmentor.py          # GT: CityScapes颜色匹配 → 路面掩码
│   ├── road_segmentor_ai.py       # VLLiNet: MobileNetV3 → 路面掩码
│   ├── road_segmentor_luna.py     # LUNA-Net: Swin-T + SNE → 路面掩码
│   ├── edge_extractor.py          # 路面掩码 → 左右边缘像素
│   ├── depth_projector.py         # 像素+深度 → 世界坐标
│   └── perception_pipeline.py     # 3模态切换 (PerceptionMode枚举)
├── planning/
│   ├── lane_planner.py            # V1 Map API规划 + 道路几何
│   └── vision_path_planner.py     # V2-V5 视觉规划 + 多项式外推 + 曲率前馈
├── control/
│   ├── marker_vehicle.py          # V1 Pure Pursuit
│   └── marker_vehicle_v2.py       # V2-V5 Pure Pursuit + 动态路径
├── evaluation/
│   ├── trajectory_evaluator.py    # Map API GT对比 + 8列detail CSV
│   ├── perception_metrics.py      # mask IoU + edge deviation
│   ├── frame_logger.py            # 31列帧日志CSV
│   ├── visualize_eval.py          # 评估图 + 8面板时序图
│   └── visualize_map.py           # 7种地图可视化 (Nature风格, PDF/SVG)
├── ros_interface/
│   ├── topic_config.py            # ROS话题常量
│   ├── rviz_publisher.py          # RVIZ发布 + 多项式曲线
│   └── autostripe_node.py         # V4 ROS节点 (CARLA-ROS Bridge)
├── datasets/
│   └── carla_highway/
│       ├── collect_night_dataset.py    # 统一数据集采集脚本
│       └── visualize_trajectory.py     # 轨迹可视化 (matplotlib矢量输出)
├── VLLiNet_models/
│   ├── models/vllinet.py          # VLLiNet_Lite 模型
│   ├── models/backbone.py         # MobileNetV3 + LiDAREncoder
│   └── checkpoints_carla/best_model.pth
├── configs/rviz/                  # RVIZ布局文件
├── launch/                        # ROS launch文件
└── docs/
    └── Project_Design.md          # 本文档
```

---

## 8. 创新点总结

| # | 创新点 | 描述 |
|---|--------|------|
| 1 | **三模态感知切换** | GT / VLLiNet / LUNA-Net 实时切换对比，支持A/B测试 |
| 2 | **全天候施工能力** | LUNA-Net LLEM模块支持夜间 (F1=97.18%)，4种天气预设 |
| 3 | **SNE表面法线融合** | 深度→表面法线→道路分割，增强低光照场景鲁棒性 |
| 4 | **闭环自动喷涂控制** | PD控制器 + 曲率前馈 + 迟滞状态机，弯道不中断喷涂 |
| 5 | **CARLA仿真验证平台** | 完整的采集→训练→推理→评估闭环 |
| 6 | **定量评估体系** | Map API GT对比 + 感知精度指标 + 31列帧日志 + 矢量可视化 |

---

## 9. 版本演进

| 版本 | 内容 | 状态 |
|------|------|------|
| V1 | Map API + Pure Pursuit 验证控制逻辑 | ✓ 已完成 |
| V2 | 视觉感知闭环 (语义相机GT) | ✓ 已完成 |
| V3 | 手动控制 + CityScapes感知 + 增强可视化 | ✓ 已完成 |
| V4 | VLLiNet AI感知 + 多项式外推 + 1248×384原生分辨率 | ✓ 已完成 |
| V4.1 | 自适应转向 + 动态偏移P控制 + 自动喷涂状态机 | ✓ 已完成 |
| V4.2 | PD控制 + 评估系统 + 虚线模式 + 感知精度指标 | ✓ 已完成 |
| V4.3 | 曲率前馈 + 自适应平滑 (弯道不中断喷涂) | ✓ 已完成 |
| V5 | LUNA-Net三模态感知 + SNE + 夜间模式 | ✓ 已完成 |
| V5+ | 数据集采集 (3600帧, 3地图×4天气) | ✓ 已完成 |

---

## 10. 参考资料

- LUNA-Net: Low-light Urban Navigation and Analysis Network
- SNE-RoadSegV2 (IEEE TIM 2025)
- CARLA Simulator: https://carla.org/
- GB 5768-2009 道路交通标志和标线
- JTG D20-2017 公路路线设计规范

---

## 12. 项目文档

| 文档 | 说明 |
|------|------|
| `docs/Project_Design.md` | 本文档：项目设计 + 实际实现进展 (V1-V5) |
| `CLAUDE.md` | 项目上下文文档（供 AI 助手使用，含完整技术参数） |

---

**Last Updated**: 2026-02-13
**Status**: V5 已完成（三模态感知 + 数据集采集），下一步：用新数据集重训练 LUNA-Net / VLLiNet

## 11. 实际实现进展

### 11.1 V1 版本：Map API 验证版（已完成）

**实现时间**：2026-02-08

**核心思路**：跳过感知模块，直接使用 CARLA Map API 获取完美路径，验证控制和喷涂逻辑。

#### V1 架构

```
CARLA Map API (完美路径)
        ↓
  Lane Planner (路径点生成)
        ↓
  Pure Pursuit Controller (路径跟踪)
        ↓
  Nozzle Trajectory Painting (喷嘴轨迹画线)
        ↓
  Overhead Display (俯视可视化)
```

#### V1 关键技术

| 模块 | 实现方式 | 说明 |
|------|----------|------|
| 感知 | **无** (使用 Map API) | 跳过感知，直接获取完美路径 |
| 规划 | `lane_planner.py` | 从 Map API 生成 200 个路径点 |
| 控制 | `marker_vehicle.py` | Pure Pursuit 算法 (wheelbase=2.875, Kdd=4.0) |
| 喷涂 | 喷嘴轨迹画线 | 车辆位置 + 右偏移 2.0m → 喷嘴位置 |
| 可视化 | 俯视相机 + OpenCV | z=25m 俯视图 + 轨迹叠加 |

#### V1 核心发现

**喷涂逻辑**：车辆本身就是划线机，标线 = 车辆实际轨迹 + 右侧偏移 2.0m

```python
# 每帧计算喷嘴位置
yaw = vehicle.get_transform().rotation.yaw
nozzle_x = veh_x + 2.0 * cos(yaw + pi/2)  # 右侧偏移
nozzle_y = veh_y + 2.0 * sin(yaw + pi/2)

# 画线：从上一帧喷嘴位置到当前喷嘴位置
world.debug.draw_line(prev_nozzle, curr_nozzle, 
                      color=(255,255,0), thickness=0.3, life_time=1000)
```

#### V1 性能指标

- **测试距离**：200m+
- **平均速度**：~5 m/s
- **横向偏差**：< 0.5m（Pure Pursuit 跟踪精度）
- **标线连续性**：完美（每帧画线）
- **可视化**：俯视图实时显示车辆轨迹和标线

#### V1 局限性

| 问题 | 说明 |
|------|------|
| **不符合真实工况** | Map API 提供完美路径，无感知误差 |
| **无感知模块** | 跳过了最核心的视觉感知部分 |
| **无法验证鲁棒性** | 无法测试感知误差对系统的影响 |
| **不可迁移** | 真实场景无 Map API |

**结论**：V1 成功验证了控制和喷涂逻辑，但需要 V2 补充真实感知模块。


---

### 11.2 V2 版本：视觉感知驱动版（已完成）

**实现时间**：2026-02-09

**核心思路**：用视觉感知替代 Map API，实现感知→规划→控制→喷涂的完整闭环。

#### V2 架构

```
语义分割相机 + 深度相机 (CARLA Sensors)
        ↓
  Perception Pipeline
    ├── Road Segmentor (路面分割)
    ├── Edge Extractor (边缘提取)
    └── Depth Projector (深度投影)
        ↓
  Vision Path Planner (视觉路径规划)
    ├── 右路沿 → 驾驶路径 (左偏移 5m)
    └── 驾驶路径 → 喷嘴路径 (右偏移 2m)
        ↓
  Pure Pursuit Controller V2 (动态路径更新)
        ↓
  Nozzle Trajectory Painting (喷嘴轨迹画线)
        ↓
  Real-time Visualization (实时可视化)
```

#### V2 传感器配置

| 传感器 | 类型 | 位置 | 参数 |
|--------|------|------|------|
| 语义分割相机 | `sensor.camera.semantic_segmentation` | x=2.5, z=2.8, pitch=-15 | 800x600, FOV=90° |
| 深度相机 | `sensor.camera.depth` | x=2.5, z=2.8, pitch=-15 | 800x600, FOV=90° |
| 俯视RGB相机 | `sensor.camera.rgb` | z=25, pitch=-90 | 1800x1600, FOV=90° |
| 语义LiDAR | `sensor.lidar.ray_cast_semantic` | z=1.8 | 32ch, 30m range |

**注**：V2 使用 CARLA 语义相机（完美语义标签），未使用 LUNA-Net。这是向真实感知过渡的中间步骤。


#### V2 关键技术

**1. 感知模块 (perception/)**

| 组件 | 功能 | 关键参数 |
|------|------|----------|
| `road_segmentor.py` | 语义标签 → 路面掩码 | tags: 0,1,6 (Unlabeled, Road, RoadLines) |
| `edge_extractor.py` | 路面掩码 → 左右边缘像素 | MIN_ROAD_RUN=20, MAX_DEPTH=30m, MAX_LEFT_SCAN=200px |
| `depth_projector.py` | 像素+深度 → 世界坐标 | fx=fy=400, CARLA深度解码 |
| `perception_pipeline.py` | 组合以上三步 | 每帧输出世界坐标路沿点 |

**边缘提取策略**：
- 从图像中心向外扫描，找第一个路沿标签像素
- 要求先经过 ≥20 个连续路面像素（过滤路标轮廓）
- Pole/Vehicle 标签透明（不中断路面连续性）
- 深度过滤：1.5m < depth < 30m（过滤车辆引擎盖和远处山/树）

**2. 规划模块 (planning/vision_path_planner.py)**

```python
# 偏移距离
line_offset = 3.0m      # 喷嘴距右路沿距离
nozzle_arm = 2.0m       # 喷嘴距车辆中心距离
driving_offset = 5.0m   # 车辆中心距右路沿距离

# 路径生成流程
右路沿点 → 按纵向排序 → 重采样(1m间隔) → 滑动窗口平滑
         → 向左偏移5m → 驾驶路径 (蓝色线)
         → 向右偏移2m → 喷嘴路径 (黄色线)
```

**关键理解**：蓝色引导线 = 当前感知的实时引导，每帧替换（不累积）

**3. 控制模块 (control/marker_vehicle_v2.py)**

| 参数 | 数值 | 说明 |
|------|------|------|
| LOOKAHEAD_WPS | 15 | 前瞻点数（更平滑转向） |
| TARGET_SPEED | 3.0 m/s | 目标巡航速度 |
| 前瞻距离 | max(actual_dist, 5.0) | 使用实际距离，最小5m |
| 速度维持 | 动态油门 0.2-0.5 | 防止转向时失速 |


#### V2 性能指标

**测试环境**：Town05 高速公路路段，起点 (10, -210, 1.85), yaw=180°

| 类别 | 指标 | 数值 | 说明 |
|------|------|------|------|
| **感知** | 路面识别率 | 52.7-52.9% | 图像下半部分路面像素占比 |
| **感知** | 路径点数 | 24-26 点/帧 | 稳定在此范围 |
| **感知** | 路径前瞻距离 | 28-31m | 符合 MAX_DEPTH=30m 限制 |
| **规划** | 路径更新频率 | 每帧 | 实时替换缓冲区 |
| **规划** | 偏移精度 | ±0.2m | 驾驶路径距右路沿 5.0±0.2m |
| **控制** | 平均速度 | 3.6 m/s | 略高于目标 3.0 m/s |
| **控制** | 横向偏差 | 0.4m | 收敛到稳定值 |
| **系统** | 测试距离 | 156-400m | 多次测试 |
| **系统** | 稳定性 | ✓ 通过 | 直道、弯道、速度维持全部通过 |

**典型运行数据**：
- F100: 车辆行驶 ~12m，路径前瞻 22m
- F300: 车辆行驶 ~35m，路径前瞻 22m
- F500: 车辆行驶 ~71m，路径前瞻 22m
- F650: 车辆行驶 ~128m，路径前瞻 28m


#### V2 调试过程总结

V2 开发过程中解决了 9 个关键问题（详见 `experiment_log_v2.md`）：

| 阶段 | 问题 | 解决方案 | 效果 |
|------|------|----------|------|
| 1 | 语义标签识别错误 | 增加 found_road 约束 | 边缘检测正确 |
| 2 | 路标轮廓误识别 | MIN_ROAD_RUN=20 + Pole透明 | 过滤小路面碎片 |
| 3 | tag 1 误识别为 Building | 确认 tag 1 = Road | 路面识别率 >50% |
| 4 | 路径偏移方向错误 | 左偏移 = (dy, -dx) | 车辆正确在路面内 |
| 5 | 车辆抖动 | LOOKAHEAD_WPS=15, STEER_FILTER=0.15 | 平稳行驶 |
| 6 | 左边缘跨越对向车道 | MAX_LEFT_SCAN=200 | 限制在本车道 |
| 7 | 路径不更新 | 每帧替换缓冲区 | 蓝色线实时刷新 |
| 8 | 车辆速度衰减至零 | ld=max(actual,5.0) + 速度维持 | 稳定行驶 >100m |
| 9 | 远处山/树误检 | MAX_DEPTH=30m | 无远处误检 |

**关键技术突破**：
- CARLA 左手坐标系的正确理解（左偏移方向）
- 实时路径规划策略（替换 vs 累积）
- Pure Pursuit 在低速时的失速问题（速度维持机制）


---

### 11.3 V1 vs V2 对比分析

| 维度 | V1 (Map API) | V2 (Vision-based) |
|------|-------------|-------------------|
| **路径来源** | CARLA Map API (完美) | 语义分割 + 深度投影 (真实) |
| **感知模块** | 无 (作弊) | 语义相机 + 深度相机 + 边缘提取 |
| **路径更新** | 启动时一次性生成 200 点 | 每帧实时生成 24-26 点 |
| **前瞻距离** | 200m (全局路径) | 28-30m (感知范围) |
| **真实性** | 不符合真实工况 | 符合真实划线机 |
| **鲁棒性** | 完美路径，无误差 | 受感知质量影响 |
| **速度** | ~5 m/s | ~3.6 m/s |
| **横向偏差** | < 0.5m | 0.4m |
| **开发难度** | 低（跳过感知） | 高（完整闭环） |
| **可迁移性** | 不可迁移到真实场景 | 可迁移（替换感知模型） |

**核心区别**：
- V1 验证了"控制+喷涂"逻辑，但跳过了最核心的感知部分
- V2 实现了完整的"感知→规划→控制→喷涂"闭环，更接近真实系统
- V3 在 V2 基础上增加手动控制、增强可视化、CityScapes 感知方案


---

### 11.4 V3 版本：手动喷涂控制 + 增强可视化（已完成）

**实现时间**：2026-02-09

**核心思路**：在 V2 视觉感知闭环基础上，增加手动/自动驾驶切换、喷涂开关控制、坡道自适应可视化、喷嘴边距可视化。

#### V3 架构

```
语义分割相机 → CityScapes 调色板转换
        ↓
  Perception Pipeline (CityScapes 颜色匹配)
    ├── Road Segmentor (紫色128,64,128 → 路面掩码)
    ├── Edge Extractor (边缘提取)
    └── Depth Projector (深度投影，保留z坐标)
        ↓
  Vision Path Planner (视觉路径规划，含z坐标)
        ↓
  Manual/Auto Control (手动/自动驾驶切换)
        ↓
  Paint Control (喷涂开关，间断喷涂支持)
        ↓
  Enhanced Visualization
    ├── 蓝色点标记 (驾驶路径，坡道z自适应)
    ├── 绿色垂线 (喷嘴到路沿距离)
    ├── 黄色画线 (喷涂轨迹，支持间断)
    └── pygame 前视图 + OpenCV 俯视图
```

#### V3 关键改进

| 改进项 | V2 方案 | V3 方案 | 效果 |
|--------|---------|---------|------|
| 路面分割 | 原始语义标签 ID 匹配 | CityScapes 调色板颜色匹配 | 兼容非标准标签 ID |
| 驾驶路径显示 | 蓝色线段 (draw_line) | 蓝色点标记 (draw_point) | 更清晰 |
| 路径z坐标 | 固定 z=0.5 | 车辆 pitch 推算坡道z | 坡道/天桥正确显示 |
| 边距计算 | 车辆到最近路沿点 | 喷嘴到车身垂线路沿交点 | 物理意义正确 |
| 边距可视化 | 无 | 绿色垂线 (喷嘴→路沿) | 直观显示距离 |
| 喷涂控制 | 始终喷涂 | SPACE 键开关，支持间断 | 灵活控制 |
| 驾驶模式 | 仅自动 | TAB 切换自动/手动 | 支持手动干预 |
| 前视显示 | OpenCV 窗口 | pygame 窗口 + 路面掩码叠加 | 集成控制+显示 |

#### V3 操作说明

| 按键 | 功能 |
|------|------|
| SPACE | 切换喷涂 ON/OFF |
| TAB | 切换自动/手动驾驶 |
| WASD/方向键 | 手动驾驶（油门/转向/刹车） |
| Q | 切换倒车模式 |
| X | 手刹 |
| ESC | 退出 |

#### V3 调试过程总结

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 原始标签 ID 不匹配 | CARLA 0.9.15-dirty 非标准标签 | 改用 CityScapes 颜色匹配 |
| Front Perception 窗口黑屏 | 创建但未写入 | 移除该窗口 |
| 黄线暂停恢复连接旧位置 | last_nozzle_loc 未重置 | 暂停时置 None + 间断标记 |
| 蓝线在天桥下方 | z 固定为 0.5 | 使用车辆 pitch 推算 z |
| 蓝线嵌入地面 | 车辆 z 在底盘位置 | 加 +0.5m 偏移 |
| 蓝线爬坡穿入坡面 | 所有点用同一 z | 按纵向距离×坡度推算每点 z |
| 绿线斜向前方 | 前视相机无侧方路沿点 | 用路沿横向距离中位数构造垂直交点 |
| 绿线嵌入地面 | z 未加偏移 | 喷嘴端和路沿端均 +0.5m |


---

### 11.5 V4 版本：VLLiNet AI感知 + 多项式外推（已完成）

**实现时间**：2026-02-10

**核心思路**：用训练好的 VLLiNet 深度学习模型替代 CARLA 语义相机，实现真正的 AI 感知闭环。同时升级相机分辨率至 1248×384 原生匹配模型输入。

#### V4 架构

```
RGB相机 (1248×384) + 深度相机
        ↓
  VLLiNet_Lite (MobileNetV3 backbone)
    ├── RGB: ImageNet归一化 → [1, 3, 384, 1248]
    └── Depth: CARLA解码 → min-max归一化 → [1, 3, 384, 1248]
        ↓
  sigmoid > 0.5 → 上采样至1248×384 → 路面掩码
        ↓
  Edge Extractor + Depth Projector → 世界坐标路沿点
        ↓
  Vision Path Planner + 多项式外推 (盲区距离估计)
        ↓
  Pure Pursuit Controller + 喷涂控制
```

#### V4 关键改进

| 改进项 | V3 方案 | V4 方案 |
|--------|---------|---------|
| 感知 | CARLA语义相机 (GT) | VLLiNet AI模型 (MaxF 98.33%) |
| 分辨率 | 800×600 | 1248×384 (原生匹配模型) |
| 相机位置 | x=2.5, z=2.8 | x=1.5, z=2.4 (匹配训练数据) |
| 距离估计 | 中位数法 | 多项式二次拟合外推 |
| 模式切换 | 无 | G键切换 AI/GT 对比 |
| ROS集成 | 无 | CARLA-ROS Bridge订阅 |

#### V4.1 自适应控制优化

- **自适应转向滤波**：横向误差大时激进(0.50)，小时平滑(0.15)
- **动态驾驶偏移**：P控制器自动收敛喷嘴距离到目标3.0m
- **自动喷涂状态机**：CONVERGING → STABILIZED(60帧) → PAINTING

#### V4.2 PD控制 + 评估系统

- **PD控制器**：微分项抑制振荡 (Kp=0.5, Kd=0.3)
- **迟滞状态机**：进入/退出容差分离 + 宽限帧，防止抖动
- **评估管线**：E键触发 Map API GT对比，生成CSV + 可视化
- **虚线模式**：D键切换实线/虚线 (3m画/3m间隔)
- **帧日志**：31列CSV记录每帧完整状态
- **感知精度**：mask IoU + edge deviation (AI vs GT逐帧对比)

#### V4.3 曲率前馈

- **曲率前馈**：多项式二次系数预测前方弯道，提前增大偏移量
- **自适应平滑**：弯道时OFFSET_SMOOTH从0.12提升至0.25
- **效果**：消除弯道入口喷涂中断问题

---

### 11.6 V5 版本：LUNA-Net 三模态感知 + 夜间模式（已完成，当前版本）

**实现时间**：2026-02-11

**核心思路**：集成 LUNA-Net 作为第三种感知模式，利用 Swin Transformer + SNE 表面法线估计，专攻夜间/低光照场景的道路分割。

#### V5 架构

```
RGB相机 (1248×384) + 深度相机
        ↓
  G键循环: GT → VLLiNet → LUNA-Net → GT
        ↓
  [LUNA-Net 模式]
    RGB: /255.0 → [0,1] → [1, 3, 384, 1248]
    Depth: CARLA解码 → meters → SNE → normal (3, H, W)
    Normal: → [1, 3, 384, 1248]
    LUNA-Net(rgb, normal, is_normal=True) → 2-class logits → argmax → mask
        ↓
  MASK_TOP_RATIO=0.35 裁剪 → Edge Extractor → Depth Projector
        ↓
  Vision Path Planner (多项式外推 + 曲率前馈)
        ↓
  PD Controller + 自动喷涂状态机
```

#### V5 三模态感知对比

| 维度 | GT (CityScapes) | VLLiNet | LUNA-Net |
|------|-----------------|---------|----------|
| 输入 | 语义相机标签 | RGB+Depth | RGB+SNE Normal |
| 骨干网络 | 无 (颜色匹配) | MobileNetV3 | Swin Transformer Tiny |
| 输出分辨率 | 1248×384 | 624×192 (需上采样) | 1248×384 (原生) |
| 输出格式 | 颜色阈值 | sigmoid > 0.5 | argmax 2-class |
| 额外计算 | 无 | 无 | SNE: depth→normal (CPU) |
| 优势场景 | 完美GT基准 | 通用道路分割 | 夜间/低光照 |
| 精度 | 100% (GT) | MaxF 98.33% | ClearNight F1=97.18% |

#### V5 关键新增

| 功能 | 说明 |
|------|------|
| LUNA-Net感知模式 | Swin-T + LLEM + IAF + NAA decoder + Edge head |
| SNE表面法线 | depth(H,W) + cam_param(3,4) → normal(3,H,W), CPU计算 |
| N键夜间模式 | 切换ClearNight天气 (sun=-30, cloud=10, fog=0) |
| 三模态循环 | G键: GT → VLLiNet → LUNA-Net → GT |
| HUD颜色区分 | GT=白色, VLLiNet=绿色, LUNA-Net=青色 |
| SNE计时 | 帧日志新增sne_time_ms列 (32列) |

#### V5 操作说明

| 按键 | 功能 |
|------|------|
| SPACE | 切换喷涂 ON/OFF |
| TAB | 切换自动/手动驾驶 |
| G | 循环感知模式 (GT → VLLiNet → LUNA-Net) |
| N | 切换ClearNight夜间天气 |
| D | 切换虚线/实线模式 |
| E | 切换评估记录 (开始/停止 + GT对比) |
| R | 切换视频录制 |
| WASD/方向键 | 手动驾驶 |
| V | 切换观察者跟随/自由相机 |
| ESC | 退出 |

---

### 11.7 数据集采集（已完成）

**采集工具**：`datasets/carla_highway/collect_night_dataset.py`

#### 采集配置

| 参数 | 值 |
|------|-----|
| 相机 | 1248×384, x=2.5, z=3.5, pitch=-15, FOV=90 |
| 地图 | Town04 + Town05 + Town06 |
| 天气 | ClearNight, ClearDay, HeavyFoggyNight, HeavyRainFoggyNight |
| 帧数 | 3600 (每地图1200, 每天气300) |
| 车辆 | vehicle.tesla.model3 + autopilot |
| 跳帧 | 每15帧采集1帧 (可选距离跳帧3m) |

#### 数据集结构

```
CARLA_Unified_Dataset/ (28GB, 压缩后1.2GB)
├── training/        (2400帧: Town04 + Town06)
│   ├── image_2/          # RGB PNG (1248×384, uint8)
│   ├── depth_u16/        # 16-bit depth PNG (毫米)
│   ├── depth_meters/     # float32 NPY (米, SNE验证用)
│   ├── normal/           # float32 NPY (3, 384, 1248) SNE预计算
│   ├── gt_image_2/       # 单通道二值mask PNG (road=255, bg=0)
│   └── calib/            # KITTI标定 + 车辆位置
└── validation/      (1200帧: Town05)
    └── [同上]
```

文件命名：`{Map}_{Weather}_{FrameNum:06d}.{ext}`

#### GT提取说明

CARLA 0.9.15-dirty 使用非标准语义标签ID：
- **Road = 1**（标准CARLA中1=Building，但0.9.15-dirty中1=Road，占47.7%）
- **RoadLine = 24**（标准CARLA中24不存在，0.9.15-dirty中24=RoadLine，占2.1%）
- 使用原始语义标签R通道，不经过CityScapes调色板转换

---

### 11.8 下一步计划

| 任务 | 描述 | 优先级 |
|------|------|--------|
| **重训练LUNA-Net** | 用新采集的3600帧数据集重训练，匹配当前相机参数 | ★★★★★ |
| **重训练VLLiNet** | 同上，确保训练-测试分布一致 | ★★★★★ |
| **多天气评估** | 在4种天气下分别运行评估，对比感知精度 | ★★★★☆ |
| **中心线支持** | 检测道路中心线，支持中央分隔线标线 | ★★★☆☆ |
| **真实场景迁移** | 从CARLA训练迁移到真实相机 | ★★★☆☆ |


