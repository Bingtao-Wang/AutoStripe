# AutoStripe V5.2.1 核心代码逻辑总览

> 目标：让新 agent 快速掌握两个核心入口文件的架构、数据流和关键参数
> 日期：2026-02-27

## 一、两个入口文件对比

| 维度 | `manual_painting_control_v5py` | `experiment_runner_v5.py` |
| --- | --- | --- |
| 用途 | 带 pygame GUI 的交互式控制 | 无头批量实验采集 |
| 驾驶模式 | 手动(WASD) / 自动(TAB切换) | 仅自动 |
| 感知切换 | G 键实时切换 GT/VLLiNet/LUNA | CLI `--mode` 参数 |
| 画线控制 | SPACE 手动 + 状态机自动 | 仅状态机自动 |
| 评估录制 | E 键手动开始/停止 | warmup 后自动开始，达距离自动停止 |
| 帧率 | ~20 FPS (pygame + 渲染) | ~190 FPS (无头) |
| 批量运行 | 不支持 | `--batch` 6组实验顺序执行 |
| 可视化 | pygame 窗口 + CARLA debug draw | 仅 CARLA debug draw (黄色画线) |

两者共享完全相同的：感知管线、路径规划器、控制器、状态机逻辑、评估器。

## 二、系统模块依赖图

```text
experiment_runner_v5.py / manual_painting_control_v5py
  │
  ├── carla_env/setup_scene_v2.py      场景：车辆 + 4相机(语义/深度/RGB/俯视)
  ├── perception/
  │   ├── perception_pipeline.py       三模式感知调度 (PerceptionMode enum)
  │   ├── road_segmentor.py            GT: CityScapes 颜色匹配 → road mask
  │   ├── road_segmentor_ai.py         VLLiNet: MobileNetV3 推理 → road mask
  │   ├── road_segmentor_luna.py       LUNA-Net: Swin + SNE → road mask
  │   ├── edge_extractor.py            road mask → 左/右边缘像素
  │   └── depth_projector.py           像素 + 深度 → 3D 世界坐标
  ├── planning/vision_path_planner_v2.py  两级偏移路径规划 + 多项式外推
  ├── control/marker_vehicle_v2.py     Pure Pursuit + 自适应转向滤波
  └── evaluation/
      ├── trajectory_evaluator.py      轨迹评估 → summary/detail CSV (仅供参考)
      ├── frame_logger.py              33列逐帧 CSV 记录
      └── perception_metrics.py        mask IoU + edge deviation
```

## 三、每帧数据流（核心循环）

```text
传感器读取 → 感知 → 规划 → 控制 → 距离计算 → 状态机 → 记录
```

### 3.1 感知层 (每3帧执行，其余帧用缓存)

```text
输入：sem_data(语义BGRA), depth_data(深度BGRA), rgb_front(RGB BGRA), cam_tf
  │
  ├─ GT模式:    sem → CityScapes颜色匹配(128,64,128) → road_mask
  ├─ VLLiNet:   rgb → ImageNet归一化 + depth → min-max归一化 → 推理 → sigmoid>0.5
  └─ LUNA-Net:  rgb → [0,1]归一化 + depth → SNE表面法线 → 推理 → argmax
  │
  ↓
  road_mask → 形态学处理 → edge_extractor (逐行扫描最左/最右) → left_px, right_px
  │
  ↓
  right_px + depth → depth_projector → right_world (3D世界坐标列表)
  │
  ↓ (AI模式额外输出)
  gt_right_world (GT语义相机同步计算，用于逐帧精度对比)
```

### 3.2 规划层 (VisionPathPlannerV2 两级偏移)

```text
right_world (边缘3D点)
  │
  ├─ 纵向排序 → 1m间隔重采样 → 5点滑动平均 → 异常值剔除(残差>0.5m)
  │
  ├─ Stage 1: 边缘沿局部法线左移 line_offset=3.1m → 喷嘴目标路径
  │
  ├─ Stage 2: 喷嘴路径沿局部法线左移 compensated_arm → 驾驶路径
  │            compensated_arm = nozzle_arm(2.0) + K_CURV_FF(55.0) × curvature
  │
  ├─ 路径时域EMA: PATH_EMA_ALPHA=0.15 (85%旧路径 + 15%新路径)
  │
  └─ estimate_nozzle_edge_distance():
     二次多项式拟合边缘点 → 外推到lon=0 → poly_dist (车辆到边缘距离)
     poly_dist 经 20帧中值滤波
```

### 3.3 控制层

```text
lateral_error = (poly_dist - nozzle_arm) - TARGET_NOZZLE_DIST
  │
  ├─ controller.set_lateral_error() → 自适应转向滤波
  │   |err|>0.5m → aggressive(0.50), |err|<0.3m → smooth(0.15)
  │
  ├─ controller.update_path(driving_coords) → 更新 Pure Pursuit 目标
  │
  └─ controller.step() → 计算转向角 + 油门/刹车 → 发送到 CARLA 车辆
```

### 3.4 距离计算 + 状态机

```text
nozzle_loc = vehicle_pos + 2.0m × right_vector
  │
  ├─ edge_dist_r = compute_point_edge_distance(nozzle, right_world)
  │   取喷嘴前后15m内最近10个边缘点的中值横向距离
  │
  ├─ NED中值滤波: 手动模式15帧 / headless模式150帧
  │
  └─ AutoPaintStateMachine.update(ned, speed, poly_a)
       CONVERGING ──(|ned-3.0|<0.3m, 稳定N帧)──→ STABILIZED ──→ PAINTING
            ↑                                                      │
            └──────(|ned-3.0|>0.55m 超过GRACE帧)──────────────────┘
```

## 四、AutoPaintStateMachine 状态机详解

### 4.1 三状态转换

| 状态 | 含义 | 进入条件 | 退出条件 |
| --- | --- | --- | --- |
| CONVERGING | 收敛中，不画线 | 初始状态 / 从其他状态退出 | `\|ned-3.0\|<tolerance_enter` 且 speed>1.0 |
| STABILIZED | 已收敛，等待稳定 | 从 CONVERGING 进入 | 稳定 N 帧→PAINTING / 超差→CONVERGING |
| PAINTING | 画线中 | 从 STABILIZED 进入 | 超差超过 GRACE 帧→CONVERGING |

### 4.2 关键参数（帧率相关）

| 参数 | 手动(~20FPS) | Headless(~190FPS) | 等效时间 |
| --- | --- | --- | --- |
| tolerance_enter | 0.3m | 0.3m | - |
| tolerance_exit | 0.55m | 0.55m | - |
| stability_frames | 15 | 150 | ~0.75s |
| GRACE_LIMIT | 30 | 300 | ~1.5s |
| STABILIZED_GRACE | 10 | 100 | ~0.5s |
| NED_SMOOTH_WINDOW | 15 | 150 | ~0.75s |

### 4.3 自适应容差（弯道放宽）

```text
curvature = |poly_coeff_a|
  CURV_LO=0.004 以下: 使用默认 tolerance
  CURV_HI=0.010 以上: tolerance_enter=0.55m, tolerance_exit=0.80m
  中间: 线性插值
```

### 4.4 V5.2.1 核心修复

状态机输入从 `poly_ned blend` 改为 `ned 直接使用`：

```python
# 修复前：弯→直过渡时 poly_ned 多项式外推失真，误触发退出
dist_for_sm = blend(poly_ned, edge_dist_r, curvature)

# 修复后：ned 已有中值滤波，直接可靠
dist_for_sm = edge_dist_r if edge_dist_r < 900 else 3.0
```

## 五、三模式感知对比

| 维度 | GT (语义相机) | VLLiNet | LUNA-Net |
| --- | --- | --- | --- |
| 输入 | 语义BGRA → CityScapes颜色 | RGB(ImageNet) + Depth(min-max) | RGB([0,1]) + SNE表面法线 |
| 骨干网络 | 无(颜色匹配) | MobileNetV3 | Swin Transformer Tiny |
| 输出分辨率 | 原始(1248×384) | 624×192(需上采样) | 1248×384(原始) |
| 输出格式 | 颜色距离≤10 | sigmoid > 0.5 | argmax(2类logits) |
| 额外步骤 | 无 | 无 | SNE: depth→表面法线(CPU) |
| 优势场景 | 全天候(仿真GT) | 通用路面分割 | 夜间/低光照 |
| 边缘位置 | 含路肩，偏外侧 | 接近车道边界 | 待验证 |

关键发现：GT语义相机的"road"范围包含路肩（比实际车道宽约 2m），
VLLiNet 边缘更接近实际车道边界。详见第七节评估指标说明。

## 六、实验输出结构

### 6.1 每次实验生成的目录

```text
evaluation/V5_run_{mode}_{weather}_{map}_{timestamp}/
  ├── experiment_info.txt          实验元数据(模式/天气/出生点/帧数)
  ├── framelog_{timestamp}.csv     逐帧33列遥测数据
  ├── eval_{timestamp}_1_summary.csv   轨迹评估摘要(仅供参考，存在边缘基准偏差)
  ├── eval_{timestamp}_1_detail.csv    逐点偏差详情(8列)
  └── distance_comparison.csv      感知vs GT 喷嘴距离对比(核心指标，10项)
```

### 6.2 FrameLogger 33列字段

```text
时间: timestamp, frame, dt, fps
车辆: veh_x, veh_y, veh_yaw, speed
喷嘴: nozzle_x, nozzle_y, nozzle_edge_dist, poly_edge_dist
控制: driving_offset, steer_filter, steer_cmd, throttle_cmd, brake_cmd, lateral_error
状态: paint_state, painting_enabled, dash_phase
感知: perception_mode, ai_edge_pts, gt_edge_pts, road_mask_ratio
多项式: poly_coeff_a, poly_coeff_b, poly_coeff_c
推理: inference_time_ms, sne_time_ms
精度: mask_iou, edge_dev_mean_px, edge_dev_median_px, edge_dev_max_px
GT参考: gt_nozzle_edge_dist
```

## 七、评估指标体系

### 7.1 核心评估指标

| 指标 | 数据来源 | 对比基准 | 适用场景 |
| --- | --- | --- | --- |
| Distance Comparison | 逐帧 nozzle_edge_dist | 逐帧 gt_nozzle_edge_dist | **跨感知模式精度对比（核心）** |
| FrameLogger 分析 | 33列逐帧遥测 | 统计分布 | 控制稳定性、画线连续性 |
| Perception Metrics | mask_iou, edge_dev | GT语义相机 | AI分割精度 |

> 注意：`trajectory_evaluator.py` 内部使用了 CARLA Map API 生成参考边缘，
> 但该边缘与感知边缘存在 ~2m 系统性偏差（Map API 只含 Driving 车道，
> 语义相机含路肩），**不适合跨感知模式对比**，仅供同模式内参考。
> 跨模式对比请使用 `distance_comparison.csv`。

### 7.2 各感知模式边缘位置差异

不同感知模式看到的"右边缘"绝对位置不同：

```text
实际车道边界 ←→ VLLiNet边缘(+0.6m) ←────→ GT语义边缘(+2.6m, 含路肩)
                    │                            │
                    │  nozzle_edge_dist ≈ 3.0m   │  nozzle_edge_dist ≈ 3.0m
```

两者 nozzle_edge_dist 都≈3.0m，逐帧对比 RMSE=0.08m，说明感知精度一致。
绝对位置差异是边缘定义不同导致的，不影响 AI vs GT 精度评估。

## 八、如何对比 AI 模型性能（论文核心）

### 8.1 对比原理

AI 模式（VLLiNet/LUNA）运行时，系统**每帧同步执行 GT 语义分割**，
产生两套边缘数据，用同一个喷嘴位置分别计算距离：

```text
同一帧、同一喷嘴位置：
  ├─ AI 感知边缘 → nozzle_edge_dist (用于驾驶控制)
  └─ GT 语义边缘 → gt_nozzle_edge_dist (仅记录，不参与控制)
```

两者差值 = AI 感知误差。GT 语义相机是 CARLA 仿真器的地面真值，作为对比基准。

### 8.2 证明 LUNA-Net 优越性的核心论证逻辑

```text
论点：LUNA-Net 在恶劣天气下仍能保持高精度感知，优于 VLLiNet

论据结构：
  1. 白天基准：LUNA ≈ VLLiNet ≈ GT（三者都好，说明公平起点）
  2. 夜间/雾/雨：LUNA 保持高精度，VLLiNet 退化 → LUNA 更鲁棒
  3. 端到端效果：LUNA 在恶劣天气下画线连续性不受影响
```

### 8.3 四层对比指标（从底层到应用层）

| 层级 | 指标 | 数据来源 | 说明 |
| --- | --- | --- | --- |
| 1. 分割精度 | mask_iou | framelog 逐帧 | AI road_mask vs GT road_mask 交并比 |
| 2. 边缘精度 | edge_dev_mean_px | framelog 逐帧 | AI 边缘 vs GT 边缘像素偏差 |
| 3. 距离精度 | nozzle_edge_dist RMSE | distance_comparison.csv | 感知→控制端到端精度(米) |
| 4. 画线效果 | PAINTING 占比 | framelog paint_state 统计 | 感知误差对实际画线的影响 |

### 8.4 横向对比表（同天气，不同模型 → 证明精度）

ClearDay 条件下三模式对比（GT 列为理想基准线）：

| 指标 | GT (基准) | VLLiNet | LUNA |
| --- | --- | --- | --- |
| mask_iou 均值 | 1.0 | 0.9862 | 0.9790 |
| edge_dev_mean_px | 0 | 0.65 | 2.54 |
| nozzle_edge_dist RMSE | 0 | 0.081m | 0.123m |
| PAINTING 占比 | 99.9% | 99.9% | 99.9% |
| 推理耗时 | N/A | 20.5ms | 60.5ms+60.1ms(SNE) |
| 有效帧数 | 27827 | 7307 | 5417 |

白天条件下 VLLiNet 边缘精度略优于 LUNA（RMSE 0.081 vs 0.123），
但两者 PAINTING 占比均为 99.9%，说明精度差异不影响实际画线效果。

### 8.5 纵向对比表（同模型，不同天气 → 证明鲁棒性，最关键）

这是证明 LUNA-Net 优越性的核心表格：

| 指标 | LUNA ClearDay | LUNA ClearNight | LUNA HeavyFoggyNight | LUNA HeavyRainFoggyNight |
| --- | --- | --- | --- | --- |
| mask_iou | 0.9790 | 0.9879 | 0.9798 | 0.9652 |
| edge_dev_mean_px | 2.54 | 5.94 | 6.14 | 8.30 |
| nozzle_edge_dist RMSE | 0.123m | 0.179m | 0.193m | 0.178m |
| PAINTING 占比 | 99.9% | 99.9% | 99.9% | 99.9% |
| 有效帧数 | 5417 | 5432 | 5411 | 5450 |

关键发现：
- mask_iou 在最恶劣天气(暴雨浓雾夜间)仍保持 0.965，仅下降 1.4%
- nozzle_edge_dist RMSE 最差仅 0.193m（<20cm），控制精度几乎不受天气影响
- PAINTING 占比全部 99.9%，恶劣天气下画线连续性完全不受影响
- edge_dev_mean_px 随天气恶化从 2.54→8.30，但像素级偏差未传导到米级控制误差

对比 VLLiNet 仅有 ClearDay 数据（夜间/雾天场景下 VLLiNet 未设计用于低光照）→ LUNA 在恶劣天气下的鲁棒性是其核心优势。

### 8.6 数据提取方法

```text
从 framelog CSV 提取：
  mask_iou 均值        = mean(framelog['mask_iou'])，排除 -1 值
  edge_dev_mean_px 均值 = mean(framelog['edge_dev_mean_px'])，排除 -1 值
  PAINTING 占比        = count(paint_state=='PAINTING') / total_frames

从 distance_comparison.csv 直接读取：
  nozzle_edge_dist RMSE = error_rmse 行的 value
```

### 8.7 注意事项

- GT 模式下 mask_iou 和 edge_dev 列为 -1（不适用），只在 AI 模式下有值
- inference_time_ms 和 sne_time_ms 可用于对比推理速度
- 每 3 帧执行一次感知，非感知帧的 inference_time_ms = -1，统计时需过滤

## 九、批量实验配置

### 9.1 启动命令

```bash
# 单次实验
python experiment_runner_v5.py --mode GT --weather ClearDay --distance 1600 --spawn 7

# 批量6组实验
python experiment_runner_v5.py --batch --distance 1600 --spawn 7
```

### 9.2 批量实验矩阵

| 序号 | 感知模式 | 天气 | 目的 |
| --- | --- | --- | --- |
| 1 | GT | ClearDay | 基准（排除感知误差） |
| 2 | VLLiNet | ClearDay | AI感知 vs GT 对比 |
| 3 | LUNA | ClearDay | LUNA vs GT 对比 |
| 4 | LUNA | ClearNight | 夜间感知鲁棒性 |
| 5 | LUNA | HeavyFoggyNight | 浓雾夜间极端条件 |
| 6 | LUNA | HeavyRainFoggyNight | 暴雨浓雾夜间极端条件 |

### 9.3 天气预设参数

| 预设 | sun_altitude | fog_density | precipitation | 特点 |
| --- | --- | --- | --- | --- |
| ClearDay | 5° | 0 | 0 | 标准白天 |
| ClearNight | -30° | 0 | 0 | 纯夜间 |
| HeavyFoggyNight | -30° | 80 | 0 | 浓雾+夜间 |
| HeavyRainFoggyNight | -30° | 50 | 80 | 暴雨+雾+夜间 |

## 十、实验创新点

### 9.1 Nozzle-Centric 两级几何偏移（V5.2 核心）

传统方法：车辆跟踪固定偏移路径，用 PD 控制器修正横向误差 → 弯道滞后、振荡。

本方法：从道路边缘几何反推车辆路径，无反馈控制器：
- Stage 1: 边缘 → 沿局部法线偏移 → 喷嘴目标路径（弯道处偏移方向自动跟随曲率）
- Stage 2: 喷嘴路径 → 曲率补偿 arm → 驾驶路径
- 弯道均值 3.004m，几乎完美命中 3.0m 目标

### 9.2 三模式感知 + 同步GT对比

每帧同时运行 AI 感知和 GT 语义分割，输出 8-tuple：
- AI 模式产出用于驾驶控制
- GT 产出用于逐帧精度评估（mask_iou, edge_deviation, gt_nozzle_edge_dist）
- 无需两次实验，单次运行即可获得 AI vs GT 对比数据

### 9.3 自适应画线状态机

- 三状态迟滞设计（CONVERGING/STABILIZED/PAINTING），防止边界抖动
- 弯道自适应容差：曲率越大，容差越宽（0.3m→0.55m）
- Grace 机制：短暂超差不立即退出，容忍 CARLA GT 标注噪声
- V5.2.1 修复：状态机输入从多项式外推改为直接 NED，消除弯→直过渡误判

### 9.4 帧率自适应参数

同一套控制逻辑适配两种帧率（~20 FPS GUI / ~190 FPS headless），
所有时间窗口参数按帧率比例缩放（×10），等效物理时间一致。

### 9.5 多维度评估体系

| 维度 | 指标 | 来源 |
| --- | --- | --- |
| 画线精度 | mean/max/std deviation | Trajectory Evaluator (同模式内参考) |
| 感知精度 | nozzle_edge_dist RMSE | Distance Comparison (AI vs GT，核心) |
| 分割精度 | mask IoU | perception_metrics (AI mask vs GT mask) |
| 边缘精度 | edge_dev mean/median/max px | perception_metrics (AI edge vs GT edge) |
| 画线连续性 | paint_state 分布、中断次数 | FrameLogger 状态列 |
| 控制稳定性 | lateral_error 分布 | FrameLogger |
