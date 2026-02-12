# AutoStripe V4 版本总结

---

## 一、V4 版本定位

V4 是 AutoStripe 从**仿真验证**迈向**真实部署**的关键版本。核心变化是将 V3 依赖的 CARLA 地面真值（CityScapes 语义相机）替换为**训练好的深度学习模型 VLLiNet** 进行道路分割，同时引入**多项式曲线外推**解决相机盲区问题。

| 指标 | V3 (GT) | V4 (AI) |
|------|---------|---------|
| 感知来源 | CARLA CityScapes 语义相机 | VLLiNet 深度学习模型 |
| 相机分辨率 | 800×600 | 1248×384（匹配训练分辨率） |
| 盲区距离估计 | 仅中值投影（近处） | 多项式外推（远→近推断） |
| 模型精度 | 100%（地面真值） | MaxF=98.33%, IoU=96.72% |
| 实时性 | ~30 FPS | ~10 FPS（skip-frame 后有效 ~30 FPS） |

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    manual_painting_control_v4.py                 │
│                         (V4 主入口 + 主循环)                     │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────────┐ │
│  │ Pygame   │  │ ManualPaint  │  │ Video   │  │ Spawn Point  │ │
│  │ 键盘/显示│  │ Control      │  │Recorder │  │ 预设 (5个)   │ │
│  └──────────┘  └──────────────┘  └─────────┘  └──────────────┘ │
└────┬───────────────┬──────────────┬──────────────┬──────────────┘
     │               │              │              │
┌────▼────┐   ┌──────▼──────┐  ┌───▼────┐  ┌─────▼──────┐
│ 感知层  │   │   规划层    │  │ 控制层 │  │  环境层    │
│Perception│  │  Planning   │  │Control │  │ carla_env  │
└────┬────┘   └──────┬──────┘  └───┬────┘  └─────┬──────┘
     │               │              │              │
     ▼               ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CARLA 0.9.15 Server                        │
│                        Town05 Highway                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、模块详解

### 3.1 环境层 — `carla_env/setup_scene_v2.py`

负责 CARLA 连接、车辆生成、传感器配置。

**传感器配置（5个）：**

| 传感器 | 分辨率 | 位置 | 用途 |
|--------|--------|------|------|
| Semantic Camera | 1248×384 | x=2.5, z=3.5, pitch=-15 | 语义标签（GT 边缘提取） |
| Depth Camera | 1248×384 | 同上 | 深度图（3D 投影） |
| RGB Front Camera | 1248×384 | 同上 | VLLiNet 输入 |
| Overhead Camera | 1800×1600 | z=25, pitch=-90 | 俯视图显示 |
| Semantic LiDAR | — | 车顶 | 点云（预留） |

**数据传递机制：** 每个传感器有独立的 `threading.Lock` + 共享字典缓冲区，回调线程写入，主循环读取。

---

### 3.2 感知层 — `perception/`

感知层是 V4 最核心的变化，由 5 个模块组成：

```
RGB 图像 ──┐                ┌── left_world  (世界坐标)
            ├→ [Segmentor] → road_mask ─┤
Depth 图像 ─┘        │                  ├── right_world (世界坐标)
                      │                  ├── left_px     (像素坐标)
              ┌───────┴───────┐          ├── right_px    (像素坐标)
              │  AI: VLLiNet  │          └── road_mask   (二值掩码)
              │  GT: CityScapes│
              └───────────────┘
                      ↓
              [Edge Extractor] → 左/右边缘像素
                      ↓
              [Depth Projector] → 世界坐标 3D 点
```

#### 3.2.1 AI 分割器 — `road_segmentor_ai.py`

```python
class RoadSegmentorAI:
    def __init__(checkpoint_path, device='cuda', model_h=384, model_w=1248)
    def segment(rgb_bgra, depth_bgra) -> np.ndarray  # (H,W) uint8, 255=road
```

**推理流程：**
1. RGB: BGRA → RGB → resize 1248×384 → ImageNet 归一化 → `[1,3,384,1248]` tensor
2. Depth: CARLA 解码 → min-max 归一化 → 堆叠为 N 通道 → `[1,N,384,1248]` tensor
3. VLLiNet_Lite 推理（mixed precision）→ sigmoid → 阈值 0.5
4. 上采样回原始分辨率 → 裁掉顶部 35%（天空区域）
5. 返回 uint8 二值掩码

**模型细节：**
- Checkpoint: `VLLiNet_models/checkpoints_carla/best_model.pth`
- 深度通道数从 checkpoint 自动检测（通常为 3）
- 输出分辨率为输入的 1/2（624×192），需上采样

#### 3.2.2 GT 分割器 — `road_segmentor.py`

```python
class RoadSegmentor:
    def segment(cityscapes_bgra) -> np.ndarray  # (H,W) uint8
```

- 匹配 CityScapes 道路颜色 BGR(128,64,128)，容差=10
- 形态学闭运算(15×15) + 开运算(5×5) 去噪

#### 3.2.3 双模式管线 — `perception_pipeline.py`

```python
class PerceptionPipeline:
    def __init__(img_w, img_h, fov_deg, use_ai=False)
    def process_frame(semantic_bgra, depth_bgra, cam_tf,
                      cityscapes_bgra=None, rgb_bgra=None)
```

**AI 模式返回 7 元素：**
`(left_world, right_world, road_mask, left_px, right_px, gt_right_world, gt_right_px)`

**GT 模式返回 7 元素（后两个为 None）：**
`(left_world, right_world, road_mask, left_px, right_px, None, None)`

AI 模式同时计算 GT 参考边缘，用于 A/B 对比。

#### 3.2.4 边缘提取 — `edge_extractor.py`

两个函数，分别处理语义标签和二值掩码：

| 函数 | 输入 | 扫描方式 |
|------|------|----------|
| `extract_road_edges_semantic()` | 语义 BGRA（R=tag ID） | 从图像中心向两侧扫描，找 road→non-road 跳变 |
| `extract_road_edges_mask()` | 二值掩码 | 同上，找 mask>0 → mask==0 跳变 |

**关键参数：** `MIN_ROAD_RUN=20px`, `MAX_DEPTH=30m`, `ROW_START_RATIO=0.3`

#### 3.2.5 深度投影 — `depth_projector.py`

```python
decode_depth_image(depth_bgra) -> float32  # CARLA 深度解码
class DepthProjector:
    def __init__(img_w, img_h, fov_deg)     # 计算 fx, fy, cx, cy
    def project_pixels(pixels, depth_image, cam_tf) -> [carla.Location]
```

投影链：像素(u,v) + 深度 → 相机坐标 → UE4 本地坐标 → 世界坐标

---

### 3.3 规划层 — `planning/vision_path_planner.py`

```python
class VisionPathPlanner:
    def __init__(nozzle_arm=2.0, line_offset=3.0, smooth_window=5,
                 max_range=20.0, memory_frames=10)
    def update(right_edges, vehicle_tf) -> (driving_coords, nozzle_locs)
    def estimate_nozzle_edge_distance(right_world, vehicle_tf) -> (dist, coeffs)
```

#### 路径生成流程

```
右侧边缘点 (world)
    ↓ 按纵向距离排序
    ↓ 1m 纵向分箱重采样
    ↓ 滑动平均平滑 (window=5)
    ↓ 向左偏移 driving_offset=5.0m
    ↓
驾驶路径 (driving_coords)
```

**短期记忆机制：** 感知失败时（<2 个边缘点），沿用上一次有效路径，最多持续 10 帧。

#### 多项式外推（V4 新增）

这是 V4 解决**相机盲区**的核心算法：

```
问题：相机看到 3m~20m 前方的路缘，但喷嘴在车辆侧面 (lon=0)
解决：用二次多项式拟合边缘点，外推到 lon=0

边缘点 (lon, lat) 在车辆坐标系:
    lon ∈ [3m, 20m]  (纵向，前方)
    lat > 0           (横向，右侧)

拟合: lat = a·lon² + b·lon + c
外推: nozzle_distance = c (lon=0 处的横向距离)

约束: 最少 5 个点, 0.5m < c < 15.0m
平滑: 10 帧中值滤波
```

---

### 3.4 控制层 — `control/marker_vehicle_v2.py`

```python
class MarkerVehicleV2:
    def __init__(vehicle, wheelbase=2.875, kdd=3.0)
    def update_path(new_coords)   # 接收新路径
    def step() -> int             # 一步 Pure Pursuit
```

**Pure Pursuit 算法：**
- 找最近路径点 → 前看 8 个点作为目标
- 转向角: `δ = atan2(2·L·sin(α), ld)`
- 低通滤波: `steer = 0.85·old + 0.15·new`
- 目标速度: 3.0 m/s，油门自适应

---

### 3.5 喷涂控制 — `ManualPaintingControl`（主文件内）

```python
class ManualPaintingControl:
    def toggle_painting()          # SPACE 键
    def paint_line(world, nozzle)  # 画黄色线段
    def toggle_drive_mode()        # TAB 键
    def update_manual_control(keys)# WASD 手动驾驶
    def apply_manual_control()     # 应用手动控制
```

**喷嘴位置计算：** 车辆中心 + 右侧 2.0m 偏移（基于 yaw 角）

---

### 3.6 视频录制 — `utils/video_recorder.py`

```python
class VideoRecorder:
    def toggle(front_size, overhead_size)  # R 键
    def write_front(frame)                 # 非阻塞入队
    def write_overhead(frame)              # 非阻塞入队
    def release()                          # 释放资源
```

**后台线程编码：** `deque(maxlen=40)` 队列，满时自动丢弃最旧帧，不阻塞主循环。

---

## 四、主循环数据流

```
每帧执行 (目标 30 FPS):
┌─────────────────────────────────────────────────────────────────┐
│ 1. 读取传感器 (加锁)                                            │
│    sem_data, depth_data, rgb_front, overhead_data               │
│                                                                  │
│ 2. 感知 (每 3 帧执行一次 VLLiNet，其余帧用缓存)                 │
│    → left_world, right_world, road_mask                         │
│    → gt_right_world (AI 模式下的 GT 参考)                       │
│                                                                  │
│ 3. 规划                                                         │
│    → driving_coords (驾驶路径)                                  │
│    → poly_dist, poly_coeffs (多项式外推距离)                    │
│                                                                  │
│ 4. 控制                                                         │
│    AUTO: Pure Pursuit 跟踪 driving_coords                       │
│    MANUAL: WASD 键盘输入                                        │
│                                                                  │
│ 5. 喷涂                                                         │
│    计算喷嘴位置 → 画黄色线段                                    │
│                                                                  │
│ 6. 可视化                                                        │
│    3D: 红点(边缘)+蓝点(路径)+品红曲线(多项式)+绿线(距离)        │
│    俯视图: OpenCV 窗口 + 2D 叠加                                │
│    前视图: Pygame 窗口 + road mask + HUD                        │
│                                                                  │
│ 7. 录制 (R 键开关，后台线程编码)                                 │
│                                                                  │
│ 8. 事件处理 (SPACE/TAB/G/R/V/Q/X/ESC)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、关键参数汇总

| 类别 | 参数 | 值 |
|------|------|----|
| **相机** | 前视分辨率 | 1248×384 |
| | FOV | 90° |
| | 位置 | x=2.5, z=3.5, pitch=-15 |
| | 俯视分辨率 | 1800×1600 |
| **VLLiNet** | 输入 | RGB(ImageNet 归一化) + Depth(min-max 归一化) |
| | 输出 | 624×192 → 上采样至 1248×384 |
| | 阈值 | 0.5 |
| | Skip-frame | 每 3 帧推理一次 |
| **边缘提取** | 顶部裁切 | 35% |
| | 最小道路宽度 | 20px |
| | 深度范围 | 1.5m ~ 30m |
| **路径规划** | 喷嘴臂长 | 2.0m |
| | 线偏移 | 3.0m |
| | 驾驶偏移 | 5.0m (line_offset + nozzle_arm) |
| | 最大范围 | 20m |
| | 记忆帧数 | 10 帧 |
| **多项式** | 拟合范围 | 3m ~ 20m |
| | 最少点数 | 5 |
| | 合理范围 | 0.5m ~ 15.0m |
| | 平滑窗口 | 10 帧中值滤波 |
| **控制** | 目标速度 | 3.0 m/s |
| | 前看点数 | 8 |
| | 轴距 | 2.875m |
| | Kdd | 3.0 |

---

## 六、V4 vs V3 关键差异

| 方面 | V3 | V4 |
|------|----|----|
| 感知 | CityScapes GT（100% 准确） | VLLiNet AI（MaxF 98.33%） |
| 分辨率 | 800×600 | 1248×384 |
| 盲区处理 | 中值横向投影（仅近处可见点） | 二次多项式外推（远→近推断） |
| 推理频率 | 每帧 | 每 3 帧（skip-frame） |
| AI/GT 切换 | 无 | G 键实时切换 |
| GT 参考 | 无 | AI 模式下同时计算 GT 边缘用于对比 |
| 视频录制 | 无 | R 键开关，后台线程编码 |
| FPS 显示 | 无 | 前视 + 俯视双窗口显示 |
| 可视化 | 红点+蓝点+绿线+青线 | +品红多项式曲线+紫色 GT 路径 |
| 入口文件 | `manual_painting_control.py` | `manual_painting_control_v4.py` |

---

## 七、键盘控制

| 按键 | 功能 |
|------|------|
| `SPACE` | 开关喷涂 |
| `TAB` | 切换自动/手动驾驶 |
| `G` | 切换 AI/GT 感知模式 |
| `R` | 开关视频录制 |
| `WASD` / 方向键 | 手动驾驶 |
| `Q` | 切换倒车 |
| `V` | 切换观察者跟随/自由相机 |
| `X` | 手刹 |
| `ESC` | 退出 |

---

## 八、文件清单

```
AutoStripe/
├── manual_painting_control_v4.py       # V4 主入口（AI/GT + 多项式 + 手动控制）
├── manual_painting_control.py          # V3 主入口（保留）
├── main_v2.py                          # V2 入口（保留）
├── main_v1.py                          # V1 入口（保留）
├── diag_vllinet.py                     # VLLiNet 模型独立验证脚本
├── carla_env/
│   ├── setup_scene_v2.py               # V2-V4 场景：1248×384 相机
│   └── setup_scene.py                  # V1 场景
├── perception/
│   ├── road_segmentor_ai.py            # V4: VLLiNet 包装器
│   ├── road_segmentor.py               # GT: CityScapes 颜色匹配
│   ├── perception_pipeline.py          # AI/GT 双模式管线
│   ├── edge_extractor.py              # 道路掩码 → 左/右边缘像素
│   └── depth_projector.py             # 像素 + 深度 → 世界坐标
├── planning/
│   ├── vision_path_planner.py          # 视觉路径规划 + 多项式外推
│   └── lane_planner.py                # V1 Map API 规划器
├── control/
│   ├── marker_vehicle_v2.py            # V2-V4 Pure Pursuit 控制器
│   └── marker_vehicle.py              # V1 控制器
├── utils/
│   └── video_recorder.py              # 后台线程视频录制
├── ros_interface/
│   ├── autostripe_node.py             # ROS 节点
│   ├── rviz_publisher.py              # RVIZ 发布器
│   └── topic_config.py               # Topic 名称常量
├── VLLiNet_models/
│   ├── models/vllinet.py              # VLLiNet_Lite 模型
│   ├── models/backbone.py             # MobileNetV3 + LiDAREncoder
│   └── checkpoints_carla/best_model.pth
├── configs/rviz/
├── launch/
└── docs/
```

---

## 九、下一步计划

### 9.1 喷涂距离闭环控制（优先级：高）

**目标**：从手动 SPACE 开关喷涂，升级为基于边距的自动喷涂控制。

**现状**：V4 已有两个实时距离读数：
- `edge_dist_r`：喷嘴到路缘的直接中值投影距离（近处）
- `poly_dist`：多项式外推距离（盲区推断）

**方案**：
- 定义目标喷涂距离范围（如 2.5m ~ 3.5m）
- 当 `poly_dist` 进入目标范围时自动开启喷涂
- 偏离范围时自动暂停
- 保留 SPACE 键手动覆盖（优先级高于自动控制）
- HUD 显示自动喷涂状态和目标范围

**意义**：从"人看数字手动操作"到"机器自主决策"的关键一步。

---

### 9.2 AI vs GT 定量评估（优先级：高）

**目标**：量化 VLLiNet AI 感知与 GT 的偏差，验证实际精度。

**现状**：AI 模式下同时输出 GT 参考路径（紫色点），但只能肉眼对比。

**方案**：
- 逐帧计算 AI 驾驶路径与 GT 驾驶路径的横向偏差
- 统计指标：均值、最大值、标准差
- 输出到 CSV 文件，按路段分析
- 在 HUD 上实时显示当前偏差值

**预期输出**：
```
frame, ai_pts, gt_pts, mean_lateral_error, max_lateral_error, poly_dist, gt_edge_dist
50, 12, 12, 0.15, 0.42, 5.1, 5.3
100, 12, 12, 0.08, 0.31, 5.0, 5.2
```

---

### 9.3 多出生点自动测试（优先级：中）

**目标**：在不同路段验证 V4 系统稳定性。

**现状**：仅在 Spawn #5（Highway eastbound）测试。已有 5 个预设出生点。

**方案**：
- 逐个出生点自动跑固定距离（如 200m）
- 收集各路段数据：边缘点数、距离波动、感知失败率
- 找出 VLLiNet 表现差的场景（弯道、阴影、桥梁）
- 生成测试报告

---

### 9.4 TensorRT 推理加速（优先级：中）

**目标**：消除 VLLiNet 推理瓶颈（200~300ms/帧），去掉 skip-frame。

**现状**：PyTorch + mixed precision，RTX 4060 Ti 上 ~280ms/帧。

**方案**：
- 导出 VLLiNet_Lite 为 ONNX 格式
- 用 TensorRT 编译为 engine（FP16）
- 预期推理降到 50~100ms/帧
- 去掉 skip-frame 后实际 FPS 可达 10~15

---

### 9.5 ROS Bridge 集成（优先级：低）

**目标**：将 V4 接入 CARLA-ROS Bridge，实现 RVIZ 可视化。

**现状**：Plan 中 Phase B 尚未实施，`autostripe_node.py` 仍使用直接 CARLA API。

**方案**：
- 重写 `autostripe_node.py` 订阅 CARLA-ROS Bridge Topics
- 新增多项式曲线 Marker 发布
- 创建 `autostripe_v4.launch` + `autostripe_v4.rviz`

---

### 已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| CARLA 段错误 | 未解决 | 长时间运行后偶发 segfault，CARLA 0.9.15 已知问题 |
| 前视图坐标空间不一致 | 保持现状 | 红点(2D 像素)与品红曲线(3D 投影)经过不同路径，有轻微偏差 |
| Nozzle-Edge 偶发 999.0m | 未解决 | 感知失败时无有效路缘点，返回默认值 |
| GT 模式 RuntimeWarning | 未解决 | CityScapes np.sqrt 遇到 NaN，不影响功能 |
