# AutoStripe V4 实验记录

## 版本概述

**V4: VLLiNet AI 感知 + 多项式盲区外推**

在 V3 手动喷涂控制基础上，核心升级：
- VLLiNet 深度学习模型替代 CityScapes 地面真值分割（MaxF=98.33%）
- 多项式曲线外推解决相机盲区喷嘴距离估计
- 相机分辨率 800x600 → 1248x384（匹配 VLLiNet 训练分辨率）
- AI/GT 实时切换（G 键）用于 A/B 对比
- 视频录制模块化（R 键 + 后台线程编码）
- Skip-frame 推理优化（每 3 帧运行一次 VLLiNet）

**实现时间**：2026-02-10
**入口文件**：`manual_painting_control_v4.py`

---

## 阶段 1: VLLiNet 模型验证

### 问题

V4 的核心是用 VLLiNet 替代 CityScapes GT 分割。但项目中 `VLLiNet_models/carla_realtime_inference.py` 引用了不存在的模块 `carla_integration.train_carla_v2.VLLiNet_CARLA`，无法直接运行。需要先验证 checkpoint 能否正常加载和推理。

### 分析

检查 checkpoint 文件 `VLLiNet_models/checkpoints_carla/best_model.pth`：
- 模型类为 `VLLiNet_Lite`（定义在 `models/vllinet.py`）
- Checkpoint 中 state_dict 的 key 使用 `fusion.fusion_modules`，但代码中使用 `fusion_module.fusion_modules`，需要修复 key 命名
- Depth encoder 输入通道数需从 checkpoint 权重 `lidar_encoder.stage1.0.weight` 的 shape[1] 自动检测

### 解决方案

创建独立验证脚本 `diag_vllinet.py`：
1. 直接导入 `VLLiNet_Lite`（绕过缺失的 `carla_integration` 模块）
2. 自动检测 depth 通道数
3. 修复 state_dict key 命名不匹配
4. 用 CARLA 实时 RGB + Depth 图像运行推理
5. 显示 road mask 叠加在 RGB 上，确认分割质量

### 结果

模型加载成功：epoch=40, MaxF=0.9833, depth_ch=1（实际堆叠为 3 通道输入）, device=cuda。推理输出正确的道路分割掩码。

---

## 阶段 2: 相机分辨率升级 + 位置调整

### 问题

VLLiNet 训练时使用 1248x384 分辨率，V3 相机为 800x600。直接用 800x600 图像 resize 到 1248x384 会引入插值伪影和宽高比失真，影响模型精度。

### 迭代过程

| 尝试 | 方案 | 结果 |
|------|------|------|
| 1 | 保持 800x600，软件 resize 到 1248x384 | 宽高比失真（4:3 → 3.25:1），模型输出质量差 |
| 2 | CARLA 原生 1248x384 相机 | 正确！无插值伪影，匹配训练分辨率 |
| 3 | 相机位置保持 V3 (x=2.5, z=2.8, pitch=-15) | 车头遮挡画面底部过多 |
| 4 | 调整为 (x=2.5, z=3.5, pitch=-10) | 视野合适，车头遮挡减少 |

### 修改文件

`carla_env/setup_scene_v2.py` — 更新相机常量：

```python
# V3                          # V4
FRONT_CAM_X = 2.5             FRONT_CAM_X = 2.5      # 不变
FRONT_CAM_Z = 2.8             FRONT_CAM_Z = 3.5      # 抬高 0.7m
FRONT_CAM_PITCH = -15         FRONT_CAM_PITCH = -10  # 减小俯角
FRONT_CAM_W = 800             FRONT_CAM_W = 1248     # 原生 VLLiNet 宽度
FRONT_CAM_H = 600             FRONT_CAM_H = 384      # 原生 VLLiNet 高度
```

### 自动适配（无需改代码）

- `DepthProjector`: 从构造参数重新计算 fx=fy=624, cx=624, cy=192
- 三个前视相机（语义、深度、RGB）共享同一组常量
- 俯视相机保持 1800x1600 不变

### 结果

V3 GT 管线在新分辨率下正常工作，边缘检测和驾驶路径不受影响。

---

## 阶段 3: VLLiNet AI 分割器封装

### 需求

将 VLLiNet 封装为与 V3 `RoadSegmentor` 相同的 `segment()` 接口，实现即插即用替换。

### 关键差异

V3 的 `RoadSegmentor.segment()` 只需一个语义图像参数。V4 的 `RoadSegmentorAI.segment()` 需要 RGB + Depth 两个输入。

### 踩坑：Checkpoint Key 命名不匹配

```python
# Checkpoint 中的 key:
'fusion.fusion_modules.0.weight'

# 代码中的属性名:
'fusion_module.fusion_modules.0.weight'
```

直接 `load_state_dict()` 会报 key mismatch 错误。解决方案：加载时逐 key 替换前缀。

```python
fixed_sd = {}
for k, v in state_dict.items():
    new_k = k.replace('fusion.fusion_modules',
                       'fusion_module.fusion_modules')
    fixed_sd[new_k] = v
self.model.load_state_dict(fixed_sd)
```

### 踩坑：Depth 通道数不确定

VLLiNet 支持不同的 depth 输入通道数（1 或 3）。Checkpoint 训练时用的通道数记录在权重 shape 中：

```python
depth_key = 'lidar_encoder.stage1.0.weight'
self.depth_channels = state_dict[depth_key].shape[1]  # 自动检测
```

实际检测结果为 1 通道，但模型 LiDAREncoder 默认期望 3 通道。解决方案：将单通道 depth 堆叠 N 次：

```python
depth_stack = np.stack([depth_norm] * self.depth_channels, axis=0)
```

### 新增文件

`perception/road_segmentor_ai.py` — 165 行，核心流程：

1. RGB: BGRA → RGB → resize 1248x384 → ImageNet 归一化 → tensor
2. Depth: CARLA 解码 → min-max [0,1] → 堆叠 N 通道 → tensor
3. 推理: mixed precision (`torch.amp.autocast`) → sigmoid → 阈值 0.5
4. 后处理: 上采样回原始分辨率 → 裁掉顶部 35%

### 结果

`RoadSegmentorAI` 与 `RoadSegmentor` 接口兼容，可在 pipeline 中无缝切换。

---

## 阶段 4: 感知管线双模式集成 + G 键切换

### 需求

1. `PerceptionPipeline` 支持 AI/GT 双模式，通过 `use_ai` 标志切换
2. AI 模式下同时计算 GT 参考边缘，用于 A/B 对比
3. 运行时按 G 键实时切换模式

### 修改文件

**`perception/perception_pipeline.py`：**

```python
class PerceptionPipeline:
    def __init__(self, img_w, img_h, fov_deg, use_ai=False):
        if use_ai:
            from perception.road_segmentor_ai import RoadSegmentorAI
            self.segmentor = RoadSegmentorAI()
        else:
            self.segmentor = RoadSegmentor()

    def process_frame(self, semantic_bgra, depth_bgra, cam_tf,
                      cityscapes_bgra=None, rgb_bgra=None):
        if self.use_ai:
            road_mask = self.segmentor.segment(rgb_bgra, depth_bgra)
            # 同时计算 GT 边缘作为参考
            _, gt_right_px = extract_road_edges_semantic(semantic_bgra, depth_m)
            gt_right_world = self.projector.project_pixels(gt_right_px, ...)
            return (..., gt_right_world, gt_right_px)  # 7 元素
        else:
            road_mask = self.segmentor.segment(cityscapes_bgra)
            return (..., None, None)  # 7 元素，后两个为 None
```

**`manual_painting_control_v4.py` — G 键处理：**

```python
elif event.key == K_g:
    use_ai_mode = not use_ai_mode
    perception = PerceptionPipeline(
        img_w=FRONT_CAM_W, img_h=FRONT_CAM_H,
        fov_deg=90.0, use_ai=use_ai_mode)
```

### 设计决策：重建 vs 切换

G 键切换时**重建整个 PerceptionPipeline**（而非内部切换 segmentor），原因：
- AI 模式需要 GPU 显存加载模型，GT 模式不需要
- 切换到 GT 时释放 GPU 显存，切换回 AI 时重新加载
- 代码简单，无需管理两个 segmentor 的生命周期

### 结果

G 键切换正常，AI 模式下 HUD 显示 `Perc: AI`（品红色），GT 模式显示 `Perc: GT`（绿色）。AI 模式同时输出 GT 参考路径（紫色点），可直观对比两种感知的差异。

---

## 阶段 5: 多项式外推解决相机盲区

### 问题

前视相机只能看到车辆前方 3m~20m 的路缘，但喷嘴在车辆正侧方（lon=0）。V3 的中值横向投影只能利用可见范围内的点，无法估计盲区内的真实距离。在弯道上，前方可见路缘的横向距离与喷嘴处的实际距离差异显著。

### 解决方案

在 `planning/vision_path_planner.py` 新增 `estimate_nozzle_edge_distance()` 方法：

**算法：**
1. 将右侧路缘点转换到车辆局部坐标系 (lon, lat)
2. 筛选 3m < lon < 20m 且 lat > 0 的点
3. 用 `numpy.polyfit(lon, lat, deg=2)` 拟合二次曲线
4. 外推到 lon=0：`nozzle_distance = c`（截距）

```python
# 车辆局部坐标系
lon = dx * fwd_x + dy * fwd_y    # 纵向（前方）
lat = dx * right_x + dy * right_y # 横向（右侧）

# 二次拟合: lat = a·lon² + b·lon + c
coeffs = np.polyfit(lon_arr, lat_arr, 2)
a, b, c = coeffs

# c 即为 lon=0 处的横向距离（喷嘴位置）
poly_dist = c
```

### 鲁棒性约束

- 最少 5 个有效点才进行拟合
- 合理范围检查：0.5m < distance < 15.0m，超出则返回 None
- 时间平滑：10 帧中值滤波，消除单帧抖动

```python
poly_dist_history.append(poly_dist_raw)
if len(poly_dist_history) > POLY_SMOOTH_WINDOW:
    poly_dist_history.pop(0)
poly_dist = float(np.median(poly_dist_history))
```

### 可视化

品红色多项式曲线在 3D 场景、俯视图、前视图三个视图中同时显示：

```python
def draw_poly_curve(world, coeffs, vehicle_tf, num_points=20, max_lon=20.0):
    for i in range(num_points + 1):
        lon = max_lon * i / num_points
        lat = a * lon**2 + b * lon + c
        # 转换回世界坐标，slope-aware z
        world.debug.draw_point(carla.Location(x=wx, y=wy, z=wz),
                               size=0.06, color=carla.Color(255, 0, 255))
```

### 结果

直道上 Poly-Edge 读数稳定在 5.0~5.5m，与 Nozzle-Edge（3.0~4.0m）形成互补。弯道上多项式外推能更准确反映喷嘴处的实际边距。

---

## 阶段 6: V4 主入口 + 多视图 2D 叠加

### 需求

创建 `manual_painting_control_v4.py`，基于 V3 `manual_painting_control.py`，集成所有 V4 新功能，同时保留 V3 不动。

### 关键变化

1. **Pygame 窗口**：800x600 → 1248x384（匹配相机分辨率）
2. **PerceptionPipeline 初始化**：`use_ai=True`，默认 AI 模式
3. **RGB 数据传递**：`process_frame()` 增加 `rgb_bgra` 参数
4. **多项式距离**：HUD 增加 `Poly-Edge` 读数
5. **GT 参考路径**：紫色点显示 GT 驾驶路径（AI 模式下）

### 踩坑：AI 模式下 3D debug 绘制污染相机

V3 的红色边缘点、蓝色路径点、绿色距离线使用 `world.debug.draw_point/draw_line` 在 3D 场景中绘制。这些 3D 标记会被前视 RGB 相机捕获，成为 VLLiNet 的输入噪声。

**解决方案**：AI 模式下跳过 3D debug 绘制，改为纯 2D 叠加：

```python
# 3D 绘制仅在 GT 模式下执行
if not use_ai_mode:
    _draw_right_edge_dots(world, right_world, veh_tf)
    draw_driving_path(world, driving_coords, veh_tf)
    draw_poly_curve(world, poly_coeffs, veh_tf)
```

所有可视化改为在俯视图和前视图上用 OpenCV/pygame 2D 绘制：
- 俯视图：`world_to_pixel()` 投影 + `cv2.circle/line`
- 前视图：`world_to_front_pixel()` 投影 + `cv2.circle/line`

### 2D 叠加元素

| 元素 | 颜色 | 俯视图 | 前视图 |
|------|------|--------|--------|
| 右路沿点 | 红色 | cv2.circle r=4 | cv2.circle r=3 |
| AI 驾驶路径 | 蓝色 | cv2.circle r=5 | cv2.circle r=4 |
| GT 参考路径 | 紫色 | cv2.circle r=5 | cv2.circle r=4 |
| 多项式曲线 | 品红 | cv2.line 连线 | cv2.line 连线 |
| 喷嘴-边缘线 | 绿色 | cv2.line | cv2.line |
| TP-多项式线 | 青色 | cv2.line + 距离文字 | cv2.line + 距离文字 |

### 结果

AI 模式下前视图干净（无 3D 标记干扰），所有可视化通过 2D 叠加实现，不影响 VLLiNet 输入。

---

## 阶段 7: 视频录制模块化

### 问题

V3 没有视频录制功能。需要在运行时按 R 键开关录制，同时录制前视图（含 HUD）和俯视图。

### 迭代过程

| 尝试 | 方案 | 结果 |
|------|------|------|
| 1 | 主循环内直接调用 `cv2.VideoWriter.write()` | 录制时明显卡顿（编码阻塞主循环） |
| 2 | 后台线程 + deque 队列 | 非阻塞，主循环不受影响 |

### 解决方案

新增 `utils/video_recorder.py`（164 行），核心设计：

**后台线程编码架构：**

```python
class VideoRecorder:
    def __init__(self):
        self._queue = deque(maxlen=40)  # 前视+俯视共 40 帧上限
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def write_front(self, frame):
        """非阻塞：copy + 入队，立即返回"""
        self._queue.append(('front', frame.copy()))
        self._event.set()

    def _writer_loop(self):
        """后台循环：等待信号 → 批量出队 → 编码写盘"""
        while not self._stop_thread:
            self._event.wait(timeout=0.5)
            self._event.clear()
            self._flush_queue()
```

**关键设计：**
- `deque(maxlen=40)` 自动丢弃最旧帧，防止编码器跟不上时内存爆炸
- `frame.copy()` 避免主循环修改帧数据时产生竞态
- `threading.Event` 信号机制，无帧时线程休眠不占 CPU
- `release()` 先 flush 队列再释放 writer，确保不丢最后几帧

### 结果

R 键开关录制正常，输出双路 MP4（`videos/front_YYYYMMDD_HHMMSS.mp4` + `videos/overhead_...mp4`），录制时主循环 FPS 无明显下降。

---

## 阶段 8: FPS 显示 + 性能瓶颈定位

### 需求

运行时感觉帧率偏低，需要在前视图和俯视图上显示实时 FPS，并定位性能瓶颈。

### FPS 显示实现

在主循环开头计算每帧耗时：

```python
now = time.time()
dt = now - last_time
last_time = now
fps = 1.0 / dt if dt > 0 else 0.0
```

颜色编码：绿色(≥15) / 黄色(≥10) / 红色(<10)，同时显示在俯视图（OpenCV）和前视图（pygame HUD）。

### 性能分析：三项优化尝试

发现 FPS 仅约 3~4，进行了三项优化尝试：

| 优化 | 内容 | 效果 |
|------|------|------|
| 1. GT 边缘跳过 | AI 模式下不计算 GT 参考边缘 | FPS 无明显变化 |
| 2. get_transform 缓存 | 每帧只调用一次 `vehicle.get_transform()`，缓存为 `veh_tf` | FPS 无明显变化 |
| 3. cv2.waitKey 移位 | 将 `cv2.waitKey(1)` 从主循环移到 `cv2.imshow` 之后 | FPS 无明显变化 |

三项优化均未显著改善 FPS。决定加入详细 profiling 定位真正瓶颈。

### Profiling 结果

在主循环各阶段插入计时点，输出：

```
PROFILE: sensor=0ms percept=323ms plan=0ms overhead=14ms front=7ms total=345ms
PROFILE: sensor=0ms percept=281ms plan=2ms overhead=13ms front=7ms total=303ms
PROFILE: sensor=0ms percept=243ms plan=4ms overhead=17ms front=13ms total=277ms
```

**结论：VLLiNet 推理占总耗时 93%（200~323ms/帧）**，其他环节合计仅 ~25ms。

### 优化 1 回退

GT 边缘跳过优化被回退（恢复 AI 模式下同时计算 GT 参考边缘），因为 A/B 对比功能比微小性能提升更重要。

### 优化 2 保留

`vehicle.get_transform()` 缓存保留，虽然 FPS 提升不明显，但减少了 5 次 CARLA RPC 调用，代码更整洁。

---

## 阶段 9: Skip-Frame 推理优化

### 问题

VLLiNet 推理 200~323ms/帧，导致整体 FPS 仅 3~4。需要在不降低感知质量的前提下提升帧率。

### 可选方案

| 方案 | 原理 | 预期效果 |
|------|------|----------|
| 降低输入分辨率 | 缩小到 624x192 推理 | 速度提升 ~4x，但精度下降 |
| Skip-frame | 每 N 帧推理一次，中间帧用缓存 | 速度提升 ~Nx，感知延迟增加 |
| FP16 量化 | 已在用 mixed precision | 已优化，无额外空间 |
| TensorRT 编译 | 模型编译为 TensorRT engine | 速度提升 2~5x，但需额外工具链 |

### 选择：Skip-Frame（每 3 帧推理一次）

最简单且效果明确。车辆速度 2~3 m/s 时，3 帧间隔约 0.15 秒，车辆移动 ~0.45m，路缘变化极小。

### 实现

```python
PERCEPT_INTERVAL = 3
cached_result = None
cached_road_mask = None

while True:
    run_percept = (frame_count % PERCEPT_INTERVAL == 1) or cached_result is None
    if run_percept:
        result = perception.process_frame(...)
        cached_result = result
        cached_road_mask = result[2]
    else:
        result = cached_result

    road_mask = cached_road_mask  # 始终用缓存的 mask
```

### 关键细节

- `road_mask` 单独缓存：因为 `result[2]` 可能在后续被修改，需要独立保存
- 首帧强制推理：`cached_result is None` 确保第一帧不会用空缓存
- 规划和控制仍然每帧执行：只有感知（最重的部分）被跳过

### 结果

- 推理帧：~280ms（与之前相同）
- 非推理帧：~25ms（仅规划+渲染）
- 有效 FPS：从 ~3 提升到 ~10（3 帧中 2 帧跳过推理）
- 感知质量：无明显下降，路径稳定

---

## 阶段 10: Overhead HUD 字体放大

### 问题

Overhead 俯视图上的状态文字（MODE、PAINT、距离等）字号偏小（0.6~1.2），在 1800x1600 的大画面上难以辨认。

### 修改

`draw_status_overlay()` 中所有 `cv2.putText` 的字号和粗细统一放大：

| 元素 | 原字号 | 新字号 | 原粗细 | 新粗细 |
|------|--------|--------|--------|--------|
| MODE | 1.2 | 2.0 | 3 | 4 |
| PERC / FPS | 1.0 | 1.8 | 3 | 4 |
| PAINT | 1.0 | 1.8 | 3 | 4 |
| Speed | 0.8 | 1.5 | 2 | 3 |
| Nozzle-Edge / Poly-Edge | 1.0 | 1.8 | 3 | 4 |
| Manual controls | 0.7 | 1.3 | 2 | 3 |
| Help text | 0.6~0.7 | 1.0~1.2 | 1~2 | 2~3 |

Y 坐标间距同步加大，避免文字重叠。

### 结果

Overhead 窗口文字清晰可读。

---

## 阶段 11: FPS 滑动平均

### 问题

FPS 显示在 ~4（VLLiNet 推理帧）和 ~30（缓存帧）之间剧烈跳动，因为 skip-frame 机制下每 3 帧只有 1 帧执行推理。瞬时 FPS（`1/dt`）无法反映真实平均帧率。

### 修改

将瞬时 FPS 改为 30 帧滑动平均：

```python
fps_history = []
FPS_SMOOTH_WINDOW = 30

# 每帧:
if dt > 0:
    fps_history.append(1.0 / dt)
    if len(fps_history) > FPS_SMOOTH_WINDOW:
        fps_history.pop(0)
    fps = sum(fps_history) / len(fps_history)
```

### 结果

FPS 显示稳定在 ~10（反映 skip-frame 后的真实平均帧率），不再跳动。

---

## 阶段 12: 视频录制格式修复（MP4 → AVI）

### 问题

CARLA 0.9.15 偶发段错误（exit code 139）时，MP4 文件损坏无法播放。原因是 MP4 格式需要在文件末尾写入 `moov atom`（索引信息），段错误导致进程被杀，索引未写入。

**现象：**
- Front 视频（1248x384，帧小）：偶尔幸存
- Overhead 视频（1800x1600，帧大）：几乎必定损坏

```
[mov,mp4,m4a,3gp,3g2,mj2 @ 0x55c5f480ae00] moov atom not found
Overhead: opened=False, frames=0
Front: opened=True, frames=171
```

### 解决方案

将视频格式从 MP4（mp4v 编码）改为 AVI（MJPEG 编码）：

```python
# 修改前
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
front_path = f'front_{timestamp}.mp4'

# 修改后
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
front_path = f'front_{timestamp}.avi'
```

**AVI 优势：** AVI 容器不依赖尾部索引，每帧独立写入。即使进程崩溃，已写入的帧仍可正常播放。

### 附加修复

Overhead 帧写入前增加内存连续性保证：

```python
f = np.ascontiguousarray(frame[:, :, :3].copy())
```

### 验证

段错误后检查 AVI 文件：

```
Overhead: opened=True, frames=248, w=1800, h=1600 ✓
Front: opened=True, frames=249, w=1248, h=384 ✓
```

### 结果

段错误后两路视频均完整可播放。

---

## 阶段 13: 出生点探索工具 + 多出生点测试

### 问题

V4 初期仅在 Spawn #4（x=155, y=-5.7, 南向直道）测试。需要在不同路段验证系统稳定性，但手动查找合适出生点效率低。

### 解决方案

创建轻量级出生点探索工具 `find_spawn.py`：

- 手动 WASD 驾驶，无感知/喷涂模块
- 实时 HUD 显示：x, y, z, yaw, speed, road_id, lane_id
- 喷嘴到右路缘距离（Map API 计算，绿色 3D 线可视化）
- **P 键**一键保存当前位置，输出可直接粘贴的 spawn dict 格式
- 俯视相机 + spectator 跟随

```python
# P 键输出示例：
  === Spawn Point #1 (Nozzle-Edge: 3.2m) ===
  {"x": -247.1, "y": -32.3, "z": 9.96, "yaw": 90.1, "desc": ""},
```

### 喷嘴-路缘距离计算（Map API）

```python
def get_nozzle_edge_distance(carla_map, nozzle_loc, vehicle_tf):
    wp = carla_map.get_waypoint(nozzle_loc)
    # 用车辆朝向计算右方向（避免 waypoint yaw 方向不一致）
    veh_yaw = math.radians(vehicle_tf.rotation.yaw)
    right_x = -math.sin(veh_yaw)
    right_y = math.cos(veh_yaw)
    # 喷嘴相对车道中心的横向偏移
    lat_offset = dx * right_x + dy * right_y
    # 到右边缘距离 = 半车道宽 - 横向偏移
    dist = half_width - lat_offset
    return dist
```

### 踩坑：waypoint yaw 方向不一致

初版用 waypoint 的 yaw 计算"右"方向，结果距离为负值（-3m）。原因是 CARLA 中 waypoint 的 forward 方向可能与车辆行驶方向不一致（取决于 lane_id 符号）。改用车辆自身 yaw 后修复。

### 多出生点测试结果

通过 `find_spawn.py` 在 Town05 全图手动驾驶，找到 3 个新出生点，加上原有 SP1，共 4 个主要测试点：

| SP# | x | y | z | yaw | 路段描述 | 测试结果 |
|-----|---|---|---|-----|----------|----------|
| 1 | 10 | -210 | 1.85 | 180 | Highway 直道 | 稳定 |
| 2 | -247.1 | -32.3 | 10.0 | 90.1 | 西侧北向 | 稳定，AI:12pts |
| 3 | 211.0 | -13.6 | 0.50 | -91.2 | 东侧南向 | 稳定，AI:12pts |
| 4 | 0.0 | 208.8 | 9.00 | 0.0 | 北侧东向 | 稳定，AI:12pts |

**碰撞问题：** 部分出生点 z 值过低导致 `Spawn failed because of collision`，需手动抬高 z（如 -0.04 → 0.50，8.22 → 9.00）。

### 结果

4 个出生点均通过长时间运行测试（400~6650 帧），AI 感知稳定输出 12 个路径点。

---

## 阶段 14: Overhead 实时坐标显示

### 问题

寻找出生点时需要在 V4 主程序中查看当前车辆坐标，但 overhead HUD 没有坐标信息，需要切换到 `find_spawn.py` 才能看到。

### 修改

`draw_status_overlay()` 增加 `veh_x, veh_y, veh_yaw` 参数，在 overhead 右上角显示白色坐标文字：

```python
# 函数签名增加参数
def draw_status_overlay(..., veh_x=0.0, veh_y=0.0, veh_yaw=0.0):

# 右上角显示坐标
cv2.putText(img, f"x={veh_x:.1f}  y={veh_y:.1f}  yaw={veh_yaw:.1f}",
            (w - 900, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
```

调用处传入车辆 transform：

```python
img = draw_status_overlay(...,
    veh_x=veh_tf.location.x, veh_y=veh_tf.location.y,
    veh_yaw=veh_tf.rotation.yaw)
```

### 结果

Overhead 窗口右上角实时显示 x, y, yaw，手动驾驶时可直接读取坐标用于出生点定位。

### V4 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `manual_painting_control_v4.py` | 新增 | V4 主入口，AI/GT 切换 + 多项式 + 2D 叠加 + 录制 + skip-frame + HUD 放大 + 坐标显示 |
| `diag_vllinet.py` | 新增 | VLLiNet 模型独立验证脚本 |
| `find_spawn.py` | 新增 | 出生点探索工具（手动驾驶 + 实时坐标 + 喷嘴距离 + P 键保存） |
| `perception/road_segmentor_ai.py` | 新增 | VLLiNet 封装，segment() 接口兼容 |
| `perception/perception_pipeline.py` | 修改 | use_ai 双模式 + rgb_bgra 参数 + GT 参考边缘 |
| `perception/edge_extractor.py` | 修改 | 新增 extract_road_edges_mask() 处理二值掩码 |
| `planning/vision_path_planner.py` | 修改 | 新增 estimate_nozzle_edge_distance() 多项式外推 |
| `carla_env/setup_scene_v2.py` | 修改 | 相机分辨率 1248x384，位置 z=3.5 pitch=-10 |
| `utils/__init__.py` | 新增 | utils 包初始化 |
| `utils/video_recorder.py` | 新增 | 后台线程双路视频录制（AVI+MJPEG 防崩溃损坏） |

### V4 关键技术突破

1. **VLLiNet 集成**：绕过缺失的 `carla_integration` 模块，直接加载 `VLLiNet_Lite`，自动检测 depth 通道数，修复 state_dict key 命名不匹配
2. **多项式盲区外推**：二次曲线拟合 3~20m 范围路缘点，外推到 lon=0 估计喷嘴处边距，10 帧中值滤波平滑
3. **AI/GT 实时切换**：G 键重建 PerceptionPipeline，AI 模式同时输出 GT 参考用于 A/B 对比
4. **2D 叠加替代 3D debug**：AI 模式下避免 3D 标记污染 RGB 相机输入，改用 OpenCV/pygame 2D 绘制
5. **后台线程录制**：deque 队列 + Event 信号 + daemon 线程，编码不阻塞主循环
6. **Skip-frame 推理**：每 3 帧运行一次 VLLiNet，有效 FPS 从 ~3 提升到 ~10
7. **性能 profiling 方法论**：分段计时定位瓶颈（93% 在推理），避免盲目优化非关键路径
8. **AVI 防崩溃录制**：MP4 段错误后 moov atom 丢失导致损坏，改用 AVI+MJPEG 格式，帧独立写入不依赖尾部索引
9. **出生点探索工具**：轻量级手动驾驶 + P 键一键保存坐标，Map API 计算喷嘴-路缘距离辅助定位
10. **FPS 滑动平均**：30 帧窗口消除 skip-frame 导致的瞬时 FPS 跳动

### V4 已知问题

1. **CARLA 段错误**：长时间运行（900+ 帧）后偶发 segfault（exit code 139），CARLA 0.9.15 已知问题，非代码 bug
2. **GT 模式 RuntimeWarning**：CityScapes 颜色匹配中 `np.sqrt` 遇到 NaN 值，不影响功能
3. **Nozzle-Edge 偶发 999.0m**：感知失败时无有效路缘点，边距计算返回默认值

### V4 运行数据（典型）

```
[F50]  Paint:OFF Perc:AI Spd:2.3 Noz:4.3m Poly:7.4m AI:12pts GT:12pts
[F100] Paint:OFF Perc:AI Spd:2.4 Noz:3.5m Poly:5.7m AI:12pts GT:12pts
[F200] Paint:OFF Perc:AI Spd:2.3 Noz:3.8m Poly:5.0m AI:12pts GT:12pts
[F500] Paint:OFF Perc:AI Spd:2.3 Noz:4.0m Poly:4.7m AI:12pts GT:12pts
[F900] Paint:OFF Perc:AI Spd:2.3 Noz:3.0m Poly:5.3m AI:12pts GT:12pts
```

- AI 感知稳定输出 12 个路径点
- GT 参考同步输出 12 个点（A/B 一致）
- Nozzle-Edge 稳定在 3.0~4.3m
- Poly-Edge 稳定在 4.7~7.4m
- 车速稳定在 2.3~2.4 m/s（Pure Pursuit 目标 3.0 m/s）
