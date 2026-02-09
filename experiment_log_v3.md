# AutoStripe V3 实验记录

## 版本概述

**V3: 手动喷涂控制 + 增强可视化**

在 V2 视觉感知闭环基础上，增加：
- CityScapes 颜色匹配路面分割（替代原始标签 ID）
- 手动/自动驾驶切换 + 喷涂开关控制
- 坡道自适应蓝色点标记（驾驶路径）
- 喷嘴到路沿垂直距离可视化（绿色线）
- 黄色喷涂轨迹间断支持

**实现时间**：2026-02-09
**入口文件**：`manual_painting_control.py`

---

## 阶段 1: CityScapes 路面分割方案

### 问题

V2 使用原始语义标签 ID 匹配路面（tags 1, 24），但 CARLA 0.9.15-dirty 的标签 ID 非标准，导致路面识别不稳定。

### 分析

参考 CARLA 官方 `manual_control.py` 的按键 6（CityScapes 语义分割模式），发现：
- `image.convert(cc.CityScapesPalette)` 调用 CARLA C++ 引擎内部转换
- 即使原始标签 ID 非标准，C++ 引擎仍能正确映射到 CityScapes 标准颜色
- 路面在 CityScapes 中显示为紫色 RGB(128, 64, 128)

### 解决方案

**修改文件：**

1. `carla_env/setup_scene_v2.py` — 语义相机回调同时存储原始数据和 CityScapes 数据
2. `perception/road_segmentor.py` — 完全重写，改用 CityScapes 颜色匹配
3. `perception/perception_pipeline.py` — 增加 `cityscapes_bgra` 参数

**核心代码：**
```python
# setup_scene_v2.py - 双缓冲存储
def _semantic_cb(image):
    raw = np.frombuffer(image.raw_data, dtype=np.uint8).copy()  # 必须copy
    raw = np.reshape(raw, (image.height, image.width, 4))
    image.convert(cc.CityScapesPalette)  # 原地修改
    cs = np.frombuffer(image.raw_data, dtype=np.uint8)
    cs = np.reshape(cs, (image.height, image.width, 4))
    with sem_lock:
        sem_data["image"] = raw
        sem_data["cityscapes"] = cs.copy()

# road_segmentor.py - 颜色匹配
ROAD_COLOR_BGR = np.array([128, 64, 128], dtype=np.uint8)
COLOR_TOLERANCE = 10
```

### 结果

路面分割正确，不再受非标准标签 ID 影响。

---

## 阶段 2: 手动喷涂控制 + 前视显示

### 新增功能

创建 `manual_painting_control.py`，基于 V2 感知系统，增加 pygame 键盘控制：

| 功能 | 按键 | 说明 |
|------|------|------|
| 喷涂开关 | SPACE | 切换 ON/OFF，支持间断喷涂 |
| 驾驶模式 | TAB | 自动(Pure Pursuit) / 手动(WASD) |
| 倒车 | Q | 切换倒车模式 |
| 手刹 | X | 紧急制动 |
| 退出 | ESC | 退出程序 |

### 显示系统

- **pygame 窗口 (800x600)**：前视 RGB 相机 + 路面掩码绿色叠加 + 状态 HUD
- **OpenCV 窗口**：俯视图 + 黄色喷涂轨迹叠加

---

## 阶段 3: Bug 修复 — Front Perception 黑屏 + 黄线间断

### 问题 1: Front Perception 窗口始终黑屏

`setup_scene_v2.py` 中创建了 "Front Perception" OpenCV 窗口，但前视图已改为在 pygame 窗口渲染，该窗口从未写入数据。

**修复**：移除 `setup_scene_v2.py` 中的窗口创建代码。

### 问题 2: 黄线暂停恢复后连接到旧位置

按 SPACE 暂停喷涂 → 行驶一段距离 → 再按 SPACE 恢复，黄线瞬间连接到暂停时的终点。

**原因**：`paint_line()` 中 `last_nozzle_loc` 在喷涂关闭时仍被更新。

**修复**：
- 喷涂关闭时：`self.last_nozzle_loc = None`
- 恢复喷涂时：检测到 `last_nozzle_loc is None`，插入 `None` 间断标记
- 俯视图轨迹绘制：跳过 `None` 标记的段

---

## 阶段 4: 蓝色驾驶路径 z 坐标修复

### 问题

蓝色驾驶路径线在天桥上时显示在桥下地面（z 固定为 0.5）。

### 迭代过程

| 尝试 | 方案 | 结果 |
|------|------|------|
| 1 | 使用深度投影的 z 坐标 | 爬坡时蓝线飞到天上（深度投影 z 不准） |
| 2 | 使用 `vehicle.get_location().z` | 平地正常，但嵌入地面（底盘位置偏低） |
| 3 | `vehicle.z + 0.5` 偏移 | 平地正常，但爬坡时穿入坡面 |
| 4 | 车辆 pitch 推算每点 z | 正确！坡道自适应 |

### 最终方案

```python
fwd = vehicle_tf.get_forward_vector()
fwd_h = math.sqrt(fwd.x**2 + fwd.y**2)
slope = fwd.z / fwd_h  # tan(pitch)

# 每个点按纵向距离推算 z
lon = (dx * fwd.x + dy * fwd.y) / fwd_h
z = veh_loc.z + lon * slope + 0.5
```

### 同时改进：蓝线 → 蓝色点标记

将 `draw_line` 改为 `draw_point(size=0.1)`，每隔 2 个点画一个，视觉更清晰。

---

## 阶段 5: 喷嘴边距计算 + 绿色垂线可视化

### 需求

将显示的"车辆到路沿距离"改为"喷嘴到路沿距离"，并用绿色线可视化。

### 迭代过程

| 尝试 | 方案 | 结果 |
|------|------|------|
| 1 | 喷嘴到最近路沿点（欧氏距离） | 绿线斜向前方 45°（最近点在前方） |
| 2 | 找纵向偏移最小的路沿点 | 仍然斜（前视相机无侧方点） |
| 3 | 路沿横向距离中位数 + 构造垂直交点 | 正确！绿线垂直于车身 |

### 最终方案

```python
# 将路沿点投影到车辆坐标系
right_x = -fwd_y  # 右侧方向
right_y = fwd_x
for loc in right_world:
    lat = dx * right_x + dy * right_y  # 横向分量
    if lat > 0: lat_dists.append(lat)

# 用中位数构造垂直交点
median_lat = sorted(lat_dists)[len(lat_dists) // 2]
edge_point = veh_loc + median_lat * right_direction
```

### 关键理解

前视相机只能看到车辆**前方**的路沿，无法看到车辆**正侧方**。因此不能直接找"最近点"或"纵向偏移最小点"，而应取所有路沿点的**横向距离统计值**，在车身正侧方构造虚拟交点。

---

## 总结

### V3 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `manual_painting_control.py` | 新增 | V3 主入口，手动/自动控制 + 喷涂 |
| `carla_env/setup_scene_v2.py` | 修改 | 增加 CityScapes 双缓冲，移除黑屏窗口 |
| `perception/road_segmentor.py` | 重写 | 原始标签 → CityScapes 颜色匹配 |
| `perception/perception_pipeline.py` | 修改 | 增加 cityscapes_bgra 参数 |
| `main_v2.py` | 修改 | 蓝线改点标记 + 坡道z + CityScapes |
| `ros_interface/autostripe_node.py` | 修改 | CityScapes 数据传递 |

### V3 关键技术突破

1. **CityScapes 颜色匹配**：绕过非标准标签 ID 问题，利用 CARLA C++ 引擎的内部映射
2. **坡道 z 自适应**：用车辆 pitch 角推算前方每点的路面高度
3. **垂直边距计算**：前视相机无侧方数据时，用横向距离中位数构造虚拟垂直交点
4. **间断喷涂**：None 标记实现轨迹段间隔，支持暂停/恢复
5. **弯道抗偏移边距**：按纵向距离排序取最近 N 点的中位数，避免远处弯道点干扰
6. **多视图距离标注**：`debug.draw_string` 仅在 CARLA 编辑器可见，需 3D→2D 投影 + cv2.putText
7. **同步模式消除抖动**：异步模式下物理与渲染不同步导致相机抖动，同步模式 + 固定时间步长彻底解决

---

## 阶段 6: 追踪点边距可视化 + 多视图距离标注

### 需求

1. 给前方第一个追踪点 (driving_coords[0]) 也加上到路沿的距离可视化线（青色）
2. 距离数值在所有三个窗口（CARLA 编辑器、Overhead View、Front View）都要显示

### 问题：debug.draw_string 仅在编辑器可见

CARLA 的 `world.debug.draw_string()` 只在 CARLA 编辑器（UE4 Viewport）中渲染，**不会**出现在 `sensor.camera.rgb` 等传感器图像中。因此 Overhead View (OpenCV) 和 Front View (pygame) 看不到距离文字。

### 解决方案

**3D 世界坐标 → 2D 像素投影 + cv2.putText：**

1. 计算距离线中点的世界坐标 (`nozzle_mid`, `tp_mid`)
2. Overhead View：用 `world_to_pixel()` 投影到俯视相机像素，`cv2.putText` 绘制
3. Front View：新增 `world_to_front_pixel()` 函数，用针孔相机模型（yaw/pitch 旋转 + 焦距投影）

```python
def world_to_front_pixel(wx, wy, wz, cam_tf, img_w=800, img_h=600, fov=90):
    f = (img_w / 2.0) / math.tan(math.radians(fov / 2.0))  # 焦距
    # 世界→相机局部坐标 (yaw + pitch 旋转)
    # 针孔投影: px = cx + f * right / fwd
```

### 弯道抗偏移：统一边距函数

原始 `compute_edge_distance` 使用所有路沿点的中位数，弯道时远处点横向距离变小，拉低结果。

**改进**：`compute_point_edge_distance(ref_loc, right_world, vehicle_tf, max_lon=15.0)`
- 按纵向距离排序，只取最近 10 个点
- 用这 10 个点的中位数横向距离
- 弯道时最近的点受弯道影响最小

---

## 阶段 7: CARLA 编辑器视角跟随 + 同步模式消除抖动

### 需求

CARLA 编辑器（UE4 Viewport）的 Spectator 相机跟随车辆，并支持按键切换跟随/自由视角。

### 迭代过程

| 尝试 | 方案 | 结果 |
|------|------|------|
| 1 | `spectator.set_transform()` 直接设置（异步模式） | 严重抖动 |
| 2 | 改用 `get_forward_vector()` 计算偏移 | 仍然抖动 |
| 3 | Lerp 线性插值平滑 (`alpha=0.08`) | 减轻但仍有抖动 |
| 4 | `SpringArmGhost` 附着类型 | z-only 变换报错，程序卡死 |
| 5 | **同步模式 + 固定时间步长** | **彻底解决！** |

### 根因分析

**异步模式**下，CARLA 物理引擎和渲染引擎以不同频率运行。`spectator.set_transform()` 在物理帧之间被调用时，物理微更新导致车辆位置在两次渲染之间跳动，相机跟随放大了这种跳动。

### 最终方案：同步模式

```python
# 启用同步模式
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS 固定步长
world.apply_settings(settings)

# 主循环中：先 tick，再读取传感器，再更新相机
while True:
    world.tick()  # 推进一帧，物理+渲染同步
    # ... 读取传感器数据 ...
    # ... 更新 spectator（直接 set，无需插值） ...

# 清理时恢复异步模式（防止服务器卡死）
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
```

### SpringArmGhost 踩坑

`carla.AttachmentType.SpringArmGhost` 用于 `spawn_actor` 的相机附着，但：
- **z-only 变换不支持**：`carla.Transform(carla.Location(x=0, y=0, z=25))` 会报 "ill-formed" 警告并卡死
- 需要 x/y 偏移才能正常工作
- 对于需要精确对齐的感知相机（语义+深度+RGB），不应使用（会破坏对齐）

### V 键切换视角

新增 `V` 键切换 Spectator 跟随/自由模式：
- **FOLLOW**：每帧调用 `update_spectator()`，相机跟随车辆后上方
- **FREE**：停止更新，用户可在 CARLA 编辑器中自由拖动视角

### Pure Pursuit 前瞻距离调整

`LOOKAHEAD_WPS` 从 15 降至 8，追踪目标点更靠近车辆（~8m vs ~15m）。

---

## 阶段 7 修改文件

| 文件 | 修改 |
|------|------|
| `manual_painting_control.py` | 同步模式启用/恢复、world.tick()、V键视角切换、追踪点边距可视化、多视图距离标注 |
| `carla_env/setup_scene.py` | `update_spectator` 使用 `get_forward_vector()` |
| `control/marker_vehicle_v2.py` | `LOOKAHEAD_WPS` 15→8 |

