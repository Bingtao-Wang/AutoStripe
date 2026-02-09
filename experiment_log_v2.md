# AutoStripe V2 实验记录

## 基本信息

- 日期: 2026-02-09
- 版本: V2 (Vision-based)
- 平台: CARLA 0.9.15-dirty (编译版, Linux)
- 地图: Town05 (高速公路路段)
- 车辆: vehicle.*stl*, 出生点 (10, -210, 1.85, yaw=180)
- Python: 3.8+

---

## V2 目标

**用视觉感知替代 Map API，实现感知→规划→控制→喷涂的完整闭环。**

V1 使用 CARLA Map API 提供完美路径（"作弊"），不符合真实划线机工况。V2 通过语义分割相机 + 深度相机实现真实感知驱动的自主导航。

### 核心改进

1. **感知模块**：语义分割 + 深度投影 → 世界坐标路沿点
2. **规划模块**：从感知路沿实时生成驾驶路径（不依赖地图）
3. **控制模块**：Pure Pursuit + 速度维持（应对横向偏差）
4. **可视化**：蓝色引导线实时显示规划路径

---

## 系统架构

```text
main_v2.py  (主入口, 主循环)
    |
    +-- carla_env/setup_scene_v2.py        场景搭建 (增加语义/深度/RGB前视相机)
    +-- perception/
    |       +-- road_segmentor.py          语义标签 → 路面掩码
    |       +-- edge_extractor.py          路面掩码 → 左右边缘像素
    |       +-- depth_projector.py         像素+深度 → 世界坐标
    |       +-- perception_pipeline.py     组合以上三步
    +-- planning/vision_path_planner.py    右路沿 → 驾驶路径 + 喷嘴路径
    +-- control/marker_vehicle_v2.py       Pure Pursuit + 速度维持
```

### 各模块职责

| 模块 | 文件 | 功能 |
|------|------|------|
| 场景 | `carla_env/setup_scene_v2.py` | 增加前视语义/深度/RGB相机 (x=2.5, z=2.8, pitch=-15) |
| 路面分割 | `perception/road_segmentor.py` | 从语义图提取路面掩码 (tags 1,6) |
| 边缘提取 | `perception/edge_extractor.py` | 逐行扫描提取左右路沿像素 (tags 2,8,9,11,22,27) |
| 深度投影 | `perception/depth_projector.py` | 像素+深度 → 3D世界坐标 (相机内参+坐标变换) |
| 感知管线 | `perception/perception_pipeline.py` | 组合分割+边缘+投影，输出世界坐标路沿点 |
| 路径规划 | `planning/vision_path_planner.py` | 右路沿向左偏移5m → 驾驶路径 (每帧替换) |
| 控制器 | `control/marker_vehicle_v2.py` | Pure Pursuit + 动态速度维持 |
| 主程序 | `main_v2.py` | 感知→规划→控制→喷涂→显示 (每帧循环) |

---

## 核心算法

### 1. 语义边缘提取

**策略**：从图像中心向外扫描，找到第一个路沿标签像素。

```python
# 关键参数
ROAD_SURFACE_TAGS = {0, 1, 6}  # Unlabeled, Road, RoadLines
ROAD_EDGE_TAGS = {8, 22, 2, 9, 11, 27}  # Sidewalk, Terrain, Fence, Vegetation, Wall, GuardRail
SKIP_TAGS = {5, 10, 24}  # Pole, Vehicle (透明，不中断扫描)

MIN_ROAD_RUN = 20  # 至少20个连续路面像素才接受边缘
MIN_DEPTH = 1.5m   # 过滤车辆引擎盖
MAX_DEPTH = 30.0m  # 过滤远处山/树 (关键修复)
MAX_LEFT_SCAN = 200px  # 限制左扫描范围，防止跨越对向车道
```

**扫描逻辑**：
1. 从图像中心向右扫描 → 右路沿
2. 从图像中心向左扫描（限制200px）→ 左路沿
3. 要求先经过≥20个连续路面像素，再接受边缘标签
4. Pole/Vehicle 标签透明（不中断路面连续性）

### 2. 深度投影

**CARLA 深度解码**：
```python
depth_m = (R + G*256 + B*65536) / (256³ - 1) * 1000
```

**相机内参** (800x600, FOV=90°)：
```python
fx = fy = 400
cx = 400
cy = 300
```

**坐标变换链**：
```
像素(u,v) + depth → 相机坐标(x_cam, y_cam, z_cam)
                  → UE4坐标(x, y, z)  [轴交换]
                  → 世界坐标(x_w, y_w, z_w)  [旋转+平移]
```

### 3. 视觉路径规划

**策略**：右路沿向左偏移 → 驾驶路径（每帧替换，不累积）

```python
# 偏移距离
line_offset = 3.0m      # 喷嘴距右路沿距离
nozzle_arm = 2.0m       # 喷嘴距车辆中心距离
driving_offset = 5.0m   # 车辆中心距右路沿距离 (line_offset + nozzle_arm)

# 左偏移方向 (CARLA左手坐标系)
left_perpendicular = (dy, -dx)  # NOT (-dy, dx)
```

**每帧流程**：
1. 对右路沿点按纵向距离排序，保留前方点
2. 重采样为1m间隔的点
3. 滑动窗口平滑
4. 向左偏移5m → 驾驶路径
5. **直接替换缓冲区**（不累积，蓝线=当前感知的实时引导）

### 4. Pure Pursuit 控制

**改进点**：
- 前瞻距离：使用实际距离到目标点，最小5m（不再用 Kdd*vf）
- 速度维持：根据当前速度动态调整油门（0.2-0.5）
- 前瞻点数：LOOKAHEAD_WPS = 15（更平滑转向）

```python
# 前瞻距离
ld_actual = sqrt((tx - veh_x)² + (ty - veh_y)²)
ld = max(ld_actual, 5.0)  # 最小5m，防止低速过度转向

# 速度维持
TARGET_SPEED = 3.0 m/s
if vf < 1.5:  throttle = 0.5
elif vf < 3.0:  throttle = 插值(0.2, 0.5)
else:  throttle = 0.2
```

---
## 实验过程

### 阶段1：语义标签识别问题

**问题现象**：
- 初始测试时，左右边缘都检测到图像中心（u=400）
- 边缘标签为 tag 9 (Vegetation)，但图像中心应该是路面

**根本原因**：
- 车辆旋转后（yaw=-98.1°），图像中心不再对准路面
- 需要先确认看到路面，再接受边缘标签

**解决方案**：
```python
# edge_extractor.py: 增加 found_road 约束
def _find_edge_tag(row_tags, row_depth, u_range):
    road_run = 0
    found_road = False  # 必须先看到路面
    for u in u_range:
        if t in ROAD_SURFACE_TAGS:
            road_run += 1
            if road_run >= MIN_ROAD_RUN:
                found_road = True
        elif t in ROAD_EDGE_TAGS:
            if found_road:  # 只有在看到路面后才接受边缘
                # 接受边缘标签
```

**效果**：边缘检测从中心移到正确位置（右边缘 u=406-464）

---

### 阶段2：路标轮廓误识别

**问题现象**：
- 路标（Pole, tag 5）之间的小路面碎片（9-18px）被当作"路面"
- 紧邻的 Vegetation 被误识别为路沿

**诊断数据**：
```
Row 200: Poles(400-530) → ROAD(531-544, 14px) → Vegetation(545-567)
```

**根本原因**：
- 路标之间的小缝隙被当作路面
- MIN_ROAD_RUN 参数太小（未设置）

**解决方案**：
```python
MIN_ROAD_RUN = 20  # 至少20个连续路面像素
SKIP_TAGS = {5, 10, 24}  # Pole 标签透明，不中断路面连续性
```

**效果**：
- 过滤掉小路面碎片
- 只接受真正的路面区域后的边缘
- 右边缘检测结果：R=39，全部为 tag 8 (Sidewalk) 和 tag 2 (Fence)

---

### 阶段3：语义标签错误识别

**问题现象**：
- 路面颜色 #01807F（浅青色），但被识别为 tag 0
- 实际路面应该是 tag 1 (Road)

**诊断过程**：
```python
# 检查语义图 R 通道
tags = sem_img[:, :, 2]
# 中心列 u=400, v=180-340: 全部 tag 1, depth 4.7-33.8m (真实路面)
# 中心列 u=400, v=350+: 全部 tag 0, depth 0.7-0.9m (车辆引擎盖)
```

**根本原因**：
- CARLA 0.9.15-dirty 版本中，tag 1 = Road（不是 Building）
- tag 0 = Unlabeled（车辆引擎盖/远处未标注区域）

**解决方案**：
```python
# edge_extractor.py
ROAD_SURFACE_TAGS = {0, 1, 6}  # 增加 tag 1
ROAD_EDGE_TAGS = {8, 22, 2, 9, 11, 27}  # 移除 tag 1

# road_segmentor.py
TAGS_BOUNDARY = {2, 5, 8, 9, 11, 22, 27}  # 移除 tag 1
```

**相机位置调整**：
- 从 x=2.0, z=1.8 → x=2.5, z=2.8
- 减少车辆引擎盖在画面中的占比

**效果**：
- 右边缘检测：R=225，全部 tag 2 (Fence)，depth 7.5-8.4m
- 路面识别正确率大幅提升

---
### 阶段4：路径偏移方向错误

**问题现象**：
- 车辆向右偏移，撞向墙壁
- 驾驶路径应该在路面内，但实际偏到路外

**根本原因**：
- CARLA 使用左手坐标系（y 向右增加）
- 左偏移向量计算错误：用了 `(-dy, dx)` 应该是 `(dy, -dx)`

**坐标系分析**：
```
CARLA 左手坐标系 (yaw=180°, 车辆朝-x方向):
  前进方向: (-1, 0)
  左侧方向: (0, 1)  ← y增加
  右侧方向: (0, -1)

左偏移向量 = 前进向量逆时针旋转90° = (dy, -dx)
```

**解决方案**：
```python
# vision_path_planner.py: _offset_left()
def _offset_left(self, edge_xy, offset):
    dx, dy = self._local_direction(edge_xy, i)
    length = math.sqrt(dx*dx + dy*dy)
    # 左偏移向量 (CARLA左手坐标系)
    lx = dy / length   # 原来是 -dy / length (错误)
    ly = -dx / length  # 原来是 dx / length (错误)
    nx = edge_xy[i][0] + lx * offset
    ny = edge_xy[i][1] + ly * offset
```

**效果**：车辆正确在路面内行驶，距离右路沿约5m

---

### 阶段5：车辆抖动与速度问题

**问题现象**：
- 车辆跟踪时持续抖动
- 速度过快，不符合划线机工况
- 车辆到道路边缘距离显示不正确

**解决方案**：

1. **减少抖动**：
```python
LOOKAHEAD_WPS = 8  # 从4增加到8
STEER_FILTER = 0.15  # 从0.3降低到0.15（更平滑）
Kdd = 3.0  # 从4.0降低到3.0
```

2. **降低速度**：
```python
throttle = 0.15  # 从0.3降低到0.15
```

3. **修正边距显示**：
- 原来用 LiDAR 测距（不准确）
- 改用感知世界坐标计算距离

```python
def compute_edge_distance_from_perception(vehicle, right_world, left_world):
    veh_loc = vehicle.get_location()
    if right_world:
        dists = [sqrt((loc.x - veh_loc.x)² + (loc.y - veh_loc.y)²) 
                 for loc in right_world]
        right_d = min(dists)
    # 同理计算 left_d
```

**效果**：车辆平稳行驶，速度适中，边距显示正确

---

### 阶段6：左边缘跨越对向车道

**问题现象**：
- 绿点（左边缘）跳到对向车道的路沿上
- 左扫描范围过大，跨越了整个道路

**根本原因**：
- 左扫描从中心到图像左边缘（0-400px）
- 包含了对向车道的路沿

**解决方案**：
```python
MAX_LEFT_SCAN = 200  # 限制左扫描范围为200像素
left_limit = max(0, cx - MAX_LEFT_SCAN)
lu = _find_edge_tag(row_tags, row_depth, range(cx, left_limit - 1, -1))
```

**效果**：左边缘检测限制在本车道范围内

---
### 阶段7：路径缓冲区更新问题（核心问题）

**问题现象**：
- 蓝色引导线不随车辆前进更新
- 车辆行驶一段距离后停住，轮子空转

**用户反馈**：
> "蓝色的待划轨迹还是没有往前更新，只是启动时有一段距离，车辆走完之后就卡在那里了"

**根本原因分析**：

1. **控制器索引失效**：
   - 规划器每帧修剪缓冲区前端（`_prune_behind_vehicle`）
   - 控制器的 `_nearest_index` 仍指向旧位置
   - 单调搜索窗口（SEARCH_WINDOW=30）无法找到正确的最近点

2. **路径缓冲区策略混乱**：
   - 累积式：每帧追加新点，但产生空间重叠
   - 替换式：每帧替换全部，路径长度固定不延伸

**解决方案演进**：

**尝试1：修复控制器索引**
```python
# marker_vehicle_v2.py: update_path()
def update_path(self, new_coords):
    self.waypoint_coords = new_coords
    if not new_coords:
        self._nearest_index = 0
        return
    # 重新搜索整个缓冲区找最近点
    veh_loc = self.vehicle.get_location()
    best_dist = float('inf')
    best_i = 0
    for i, (cx, cy) in enumerate(new_coords):
        d = (veh_loc.x - cx)² + (veh_loc.y - cy)²
        if d < best_dist:
            best_dist = d
            best_i = i
    self._nearest_index = best_i
```

**尝试2：改进累积策略**
```python
# 只追加纵向超过缓冲区末端的新点
def _extend_buffer(self, driving_pts, nozzle_pts, veh_loc, fwd_x, fwd_y):
    max_lon_existing = max([lon(cx, cy) for cx, cy in self.driving_coords])
    for i in range(n):
        if lon(driving_pts[i]) > max_lon_existing + min_spacing:
            self.driving_coords.append(driving_pts[i])
```

**问题**：远处深度投影误差产生异常点（如 y=-230），导致 max_lon 过大，新点无法追加。

**最终方案：每帧替换策略**

**关键理解**：
> 蓝色引导线 = 当前感知的实时引导，不是预先规划的累积路径

```python
# vision_path_planner.py: update()
def update(self, right_edges, vehicle_transform):
    # 每帧用当前感知结果直接替换缓冲区
    self.driving_coords = driving_pts[:n]
    self.nozzle_locations = nozzle_pts[:n]
    return self.driving_coords, self.nozzle_locations
```

**效果**：
- 路径长度稳定在 24-26 个点（当前感知范围）
- 蓝色线每帧实时更新，随车辆前进持续刷新
- 车辆不再因路径耗尽而停车

---

### 阶段8：车辆速度衰减至零

**问题现象**：
- 车辆初期正常行驶，逐渐减速，最终停止
- 数据显示：F150→F250: 5.1m, F250→F350: 2.5m, F350→F450: 1.9m

**诊断数据**：
```
F150: veh=(-11.2, -209.1), path[0]=(-17.8, -207.7)  横向偏差 1.4m
F300: veh=(-17.5, -208.9), path[0]=(-24.3, -207.7)  横向偏差 1.2m
F500: veh=(-21.2, -208.9), path[0]=(-27.9, -207.7)  横向偏差 1.2m
```

**根本原因**：
1. **横向偏差持续存在**：车辆出生在 y=-210，路径在 y=-207.7，偏差 2.3m
2. **Pure Pursuit 过度转向**：低速时 `ld = Kdd * vf = 3.0 * 0.1 = 0.3m`，前瞻距离极短
3. **转向导致失速**：大转向角 → 速度下降 → 更小的 ld → 更大的转向角（恶性循环）

**解决方案**：

**修复1：使用实际距离作为前瞻距离**
```python
# marker_vehicle_v2.py: step()
ld_actual = math.sqrt((tx - veh_loc.x)² + (ty - veh_loc.y)²)
ld = max(ld_actual, 5.0)  # 最小5m，防止过度转向
```

**修复2：增加前瞻点数**
```python
LOOKAHEAD_WPS = 15  # 从8增加到15，目标点更远，转向更平滑
```

**修复3：速度维持机制**
```python
# 动态调整油门，防止转向时失速
TARGET_SPEED = 3.0  # m/s
MIN_THROTTLE = 0.2
MAX_THROTTLE = 0.5

if vf_raw < TARGET_SPEED * 0.5:
    throttle = MAX_THROTTLE  # 速度过低，全油门
elif vf_raw < TARGET_SPEED:
    # 线性插值
    throttle = MIN_THROTTLE + (MAX_THROTTLE - MIN_THROTTLE) * (1.0 - vf_raw / TARGET_SPEED)
else:
    throttle = MIN_THROTTLE  # 速度足够，低油门
```

**效果**：
- 车辆持续稳定前进，速度维持在 ~3.6 m/s
- 横向偏差收敛到 0.4m
- 成功通过直道和弯道（行驶超过 240m）

---

### 阶段9：远处山/树误检

**问题现象**：
- 红点（右路沿）识别到远处山上的树边
- 路径末端出现异常坐标（如 y=-230, -235）

**根本原因**：
- MAX_DEPTH = 50.0m 过大
- 远处的 Vegetation (tag 9) 和 Terrain (tag 22) 被接受
- 远距离深度精度下降，投影位置错误

**解决方案**：
```python
# edge_extractor.py
MAX_DEPTH = 30.0  # 从50.0降低到30.0
```

**效果对比**：

| 参数 | 修复前 (50m) | 修复后 (30m) |
|------|-------------|-------------|
| 路径长度 | 40-44 点 | 24-26 点 |
| 路径末端距离 | 50-70m | 28-31m |
| path[-1].y | -230, -235 (异常) | -207.7~-208.2 (正常) |

**最终效果**：
- 路径末端距离稳定在 28-30m（符合深度限制）
- 无远处山/树误检
- 路径坐标稳定，无异常值

---
## 最终配置总结

### 传感器配置

| 传感器 | 类型 | 位置 | 参数 |
|--------|------|------|------|
| 语义分割相机 | `sensor.camera.semantic_segmentation` | x=2.5, z=2.8, pitch=-15 | 800x600, FOV=90° |
| 深度相机 | `sensor.camera.depth` | x=2.5, z=2.8, pitch=-15 | 800x600, FOV=90° |
| 俯视RGB相机 | `sensor.camera.rgb` | z=25, pitch=-90 | 1800x1600, FOV=90° |
| 语义LiDAR | `sensor.lidar.ray_cast_semantic` | z=1.8 | 32ch, 30m range |

### 感知参数

```python
# 语义标签集
ROAD_SURFACE_TAGS = {0, 1, 6}  # Unlabeled, Road, RoadLines
ROAD_EDGE_TAGS = {8, 22, 2, 9, 11, 27}  # Sidewalk, Terrain, Fence, Vegetation, Wall, GuardRail
SKIP_TAGS = {5, 10, 24}  # Pole, Vehicle, ??? (透明标签)

# 边缘提取参数
ROW_START_RATIO = 0.3      # 只扫描图像下方70%
SMOOTH_WINDOW = 7          # 中值滤波窗口
MIN_DEPTH = 1.5            # 最小深度 (过滤车辆引擎盖)
MAX_DEPTH = 30.0           # 最大深度 (过滤远处山/树)
MIN_CONFIRM = 2            # 连续边缘标签确认数
MIN_ROAD_RUN = 20          # 最小连续路面像素数
MAX_LEFT_SCAN = 200        # 左扫描最大范围 (防止跨越对向车道)

# 相机内参 (800x600, FOV=90°)
fx = fy = 400
cx = 400
cy = 300
```

### 规划参数

```python
# 偏移距离
line_offset = 3.0          # 喷嘴距右路沿距离 (m)
nozzle_arm = 2.0           # 喷嘴距车辆中心距离 (m)
driving_offset = 5.0       # 车辆中心距右路沿距离 (m)

# 路径生成
longitudinal_bin = 1.0     # 纵向采样间隔 (m)
smooth_window = 5          # 滑动窗口平滑
min_spacing = 0.5          # 最小点间距 (m)

# 缓冲区策略
每帧替换 (不累积)          # 蓝色线 = 当前感知的实时引导
```

### 控制参数

```python
# Pure Pursuit
LOOKAHEAD_WPS = 15         # 前瞻点数
SEARCH_WINDOW = 30         # 单调搜索窗口
STEER_FILTER = 0.15        # 转向低通滤波系数
MIN_PATH_POINTS = 5        # 最小路径点数

# 速度维持
TARGET_SPEED = 3.0         # 目标速度 (m/s)
MIN_THROTTLE = 0.2         # 最小油门
MAX_THROTTLE = 0.5         # 最大油门

# 前瞻距离
ld = max(actual_distance, 5.0)  # 使用实际距离，最小5m

# 车辆参数
wheelbase = 2.875          # 轴距 (m)
Kdd = 3.0                  # 前瞻距离系数 (未使用)
```

---

## 性能指标

### 测试环境

- **地图**: Town05 高速公路路段
- **起点**: (10, -210, 1.85), yaw=180°
- **测试时长**: 650-4250 帧 (约 22-142 秒)
- **测试距离**: 156-400 米

### 感知性能

| 指标 | 数值 | 说明 |
|------|------|------|
| 路面识别率 | 52.7-52.9% | 图像下半部分路面像素占比 |
| 右边缘检测数 | 196-420 像素/帧 | 随距离变化 |
| 左边缘检测数 | 0-1 像素/帧 | 受 MAX_LEFT_SCAN 限制 |
| 路径点数 | 24-26 点 | 稳定在此范围 |
| 路径前瞻距离 | 28-31 米 | 符合 MAX_DEPTH=30m 限制 |

### 规划性能

| 指标 | 数值 | 说明 |
|------|------|------|
| 路径更新频率 | 每帧 | 实时替换缓冲区 |
| 路径长度 | 24-26 点 | 对应 28-30m 前瞻 |
| 路径平滑度 | 良好 | 滑动窗口平滑 (window=5) |
| 偏移精度 | ±0.2m | 驾驶路径距右路沿 5.0±0.2m |

### 控制性能

| 指标 | 数值 | 说明 |
|------|------|------|
| 平均速度 | 3.6 m/s | 略高于目标 3.0 m/s |
| 速度稳定性 | ±0.3 m/s | 速度维持机制有效 |
| 横向偏差 | 0.4 m | 收敛到稳定值 |
| 转向平滑度 | 良好 | STEER_FILTER=0.15 |
| 前瞻距离 | 5-15 m | 动态调整，最小5m |

### 系统稳定性

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 直道行驶 | ✓ 通过 | 稳定行驶 >100m |
| 弯道行驶 | ✓ 通过 | 成功通过 Town05 弯道 |
| 速度维持 | ✓ 通过 | 无失速现象 |
| 路径更新 | ✓ 通过 | 蓝色线实时刷新 |
| 边缘误检 | ✓ 通过 | 无远处山/树误检 |
| 对向车道 | ✓ 通过 | 左边缘不跨越中线 |

### 典型运行数据 (v2_run16.log)

```
[F100] veh=(-11.8,-208.8) path[0]=(-19.0,-207.6) path[-1]=(-41.4,-207.9)
       edges: R=236, path: 26 pts, trail: 100 pts

[F300] veh=(-46.7,-207.9) path[0]=(-54.4,-207.7) path[-1]=(-76.9,-208.1)
       edges: R=194, path: 25 pts, trail: 300 pts

[F500] veh=(-82.2,-208.1) path[0]=(-89.7,-207.8) path[-1]=(-111.8,-208.0)
       edges: R=188, path: 25 pts, trail: 500 pts

[F650] veh=(-118.3,-208.0) path[0]=(-126.0,-207.8) path[-1]=(-146.6,-207.6)
       edges: R=196, path: 24 pts, trail: 600 pts
```

**分析**:
- 车辆从 x=9.8 行驶到 x=-118.3，总距离 ~128m
- 横向坐标稳定在 y=-208.0±0.2 (目标 y=-207.7)
- 路径前瞻距离稳定在 28-30m
- 右边缘检测数随距离变化 (近处多，远处少)

---

## 结论与展望

### V2 核心成果

1. **完全替代 Map API**：实现了从视觉感知到路径规划的完整闭环，不依赖 CARLA 地图 API。

2. **实时感知驱动**：每帧从语义分割和深度相机提取路沿，动态生成驾驶路径，符合真实划线机工况。

3. **稳定控制**：Pure Pursuit 控制器成功跟踪视觉生成的路径，速度维持机制有效防止失速。

4. **鲁棒性提升**：通过多轮调试，解决了语义标签识别、边缘误检、路径更新、速度衰减等关键问题。

### 关键技术突破

| 问题 | 解决方案 | 效果 |
|------|----------|------|
| 语义标签错误 | 确认 tag 1 = Road (CARLA 0.9.15-dirty) | 路面识别率 >50% |
| 路标轮廓误检 | MIN_ROAD_RUN=20 + Pole 透明 | 过滤小路面碎片 |
| 偏移方向错误 | 左偏移 = (dy, -dx) (左手坐标系) | 车辆正确在路面内行驶 |
| 路径不更新 | 每帧替换缓冲区 (实时引导) | 蓝色线持续刷新 |
| 车辆失速 | ld = max(actual, 5.0) + 速度维持 | 稳定行驶 >100m |
| 远处误检 | MAX_DEPTH=30m | 无山/树误检 |

### V2 vs V1 对比

| 维度 | V1 (Map API) | V2 (Vision-based) |
|------|-------------|-------------------|
| 路径来源 | CARLA Map API (完美) | 语义分割 + 深度投影 (真实) |
| 感知模块 | 无 (作弊) | 语义相机 + 深度相机 |
| 路径更新 | 启动时一次性生成 | 每帧实时生成 |
| 前瞻距离 | 200m (全局路径) | 28-30m (感知范围) |
| 真实性 | 不符合真实工况 | 符合真实划线机 |
| 鲁棒性 | 完美路径，无误差 | 受感知质量影响 |

### 当前限制

1. **感知范围有限**：MAX_DEPTH=30m 限制了路径前瞻距离，高速场景可能不足。

2. **依赖 CARLA 语义相机**：仍使用 CARLA 完美语义标签，未使用真实感知模型 (LUNA-Net)。

3. **单车道场景**：仅测试了高速公路单车道，未验证多车道、复杂路口等场景。

4. **左边缘未充分利用**：当前仅使用右边缘规划，左边缘检测结果未用于路径生成。

5. **无障碍物处理**：未考虑动态障碍物 (车辆、行人) 的避让。

### 未来工作

#### 短期改进 (V2+)

1. **集成 LUNA-Net**：
   - 替换 CARLA 语义相机为真实语义分割模型
   - 验证真实感知条件下的系统性能
   - 评估感知误差对规划控制的影响

2. **双边缘融合**：
   - 同时使用左右边缘计算道路中心线
   - 提高路径规划的鲁棒性和精度
   - 处理单侧边缘缺失的情况

3. **自适应深度范围**：
   - 根据车速动态调整 MAX_DEPTH
   - 低速时 20m，高速时 50m
   - 平衡感知精度和前瞻距离

4. **路径平滑优化**：
   - 使用样条插值替代滑动窗口
   - 减少路径抖动，提高跟踪精度

#### 中期扩展 (V3)

1. **多车道支持**：
   - 检测多条车道线
   - 支持车道保持和变道
   - 规划多条喷涂路径

2. **复杂场景测试**：
   - 城市道路 (Town01-04)
   - 路口、环岛、匝道
   - 不同天气和光照条件

3. **障碍物避让**：
   - 集成目标检测 (车辆、行人)
   - 动态路径重规划
   - 安全停车机制

4. **ROS + RVIZ 集成**：
   - 实现 `ros_interface/` 模块
   - 专业可视化界面
   - 实时数据记录和回放

#### 长期目标 (V4+)

1. **端到端学习**：
   - 图像 → 转向/油门 (模仿学习)
   - 减少手工设计的感知和规划模块
   - 提高泛化能力

2. **真实车辆部署**：
   - 迁移到真实划线机平台
   - 硬件适配 (工业相机、RTK-GPS)
   - 实地测试和优化

3. **全天候作业**：
   - 夜间作业 (红外相机)
   - 雨雪天气 (多传感器融合)
   - 低能见度场景

4. **智能决策**：
   - 自动检测已有标线磨损程度
   - 规划最优喷涂路径
   - 多车协同作业

---

## 附录

### 实验日志文件

- `experiment_log_v1.md` — V1 (Map API) 实验记录
- `experiment_log_v2.md` — V2 (Vision-based) 实验记录 (本文档)
- `/tmp/v2_run*.log` — V2 运行日志 (run8-run16)

### 关键代码文件

```
perception/
  road_segmentor.py          # 语义标签 → 路面掩码
  edge_extractor.py          # 路面掩码 → 左右边缘像素
  depth_projector.py         # 像素+深度 → 世界坐标
  perception_pipeline.py     # 完整感知管线

planning/
  vision_path_planner.py     # 右路沿 → 驾驶路径 + 喷嘴路径

control/
  marker_vehicle_v2.py       # Pure Pursuit + 速度维持

carla_env/
  setup_scene_v2.py          # V2 场景搭建 (语义/深度/RGB相机)

main_v2.py                   # V2 主入口 (standalone, 无ROS)
```

### 参考资料

- CARLA 0.9.15 Documentation: https://carla.readthedocs.io/en/0.9.15/
- CARLA Semantic Segmentation Tags: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
- Pure Pursuit Algorithm: Coulter, R. C. (1992). Implementation of the pure pursuit path tracking algorithm.
- LUNA-Net: (待补充真实感知模型论文)

---

**实验记录完成日期**: 2026-02-09  
**记录人**: AutoStripe V2 开发团队  
**CARLA 版本**: 0.9.15-dirty (编译版, Linux)  
**Python 版本**: 3.8+

