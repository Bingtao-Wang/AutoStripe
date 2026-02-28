# V6: RViz 三面板显示

## 概述

V6 在 V5 三模式感知管线基础上做了两项核心升级：

1. **路径规划**：`VisionPathPlannerV2` (Nozzle-Centric 两级偏移) 替代 PD 控制器，
   几何推导驾驶路径消除弯道系统性偏移
2. **RViz 可视化**：新增实时图像发布（三面板 + 2D 地图），
   无需 CARLA-ROS Bridge，主循环直接通过 `rospy` 发布 `sensor_msgs/Image`

ROS 为可选依赖：未安装 ROS 时，V6 运行效果与 V5 完全一致（但使用新路径规划器）。

## 面板定义

| 面板 | Topic | 内容 | 分辨率 |
| ---- | ----- | ---- | ------ |
| 1 | `/autostripe/v6/front_overlay` | RGB 前视相机 + 道路掩码 + 边缘/路径叠加 | 1248x384 |
| 2 | `/autostripe/v6/perception_detail` | SNE 法线图 (LUNA) 或深度热力图 (VLLiNet/GT) | 1248x384 |
| 3 | `/autostripe/v6/overhead` | 俯视鸟瞰图 + 喷涂轨迹 + 状态叠加 | 900x800 |

面板 2 根据感知模式自适应切换：

- LUNA 模式：SNE 表面法线可视化为 RGB `((normal+1)/2 * 255)`
- VLLiNet/GT 模式：深度相机 COLORMAP_MAGMA 热力图

## 文件架构

```text
manual_painting_control_v6.py          -- V6 入口 (Nozzle-Centric 规划 + ROS 发布)
planning/vision_path_planner_v2.py     -- Nozzle-Centric 两级偏移路径规划器
ros_interface/rviz_publisher_v6.py     -- 3路 Image + MapView 发布器
ros_interface/topic_config.py          -- + V6 topic 常量 (5个)
configs/rviz/autostripe_v6.rviz       -- RViz 布局 (3个图像面板)
launch/autostripe_v6.launch           -- 仅 RViz 的 launch 文件
```

## 路径规划方案变更：Nozzle-Centric (V5.2)

V6 采用 `VisionPathPlannerV2` 替代 V5 的 `VisionPathPlanner`，
用几何推导替代 PD 反馈控制器，从根本上解决弯道画线距离偏移问题。

### 核心思路

不再让车辆"追"一个目标距离（PD 控制 `driving_offset`），
而是从道路边缘**几何反推**出车辆应该走的路径：

```
right_edge ──[line_offset=3.1m]──> nozzle_path ──[compensated_arm]──> driving_path
                                                        ↑
                                          nozzle_arm(2.0) + K_CURV_FF(55.0) × curvature
```

| 对比项 | V5 (PD + 曲率前馈) | V6 (Nozzle-Centric) |
|--------|----------------------|------------------------|
| 控制方式 | PD 控制 driving_offset | 几何两级偏移，无 PD |
| 路径生成 | 边缘 → 固定 offset → 驾驶路径 | 边缘 → 喷嘴路径 → 驾驶路径 |
| 弯道补偿 | 曲率前馈修正 offset | 局部法线偏移 + 曲率补偿 arm |
| 偏移方向 | 全局横向 | 每点沿局部切线法向量 |
| 参数 | Kp, Kd, OFFSET_SMOOTH, FF_GAIN | line_offset, nozzle_arm, K_CURV_FF |

### 每帧流程

1. 右侧边缘点按纵向排序 → 1m 间隔重采样（起点 3m）→ 5 点滑动平均
2. **空间异常值剔除** `_reject_outliers()`：二次拟合，剔除横向残差 > 0.5m 的离群点
3. **Stage 1**：边缘沿局部法线向左偏移 `line_offset=3.1m` → 喷嘴目标路径
4. 估计曲率：边缘点二次拟合 |a₂|，EMA(α=0.10) + rate_limit(0.0005/frame)
5. **Stage 2**：喷嘴路径沿局部法线向左偏移 `compensated_arm` → 车辆驾驶路径
6. **路径时域 EMA**：`PATH_EMA_ALPHA=0.15`（15% 新 + 85% 旧），抑制帧间跳变
7. Pure Pursuit 跟踪驾驶路径

### Nozzle-Centric 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `line_offset` | 3.1m | 边缘→喷嘴路径偏移（含 0.1m baseline 修正） |
| `nozzle_arm` | 2.0m | 喷嘴→车辆中心距离 |
| `K_CURV_FF` | 55.0 | 曲率前馈增益（弯道增大 arm） |
| `PATH_EMA_ALPHA` | 0.15 | 路径时域平滑系数 |
| `POLY_EMA_ALPHA` | 0.4 | 多项式系数 EMA 平滑 |
| curvature EMA α | 0.10 | 曲率估计平滑（慢响应防过冲） |
| curvature rate_limit | 0.0005/frame | 曲率变化速率限制 |
| `POLY_SMOOTH_WINDOW` | 10 | poly_dist 中值滤波窗口 |
| `NED_SMOOTH_WINDOW` | 150 | nozzle-edge 距离中值滤波窗口 |

### AutoPaint 状态机参数（Headless 帧率适配 ×10）

| 参数 | V5 手动模式 | V6 当前值 |
|------|----------|-----------|
| `tolerance_enter` | 0.30m | 0.30m |
| `tolerance_exit` | 0.45m | 0.55m |
| `stability_frames` | 60 | 150 |
| `GRACE_LIMIT` | 15 | 300 |
| `STABILIZED_GRACE` | — | 100 |
| `TOL_ENTER_CURVE` | 0.55 | 0.55 |
| `TOL_EXIT_CURVE` | 0.80 | 0.80 |

### 感知缓存

`PERCEPT_INTERVAL=3`：AI 感知每 3 帧运行一次，中间帧复用上次结果，降低 GPU 负载。

### 3D 调试点禁用

V6 禁用 CARLA `world.debug.draw_point/draw_line`（边缘红点、驾驶路径蓝点），
因为 3D 调试绘制会污染深度/语义相机传感器输出。仅保留非 AI 模式下的绿色喷嘴距离线。

## 相对 V5 的变更

### 修改的文件

- `perception/road_segmentor_luna.py` — 新增 `self.last_normal` 存储 SNE 输出
- `perception/perception_pipeline.py` — 新增 `last_normal` 属性 (仅 LUNA 模式返回)
- `ros_interface/topic_config.py` — 新增 5 个 V6 topic 常量

### 新增的文件

- `manual_painting_control_v6.py` — V6 入口 (V5 + Nozzle-Centric 规划 + ROS 发布)
- `planning/vision_path_planner_v2.py` — Nozzle-Centric 两级偏移路径规划器
- `ros_interface/rviz_publisher_v6.py` — 轻量级 3 路 Image + MapView 发布器
- `configs/rviz/autostripe_v6.rviz` — RViz 三面板配置
- `launch/autostripe_v6.launch` — 仅启动 RViz (无需 Bridge)

### 主循环中的 V6 关键改动

1. 路径规划：`VisionPathPlanner` → `VisionPathPlannerV2`，去掉 `set_dynamic_offset()` PD 控制
2. 启动时 `rospy.init_node('autostripe_v6', disable_signals=True)` (try/except 包裹)
3. `_render_front_view()` 返回 BGR 合成图 (面板 1)
4. `_build_perception_detail()` 构建面板 2 (SNE 法线图或深度热力图)
5. 每帧调用 `rviz_pub.publish_all(front, detail, overhead, map, dashboard)` 发布
6. `PERCEPT_INTERVAL=3` 感知缓存，AI 模式每 3 帧推理一次
7. 3D 调试绘制禁用（防止污染深度相机）
8. finally 块中调用 `rospy.signal_shutdown()` 清理

## 运行方式

```bash
# 无 ROS (与 V5 完全一致):
python manual_painting_control_v6.py

# 有 ROS + RViz:
# 终端 1: ./CarlaUE4.sh
# 终端 2: python manual_painting_control_v6.py
# 终端 3: roslaunch autostripe autostripe_v6.launch
#   或: rviz -d configs/rviz/autostripe_v6.rviz
```

## 实现日志

- 2026-02-28: V6 路径规划方案变更 — 采用 `VisionPathPlannerV2` (Nozzle-Centric 两级偏移) 替代 `VisionPathPlanner` (PD 控制)。去掉 `set_dynamic_offset()` 反馈控制，改为几何两级偏移：边缘 → 喷嘴路径 (line_offset=3.1m) → 驾驶路径 (nozzle_arm + 曲率补偿)。新增空间异常值剔除 `_reject_outliers()` + 路径时域 EMA (α=0.15)。详见 `docs/V5_2_改良控制逻辑_Log.md`。
- 2026-02-28: V6 实现完成，共创建/修改 8 个文件，按计划全部完成。
- 2026-02-28: 新增 3D Marker 发布 — 复用 `RvizPublisher`，3D 视口显示道路边缘、驾驶路径、多项式曲线、喷涂轨迹、车辆位置、状态文字。
- 2026-03-01: 新增 Town05 地图背景 — `publish_map_roads()` 通过 `generate_waypoints(2.0)` 采样全地图，按 (road_id, lane_id) 分组，发布车道边缘 (LINE_STRIP) + 路面填充 (TRIANGLE_LIST)。latch=True 启动时发布一次，RViz 订阅即可见。Topic: `/autostripe/v6/map_roads`。
- 2026-03-01: 地图改为 2D 图像渲染 — 废弃 3D MarkerArray 方案（RViz 视口缩放/平移不可用），改用 `MapView` 类在 Python 端渲染 2D 俯视地图图像，发布为 `sensor_msgs/Image`。支持按键控制：`[ ]` 缩放、`Shift+方向键` 平移、`, .` 旋转、`\` 重置、`M` 跟随/自由切换。地图上叠加车辆(绿色三角)、喷涂轨迹(黄色)、驾驶路径(蓝色)。
- 2026-03-01: 喷涂轨迹渐变色 — 2D 地图上的喷涂轨迹根据喷嘴到道路边缘距离显示发散型渐变色。采用论文一致的 colormap 结构（TwoSlopeNorm + 中间 30% 纯绿），`_nozzle_dist_color()` 模块级函数实现。
- 2026-03-01: 渐变色区间收紧 — `vmin=2.0,vmax=4.0` → `vmin=2.6,vmax=3.4`，0.8m 总跨度，纯绿区间 2.88~3.12m (±0.12m)，偏差 0.2m 即可看到明显颜色变化。
- 2026-03-01: 渐变色亮度提升 — 暗色调论文配色改为高亮色适配深色地图背景：蓝 #2166ac→#3ca0ff、绿 #1a9850→#00ff00、橙 #fc8d59→#ffa050、红 #d73027→#ff3232。
- 2026-03-01: ROS 连接优化 — `rospy.init_node()` 前增加 `rosgraph.is_master_online()` 预检查，master 不在线时立即跳过（不再阻塞等待）。
- 2026-03-01: CARLA 调试绘制全部禁用 — 绿色喷嘴距离线 (`draw_line`/`draw_string`) 也一并关闭，V6 不再向 CARLA 编辑器写入任何 `debug.draw_*`，避免污染深度/语义传感器。
