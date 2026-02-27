# AutoStripe V1 实验记录

## 基本信息

- 日期: 2026-02-08
- 版本: V1
- 平台: CARLA 0.9.15 (编译版, Linux)
- 地图: Town05 (高速公路路段)
- 车辆: vehicle.*stl*, 出生点 (10, -210, 1.85, yaw=180)

---

## V1 目标

跳过 LUNA-Net 感知模块，使用 CARLA Map API 提供路径规划（Pure Pursuit），
验证**划线机车辆行驶并实时喷涂标线**的控制流程。

---

## 系统架构

```text
main_v1.py  (主入口, 主循环)
    |
    +-- carla_env/setup_scene.py   场景搭建 / 视角跟随 / 俯视显示
    +-- planning/lane_planner.py   Map API 路径点生成
    +-- control/marker_vehicle.py  Pure Pursuit 控制器
```

### 各模块职责

| 模块 | 文件 | 功能 |
|------|------|------|
| 场景 | `carla_env/setup_scene.py` | 连接CARLA、加载地图、生成车辆/俯视摄像头、跟随视角、OpenCV显示 |
| 规划 | `planning/lane_planner.py` | 从Map API生成200个中心车道路径点供Pure Pursuit使用 |
| 控制 | `control/marker_vehicle.py` | MarkerVehicle类, 封装Pure Pursuit算法驱动车辆沿路径行驶 |
| 主程序 | `main_v1.py` | 编排全流程: 场景初始化 -> 路径生成 -> 主循环(驾驶+喷涂+显示) |

---

## 核心算法

### 1. Pure Pursuit 路径跟踪

- 轮距 (wheelbase): 2.875 m
- 前视距离系数 (Kdd): 4.0
- 油门: 0.3 (恒定)
- 转向角: `delta = atan2(2 * L * sin(alpha), ld)`
- 路径点: Map API `wp.next(1.0)` 链式生成 200 个点, 间距 1.0 m

### 2. 喷嘴轨迹画线 (核心逻辑)

划线机的工作原理: **车辆行驶到哪里, 线就画到哪里。**

每帧执行:
1. 获取车辆实际位置 `(x, y, z)` 和朝向 `yaw`
2. 计算右侧喷嘴位置 (偏移 2.0 m):
   ```
   dx = offset * cos(yaw + pi/2)
   dy = offset * sin(yaw + pi/2)
   nozzle = (veh.x + dx, veh.y + dy, veh.z)
   ```
3. 从上一帧喷嘴位置到当前喷嘴位置画线段 (`debug.draw_line`)
4. 将喷嘴位置加入轨迹列表, 用于俯视图叠加显示

关键区别: 线条是车辆实际轨迹的右侧偏移, 不是地图预计算的标准车道边缘。

### 3. 跟随视角

基于车辆朝向计算摄像机位置, 始终在车辆正后方:
```
cam_x = veh.x - behind * cos(yaw)
cam_y = veh.y - behind * sin(yaw)
cam_z = veh.z + height
```
参数: behind=10m, height=6m, pitch=-20

### 4. 俯视图轨迹叠加

`debug.draw_line` 只在CARLA服务端视窗渲染, 摄像头传感器拍不到。
解决方案: 将喷嘴轨迹世界坐标投影到俯视摄像头图像坐标, 用 `cv2.line` 叠加绘制。

投影参数: 摄像头高度 25m, FOV 90度, 图像 1800x1600。

---

## 画线参数

| 参数 | 值 | 说明 |
|------|----|------|
| 颜色 | 黄色 (255, 255, 0) | 与路面自带白线区分 |
| 线宽 | 0.3 m | 从高空可见 |
| 喷嘴偏移 | 2.0 m | 车辆右侧 |
| 画线方式 | 每帧一段 | 上一帧喷嘴位置 -> 当前帧喷嘴位置 |
| 持久性 | life_time=1000, persistent_lines=True | 线条不消失 |

---

## 运行方式

```bash
# 终端1: 启动 CARLA
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15
./CarlaUE4.sh

# 终端2: 运行 V1
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15/0MyCode/AutoStripe
python main_v1.py

# Ctrl+C 退出
```

---

## 调试过程与问题解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Overhead窗口白屏 | `cv2.imshow` 在回调线程调用, OpenCV要求主线程显示 | 共享缓冲区 + 主循环中显示 |
| 画线逻辑错误(v1) | 预计算地图标准边缘并提前画出 = 染色地图 | 改为车辆实际轨迹 + 右侧偏移 |
| 画线逻辑错误(v2) | 同时画两侧 = 不符合划线机一次只能画一侧的实际 | 只画右侧 |
| 跟随视角转弯丢车 | 固定世界坐标偏移 `(x=13)` 不随车辆朝向旋转 | 基于 yaw 计算正后方位置 |
| Overhead无黄线 | `debug.draw_line` 不渲染到摄像头传感器 | 世界坐标投影到图像 + cv2叠加 |

---

## 实验结果

- 车辆在 Town05 高速公路上沿 Pure Pursuit 路径稳定行驶
- 车辆右侧 2m 处实时喷出黄色标线, 线条随车辆轨迹自然延伸
- CARLA 服务端视窗: 黄色 debug 线条可见
- OpenCV 俯视窗口: 黄色轨迹叠加可见
- 跟随视角在转弯时保持车辆居中

---

## 后续计划 (V2+)

- [ ] 接入 LUNA-Net 感知模块替代 Map API, 实现视觉驱动的路径规划
- [ ] 支持虚线车道分隔线 (6m线段 / 9m间隔)
- [ ] 支持左侧边缘线 (第二趟行驶)
- [ ] 评估模块: 喷涂线与标准车道边缘的横向误差统计
- [ ] 全天候测试 (夜间/雾天)
