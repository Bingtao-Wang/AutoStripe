# ORB-SLAM3实验记录 - exp_001_v7_initial_setup

## 基本信息
- **日期**: 2026-03-14
- **地图**: Town05
- **出生点**: (10, -210, 1.85)
- **车速**: 1 m/s (autopilot)
- **测试时长**: 开发调试阶段

## ORB-SLAM3配置
- **特征点数**: 2000
- **尺度因子**: 1.2
- **FAST阈值**: iniThFAST=10, minThFAST=5
- **IMU频率**: 200Hz
- **相机分辨率**: 752×480
- **基线**: 0.6m
- **焦距**: fx=fy=376
- **主点**: cx=376, cy=240

## 实验目标
创建独立的ORB-SLAM3测试项目，与AutoStripe划线系统分离，专注于SLAM性能评估。

## 实验过程

### 阶段1：项目搭建
- 创建独立项目结构
- 复用V7的传感器配置和SLAM接口
- 实现自动驾驶场景（autopilot）
- 实现实时可视化（双目+轨迹）

### 阶段2：Bug修复
1. **类名大小写问题** - `ORBSLAM3Wrapper` → `ORBSlam3Wrapper`
2. **方法名不匹配** - 添加`publish_imu_only`和`publish_stereo_only`方法
3. **ROS节点初始化** - 在创建wrapper前初始化ROS节点
4. **初始化顺序** - SLAM在传感器回调前初始化，避免早期数据丢失
5. **车速控制** - 修复traffic manager参数（正值=减速），设置1m/s
6. **车速平滑** - 禁用变道、忽略交通灯和其他车辆
7. **出生点调整** - 从(-50,100)改为(10,-210)避免撞墙
8. **f-string兼容性** - 改为.format()格式兼容Python 3.5+

### 阶段3：关键优化（轨迹精度）
1. **时间戳同步** - 相机和IMU都使用传感器自带timestamp
2. **IMU协方差矩阵** - 添加ORB-SLAM3需要的协方差参数
3. **IMU坐标系转换** ⭐ - 车辆坐标系转换为相机坐标系
   - 车辆系：X前，Y右，Z上（左手系）
   - 相机系：X右，Y下，Z前（右手系，OpenCV）
   - 转换：X_cam=Y_veh, Y_cam=-Z_veh, Z_cam=X_veh
4. **可视化优化** - 显示ORB特征点（2000个绿色关键点）

## 关键发现

### 1. IMU坐标系转换是核心
ORB-SLAM3轨迹漂移的主要原因是IMU数据坐标系不匹配。CARLA的IMU数据在车辆坐标系中，而ORB-SLAM3期望相机坐标系。必须进行坐标系转换。

### 2. 时间戳同步至关重要
使用传感器自带的timestamp而不是world snapshot，确保相机和IMU完美同步。

### 3. 初始化顺序影响数据完整性
SLAM接口必须在传感器回调设置之前初始化，否则早期的IMU数据会丢失，导致"not IMU meas"错误。

## 实验结果
- **状态**: 开发调试阶段，系统基本可运行
- **初始化**: ORB-SLAM3可成功初始化并创建地图
- **特征提取**: 2000个ORB特征点正常提取和显示
- **轨迹精度**: 待测试（IMU坐标系转换后需重新评估）

## 待解决问题
1. 轨迹精度验证 - 需要运行完整测试评估ATE/RPE
2. 车速稳定性 - autopilot可能仍有加减速，需进一步优化
3. 长时间稳定性 - 需测试长时间运行是否会崩溃

## 下一步计划
1. 运行完整测试，记录ATE/RPE数据
2. 对比不同场景（Town01/Town03/Town05）
3. 测试不同天气条件
4. 优化车速控制，实现真正匀速

## 技术总结
本次实验成功创建了独立的ORB-SLAM3测试项目，解决了多个关键技术问题：
- IMU坐标系转换（最关键）
- 时间戳同步
- 初始化顺序
- 车速控制

项目代码简洁（<500行），功能完整，为后续SLAM性能评估奠定了基础。

## 备注
- 项目位置：`SLAM/V7_ORB-SLAM3/`
- 主程序：`main_orb_slam3.py`
- 配置文件：`config/orb_slam3_carla.yaml`
- 实验数据将保存在：`experiments/run_YYYYMMDD_HHMMSS/`

---

## 阶段4：弯道漂移问题诊断与优化

### 问题描述
初始测试发现：
- 直道表现良好，轨迹基本准确
- **弯道出现严重漂移**，过弯后持续漂移，越来越偏离GT
- 这是立体-惯性SLAM的典型问题

### 优化尝试1：车速和特征点优化
**假设**：车速太慢（1m/s）导致弯道视差不足，特征跟踪失败

**措施**：
1. 车速提升：1m/s → 2m/s
2. 特征点增加：2000 → 3000
3. FAST阈值降低：(10,5) → (8,3)

**结果**：弯道漂移仍然存在

### 优化尝试2：IMU噪声参数调整
**假设**：IMU数据在弯道时误差较大，ORB-SLAM3过度信任IMU导致漂移

**措施**：逐步增加IMU噪声参数，降低IMU权重
- 第1次：NoiseGyro/Acc 0.001 → 0.01（提高10倍）
- 第2次：NoiseGyro/Acc 0.01 → 0.1（提高100倍，几乎纯视觉）

**结果**：弯道漂移问题持续存在，说明问题可能不仅在IMU权重

### 优化尝试3：2D地图可视化
**目的**：更直观地观察轨迹漂移情况

**实现**：
1. 从V6复用MapView类
2. 从CARLA地图采样waypoints构建道路多边形
3. 实时渲染GT轨迹（绿色）和ORB轨迹（蓝色）
4. 900×800像素俯视图，自动缩放

**效果**：可以清晰看到弯道时的轨迹偏离

## 当前状态（2026-03-14 下午）

### 已完成功能
- ✅ 独立测试项目搭建
- ✅ 双目ORB特征点可视化（3000个特征点）
- ✅ 2D地图俯视图（实时轨迹对比）
- ✅ IMU坐标系转换（车辆系→相机系）
- ✅ 时间戳完全同步
- ✅ 车速优化（2m/s）

### 待解决问题
1. **弯道漂移** ⚠️ - 核心问题，直道正常但弯道严重漂移
2. 可能原因：
   - IMU角速度数据在弯道时误差累积
   - 视觉特征在弯道时跟踪质量下降
   - IMU预积分模型与CARLA数据不完全匹配

### 下一步计划
1. 尝试纯立体视觉模式（禁用IMU）
2. 分析ORB-SLAM3日志，查看弯道时的跟踪状态
3. 考虑KISS-ICP作为备选方案
4. 记录详细的弯道测试数据

## 技术总结（更新）
完成了独立SLAM测试项目的搭建和多轮优化，识别了弯道漂移这一关键问题，需要进一步研究视觉-惯性融合策略。

---

## 阶段5：同步模式 + Tbc/Tlr坐标系修复 + 车道线地图

### 问题1："Empty IMU measurements vector" + ORB-SLAM3崩溃

**根因**：异步模式下IMU `sensor_tick=0.005`不可靠，实际回调频率远低于200Hz，两帧图像之间IMU数据不足，ORB-SLAM3预积分失败后segfault (exit code -11)。

**修复**：切换为同步模式
- `fixed_delta_seconds=0.005`（200Hz tick），IMU每tick触发（200Hz）
- 相机`sensor_tick=0.05`（每10个tick触发 = 20Hz）
- 主循环用`world.tick()` × 10代替`time.sleep(0.05)`
- Traffic Manager同步设置`set_synchronous_mode(True)`
- cleanup时恢复异步模式防止CARLA退出后卡死
- 每帧精确10条IMU数据，彻底消除"Empty IMU measurements vector"

### 问题2：车辆运动卡顿

**根因**：
1. 异步模式下仿真不等处理完就继续跑，帧处理跟不上导致位置跳跃
2. 2D地图每帧重绘Town05全部道路多边形（fillPoly），CPU开销大

**修复**：
1. 同步模式：仿真等主循环处理完才推进，不会卡顿
2. 移除2D地图可视化（MapView类），只保留双目ORB特征点窗口
3. 关闭ORB-SLAM3内置Pangolin viewer（`enable_pangolin=false`），省CPU/GPU

### 问题3：弯道漂移根因 ⭐⭐⭐

**发现**：Tbc和Tlr的平移量写在了错误的坐标系中！

`slam_interface.py`将IMU数据从车辆系转到相机系发布，ORB-SLAM3认为"body frame"就是相机系方向。但yaml中Tbc/Tlr的平移量仍是车辆系的值。

坐标系映射：`cam_x = veh_y, cam_y = -veh_z, cam_z = veh_x`

| 矩阵 | 修复前（车辆系） | 修复后（相机系） |
|------|----------------|----------------|
| Tbc平移 | (2.5, -0.3, 2.0) | (-0.3, -2.0, 2.5) |
| Tlr平移 | (0, 0.6, 0) | (0.6, 0, 0) |

**影响**：Tbc的lever arm方向错误，弯道时ORB-SLAM3的离心加速度补偿方向反了——左弯本该往左修正，结果往右修正，轨迹就往右偏。这完美解释了"左弯道后半段往右偏"的现象。

**修复文件**：
- `SLAM/V7_ORB-SLAM3/config/orb_slam3_carla.yaml`
- `slam/configs/orb_slam3_carla_stereo_inertial.yaml`（launch文件实际引用的）

### 问题4：地图缺少车道线纹理

**分析**：原Town05地图车道线被删除，纯沥青路面几乎无纹理，ORB特征点集中在远处建筑/树木。弯道时近处路面无特征可跟踪，视觉跟踪质量下降。

**修复**：切换到`Town05_line`地图（有车道线版本）
- 复制`Town05.xodr` → `Town05_line.xodr`（路网相同，仅视觉差异）
- 默认地图改为`/Game/Carla/Maps/Town05_line`
- 车道线提供高对比度边缘特征，填补图像下半部分（近处路面）的特征空白

### 阶段5修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `carla_setup.py` | 同步模式、camera sensor_tick、cleanup恢复异步、默认Town05_line |
| `main_orb_slam3.py` | world.tick()×10驱动、new_stereo flag、warmup、默认Town05_line |
| `visualization.py` | 移除MapView类和2D地图，只保留双目ORB显示 |
| `config/orb_slam3_carla.yaml` | Tbc/Tlr平移量转到相机坐标系 |
| `slam/configs/orb_slam3_carla_stereo_inertial.yaml` | 同上 + 同步ORB/IMU参数 |
| `autostripe_v7.launch` | 添加`enable_pangolin=false` |
| `Town05_line.xodr` | 从Town05.xodr复制，支持Town05_line地图加载 |

## 当前状态（2026-03-14 晚）

### 已完成功能
- ✅ 独立测试项目搭建
- ✅ 双目ORB特征点可视化（3000个特征点）
- ✅ IMU坐标系转换（车辆系→相机系）
- ✅ 时间戳完全同步
- ✅ 车速优化（2m/s）
- ✅ 同步模式（200Hz tick，精确10条IMU/帧）
- ✅ Tbc/Tlr坐标系修复（lever arm方向修正）
- ✅ Town05_line车道线地图
- ✅ 关闭Pangolin viewer省资源

### 待验证
1. **弯道漂移是否改善** - Tbc修复 + 车道线特征，需重新测试
2. **ATE/RPE指标** - 运行完整测试记录定量数据
3. **长时间稳定性** - 同步模式下是否稳定运行

### 硬件环境
- CPU: i5-13400 (10核16线程, 4.6GHz)
- GPU: RTX 4060 Ti 16GB
- 内存: 32GB
- 200Hz tick率下GPU占用约39%，性能充足
