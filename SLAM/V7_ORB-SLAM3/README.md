# V7_ORB-SLAM3 独立测试项目

独立的ORB-SLAM3性能测试项目，专注于自动驾驶场景中的立体-惯性SLAM评估。

## 项目特点

- **独立项目**：与AutoStripe划线机系统分离
- **自动驾驶**：使用CARLA autopilot，无需手动控制
- **立体-惯性SLAM**：752×480立体相机 + 200Hz IMU
- **实时可视化**：立体相机视图 + 轨迹对比图
- **评估记录**：ATE/RPE计算 + CSV导出
- **实验模板**：标准化实验记录流程

## 依赖安装

### 1. CARLA 0.9.15
```bash
# 已安装在 /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15
```

### 2. ORB-SLAM3 + ROS Wrapper
```bash
# 确保已编译ORB-SLAM3和ROS wrapper
# 参考：https://github.com/UZ-SLAMLab/ORB_SLAM3
```

### 3. Python依赖
```bash
pip install numpy opencv-python
# rospy通过ROS安装
```

## 快速启动

### 终端1：启动CARLA
```bash
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15
./CarlaUE4.sh
```

### 终端2：启动ORB-SLAM3
```bash
source ~/catkin_ws/devel/setup.bash
roslaunch orb_slam3_ros_wrapper autostripe_v7.launch
```

### 终端3：运行测试程序
```bash
cd SLAM/V7_ORB-SLAM3
python main_orb_slam3.py --map Town05 --spawn-x -50 --spawn-y 100
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--map` | Town05 | CARLA地图名称 |
| `--spawn-x` | -50 | 出生点X坐标 |
| `--spawn-y` | 100 | 出生点Y坐标 |
| `--spawn-z` | 1.85 | 出生点Z坐标 |
| `--spawn-yaw` | 180 | 出生点朝向角度 |

## 键盘控制

| 按键 | 功能 |
|------|------|
| S | 开启/关闭可视化窗口 |
| E | 开始/停止评估记录 |
| ESC | 退出程序 |

## 使用流程

1. **启动系统**（三个终端）
2. **等待初始化**：ORB-SLAM3需要5-10秒初始化
3. **开启可视化**：按S键查看立体相机和轨迹
4. **开始记录**：按E键开始评估记录
5. **自动驾驶**：车辆自动行驶30-60秒
6. **停止记录**：按E键停止，自动生成结果
7. **查看结果**：在 `experiments/run_YYYYMMDD_HHMMSS/` 查看CSV和图表

## 输出结构

每次评估记录会生成一个时间戳文件夹：
```
experiments/run_20260314_120000/
  slam_poses_detail.csv       # 每帧pose详情
  slam_eval_summary.csv       # ATE/RPE汇总
  trajectory_plot.png         # 轨迹对比图
```

## 实验记录

使用 `experiments/experiment_template.md` 记录实验：
1. 复制模板并重命名（如 `exp_001_town05.md`）
2. 运行实验并记录观察
3. 填写实验结果和分析

详见 `experiments/README.md`

## 传感器配置

- **立体相机**：752×480，FOV=90°，基线0.6m
- **IMU**：200Hz，位置(0, 0, 1.5)
- **相机位置**：车辆前方(2.5, 0, 3.5)，俯仰-15°

## 故障排查

### ORB-SLAM3初始化失败
- **现象**：长时间无蓝色轨迹
- **原因**：场景纹理不足或运动不够
- **解决**：等待车辆行驶到纹理丰富区域

### 轨迹混乱
- **现象**：蓝色轨迹与绿色GT偏差很大
- **原因**：ORB-SLAM3跟踪丢失
- **解决**：重启测试，检查ORB-SLAM3配置

### ROS连接问题
- **现象**：无法获取ORB pose
- **原因**：ROS话题未正确发布/订阅
- **解决**：检查 `rostopic list`，确认话题存在

### 可视化窗口无响应
- **现象**：按S键后无窗口
- **原因**：传感器数据未就绪
- **解决**：等待2-3秒后重试

## 项目结构

```
V7_ORB-SLAM3/
  main_orb_slam3.py           # 主程序
  carla_setup.py              # CARLA场景设置
  slam_interface.py           # ORB-SLAM3接口
  visualization.py            # 可视化模块
  evaluator.py                # 评估模块
  config/
    orb_slam3_carla.yaml      # ORB-SLAM3配置
  experiments/
    experiment_template.md    # 实验记录模板
    README.md                 # 实验说明
  README.md                   # 本文件
```

## 与AutoStripe的区别

| 特性 | AutoStripe V7 | V7_ORB-SLAM3 |
|------|---------------|--------------|
| 功能 | 划线机系统 | SLAM测试 |
| 控制 | 手动/自动 | 仅autopilot |
| 感知 | LUNA-Net/VLLiNet | 无 |
| 规划 | 路径规划 | 无 |
| SLAM | ORB+KISS-ICP | 仅ORB-SLAM3 |
| 复杂度 | 高 | 低 |
| 用途 | 完整系统 | SLAM评估 |

## 参考资料

- ORB-SLAM3论文：https://arxiv.org/abs/2007.11898
- CARLA文档：https://carla.readthedocs.io/
- AutoStripe V7文档：`../../docs/V7_Technical_Summary.md`

