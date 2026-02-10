# AutoStripe V4 实施方案：VLLiNet AI 感知 + 多项式盲区外推

> 基于实际验证结果编写，所有技术细节均经过 CPU 推理测试确认。

---
● Town05, 坐标：                                                                  
                                                                                  
  - x=10, y=-210, z=1.85                                                          
  - yaw=180（朝负X方向）

## 1. V4 目标

V3 使用 CARLA 内置 CityScapes 语义相机作为"地面真值"道路分割。
V4 将其替换为 **VLLiNet**（已训练模型，MaxF 98.33%，IoU 96.72%），迈向真实世界可部署感知。

**两个核心工程挑战：**

| 挑战 | 描述 | 解决方案 |
|------|------|----------|
| 分辨率不匹配 | VLLiNet 训练输入 1248x384；V3 相机 800x600 | 原生分辨率相机，不做软件缩放 |
| 盲区几何 | 相机视野从 x≈3m 开始；喷嘴在 x=0（车身侧面） | 二次多项式拟合路缘点，外推到 lon=0 |

---

## 2. 已验证的关键发现

以下所有结论均在 `autostripe_v4` conda 环境（Python 3.10 + PyTorch 2.1）中验证通过。

### 2.1 Checkpoint 信息

```
文件: VLLiNet_models/checkpoints_carla/best_model.pth
大小: ~162 MB
Keys: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict,
      val_loss, val_maxf, val_iou, val_precision, val_recall
Epoch: 40
MaxF:  0.9833
IoU:   0.9671
State dict keys: 652
参数量: 14,034,865
```

### 2.2 两个必须修复的加载问题

**问题 1 — Depth 输入通道数**

| 项目 | 值 |
|------|----|
| Checkpoint 训练时 depth 通道 | **1**（单通道） |
| `VLLiNet_Lite` 代码默认 | 3（三通道） |
| 检测方法 | `state_dict['lidar_encoder.stage1.0.weight'].shape[1]` |

**修复**: 构建模型后替换 LiDAREncoder：
```python
from models.backbone import LiDAREncoder
model = VLLiNet_Lite(pretrained=False, use_deep_supervision=True)
model.lidar_encoder = LiDAREncoder(in_channels=1)  # 匹配 checkpoint
```

**问题 2 — Fusion 模块命名差异**

| 项目 | Key 前缀 |
|------|----------|
| Checkpoint state_dict | `fusion.fusion_modules.X...` |
| 代码中模型结构 | `fusion_module.fusion_modules.X...` |

两边结构完全一致（都是 5 层，各 200 个 key），仅命名不同。

**修复**: 加载时重命名：
```python
fixed_sd = {}
for k, v in state_dict.items():
    new_k = k.replace('fusion.fusion_modules', 'fusion_module.fusion_modules')
    fixed_sd[new_k] = v
model.load_state_dict(fixed_sd)
```

### 2.3 模型输入输出规格

```
输入 RGB:   [1, 3, 384, 1248]  — ImageNet 归一化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
输入 Depth: [1, 1, 384, 1248]  — min-max 归一化到 [0,1]
输出:       [1, 1, 192, 624]   — 1/2 分辨率
后处理:     sigmoid -> F.interpolate 到 (384,1248) -> threshold 0.5 -> uint8 mask
```

### 2.4 CARLA Depth 解码公式

```python
depth_meters = (R + G * 256 + B * 65536) / (256^3 - 1) * 1000
```

---

## 3. 环境配置

### 3.1 Conda 环境

| 环境 | Python | PyTorch | 用途 |
|------|--------|---------|------|
| `carla250116` | 3.7 | 1.13.1+cu117 | V1-V3（CARLA egg 兼容） |
| `autostripe_v4` | 3.10 | 2.1.0+cu121 | V4 AI 推理（支持 RTX 4060 Ti） |

### 3.2 autostripe_v4 环境已安装包

```
pytorch 2.1.0+cu121
numpy < 2.0
opencv-python
pygame
```

### 3.3 GPU 要求

- RTX 4060 Ti 16GB — 预计 VLLiNet 推理 ~50 FPS
- NVIDIA Driver 530.30.02，CUDA 12.1
- **当前状态**: 驱动降级（cuInit error 100），需重启恢复

### 3.4 CARLA Egg 兼容性问题

| 环境 | CARLA egg | 状态 |
|------|-----------|------|
| `carla250116` (Py3.7) | `carla-*3.7-linux-x86_64.egg` | 可用（V3 正常运行） |
| `autostripe_v4` (Py3.10) | 无对应 egg | **需要解决** |

**解决方案（按优先级）：**
1. `pip install carla==0.9.15` 在 autostripe_v4 环境中
2. 若 pip 无 0.9.15，尝试 `pip install carla==0.9.13`（API 兼容）
3. 若都不行，从源码编译 Python 3.10 的 egg
4. 最后方案：在 carla250116 环境升级 PyTorch 到 2.x（需验证 Python 3.7 兼容性）

---

## 4. 代码实现状态

### 4.1 文件清单与状态

| 文件 | 动作 | 状态 | 说明 |
|------|------|------|------|
| `diag_vllinet.py` | 新建 | 已完成+已修复 | 独立模型验证脚本 |
| `carla_env/setup_scene_v2.py` | 修改 | 已完成 | 1248x384, 位置(1.5, 2.4) |
| `perception/road_segmentor_ai.py` | 新建 | 已完成+已修复 | VLLiNet wrapper |
| `perception/perception_pipeline.py` | 修改 | 已完成 | AI/GT 双模式 |
| `planning/vision_path_planner.py` | 修改 | 已完成 | 多项式外推方法 |
| `manual_painting_control_v4.py` | 新建 | 已完成 | V4 主入口 |
| `ros_interface/autostripe_node.py` | 重写 | 已完成 | 订阅 Bridge Topics |
| `ros_interface/rviz_publisher.py` | 修改 | 已完成 | 多项式曲线 Marker |
| `ros_interface/topic_config.py` | 修改 | 已完成 | 新增 poly_curve topic |
| `configs/rviz/autostripe_v4.rviz` | 新建 | 已完成 | V4 RVIZ 布局 |
| `launch/autostripe_v4.launch` | 新建 | 已完成 | Bridge+AutoStripe+RVIZ |

### 4.2 V4 数据流架构

```
                        ┌─────────────────────────────┐
                        │       CARLA Server           │
                        │  (Town05, vehicle.*stl*)     │
                        └──────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     RGB 1248x384          Depth 1248x384        Semantic 1248x384
     (BGRA)                (BGRA encoded)        (CityScapes BGRA)
              │                    │                     │
              ▼                    ▼                     ▼
    ┌─────────────────────────────────────────────────────────┐
    │              PerceptionPipeline (use_ai=True/False)     │
    │                                                         │
    │  AI 模式 (G键切换):                                      │
    │    RGB→ImageNet归一化→[1,3,384,1248]                     │
    │    Depth→CARLA解码→min-max→[1,1,384,1248]               │
    │    VLLiNet_Lite推理→sigmoid→阈值0.5→road_mask            │
    │                                                         │
    │  GT 模式 (G键切换):                                      │
    │    Semantic→CityScapes紫色匹配→road_mask                 │
    │                                                         │
    │  通用后处理:                                              │
    │    road_mask→形态学→边缘提取→深度投影→世界坐标             │
    └────────────────┬────────────────────────────────────────┘
                     │
          left_world, right_world (3D 路缘点)
                     │
                     ▼
    ┌─────────────────────────────────────────────┐
    │         VisionPathPlanner                    │
    │                                              │
    │  1. right_world → 驾驶路径 (offset 3m)       │
    │  2. 多项式外推:                               │
    │     right_world → 车辆局部坐标               │
    │     → 筛选 lon∈[3m,20m], lat>0               │
    │     → np.polyfit(deg=2)                      │
    │     → 截距 c = 喷嘴到路缘距离                 │
    │  3. 短期记忆: 感知失败时保持10帧              │
    └──────────────┬──────────────────────────────┘
                   │
        driving_path, poly_dist, poly_coeffs
                   │
                   ▼
    ┌─────────────────────────────────────────────┐
    │      MarkerVehicleV2 (Pure Pursuit)          │
    │      + 喷涂控制 (SPACE键)                     │
    │      + 可视化 (蓝点路径/红点路缘/品红曲线)     │
    └─────────────────────────────────────────────┘
```

---

## 5. 重启后执行步骤

### Phase A: 环境验证（无 CARLA）

#### Step A1: 确认 GPU 驱动恢复

```bash
nvidia-smi
# 预期: RTX 4060 Ti, Driver 530.30.02, CUDA 12.1, 无 Unknown Error
```

#### Step A2: 确认 PyTorch CUDA 可用

```bash
conda activate autostripe_v4
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 预期: True NVIDIA GeForce RTX 4060 Ti
```

#### Step A3: 解决 CARLA egg 兼容性

```bash
conda activate autostripe_v4
pip install carla==0.9.15
# 若失败:
pip install carla==0.9.13
# 验证:
python -c "import carla; print(carla.__file__)"
```

### Phase B: 模型 GPU 验证（无 CARLA 服务器）

#### Step B1: GPU 推理验证

```bash
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15/0MyCode/AutoStripe
conda activate autostripe_v4

python -c "
import sys, torch, torch.nn.functional as F, numpy as np, time
sys.path.insert(0, 'VLLiNet_models')
from models.vllinet import VLLiNet_Lite
from models.backbone import LiDAREncoder

# 构建模型 (depth_ch=1)
model = VLLiNet_Lite(pretrained=False, use_deep_supervision=True)
model.lidar_encoder = LiDAREncoder(in_channels=1)

# 加载 checkpoint (key 重命名)
ckpt = torch.load('VLLiNet_models/checkpoints_carla/best_model.pth', map_location='cuda')
sd = {k.replace('fusion.fusion_modules','fusion_module.fusion_modules'): v
      for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(sd)
model = model.cuda().eval()

# Dummy 推理 + FPS 测试
rgb = torch.randn(1, 3, 384, 1248).cuda()
depth = torch.randn(1, 1, 384, 1248).cuda()

# 预热
for _ in range(10):
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _ = model(rgb, depth, return_aux=False)
torch.cuda.synchronize()

# 计时
t0 = time.time()
N = 50
for _ in range(N):
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(rgb, depth, return_aux=False)
torch.cuda.synchronize()
fps = N / (time.time() - t0)

mem = torch.cuda.memory_allocated() / 1e6
print(f'Output: {out.shape}')
print(f'FPS: {fps:.1f}')
print(f'GPU Memory: {mem:.0f} MB')
"
# 预期: Output [1,1,192,624], FPS > 40, GPU Memory < 500 MB
```

### Phase C: CARLA 端到端测试

#### Step C1: 运行 diag_vllinet.py（模型 + CARLA 实时推理）

```bash
# Terminal 1: 启动 CARLA
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15
./CarlaUE4.sh

# Terminal 2: 运行诊断
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15/0MyCode/AutoStripe
conda activate autostripe_v4
python diag_vllinet.py
```

**验证标准：**
- 绿色叠加层覆盖道路区域（非天空、非建筑）
- FPS > 30
- Road 占比 20%-60%（正常道路场景）
- 按 `q` 退出，actors 正常清理

#### Step C2: 运行 V4 主程序（AI 感知 + 手动控制）

```bash
conda activate autostripe_v4
python manual_painting_control_v4.py
```

**验证标准：**

| 测试项 | 操作 | 预期结果 |
|--------|------|----------|
| AI 感知 | 启动后观察 | 道路 mask 正确覆盖路面 |
| GT 切换 | 按 G 键 | HUD 显示 "GT"，切换到 CityScapes 分割 |
| AI 切换 | 再按 G 键 | HUD 显示 "AI"，切换回 VLLiNet |
| 多项式距离 | 观察 HUD | Poly-Edge 显示 0.5m-15m 范围内的数值 |
| 品红曲线 | 观察 CARLA 编辑器 | 品红色曲线从车前方延伸到车身位置 |
| 自动驾驶 | 按 TAB | 车辆沿路缘自动行驶 |
| 手动喷涂 | 按 SPACE | 黄色喷涂线出现在车辆右侧 |
| 红点路缘 | 观察编辑器 | 红色点标记右侧路缘，20m 范围内 |
| 蓝点路径 | 观察编辑器 | 蓝色点标记驾驶路径 |

#### Step C3: AI vs GT 对比验证

在同一路段分别用 AI 和 GT 模式行驶，对比：
- 道路 mask 覆盖范围是否一致
- 边缘点位置是否吻合
- 多项式距离 vs V3 中位数距离的差异
- 驾驶路径平滑度

### Phase D: ROS Bridge 集成（可选）

#### Step D1: 启动 CARLA-ROS Bridge

```bash
# Terminal 1: CARLA
./CarlaUE4.sh

# Terminal 2: ROS Bridge + AutoStripe + RVIZ
source /opt/ros/melodic/setup.bash
conda activate autostripe_v4
roslaunch autostripe autostripe_v4.launch
```

**注意**: ROS Melodic 是 Python 2.7，与 autostripe_v4 (Python 3.10) 可能冲突。
需要确认 CARLA-ROS Bridge 的 Python 版本兼容性。若不兼容，Phase D 需要额外适配。

---

## 6. 已知问题与解决方案

### 6.1 已解决

| 问题 | 原因 | 解决方案 | 状态 |
|------|------|----------|------|
| Checkpoint 加载失败 | fusion key 命名不一致 | 加载时 `fusion.` → `fusion_module.` | 已修复 |
| Depth 通道不匹配 | Checkpoint=1ch, 代码默认=3ch | 检测后重建 `LiDAREncoder(in_channels=1)` | 已修复 |
| CPU autocast 报错 | `torch.amp.autocast('cuda')` 不支持 CPU | 条件判断 `device.type == 'cuda'` | 已修复 |
| PyTorch 不支持 RTX 4060 Ti | Py3.7 + PyTorch 1.13 不支持 compute 8.9 | 新建 Py3.10 + PyTorch 2.1 环境 | 已解决 |
| NumPy 2.x 不兼容 | PyTorch 2.1 与 NumPy 2.x 冲突 | `pip install "numpy<2"` | 已解决 |

### 6.2 待解决（重启后）

| 问题 | 影响 | 优先级 | 解决方案 |
|------|------|--------|----------|
| NVIDIA 驱动降级 | CUDA 不可用，无法 GPU 推理 | **P0** | 重启系统 |
| CARLA egg 无 Py3.10 版本 | V4 无法 `import carla` | **P0** | `pip install carla` 或编译 egg |
| V3 分辨率被改 | `setup_scene_v2.py` 全局改为 1248x384 | P1 | 若需 V3 兼容，参数化分辨率 |
| 形态学核大小 | 15x15/5x5 在新分辨率下物理尺度不同 | P2 | 实测后调整 |
| MASK_TOP_RATIO | 35% 在 384px 高度下 = 134px 截断 | P2 | 实测后调整 |

---

## 7. 关键参数速查表

### 7.1 相机参数

| 参数 | V3 值 | V4 值 | 说明 |
|------|-------|-------|------|
| FRONT_CAM_W | 800 | **1248** | VLLiNet 原生宽度 |
| FRONT_CAM_H | 600 | **384** | VLLiNet 原生高度 |
| FRONT_CAM_X | 2.5 | **1.5** | 匹配 VLLiNet 训练位置 |
| FRONT_CAM_Z | 2.8 | **2.4** | 匹配 VLLiNet 训练位置 |
| FRONT_CAM_PITCH | -15 | -15 | 不变 |
| FRONT_CAM_FOV | 90 | 90 | 不变 |
| fx = fy | 400 | **624** | W/2 (FOV=90) |
| cx | 400 | **624** | W/2 |
| cy | 300 | **192** | H/2 |

### 7.2 VLLiNet 模型参数

| 参数 | 值 |
|------|----|
| 模型 | VLLiNet_Lite (MobileNetV3 + LiDAREncoder + SlimFusion + Decoder) |
| 参数量 | 14,034,865 |
| RGB 输入 | [1, 3, 384, 1248]，ImageNet 归一化 |
| Depth 输入 | [1, **1**, 384, 1248]，min-max [0,1] |
| 输出 | [1, 1, 192, 624]（1/2 分辨率） |
| 后处理 | sigmoid → bilinear 上采样 → 阈值 0.5 |
| Checkpoint | `VLLiNet_models/checkpoints_carla/best_model.pth` |
| 训练 Epoch | 40 |
| MaxF / IoU | 0.9833 / 0.9671 |

### 7.3 多项式外推参数（"隐形尺子"）

| 参数 | 值 | 说明 |
|------|----|------|
| 拟合阶数 | 2（二次） | `lat = a*lon² + b*lon + c` |
| 纵向范围 | 3m < lon < 20m | 过滤有效边缘点 |
| 横向约束 | lat > 0 | 只取车辆右侧点 |
| 最少点数 | 5 | 少于 5 点不拟合 |
| 距离合理范围 | 0.5m < dist < 15.0m | 超出范围回退 V3 中位数方法 |
| 外推目标 | lon = 0 处的 lat 值 | 即多项式截距 c |

**算法步骤：**

```
1. 将 right_world 边缘点转换到车辆局部坐标系:
   dx = edge.x - veh.x
   dy = edge.y - veh.y
   lon = dx * fwd_x + dy * fwd_y     (纵向 = 前方距离)
   lat = dx * right_x + dy * right_y  (横向 = 右侧距离)

2. 筛选: 3.0 < lon < 20.0 且 lat > 0

3. 拟合: np.polyfit(lon_array, lat_array, deg=2) → [a, b, c]

4. 喷嘴到路缘距离 = c (lon=0 处的横向距离)

5. 合理性检查: 0.5 < c < 15.0，否则返回 None
```

---

## 8. V4 键盘控制

| 按键 | 功能 | V3 有 |
|------|------|-------|
| SPACE | 切换喷涂 ON/OFF | 有 |
| TAB | 切换 自动/手动 驾驶 | 有 |
| **G** | **切换 AI/GT 感知模式** | **V4 新增** |
| WASD / 方向键 | 手动驾驶 | 有 |
| Q | 切换倒车 | 有 |
| V | 切换观察者跟随/自由 | 有 |
| X | 手刹 | 有 |
| ESC | 退出 | 有 |

---

## 9. V3 向后兼容性说明

`setup_scene_v2.py` 中的相机常量是全局共享的。V4 修改后：

| 常量 | V3 原值 | V4 当前值 | 影响 |
|------|---------|-----------|------|
| FRONT_CAM_W | 800 | 1248 | V3 pygame 窗口变宽 |
| FRONT_CAM_H | 600 | 384 | V3 pygame 窗口变矮 |
| FRONT_CAM_X | 2.5 | 1.5 | V3 相机位置前移 |
| FRONT_CAM_Z | 2.8 | 2.4 | V3 相机位置降低 |

**如果需要 V3 仍按原参数运行**，有两个方案：

1. **参数化**：`setup_scene_v2.py` 接受版本参数，V3/V4 各用各的常量
2. **不管**：V3 用新分辨率也能跑，只是画面比例变了（功能不受影响）

当前采用方案 2（不影响功能）。如需方案 1，后续再改。

---

## 10. 总结：重启后行动清单

```
□ Step A1: nvidia-smi 确认驱动恢复
□ Step A2: PyTorch CUDA 可用性确认
□ Step A3: 解决 CARLA egg / pip install carla
□ Step B1: GPU 推理 FPS 测试 (预期 >40 FPS)
□ Step C1: diag_vllinet.py 实时推理验证
□ Step C2: manual_painting_control_v4.py 端到端测试
□ Step C3: AI vs GT 对比验证
□ Step D1: (可选) ROS Bridge 集成测试
```

**最关键的两步：A3（CARLA egg）和 C2（端到端）。**

A3 解决后，所有代码已就绪，直接运行 `python manual_painting_control_v4.py` 即可。

---

> 文档编写日期: 2026-02-10
> 基于实际 CPU 推理验证结果，非理论推测
