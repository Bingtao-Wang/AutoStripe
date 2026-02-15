# AutoStripe V5+ 实验记录

## 版本概述

**V5+: 统一数据集采集 + LUNA-Net 重训练 + 新权重验证**

在 V5 三模态感知系统基础上，完成数据采集→模型重训练→权重验证的完整闭环：
- 统一数据集采集脚本（3地图 × 4天气 × 300帧 = 3600帧）
- KITTI格式 + SNE预计算 + 车辆轨迹记录
- LUNA-Net 用新数据集重训练，生成 best_net_LUNA.pth
- 新权重在 V5 管线中验证通过（2100+ 帧稳定运行）

**实现时间**：2026-02-12 ~ 2026-02-13
**关键文件**：`datasets/carla_highway/collect_night_dataset.py`, `datasets/carla_highway/visualize_trajectory.py`

---

## 阶段 1: 采集脚本改造

### 目标

现有采集脚本相机参数 (1600×900, x=1.5, z=2.4, pitch=-5) 与当前测试环境 (1248×384, x=2.5, z=3.5, pitch=-15) 不一致，导致训练-测试分布偏移。需要改造采集脚本使其与 V5 测试环境完全匹配。

### 修改内容

| 参数 | 原值 | 新值 |
|------|------|------|
| DEFAULT_WIDTH | 1600 | 1248 |
| DEFAULT_HEIGHT | 900 | 384 |
| CAMERA_X | 1.5 | 2.5 |
| CAMERA_Z | 2.4 | 3.5 |
| CAMERA_PITCH | -5.0 | -15.0 |
| LABEL_ROAD | 7 | 1 (CARLA 0.9.15-dirty) |
| LABEL_ROADLINE | 6 | 24 (CARLA 0.9.15-dirty) |

新增功能：
- SNE 表面法线预计算（depth → normal, 保存为 float32 NPY）
- 深度图双格式保存（uint16 mm + float32 meters）
- 4种天气预设（ClearNight, ClearDay, HeavyFoggyNight, HeavyRainFoggyNight）
- 按地图划分训练/验证集（Town04+Town06=训练, Town05=验证）
- Spawn point 队列系统（避免重复路段）
- 断点续传（checkpoint 机制）

删除：LiDAR 相关代码

---

## 阶段 2: GT 语义标签调试

### 问题

首次采集发现 GT 提取错误，电线杆被识别为道路。

### 排查过程

1. 初始方案：使用 CityScapes 调色板颜色匹配（紫色 128,64,128）
   - 结果：40-45% 路面覆盖率，但有 int16 溢出 warning
   - 问题：逻辑复杂且不稳定

2. 尝试标准 CARLA 标签 (Road=7, RoadLine=6)
   - 结果：仅 0.2-7.1% 覆盖率，严重不足

3. 实际标签分布分析（CARLA 0.9.15-dirty）：
   - Tag 1: 47.7% → 实际是 Road（标准 CARLA 中 1=Building）
   - Tag 7: 0.1% → 标准 Road 标签几乎不存在
   - Tag 24: 2.1% → RoadLine（0.9.15-dirty 特有）

### 最终方案

```python
LABEL_ROAD = 1       # CARLA 0.9.15-dirty: Road
LABEL_ROADLINE = 24  # CARLA 0.9.15-dirty: RoadLine

labels = array[:, :, 2]  # R通道 = 语义标签ID
road_mask = (labels == LABEL_ROAD) | (labels == LABEL_ROADLINE)
```

验证结果：51-55% 路面覆盖率，正确。

### 关键结论

- CARLA 0.9.15-dirty 使用非标准语义标签 ID，不能用标准 CARLA 文档的映射
- GT 应保持完整路面区域（包括对向车道），不做 ROI 裁剪，模型自行学习分布
- ROI 裁剪（MASK_TOP_RATIO）仅在推理时应用

---

## 阶段 3: 数据集采集

### 第一轮采集（完整 3600 帧）

```
采集配置:
  地图: Town04, Town05, Town06
  天气: ClearNight, ClearDay, HeavyFoggyNight, HeavyRainFoggyNight
  每场景: 300帧 (3地图 × 4天气 × 300 = 3600帧)
  车辆: vehicle.tesla.model3 + autopilot
  跳帧: 每15帧采集1帧

结果:
  Training: 2400帧 (Town04 1200 + Town06 1200)
  Validation: 1200帧 (Town05)
  总大小: 28GB
  耗时: 3小时2分
```

### Town06 重新采集

轨迹可视化发现 Town06 路径分布异常，删除旧数据后重新采集：

```
python collect_night_dataset.py --map Town06 --num-frames 1200 --output-dir ./datasets/CARLA_Unified_Dataset
```

- 1200帧，4种天气各300帧
- 耗时1小时
- 退出码139（CARLA cleanup segfault，数据完整）

### 最终数据集统计

| 子目录 | Training | Validation |
|--------|----------|------------|
| image_2 (RGB PNG) | 2400 | 1200 |
| depth_u16 (16-bit PNG) | 2400 | 1200 |
| depth_meters (float32 NPY) | 2400 | 1200 |
| normal (float32 NPY) | 2400 | 1200 |
| gt_image_2 (binary mask PNG) | 2400 | 1200 |
| calib (KITTI txt) | 2400 | 1200 |

压缩包：CARLA_Unified_Dataset.tar.gz (1.2GB)

---

## 阶段 4: 轨迹可视化

### 改造

将 `visualize_trajectory.py` 从 OpenCV 栅格输出改为 matplotlib 矢量输出：
- 从 CARLA Map API 提取道路拓扑作为背景
- 按天气分色绘制轨迹（红/绿/蓝/黄）
- 跳跃检测（>50m 断开）+ 发光效果
- 同时输出 PNG + PDF + SVG

### 覆盖率分析（10m 网格）

| 地图 | 独立网格 | 最大重复 | 覆盖率 |
|------|----------|----------|--------|
| Town04 | 394 | 60 | 32.8% |
| Town05 | 357 | 47 | 29.8% |
| Town06 | 355 | 16 | 29.6% |

Town06 重新采集后最大重复从高值降至16次，分布更均匀。

---

## 阶段 5: LUNA-Net 重训练

使用新采集的 CARLA_Unified_Dataset 重训练 LUNA-Net。

- 训练数据：2400帧 (Town04 + Town06, 4种天气)
- 验证数据：1200帧 (Town05, 4种天气)
- 输出权重：`LUNA-Net_carla/best_net_LUNA.pth`

---

## 阶段 6: 新权重验证

### diag_luna.py 验证

```
模型加载: ✓
  - 权重: LUNA-Net_carla/best_net_LUNA.pth
  - 参数量: 73,784,707
  - 4 missing keys (Swin norm buffer, 不影响推理)
  - 配置: LLEM=True, IAF=True, NAA=True, Edge=True, 2-class
```

### 完整管线验证 (manual_painting_control_v4.py)

多次运行验证，最长一次达 6550 帧，覆盖全部功能：

```
运行 1 (2100帧): 三模态切换 + 天气切换基础验证
运行 2 (2150帧): + 录像功能 (R键) + 观察者切换 (V键)
运行 3 (6550帧): 长时间稳定性测试 + 虚线模式 (D键)

LUNA-Net 模式 (主要测试):
  - 边缘点: 稳定 13-14 pts/帧
  - 喷嘴距离: 2.5-3.2m (目标 3.0m)
  - 直道: PAINTING 状态长时间稳定 (F3000-F4300 连续1300帧)
  - 弯道: 短暂 CONVERGING 后恢复 (曲率前馈生效)
  - ClearNight + ClearDay 均正常

VLLiNet 模式:
  - 边缘点: 稳定 13 pts/帧
  - 喷嘴距离: 2.5-3.1m
  - 直道 PAINTING 连续稳定 (F1000-F2050 超过1000帧)

GT 模式: 正常 (12-13 pts, Noz 2.9-3.0m)
G键循环: GT → VLLiNet → LUNA → GT 切换正常
N键天气: ClearNight ↔ ClearDay 切换正常
D键虚线: DASHED (3m/3m) 模式正常
R键录像: AVI 录制正常
```

### 已知问题

- G键反复切换感知模式（重新加载 LUNA-Net 73.8M 参数）会导致 segfault (exit 139)
  - 原因：GPU 内存碎片 + CARLA 共享显存
  - 发生时机：第二次或第三次切换回 LUNA-Net 时
  - 数据无损失：segfault 发生在运行时，不影响已保存的评估数据
  - 解决方案：可改为启动时预加载三个模型常驻显存（待实现）

### 代码路径更新

三个文件的权重路径从 `LUNA-Net_carla/weights/best_net_LUNA_ClearNight.pth` 更新为 `LUNA-Net_carla/best_net_LUNA.pth`：
- `perception/road_segmentor_luna.py`
- `diag_luna.py`
- `CLAUDE.md`

---

## 总结

V5+ 完成了从数据采集到模型重训练再到权重验证的完整闭环：

1. 采集脚本匹配 V5 测试环境（1248×384, x=2.5, z=3.5, pitch=-15）
2. 解决 CARLA 0.9.15-dirty 非标准语义标签问题（Road=1, RoadLine=24）
3. 3600帧统一数据集（3地图 × 4天气, KITTI格式 + SNE预计算）
4. LUNA-Net 重训练权重在完整管线中验证通过
5. 轨迹可视化支持矢量输出（PDF/SVG）
