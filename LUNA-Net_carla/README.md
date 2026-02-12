# LUNA-Net CARLA 部署指南

## 概述

将 LUNA-Net（Low-light Urban Navigation and Analysis Network）部署到 CARLA 模拟器，实现夜间可行驶区域的实时感知。

模型基于 ClearNight 场景训练，性能指标：F1=97.18%, IoU=94.52%。

---

## 文件结构

```
deploy_carla/
├── carla_luna_deploy.py        # 主部署脚本
├── config.py                   # 配置文件
├── requirements.txt            # Python 依赖
├── weights/
│   └── best_net_LUNA_ClearNight.pth   # ClearNight 权重 (282MB, 软链接)
├── models_luna/                # LUNA-Net 模型架构
│   ├── luna_net.py             #   主网络
│   ├── low_light_enhance.py    #   LLEM 低光照增强
│   ├── robust_sne.py           #   R-SNE 鲁棒法线估计
│   ├── illumination_adaptive_fusion.py  # IAF 光照自适应融合
│   ├── night_aware_decoder.py  #   NAA 夜间感知解码器
│   └── adaptive_losses.py      #   自适应损失（推理时不使用）
├── modelsV2/                   # 骨干网络
│   ├── swin_backbone.py        #   Dual Swin Transformer
│   └── decoder.py              #   DSConv 解码器
└── models/
    └── sne_model.py            # 原始 SNE 法线计算
```

---

## 环境要求

- CARLA 0.9.13+（需先启动 CARLA 服务器）
- Python 3.8+
- CUDA GPU（推荐）

安装依赖：

```bash
pip install -r requirements.txt
```

> 注意：`carla` Python 包需从 CARLA 安装目录获取，不在 requirements.txt 中。
> CARLA 0.9.13+ 通常可直接 `pip install carla`。

---

## 快速开始

### 1. 启动 CARLA 服务器

```bash
# 在 CARLA 安装目录下
./CarlaUE4.sh
# 或指定端口
./CarlaUE4.sh -carla-rpc-port=2000
```

### 2. 运行部署脚本

```bash
cd deploy_carla
python carla_luna_deploy.py
```

默认连接 `localhost:2000`，使用 ClearNight 天气。

### 3. 常用参数

```bash
# 指定天气
python carla_luna_deploy.py --weather ClearNight

# 保存推理帧到目录
python carla_luna_deploy.py --save-dir ./output_frames

# 无头模式（不显示窗口）
python carla_luna_deploy.py --no-display --save-dir ./output_frames

# 指定 CARLA 地址和 GPU
python carla_luna_deploy.py --host 192.168.1.100 --port 2000 --gpu 0

# 使用自定义权重路径
python carla_luna_deploy.py --weights weights/best_net_LUNA_ClearNight.pth
```

### 4. 操作

- 窗口中按 `q` 退出
- `Ctrl+C` 终止程序
- 程序会自动清理 CARLA 中生成的车辆和传感器

---

## 推理流程

```
CARLA Simulator
    │
    ├── RGB Camera ──────────────┐
    │                            ▼
    │                     [Preprocess]
    │                     resize to 1248×384
    │                     normalize to [0,1]
    │                            │
    │                            ▼
    └── Depth Camera ──► [SNE] ──► Surface Normal
                                     │
                                     ▼
                              ┌─────────────┐
                              │  LUNA-Net    │
                              │  ┌─────┐    │
                              │  │LLEM │    │
                              │  └──┬──┘    │
                              │     ▼       │
                              │  Dual Swin  │
                              │     ▼       │
                              │  ┌─────┐    │
                              │  │ IAF │    │
                              │  └──┬──┘    │
                              │     ▼       │
                              │  ┌─────┐    │
                              │  │ NAA │    │
                              │  └──┬──┘    │
                              └────┼────────┘
                                   ▼
                            Drivable Area Mask
                                   ▼
                            Overlay on RGB
                                   ▼
                            OpenCV Display
```

---

## 配置说明

编辑 `config.py` 可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CAMERA_WIDTH` | 1600 | CARLA 相机分辨率宽 |
| `CAMERA_HEIGHT` | 900 | CARLA 相机分辨率高 |
| `CAMERA_FOV` | 90.0 | 相机视场角 |
| `CAMERA_X/Y/Z` | 1.5/0.0/2.4 | 相机安装位置（相对车辆） |
| `INPUT_WIDTH` | 1248 | 模型输入宽（与训练一致） |
| `INPUT_HEIGHT` | 384 | 模型输入高（与训练一致） |
| `OVERLAY_ALPHA` | 0.4 | 可视化叠加透明度 |
| `ROAD_COLOR` | (255,0,255) | 可行驶区域颜色（品红） |

---

## 天气预设

| 预设 | 说明 | 适用权重 |
|------|------|----------|
| `ClearNight` | 晴朗夜间 | best_net_LUNA_ClearNight.pth |
| `ClearDay` | 晴朗白天 | 需另外部署 ClearDay 权重 |
| `HeavyFoggyNight` | 大雾夜间 | 需另外部署对应权重 |
| `HeavyRainFoggyNight` | 暴雨雾夜 | 需另外部署对应权重 |

当前部署包仅包含 ClearNight 权重。如需其他天气条件，将对应的 `best_net_LUNA.pth` 复制到 `weights/` 目录并修改 `config.py` 中的 `MODEL_WEIGHTS` 路径。

---

## 注意事项

1. SNE 法线计算在 CPU 上运行（与训练时一致），是推理流程中的主要瓶颈
2. 模型输入分辨率固定为 1248×384，与 CARLA 相机分辨率无关（内部会 resize）
3. 权重文件为软链接，如需独立部署请替换为实际文件拷贝
4. 首次运行时 `timm` 可能需要下载 Swin Transformer 配置（非权重），确保网络可用或提前缓存
