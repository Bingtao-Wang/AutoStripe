"""诊断脚本：实时显示语义tag彩色图，每个tag不同颜色+标注"""
import glob, os, sys, time
import cv2
import numpy as np

try:
    sys.path.append(glob.glob(
        '../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major, sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla_env.setup_scene_v2 import setup_scene_v2

# 每个tag对应的颜色和名称
TAG_INFO = {
    0:  ((128, 128, 128), "Unlabeled"),
    1:  ((70,  70,  70),  "Building/Road?"),
    2:  ((100, 40,  40),  "Fence"),
    3:  ((55,  90,  80),  "Other"),
    4:  ((220, 20,  60),  "Pedestrian"),
    5:  ((153, 153, 153), "Pole"),
    6:  ((50,  234, 157), "RoadLines"),
    7:  ((128, 64,  128), "Road(std)"),
    8:  ((232, 35,  244), "Sidewalk"),
    9:  ((35,  142, 107), "Vegetation"),
    10: ((142, 0,   0),   "Vehicle"),
    11: ((156, 102, 102), "Wall"),
    12: ((0,   220, 220), "TrafficSign"),
    13: ((180, 130, 70),  "Sky"),
    14: ((81,  0,   81),  "Ground"),
    15: ((100, 150, 150), "Bridge"),
    22: ((100, 170, 145), "Terrain"),
    24: ((140, 150, 230), "RailTrack"),
    27: ((180, 165, 180), "GuardRail"),
    28: ((0,   220, 220), "TrafficSign2"),
}

print("连接 CARLA...")
scene = setup_scene_v2()
world = scene["world"]

print("预热传感器...")
for _ in range(40):
    world.tick()
    time.sleep(0.05)

# 抓取一帧语义数据
with scene["_semantic_lock"]:
    sem = scene["_semantic_data"]["image"]

if sem is None:
    print("ERROR: 未收到语义数据")
else:
    tags = sem[:, :, 2]  # R通道 = tag ID
    h, w = tags.shape

    # 打印tag分布
    print(f"\n图像: {w}x{h}")
    print(f"\n{'='*55}")
    print(f"  Tag分布")
    print(f"{'='*55}")
    unique, counts = np.unique(tags, return_counts=True)
    for tid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        pct = cnt / tags.size * 100
        name = TAG_INFO.get(tid, ((255,255,255), f"Unknown"))[1]
        print(f"  Tag {tid:2d} ({name:15s}): {cnt:7d} px ({pct:5.1f}%)")

    # 生成彩色图
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for tid in unique:
        color, name = TAG_INFO.get(tid, ((255, 255, 255), "Unknown"))
        color_map[tags == tid] = color

    # 在图上标注图例
    legend_y = 30
    for tid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        color, name = TAG_INFO.get(tid, ((255, 255, 255), "Unknown"))
        pct = cnt / tags.size * 100
        # 画色块
        cv2.rectangle(color_map, (w - 250, legend_y - 12),
                       (w - 230, legend_y + 4), color, -1)
        # 写文字
        label = f"T{tid}: {name} ({pct:.1f}%)"
        cv2.putText(color_map, label, (w - 225, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 20

    # 显示
    cv2.imshow("Semantic Tags (每种颜色=一个tag)", color_map)
    cv2.imwrite("/tmp/sem_colorized.png", color_map)
    print("\n已保存: /tmp/sem_colorized.png")
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 清理
from carla_env.setup_scene import cleanup
cleanup(scene["actors"])
print("完成。")
