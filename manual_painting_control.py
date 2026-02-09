#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manual Painting Control - 手动控制喷涂模式

基于 V2 视觉感知系统，增加按键控制喷涂开关功能。

按键说明：
  SPACE - 切换喷涂模式 ON/OFF
  ESC   - 退出程序
  Q     - 退出程序

使用方法：
  1. 启动 CARLA: ./CarlaUE4.sh
  2. 运行本脚本: python manual_painting_control.py
  3. 按空格键控制喷涂开关
"""

import glob
import os
import sys
import time
import math
import threading
import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_TAB, K_q
from pygame.locals import K_w, K_a, K_s, K_d, K_x, K_v
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT

try:
    sys.path.append(glob.glob(
        '../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')
    )[0])
except IndexError:
    pass

import carla
import numpy as np
import cv2

# 导入 V2 模块
from carla_env.setup_scene_v2 import setup_scene_v2
from carla_env.setup_scene import update_spectator
from perception.perception_pipeline import PerceptionPipeline
from planning.vision_path_planner import VisionPathPlanner
from control.marker_vehicle_v2 import MarkerVehicleV2


def get_nozzle_position(vehicle, offset=2.0):
    """计算喷嘴位置：车辆实际位置 + 右侧偏移（V1逻辑）"""
    veh_tf = vehicle.get_transform()
    veh_loc = vehicle.get_location()
    yaw_rad = math.radians(veh_tf.rotation.yaw)

    # 右侧偏移：yaw + 90度
    dx = offset * math.cos(yaw_rad + math.pi / 2)
    dy = offset * math.sin(yaw_rad + math.pi / 2)

    return carla.Location(
        x=veh_loc.x + dx,
        y=veh_loc.y + dy,
        z=veh_loc.z  # 与车辆同高度
    )


class ManualPaintingControl:
    """手动喷涂控制器 + 手动驾驶控制"""

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.painting_enabled = False  # 喷涂模式开关
        self.paint_trail = []          # 喷涂轨迹
        self.last_nozzle_loc = None    # 上一帧喷嘴位置

        # 驾驶模式
        self.auto_drive = True         # True=自动驾驶, False=手动驾驶

        # 手动控制状态
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.reverse = False  # 倒车模式

    def toggle_painting(self):
        """切换喷涂模式"""
        self.painting_enabled = not self.painting_enabled
        status = "ON" if self.painting_enabled else "OFF"
        print(f"\n{'='*50}")
        print(f"  喷涂模式: {status}")
        print(f"{'='*50}\n")
        return self.painting_enabled

    def paint_line(self, world, nozzle_loc):
        """执行喷涂（仅在 painting_enabled=True 时）"""
        if not self.painting_enabled:
            # 停止喷涂时重置喷嘴位置，避免恢复时连接到旧位置
            self.last_nozzle_loc = None
            return

        if self.last_nozzle_loc is not None:
            # 画线：从上一帧喷嘴位置到当前喷嘴位置
            world.debug.draw_line(
                self.last_nozzle_loc,
                nozzle_loc,
                thickness=0.3,
                color=carla.Color(255, 255, 0),  # 黄色
                life_time=1000.0,
                persistent_lines=True
            )
            # 记录轨迹（连续点）
            self.paint_trail.append((nozzle_loc.x, nozzle_loc.y))
        else:
            # 新段的第一个点（恢复喷涂后的起点）
            # 插入 None 作为段间隔标记
            if len(self.paint_trail) > 0:
                self.paint_trail.append(None)
            self.paint_trail.append((nozzle_loc.x, nozzle_loc.y))

        self.last_nozzle_loc = nozzle_loc

    def toggle_drive_mode(self):
        """切换驾驶模式（自动/手动）"""
        self.auto_drive = not self.auto_drive
        mode = "AUTO" if self.auto_drive else "MANUAL"
        print(f"\n{'='*50}")
        print(f"  驾驶模式: {mode}")
        print(f"{'='*50}\n")
        return self.auto_drive

    def toggle_reverse(self):
        """切换倒车模式"""
        self.reverse = not self.reverse
        status = "ON" if self.reverse else "OFF"
        print(f"\n{'='*50}")
        print(f"  倒车模式: {status}")
        print(f"{'='*50}\n")

    def update_manual_control(self, keys):
        """根据pygame按键状态更新手动控制（持续检测按键是否被按住）"""
        # W/↑ - 油门（按住时持续增加）
        if keys[K_w] or keys[K_UP]:
            self.throttle = min(1.0, self.throttle + 0.1)
        else:
            self.throttle = 0.0  # 松开时归零

        # S/↓ - 刹车（按住时持续增加）
        if keys[K_s] or keys[K_DOWN]:
            self.brake = min(1.0, self.brake + 0.2)
        else:
            self.brake = 0.0  # 松开时归零

        # A/← - 左转（按住时持续增加）
        if keys[K_a] or keys[K_LEFT]:
            if self.steer > 0:
                self.steer = 0
            else:
                self.steer = max(-0.7, self.steer - 0.05)
        # D/→ - 右转（按住时持续增加）
        elif keys[K_d] or keys[K_RIGHT]:
            if self.steer < 0:
                self.steer = 0
            else:
                self.steer = min(0.7, self.steer + 0.05)
        else:
            self.steer = 0.0  # 松开时归零

        # X - 手刹
        if keys[K_x]:
            self.brake = 1.0
            self.throttle = 0.0

    def apply_manual_control(self):
        """应用手动控制到车辆"""
        control = carla.VehicleControl()
        control.throttle = self.throttle
        control.steer = self.steer
        control.brake = self.brake
        control.reverse = self.reverse
        self.vehicle.apply_control(control)


def draw_status_overlay(img, painting_enabled, frame_count, speed, edge_dist_r,
                        drive_mode="AUTO", throttle=0.0, steer=0.0, brake=0.0,
                        tp_edge_dist=999.0):
    """在图像上绘制状态信息"""
    h, w = img.shape[:2]

    # 驾驶模式（最上方，大字体）
    mode_text = f"MODE: {drive_mode}"
    mode_color = (0, 255, 255) if drive_mode == "AUTO" else (255, 0, 255)  # 青色/紫色
    cv2.putText(img, mode_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, mode_color, 3)

    # 喷涂状态
    status_text = "PAINT: ON" if painting_enabled else "PAINT: OFF"
    status_color = (0, 255, 0) if painting_enabled else (0, 0, 255)
    cv2.putText(img, status_text, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

    # 车辆状态
    cv2.putText(img, f"Speed: {speed:.1f} m/s", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Nozzle-Edge: {edge_dist_r:.1f}m", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    tp_dist_text = f"TP-Edge: {tp_edge_dist:.1f}m" if tp_edge_dist < 900 else "TP-Edge: N/A"
    cv2.putText(img, tp_dist_text, (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # 手动控制状态（仅在手动模式下显示）
    if drive_mode == "MANUAL":
        cv2.putText(img, f"Throttle: {throttle:.2f}", (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"Steer: {steer:.2f}", (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"Brake: {brake:.2f}", (20, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 按键提示（底部）
    help_y = h - 150
    cv2.putText(img, "Controls:", (20, help_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(img, "TAB - Toggle Auto/Manual", (20, help_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "SPACE - Toggle Painting", (20, help_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "WASD/Arrows - Manual Drive", (20, help_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "X - Handbrake", (20, help_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "ESC/Q - Quit", (20, help_y + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return img


def draw_driving_path(world, driving_coords, vehicle_tf=None):
    """绘制蓝色驾驶路径点（根据车辆pitch推算前方路面z，适配坡道）"""
    if len(driving_coords) < 1:
        return
    if vehicle_tf is None:
        return

    veh_loc = vehicle_tf.location
    fwd = vehicle_tf.get_forward_vector()
    fwd_h = math.sqrt(fwd.x**2 + fwd.y**2)
    slope = fwd.z / fwd_h if fwd_h > 1e-6 else 0.0

    # 每隔2个点画一个蓝色大点，减少开销
    for i in range(0, len(driving_coords), 2):
        dx = driving_coords[i][0] - veh_loc.x
        dy = driving_coords[i][1] - veh_loc.y
        lon = (dx * fwd.x / fwd_h + dy * fwd.y / fwd_h) if fwd_h > 1e-6 else 0.0
        z = veh_loc.z + lon * slope + 0.5

        pt = carla.Location(x=driving_coords[i][0],
                           y=driving_coords[i][1], z=z)
        world.debug.draw_point(pt, size=0.1,
                              color=carla.Color(0, 0, 255),
                              life_time=0.1)


def compute_point_edge_distance(ref_loc, right_world, vehicle_tf, max_lon=15.0):
    """计算参考点到右路沿的垂直距离（沿车身横向）。

    以 ref_loc 为中心，按纵向距离排序取最近的点，
    用中位数横向投影构造垂直交点（弯道抗偏移）。
    返回：(距离, 路沿交点位置)
    """
    if not right_world:
        return 999.0, None

    yaw = math.radians(vehicle_tf.rotation.yaw)
    fwd_x = math.cos(yaw)
    fwd_y = math.sin(yaw)
    right_x = -fwd_y
    right_y = fwd_x

    # 收集所有右侧路沿点的 (纵向距离, 横向距离)
    candidates = []
    for loc in right_world:
        dx = loc.x - ref_loc.x
        dy = loc.y - ref_loc.y
        lon = dx * fwd_x + dy * fwd_y
        lat = dx * right_x + dy * right_y
        if abs(lon) < max_lon and lat > 0:
            candidates.append((abs(lon), lat))

    if not candidates:
        return 999.0, None

    # 按纵向距离排序，取最近的 N 个点（受弯道影响最小）
    candidates.sort(key=lambda c: c[0])
    top_n = min(10, len(candidates))
    nearest_lats = [c[1] for c in candidates[:top_n]]

    # 中位数横向距离
    nearest_lats.sort()
    median_lat = nearest_lats[len(nearest_lats) // 2]

    edge_point = carla.Location(
        x=ref_loc.x + median_lat * right_x,
        y=ref_loc.y + median_lat * right_y,
        z=ref_loc.z
    )

    dist = math.sqrt((edge_point.x - ref_loc.x)**2 +
                     (edge_point.y - ref_loc.y)**2)
    return dist, edge_point


def world_to_pixel(wx, wy, veh_tf, img_w=1800, img_h=1600, cam_h=25.0):
    """将世界坐标投影到俯视相机像素坐标（V1逻辑）"""
    vx = veh_tf.location.x
    vy = veh_tf.location.y
    yaw = math.radians(veh_tf.rotation.yaw)

    dx = wx - vx
    dy = wy - vy

    # 旋转到车辆本地坐标系
    local_fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_right = -dx * math.sin(yaw) + dy * math.cos(yaw)

    # FOV=90 => 在25m高度，可视半幅 = 25m
    scale_x = (img_w / 2.0) / cam_h
    scale_y = (img_h / 2.0) / cam_h

    px = int(img_w / 2.0 + local_right * scale_x)
    py = int(img_h / 2.0 - local_fwd * scale_y)
    return px, py


def world_to_front_pixel(wx, wy, wz, cam_tf, img_w=800, img_h=600, fov=90):
    """将世界坐标投影到前视摄像头像素坐标。
    返回 (px, py) 或 None（点在摄像头后方时）。
    """
    f = (img_w / 2.0) / math.tan(math.radians(fov / 2.0))  # 焦距=400

    # 世界→摄像头局部坐标
    dx = wx - cam_tf.location.x
    dy = wy - cam_tf.location.y
    dz = wz - cam_tf.location.z

    yaw = math.radians(cam_tf.rotation.yaw)
    pitch = math.radians(cam_tf.rotation.pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # 先绕 yaw 旋转
    x1 = dx * cy + dy * sy
    y1 = -dx * sy + dy * cy
    z1 = dz

    # 再绕 pitch 旋转（CARLA pitch: 负=向下看）
    fwd = x1 * cp + z1 * sp
    up = -x1 * sp + z1 * cp
    right = y1

    if fwd < 0.5:  # 在摄像头后方
        return None

    px = int(img_w / 2.0 + f * right / fwd)
    py = int(img_h / 2.0 - f * up / fwd)
    return px, py


def main():
    print("="*60)
    print("  AutoStripe - Manual Painting Control")
    print("  手动喷涂控制模式")
    print("="*60)
    print("\n按键说明：")
    print("  SPACE - 切换喷涂模式 ON/OFF")
    print("  ESC/Q - 退出程序")
    print("\n启动中...\n")

    # 初始化pygame（用于键盘输入 + 前视显示）
    pygame.init()
    pg_screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("AutoStripe - Front View + Control")
    pg_font = pygame.font.SysFont("monospace", 18)
    pg_clock = pygame.time.Clock()

    actors = []

    try:
        # 1. 搭建场景
        scene = setup_scene_v2()
        actors = scene['actors']
        world = scene['world']
        vehicle = scene['vehicle']

        # 同步模式可消除视角抖动，但会将帧率拉低到 ~10fps，暂时关闭
        # settings = world.get_settings()
        # settings.synchronous_mode = True
        # settings.fixed_delta_seconds = 0.05  # 20 FPS physics
        # world.apply_settings(settings)

        # 2. 初始化模块
        perception = PerceptionPipeline(
            img_w=800, img_h=600, fov_deg=90.0
        )
        planner = VisionPathPlanner(
            line_offset=3.0, nozzle_arm=2.0, smooth_window=5
        )
        controller = MarkerVehicleV2(vehicle, wheelbase=2.875, kdd=3.0)

        # 3. 初始化喷涂控制器（传入 vehicle）
        paint_ctrl = ManualPaintingControl(vehicle)

        # 4. 预热传感器
        print("预热传感器 (30 帧)...")
        for _ in range(30):
            time.sleep(0.05)
        print("传感器就绪。\n")

        # 5. 主循环
        frame_count = 0
        spectator_follow = True  # V键切换：True=跟随车辆, False=自由视角

        print("="*60)
        print("  系统就绪！按 SPACE 键开始喷涂")
        print("="*60)

        while True:
            frame_count += 1

            # 获取传感器数据（从共享缓冲区字典中提取图像）
            with scene['_semantic_lock']:
                sem_data = scene['_semantic_data']['image']
                cs_data = scene['_semantic_data'].get('cityscapes')
            with scene['_depth_lock']:
                depth_data = scene['_depth_data']['image']
            with scene['_frame_lock']:
                overhead_data = scene['_frame_data']['image']

            if sem_data is None or depth_data is None:
                time.sleep(0.05)
                continue

            # 感知：提取路沿
            cam_tf = scene['semantic_cam'].get_transform()
            left_world, right_world, road_mask, left_px, right_px = perception.process_frame(
                sem_data, depth_data, cam_tf, cityscapes_bgra=cs_data
            )

            # 规划：生成驾驶路径
            veh_tf = vehicle.get_transform()
            driving_coords, _ = planner.update(right_world, veh_tf)  # 不使用预规划的nozzle_locs

            # 控制：根据驾驶模式选择控制方式
            if paint_ctrl.auto_drive:
                # 自动驾驶：使用 Pure Pursuit 控制器
                controller.update_path(driving_coords)
                controller.step()
            else:
                # 手动驾驶：应用手动控制
                paint_ctrl.apply_manual_control()

            # 喷涂：根据喷涂模式决定是否画线（使用车辆实际位置，V1逻辑）
            nozzle_loc = get_nozzle_position(vehicle)  # 从车辆实际位置计算喷嘴位置
            paint_ctrl.paint_line(world, nozzle_loc)

            # 可视化：绘制驾驶路径（根据车辆pitch推算坡道z）
            draw_driving_path(world, driving_coords, vehicle.get_transform())

            # 更新spectator跟随视角（仅在跟随模式下）
            if spectator_follow:
                update_spectator(scene['spectator'], vehicle)

            # 计算状态信息
            veh_vel = vehicle.get_velocity()
            speed = math.sqrt(veh_vel.x**2 + veh_vel.y**2)

            # 可视化：第一个追踪点到路沿的距离线（青色）+ 距离标注
            tp_edge_dist = 999.0
            tp_mid = None
            if len(driving_coords) > 0:
                veh_tf_now = vehicle.get_transform()
                fwd = veh_tf_now.get_forward_vector()
                fwd_h = math.sqrt(fwd.x**2 + fwd.y**2)
                slope = fwd.z / fwd_h if fwd_h > 1e-6 else 0.0

                tp = driving_coords[0]
                dx_tp = tp[0] - veh_tf_now.location.x
                dy_tp = tp[1] - veh_tf_now.location.y
                lon_tp = (dx_tp * fwd.x / fwd_h + dy_tp * fwd.y / fwd_h) if fwd_h > 1e-6 else 0.0
                tp_z = veh_tf_now.location.z + lon_tp * slope + 0.5

                tp_loc = carla.Location(x=tp[0], y=tp[1], z=tp_z)
                tp_edge_dist, tp_edge_pt = compute_point_edge_distance(
                    tp_loc, right_world, veh_tf_now)

                if tp_edge_pt is not None:
                    tp_edge_pt.z = tp_z  # 与追踪点同高度
                    world.debug.draw_line(
                        tp_loc, tp_edge_pt,
                        thickness=0.08,
                        color=carla.Color(0, 255, 255),
                        life_time=0.1)
                    # 在线段中点标注距离
                    tp_mid = carla.Location(
                        x=(tp_loc.x + tp_edge_pt.x) / 2,
                        y=(tp_loc.y + tp_edge_pt.y) / 2,
                        z=tp_z + 0.3)
                    world.debug.draw_string(
                        tp_mid, f"{tp_edge_dist:.1f}m",
                        color=carla.Color(0, 255, 255),
                        life_time=0.1)

            # 喷嘴边距：实际计算喷嘴到路沿的垂直距离
            nozzle_raised = carla.Location(
                x=nozzle_loc.x, y=nozzle_loc.y, z=nozzle_loc.z + 0.5)
            edge_dist_r, nozzle_edge_pt = compute_point_edge_distance(
                nozzle_raised, right_world, vehicle.get_transform())

            # 可视化：喷嘴到路沿的连接线（绿色）
            nozzle_mid = None
            if nozzle_edge_pt is not None:
                world.debug.draw_line(
                    nozzle_raised, nozzle_edge_pt,
                    thickness=0.08,
                    color=carla.Color(0, 255, 0),
                    life_time=0.1)
                nozzle_mid = carla.Location(
                    x=(nozzle_raised.x + nozzle_edge_pt.x) / 2,
                    y=(nozzle_raised.y + nozzle_edge_pt.y) / 2,
                    z=nozzle_raised.z + 0.3)
                world.debug.draw_string(
                    nozzle_mid, f"{edge_dist_r:.1f}m",
                    color=carla.Color(0, 255, 0),
                    life_time=0.1)

            # 显示俯视图（Overhead View）+ 黄线轨迹叠加
            if overhead_data is not None:
                img = overhead_data.copy()
                veh_tf = vehicle.get_transform()

                # 叠加黄色喷涂轨迹（debug.draw_line在相机中不可见，需手动投影）
                # trail 中 None 表示段间隔，跳过不画线
                trail = paint_ctrl.paint_trail
                if len(trail) >= 2:
                    for i in range(1, len(trail)):
                        if trail[i - 1] is None or trail[i] is None:
                            continue
                        x0, y0 = trail[i - 1]
                        x1, y1 = trail[i]
                        px0, py0 = world_to_pixel(x0, y0, veh_tf)
                        px1, py1 = world_to_pixel(x1, y1, veh_tf)
                        cv2.line(img, (px0, py0), (px1, py1), (0, 255, 255), 3)

                # 在俯视图线条上叠加距离文字
                if nozzle_mid is not None:
                    npx, npy = world_to_pixel(nozzle_mid.x, nozzle_mid.y, veh_tf)
                    cv2.putText(img, f"{edge_dist_r:.1f}m", (npx, npy),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                if tp_mid is not None:
                    tpx, tpy = world_to_pixel(tp_mid.x, tp_mid.y, veh_tf)
                    cv2.putText(img, f"{tp_edge_dist:.1f}m", (tpx, tpy),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                drive_mode = "AUTO" if paint_ctrl.auto_drive else "MANUAL"
                img = draw_status_overlay(
                    img, paint_ctrl.painting_enabled,
                    frame_count, speed, edge_dist_r,
                    drive_mode, paint_ctrl.throttle, paint_ctrl.steer, paint_ctrl.brake,
                    tp_edge_dist
                )
                cv2.imshow("Overhead View", img)

            # 显示前视感知图（在pygame窗口中渲染，合并控制信息）
            with scene['_rgb_front_lock']:
                rgb_front = scene['_rgb_front_data']['image']
            if rgb_front is not None:
                front_display = rgb_front[:, :, :3].copy()
                # 如果有road_mask，叠加绿色半透明
                if road_mask is not None:
                    mask_overlay = np.zeros_like(front_display)
                    mask_overlay[road_mask > 0] = [0, 255, 0]
                    front_display = cv2.addWeighted(front_display, 0.7, mask_overlay, 0.3, 0)
                # 在前视图线条上叠加距离文字
                rgb_cam_tf = scene['rgb_front_cam'].get_transform()
                if nozzle_mid is not None:
                    fp = world_to_front_pixel(
                        nozzle_mid.x, nozzle_mid.y, nozzle_mid.z, rgb_cam_tf)
                    if fp is not None:
                        fpx, fpy = fp
                        if 0 <= fpx < 800 and 0 <= fpy < 600:
                            cv2.putText(front_display, f"{edge_dist_r:.1f}m",
                                        (fpx, fpy), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 255, 0), 3)
                if tp_mid is not None:
                    fp2 = world_to_front_pixel(
                        tp_mid.x, tp_mid.y, tp_mid.z, rgb_cam_tf)
                    if fp2 is not None:
                        fpx2, fpy2 = fp2
                        if 0 <= fpx2 < 800 and 0 <= fpy2 < 600:
                            cv2.putText(front_display, f"{tp_edge_dist:.1f}m",
                                        (fpx2, fpy2), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 255, 255), 3)
                # OpenCV BGR -> RGB，转为pygame surface
                front_rgb = cv2.cvtColor(front_display, cv2.COLOR_BGR2RGB)
                pg_surface = pygame.surfarray.make_surface(front_rgb.swapaxes(0, 1))
                pg_screen.blit(pg_surface, (0, 0))

            # 按键处理（使用pygame检测按键状态）
            # 处理pygame事件（窗口关闭、特殊按键）
            should_exit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_exit = True
                    elif event.key == K_SPACE:
                        paint_ctrl.toggle_painting()
                    elif event.key == K_TAB:
                        paint_ctrl.toggle_drive_mode()
                    elif event.key == K_q:
                        paint_ctrl.toggle_reverse()
                    elif event.key == K_v:
                        spectator_follow = not spectator_follow
                        mode = "FOLLOW" if spectator_follow else "FREE"
                        print(f"\n{'='*50}")
                        print(f"  编辑器视角: {mode}")
                        print(f"{'='*50}\n")

            if should_exit:
                print("\n退出程序...")
                break

            # 获取当前所有按键状态（用于持续按键检测）
            keys = pygame.key.get_pressed()

            # 在手动模式下，持续检测WASD按键状态
            if not paint_ctrl.auto_drive:
                paint_ctrl.update_manual_control(keys)

            # 更新OpenCV窗口显示
            cv2.waitKey(1)

            # 在pygame窗口上叠加状态信息（前视图已渲染在底层）
            lines = [
                ("Drive: AUTO" if paint_ctrl.auto_drive else "Drive: MANUAL",
                 (0, 255, 0) if paint_ctrl.auto_drive else (255, 255, 0)),
                ("Paint: ON" if paint_ctrl.painting_enabled else "Paint: OFF",
                 (0, 255, 0) if paint_ctrl.painting_enabled else (150, 150, 150)),
                ("Reverse: ON" if paint_ctrl.reverse else "",
                 (255, 100, 100)),
                (f"Speed: {speed:.1f} m/s", (255, 255, 255)),
                (f"Nozzle-Edge: {edge_dist_r:.1f}m", (0, 255, 0)),
                (f"TP-Edge: {tp_edge_dist:.1f}m" if tp_edge_dist < 900 else "TP-Edge: N/A",
                 (0, 255, 255)),
                (f"Throttle: {paint_ctrl.throttle:.1f}  Steer: {paint_ctrl.steer:.2f}  Brake: {paint_ctrl.brake:.1f}",
                 (255, 255, 255)),
                ("", (0, 0, 0)),
                ("Cam: FOLLOW" if spectator_follow else "Cam: FREE",
                 (0, 200, 255) if spectator_follow else (255, 150, 0)),
                ("TAB=Mode SPACE=Paint Q=Reverse", (200, 200, 200)),
                ("V=Camera WASD=Drive X=Brake ESC=Quit", (200, 200, 200)),
            ]
            for i, (text, color) in enumerate(lines):
                if text:
                    surf = pg_font.render(text, True, color)
                    pg_screen.blit(surf, (10, 8 + i * 24))
            pygame.display.flip()
            pg_clock.tick(30)

            # 每 50 帧打印一次状态
            if frame_count % 50 == 0:
                status = "ON" if paint_ctrl.painting_enabled else "OFF"
                tp_d = f"{tp_edge_dist:.1f}" if tp_edge_dist < 900 else "N/A"
                print(f"[F{frame_count}] 喷涂:{status}, 速度:{speed:.1f}m/s, "
                      f"喷嘴边距:{edge_dist_r:.1f}m, 追踪点边距:{tp_d}m, "
                      f"路径:{len(driving_coords)}点")

    except KeyboardInterrupt:
        print("\n用户中断...")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理
        print("\n清理资源...")
        pygame.quit()
        cv2.destroyAllWindows()
        for actor in actors:
            if actor is not None:
                actor.destroy()
        print("完成。")


if __name__ == '__main__':
    main()
