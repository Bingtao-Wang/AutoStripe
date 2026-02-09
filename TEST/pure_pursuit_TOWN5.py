# -*- coding: utf-8 -*-
#2025.3.13
#这个是让小车用pure_pursuit跟踪CARLA内部行车路径，然后画出轨迹，同时新增一个附着在车辆上方（z=20）的摄像头显示俯视视角
#TOWN05 --高速公路路段

import glob
import os
import sys
import cv2
import numpy as np
import math
import time

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

L = 2.875
Kdd = 4.0
alpha_prev = 0
delta_prev = 0

client = carla.Client('localhost', 2000)
client.set_timeout(200)

world = client.load_world("Town05")
world.set_weather(carla.WeatherParameters.ClearNoon)

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.filter('vehicle.*stl*')[0]

transform = carla.Transform()
transform.location.x = 10
transform.location.y = -210
transform.location.z = 1.85
transform.rotation.yaw = 180
vehicle = world.spawn_actor(vehicle_bp, transform)

spectator = world.get_spectator()

# 更新默认视角（跟随车辆）
def update_spectator():
    veh_transform = vehicle.get_transform()
    sp_transform = carla.Transform(
        veh_transform.location + carla.Location(z=8, x=13, y=0),
        carla.Rotation(yaw=veh_transform.rotation.yaw, pitch=-25)
    )
    spectator.set_transform(sp_transform)


control = carla.VehicleControl()
control.throttle = 0.3
vehicle.apply_control(control)

# 获取地图和初始规划路径点（仅用于计算Pure Pursuit目标点）
map = world.get_map()
wp = map.get_waypoint(vehicle.get_location(), project_to_road=True,
                      lane_type=carla.LaneType.Driving)

waypoint_list = []      # 保存 (x,y) 坐标
waypoint_obj_list = []  # 保存完整 waypoint 对象

# 生成规划路径，用于Pure Pursuit目标点计算
noOfWp = 200
t = 0
while t < noOfWp:
    wp_next = wp.next(1.0)
    if len(wp_next) > 1:
        wp = wp_next[1]
    else:
        wp = wp_next[0]
    waypoint_obj_list.append(wp)
    waypoint_list.append((wp.transform.location.x, wp.transform.location.y))
    t += 1

# 创建俯视摄像头（RGB Sensor），附着在车辆上，固定在车辆正上方z=25处
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "1800")
camera_bp.set_attribute("image_size_y", "1600")
camera_bp.set_attribute("fov", "90")

# 这里使用相对变换，设置摄像头在车辆正上方25米处，俯视（pitch=-90）
relative_transform = carla.Transform(carla.Location(x=0, y=0, z=25),
                                     carla.Rotation(pitch=-90, yaw=0, roll=0))
overhead_camera = world.spawn_actor(camera_bp, relative_transform, attach_to=vehicle)

# 创建独立OpenCV窗口用于显示俯视视角
cv2.namedWindow("Overhead View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Overhead View", 800, 600)

# 定义摄像头回调函数，将图像通过OpenCV显示
def process_img(image):
    print("Received image from camera.")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # 舍弃alpha通道
    cv2.imshow("Overhead View", array)
    cv2.waitKey(1)  # Ensure OpenCV window updates

# 开始监听摄像头数据
overhead_camera.listen(lambda image: process_img(image))

# 调试绘制函数：
# "line"：绘制车辆经过的轨迹（白色连续线）；
# "target"：绘制当前Pure Pursuit目标点（红色离散标记）。
def draw(loc1, loc2=None, type=None):
    if type == "target":
        world.debug.draw_string(
            loc1, "o",
            color=carla.Color(255, 0, 0),
            life_time=0.3, persistent_lines=False
        )
    elif type == "line" and loc2 is not None:
        world.debug.draw_line(
            loc1,
            loc2,
            thickness=0.3,
            color=carla.Color(255, 255, 0),
            life_time=1000,
            persistent_lines=True
        )
    elif type == "string":
        world.debug.draw_string(loc1, "X", life_time=2000, persistent_lines=True)

def display(disp=False):
    if disp:
        print("--" * 20)
        print("Min Index= ", min_index)
        print("Forward Vel= %.3f m/s" % vf)
        print("Lookahead Dist= %.2f m" % ld)
        print("Alpha= %.5f rad" % alpha)
        print("Delta= %.5f rad" % steer_angle)
        print("Error= %.3f m" % e)

def calc_steering_angle(alpha, ld):
    delta = math.atan2(2 * L * np.sin(alpha), ld)
    return np.clip(delta, -1.0, 1.0)

def get_target_wp_index(veh_location, waypoint_list):
    dxl, dyl = [], []
    for i in range(len(waypoint_list)):
        dxl.append(abs(veh_location.x - waypoint_list[i][0]))
        dyl.append(abs(veh_location.y - waypoint_list[i][1]))
    dist = np.hypot(dxl, dyl)
    idx = np.argmin(dist) + 4
    if idx < len(waypoint_list):
        tx, ty = waypoint_list[idx]
    else:
        tx, ty = waypoint_list[-1]
    return idx, tx, ty, dist

def get_lookahead_dist(vf, idx, waypoint_list, dist):
    return Kdd * vf


def get_right_side_position(veh_location, yaw, offset=2.0):
    """
    根据车辆当前位置和朝向，计算车辆右侧2米的点。
    """
    # 将车辆朝向从角度转换为弧度
    yaw_rad = math.radians(yaw)

    # 计算车辆右侧2米的位置（旋转90度）
    dx = offset * math.cos(yaw_rad + math.pi / 2)  # 右侧的x偏移
    dy = offset * math.sin(yaw_rad + math.pi / 2)  # 右侧的y偏移

    # 计算右侧点的位置
    right_side_location = carla.Location(veh_location.x + dx, veh_location.y + dy, veh_location.z)

    return right_side_location


# 记录车辆经过的轨迹点
past_positions = []

# Game Loop - 无限循环执行Pure Pursuit任务
while True:
    veh_transform = vehicle.get_transform()
    veh_location = vehicle.get_location()
    veh_vel = vehicle.get_velocity()
    vf = np.sqrt(veh_vel.x ** 2 + veh_vel.y ** 2)
    vf = np.clip(vf, 0.1, 2.5)

    # 记录车辆经过的轨迹
    if past_positions:
        last_pos = past_positions[-1]
        # 修改为在右侧2米处绘制轨迹
        right_side_pos = get_right_side_position(last_pos, veh_transform.rotation.yaw, offset=2.0)
        right_side_veh_loc = get_right_side_position(veh_location, veh_transform.rotation.yaw, offset=2.0)

        # 绘制车辆右侧的轨迹
        draw(right_side_pos, right_side_veh_loc, type="line")

    past_positions.append(veh_location)

    # 计算Pure Pursuit的目标点
    min_index, tx, ty, dist = get_target_wp_index(veh_location, waypoint_list)
    ld = get_lookahead_dist(vf, min_index, waypoint_list, dist)
    yaw = math.radians(veh_transform.rotation.yaw)
    alpha = math.atan2(ty - veh_location.y, tx - veh_location.x) - yaw
    if math.isnan(alpha):
        alpha = alpha_prev
    else:
        alpha_prev = alpha

    e = np.sin(alpha) * ld
    steer_angle = calc_steering_angle(alpha, ld)
    control.steer = steer_angle
    vehicle.apply_control(control)

    update_spectator()
    # 绘制当前Pure Pursuit目标点（红色离散点）
    draw(waypoint_obj_list[min_index].transform.location, type="target")
    display(disp=True)

    time.sleep(0.005)

print("Task Done!")
vehicle.destroy()

