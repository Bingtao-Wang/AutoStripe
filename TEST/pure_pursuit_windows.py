import glob
import os
import sys
import pygame
import numpy as np
import carla
import math
import time
import weakref
import random
import logging
import argparse  # 添加 argparse 导入

# 初始化 pygame
pygame.init()

# 设置显示窗口的大小
window_width, window_height = 1280, 720
display = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("CARLA Simulation")

# 设置 pygame 字体
font = pygame.font.Font(None, 36)

L = 2.875
Kdd = 4.0
alpha_prev = 0
delta_prev = 0

# 连接 CARLA 客户端
client = carla.Client('localhost', 2000)
client.set_timeout(200)

# 加载世界
world = client.load_world("Town05")
world.set_weather(carla.WeatherParameters.ClearNoon)

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.filter('vehicle.*stl*')[0]

# 车辆控制
control = carla.VehicleControl()
control.throttle = 0.3


# CameraManager 类来管理摄像头
class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self._gamma = gamma_correction
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            (carla.Transform(carla.Location(x=0, y=0, z=25), carla.Rotation(pitch=-90)), Attachment.SpringArmGhost)
        ]
        self.transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', carla.ColorConverter.Raw, 'Camera RGB', {}]
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(window_width))
                bp.set_attribute('image_size_y', str(window_height))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(self._gamma))
            item.append(bp)
        self.index = None

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.camera_manager = None
        self.restart()

    def restart(self):
        # 获取地图中的所有可用 spawn 点
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            print("No spawn points available!")
            return

        # 随机选择一个 spawn 点来生成车辆
        spawn_point = random.choice(spawn_points)

        # 确保生成点没有碰撞
        self.player = self.world.try_spawn_actor(self.world.get_blueprint_library().filter('vehicle.*')[0], spawn_point)

        if self.player is None:
            print("Failed to spawn vehicle. Trying another spawn point.")
            # 如果失败，则可以尝试再次选择其他 spawn 点
            self.restart()  # 递归调用，可以避免碰撞
        else:
            print(f"Vehicle spawned at {spawn_point.location}")
            self.camera_manager = CameraManager(self.player, gamma_correction=2.2)
            self.camera_manager.set_sensor(0, notify=False)

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)


# 游戏主循环
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
