"""
LUNA-Net Real-time Drivable Area Perception on CARLA

Connects to CARLA simulator, captures RGB + Depth from vehicle cameras,
computes surface normals via SNE, runs LUNA-Net inference, and visualizes
the drivable area overlay in real-time.

Usage:
    1. Start CARLA server: ./CarlaUE4.sh
    2. Run: python carla_luna_deploy.py [--host localhost] [--port 2000]
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Add deploy_carla root to path
DEPLOY_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEPLOY_ROOT)

import carla
import config as cfg
from models.sne_model import SNE
from models_luna.luna_net import LUNANet


def parse_args():
    parser = argparse.ArgumentParser(description="LUNA-Net CARLA Deployment")
    parser.add_argument("--host", default=cfg.CARLA_HOST)
    parser.add_argument("--port", type=int, default=cfg.CARLA_PORT)
    parser.add_argument("--weights", default=cfg.MODEL_WEIGHTS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV window (headless mode)")
    parser.add_argument("--save-dir", default=None,
                        help="Save frames to directory")
    parser.add_argument("--weather", default="ClearNight",
                        choices=["ClearNight", "ClearDay",
                                 "HeavyFoggyNight", "HeavyRainFoggyNight"],
                        help="CARLA weather preset")
    return parser.parse_args()


# ============================================================
# Model Loading
# ============================================================

def load_model(weights_path, device):
    """Load LUNA-Net with ClearNight weights."""
    model = LUNANet(
        swin_model=cfg.SWIN_MODEL,
        pretrained=False,
        num_classes=cfg.NUM_CLASSES,
        use_llem=cfg.USE_LLEM,
        use_robust_sne=cfg.USE_ROBUST_SNE,
        use_iaf=cfg.USE_IAF,
        use_naa_decoder=cfg.USE_NAA_DECODER,
        use_edge_head=cfg.USE_EDGE_HEAD,
    )

    abs_path = os.path.join(DEPLOY_ROOT, weights_path)
    state_dict = torch.load(abs_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[Model] Loaded weights from {abs_path}")
    return model


# ============================================================
# Camera Intrinsics
# ============================================================

def build_cam_intrinsics(img_w, img_h, fov_deg):
    """Compute 3x4 camera intrinsic matrix from CARLA camera params."""
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = img_w / (2.0 * np.tan(fov_rad / 2.0))
    cx, cy = img_w / 2.0, img_h / 2.0
    return np.array([
        [fx,  0, cx, 0],
        [ 0, fy, cy, 0],
        [ 0,  0,  1, 0],
    ], dtype=np.float32)


# ============================================================
# Sensor Data Processing
# ============================================================

def carla_depth_to_meters(depth_image):
    """Convert CARLA depth image (BGRA encoding) to depth in meters.

    CARLA encodes depth as: depth = (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    """
    raw = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    raw = raw.reshape(depth_image.height, depth_image.width, 4)  # BGRA
    # Decode depth
    depth = (raw[:, :, 2].astype(np.float32)
             + raw[:, :, 1].astype(np.float32) * 256.0
             + raw[:, :, 0].astype(np.float32) * 256.0 * 256.0)
    depth = depth / (256.0 ** 3 - 1.0) * 1000.0  # meters
    return depth


def carla_image_to_rgb(image):
    """Convert CARLA image (BGRA) to RGB numpy array."""
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    raw = raw.reshape(image.height, image.width, 4)
    return raw[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB


def compute_sne_normal(depth_meters, cam_param, sne_model):
    """Compute surface normal from depth using the original SNE model.

    Args:
        depth_meters: (H, W) float32 depth in meters
        cam_param: (3, 4) camera intrinsic matrix
        sne_model: SNE() instance

    Returns:
        normal: (3, H, W) float32 surface normal
    """
    depth_t = torch.from_numpy(depth_meters).float()
    cam_t = torch.from_numpy(cam_param).float()
    with torch.no_grad():
        normal = sne_model(depth_t, cam_t)  # (3, H, W)
    return normal.numpy()


# ============================================================
# Inference Pipeline
# ============================================================

def preprocess(rgb, normal, target_size=(cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT)):
    """Preprocess RGB and normal for model input.

    Args:
        rgb: (H, W, 3) uint8 RGB image
        normal: (3, H, W) float32 surface normal

    Returns:
        rgb_tensor: (1, 3, tH, tW) float32 in [0, 1]
        normal_tensor: (1, 3, tH, tW) float32
    """
    tw, th = target_size

    # RGB: resize, normalize to [0,1], to tensor
    rgb_resized = cv2.resize(rgb, (tw, th))
    rgb_t = torch.from_numpy(rgb_resized).float().permute(2, 0, 1) / 255.0

    # Normal: resize each channel
    normal_hwc = np.transpose(normal, (1, 2, 0))  # (H,W,3)
    normal_resized = cv2.resize(normal_hwc, (tw, th))
    normal_t = torch.from_numpy(normal_resized).float().permute(2, 0, 1)

    return rgb_t.unsqueeze(0), normal_t.unsqueeze(0)


def run_inference(model, rgb_tensor, normal_tensor, device):
    """Run LUNA-Net inference.

    Returns:
        pred: (H, W) uint8 prediction mask (0=background, 1=road)
        aux: dict of auxiliary outputs
    """
    rgb_tensor = rgb_tensor.to(device)
    normal_tensor = normal_tensor.to(device)

    with torch.no_grad():
        output, aux = model(rgb_tensor, normal_tensor, is_normal=True)

    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred, aux


def make_overlay(rgb, pred, alpha=cfg.OVERLAY_ALPHA):
    """Create drivable area overlay on RGB image."""
    h, w = rgb.shape[:2]
    pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = rgb.copy()
    road_mask = pred_resized == 1
    color = np.array(cfg.ROAD_COLOR, dtype=np.uint8)
    overlay[road_mask] = (
        rgb[road_mask] * (1 - alpha) + color * alpha
    ).astype(np.uint8)
    return overlay


# ============================================================
# CARLA Weather Presets
# ============================================================

WEATHER_PRESETS = {
    "ClearNight": {
        "sun_altitude_angle": -30.0,
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "fog_density": 0.0,
    },
    "ClearDay": {
        "sun_altitude_angle": 60.0,
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "fog_density": 0.0,
    },
    "HeavyFoggyNight": {
        "sun_altitude_angle": -30.0,
        "cloudiness": 80.0,
        "precipitation": 0.0,
        "fog_density": 80.0,
        "fog_distance": 10.0,
    },
    "HeavyRainFoggyNight": {
        "sun_altitude_angle": -30.0,
        "cloudiness": 90.0,
        "precipitation": 80.0,
        "precipitation_deposits": 80.0,
        "fog_density": 50.0,
        "fog_distance": 20.0,
    },
}


def set_weather(world, preset_name):
    """Apply weather preset to CARLA world."""
    weather = world.get_weather()
    preset = WEATHER_PRESETS.get(preset_name, WEATHER_PRESETS["ClearNight"])
    for attr, val in preset.items():
        setattr(weather, attr, val)
    world.set_weather(weather)
    print(f"[Weather] Set to {preset_name}")


# ============================================================
# Main Deployment
# ============================================================

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model
    model = load_model(args.weights, device)

    # SNE model (CPU, same as training)
    sne_model = SNE()

    # Camera intrinsics
    cam_param = build_cam_intrinsics(cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT, cfg.CAMERA_FOV)

    # Save directory
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ---- Connect to CARLA ----
    client = carla.Client(args.host, args.port)
    client.set_timeout(cfg.CARLA_TIMEOUT)
    world = client.get_world()
    print(f"[CARLA] Connected to {args.host}:{args.port}")

    # Set weather
    set_weather(world, args.weather)

    # Enable synchronous mode
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS simulation
    world.apply_settings(settings)

    # Spawn vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    if vehicle is None:
        print("[Error] Failed to spawn vehicle, trying another spawn point...")
        for sp in spawn_points[1:10]:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                break
    if vehicle is None:
        raise RuntimeError("Cannot spawn vehicle")
    vehicle.set_autopilot(True)
    print(f"[CARLA] Vehicle spawned: {vehicle.type_id}")

    # ---- Attach sensors ----
    cam_transform = carla.Transform(
        carla.Location(x=cfg.CAMERA_X, y=cfg.CAMERA_Y, z=cfg.CAMERA_Z)
    )
    cam_attrs = {
        "image_size_x": str(cfg.CAMERA_WIDTH),
        "image_size_y": str(cfg.CAMERA_HEIGHT),
        "fov": str(cfg.CAMERA_FOV),
    }

    # RGB camera
    rgb_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in cam_attrs.items():
        rgb_bp.set_attribute(k, v)
    rgb_sensor = world.spawn_actor(rgb_bp, cam_transform, attach_to=vehicle)

    # Depth camera (same transform)
    depth_bp = bp_lib.find("sensor.camera.depth")
    for k, v in cam_attrs.items():
        depth_bp.set_attribute(k, v)
    depth_sensor = world.spawn_actor(depth_bp, cam_transform, attach_to=vehicle)

    print("[CARLA] RGB + Depth cameras attached")

    # Sensor data queues
    rgb_queue = []
    depth_queue = []
    rgb_sensor.listen(lambda img: rgb_queue.append(img))
    depth_sensor.listen(lambda img: depth_queue.append(img))

    # ---- Main Loop ----
    actors = [vehicle, rgb_sensor, depth_sensor]
    frame_idx = 0
    print("[Run] Starting inference loop... Press Ctrl+C to stop.\n")

    try:
        # Warm up: let a few frames pass
        for _ in range(5):
            world.tick()
        rgb_queue.clear()
        depth_queue.clear()

        while True:
            world.tick()

            # Wait for both sensors
            if len(rgb_queue) == 0 or len(depth_queue) == 0:
                continue

            rgb_raw = rgb_queue.pop(0)
            depth_raw = depth_queue.pop(0)

            t0 = time.time()

            # 1. Decode sensor data
            rgb_np = carla_image_to_rgb(rgb_raw)
            depth_m = carla_depth_to_meters(depth_raw)

            # 2. Compute SNE normal
            normal = compute_sne_normal(depth_m, cam_param, sne_model)

            # 3. Preprocess
            rgb_t, normal_t = preprocess(rgb_np, normal)

            # 4. Inference
            pred, aux = run_inference(model, rgb_t, normal_t, device)

            # 5. Overlay
            overlay = make_overlay(rgb_np, pred)

            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)

            # 6. HUD info
            brightness = aux.get("brightness", None)
            br_str = f"  Brightness: {brightness.item():.3f}" if brightness is not None else ""
            info = f"Frame {frame_idx:05d}  |  {fps:.1f} FPS  |  {dt*1000:.0f} ms{br_str}"

            # Draw info bar on overlay
            cv2.putText(overlay, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 7. Display
            if not args.no_display:
                display_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imshow(cfg.WINDOW_NAME, display_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[Run] 'q' pressed, stopping...")
                    break

            # 8. Save frame
            if args.save_dir:
                fname = os.path.join(args.save_dir, f"{frame_idx:05d}.png")
                cv2.imwrite(fname, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if frame_idx % 50 == 0:
                print(info)

            frame_idx += 1

            # Prevent queue buildup
            if len(rgb_queue) > 2:
                rgb_queue.clear()
            if len(depth_queue) > 2:
                depth_queue.clear()

    except KeyboardInterrupt:
        print("\n[Run] Interrupted by user.")
    finally:
        # ---- Cleanup ----
        print("[Cleanup] Destroying actors...")
        for actor in actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        world.apply_settings(original_settings)
        if not args.no_display:
            cv2.destroyAllWindows()
        print("[Done]")


if __name__ == "__main__":
    main()
