#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VLLiNet Model Diagnostic Script — Step 0

Standalone script to verify VLLiNet checkpoint loading and inference.
Does NOT depend on the AutoStripe pipeline.

Usage:
  1. Start CARLA: ./CarlaUE4.sh
  2. Run: python diag_vllinet.py

Verifies:
  - VLLiNet_Lite model loads from checkpoint
  - Depth encoder input channel detection
  - Real-time RGB + Depth inference from CARLA
  - Road mask overlay visualization
  - FPS and GPU memory usage
"""

import glob
import os
import sys
import time
import threading

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import deque

# CARLA egg path
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

# VLLiNet model path
VLLINET_DIR = os.path.join(os.path.dirname(__file__), 'VLLiNet_models')
sys.path.insert(0, VLLINET_DIR)
from models.vllinet import VLLiNet_Lite
from models.backbone import LiDAREncoder

# --- Constants ---
CHECKPOINT_PATH = os.path.join(VLLINET_DIR, 'checkpoints_carla', 'best_model.pth')
IMG_W, IMG_H = 1248, 384
CAM_FOV = 90
CAM_X, CAM_Z, CAM_PITCH = 2.5, 3.5, -10
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint_path, device):
    """Load VLLiNet_Lite from checkpoint, auto-detecting depth channels."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect depth encoder input channels from checkpoint
    state_dict = checkpoint['model_state_dict']
    depth_key = 'lidar_encoder.stage1.0.weight'
    if depth_key in state_dict:
        depth_in_ch = state_dict[depth_key].shape[1]
        print(f"  Depth encoder input channels (from checkpoint): {depth_in_ch}")
    else:
        depth_in_ch = 3
        print(f"  Depth encoder key not found, defaulting to {depth_in_ch} channels")

    # Build model with correct depth channels
    model = VLLiNet_Lite(pretrained=False, use_deep_supervision=True)
    if depth_in_ch != 3:
        model.lidar_encoder = LiDAREncoder(in_channels=depth_in_ch)

    # Fix key naming: checkpoint 'fusion.' -> code 'fusion_module.'
    fixed_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace('fusion.fusion_modules',
                          'fusion_module.fusion_modules')
        fixed_sd[new_k] = v
    model.load_state_dict(fixed_sd)
    model = model.to(device).eval()

    epoch = checkpoint.get('epoch', '?')
    maxf = checkpoint.get('val_maxf', 0.0)
    print(f"  Epoch: {epoch}, MaxF: {maxf:.4f}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, depth_in_ch


def preprocess_rgb(bgra, device):
    """CARLA BGRA -> normalized RGB tensor [1, 3, H, W]."""
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def preprocess_depth(bgra, depth_channels, device):
    """CARLA depth BGRA -> normalized depth tensor [1, C, H, W]."""
    r = bgra[:, :, 2].astype(np.float32)
    g = bgra[:, :, 1].astype(np.float32)
    b = bgra[:, :, 0].astype(np.float32)
    depth_m = (r + g * 256.0 + b * 65536.0) / (256.0**3 - 1) * 1000.0

    depth_m = cv2.resize(depth_m, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    # Min-max normalize to [0, 1]
    d_min, d_max = depth_m.min(), depth_m.max()
    if d_max > d_min:
        depth_norm = (depth_m - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_m)

    # Expand to required channels
    depth_stack = np.stack([depth_norm] * depth_channels, axis=0)  # [C, H, W]
    tensor = torch.from_numpy(depth_stack).unsqueeze(0)  # [1, C, H, W]
    return tensor.to(device)


@torch.no_grad()
def run_inference(model, rgb_tensor, depth_tensor, device):
    """Run model inference, return binary mask at original size."""
    if device.type == 'cuda':
        with torch.amp.autocast('cuda'):
            output = model(rgb_tensor, depth_tensor, return_aux=False)
    else:
        output = model(rgb_tensor, depth_tensor, return_aux=False)
    pred = torch.sigmoid(output)
    pred = F.interpolate(pred, size=(IMG_H, IMG_W),
                         mode='bilinear', align_corners=False)
    mask = (pred > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    return mask


def main():
    print("=" * 60)
    print("  VLLiNet Diagnostic Script")
    print("=" * 60)

    # 1. Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\nWARNING: No GPU detected, running on CPU (will be slow)")

    # 2. Load model
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return
    model, depth_ch = load_model(CHECKPOINT_PATH, device)

    # 3. Connect to CARLA and spawn cameras
    print("\nConnecting to CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(20)
    world = client.load_world('Town05')

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.*stl*')[0]
    spawn_tf = carla.Transform(
        carla.Location(x=10, y=-210, z=1.85),
        carla.Rotation(yaw=180),
    )
    vehicle = world.spawn_actor(vehicle_bp, spawn_tf)
    actors = [vehicle]

    cam_tf = carla.Transform(
        carla.Location(x=CAM_X, y=0, z=CAM_Z),
        carla.Rotation(pitch=CAM_PITCH),
    )

    # RGB camera
    rgb_bp = bp_lib.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(IMG_W))
    rgb_bp.set_attribute('image_size_y', str(IMG_H))
    rgb_bp.set_attribute('fov', str(CAM_FOV))
    rgb_cam = world.spawn_actor(rgb_bp, cam_tf, attach_to=vehicle)
    actors.append(rgb_cam)

    rgb_lock = threading.Lock()
    rgb_buf = {'image': None}

    def _rgb_cb(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        with rgb_lock:
            rgb_buf['image'] = arr.copy()

    rgb_cam.listen(_rgb_cb)

    # Depth camera
    depth_bp = bp_lib.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(IMG_W))
    depth_bp.set_attribute('image_size_y', str(IMG_H))
    depth_bp.set_attribute('fov', str(CAM_FOV))
    depth_cam = world.spawn_actor(depth_bp, cam_tf, attach_to=vehicle)
    actors.append(depth_cam)

    depth_lock = threading.Lock()
    depth_buf = {'image': None}

    def _depth_cb(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        with depth_lock:
            depth_buf['image'] = arr.copy()

    depth_cam.listen(_depth_cb)

    # 4. Warm up sensors
    print("Warming up sensors (30 frames)...")
    for _ in range(30):
        time.sleep(0.05)

    # 5. Inference loop
    print("\nStarting inference loop. Press 'q' to quit.\n")
    cv2.namedWindow("VLLiNet Diag", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VLLiNet Diag", IMG_W, IMG_H)

    fps_history = deque(maxlen=30)

    try:
        while True:
            with rgb_lock:
                rgb_data = rgb_buf['image']
            with depth_lock:
                depth_data = depth_buf['image']

            if rgb_data is None or depth_data is None:
                time.sleep(0.01)
                continue

            t0 = time.time()

            rgb_tensor = preprocess_rgb(rgb_data, device)
            depth_tensor = preprocess_depth(depth_data, depth_ch, device)
            mask = run_inference(model, rgb_tensor, depth_tensor, device)

            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)

            # Visualize: green overlay on RGB
            display = cv2.cvtColor(rgb_data[:, :, :3], cv2.COLOR_BGR2RGB)
            display = cv2.resize(display, (IMG_W, IMG_H))
            overlay = display.copy()
            overlay[mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

            # HUD
            cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if device.type == 'cuda':
                mem_used = torch.cuda.memory_allocated() / 1e6
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e6
                cv2.putText(vis, f"GPU Mem: {mem_used:.0f}/{mem_total:.0f} MB",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

            road_pct = mask.sum() / mask.size * 100
            cv2.putText(vis, f"Road: {road_pct:.1f}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("VLLiNet Diag", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        for a in actors:
            if a is not None:
                a.destroy()
        print("Done.")


if __name__ == '__main__':
    main()
