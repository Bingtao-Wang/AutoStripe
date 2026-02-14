#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LUNA-Net Model Diagnostic Script

Standalone script to verify LUNA-Net checkpoint loading, SNE computation,
and inference. Does NOT depend on the AutoStripe pipeline.

Usage:
  1. Start CARLA: ./CarlaUE4.sh
  2. Run: python diag_luna.py

Verifies:
  - LUNA-Net model loads from checkpoint
  - SNE surface normal computation from depth
  - Real-time RGB + Depth + SNE -> LUNA-Net inference
  - Road mask overlay visualization
  - Timing breakdown (SNE vs inference)
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

# LUNA-Net model path
LUNA_DIR = os.path.join(os.path.dirname(__file__), 'LUNA-Net_carla')
sys.path.insert(0, LUNA_DIR)

from models.sne_model import SNE
from models_luna.luna_net import LUNANet

# --- Constants ---
CHECKPOINT_PATH = os.path.join(
    LUNA_DIR, 'best_net_LUNA.pth')
IMG_W, IMG_H = 1248, 384
CAM_FOV = 90
CAM_X, CAM_Z, CAM_PITCH = 1.5, 2.4, -15


def build_cam_intrinsics(img_w, img_h, fov_deg):
    """Compute 3x4 camera intrinsic matrix."""
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = img_w / (2.0 * np.tan(fov_rad / 2.0))
    cx, cy = img_w / 2.0, img_h / 2.0
    return np.array([
        [fx,  0, cx, 0],
        [ 0, fy, cy, 0],
        [ 0,  0,  1, 0],
    ], dtype=np.float32)


def remap_luna_checkpoint(state_dict):
    """Remap checkpoint keys to match timm 0.6.x Swin layer naming.

    The checkpoint was trained with a custom Swin that uses:
      layers_0.blocks, layers_1.downsample, layers_1.blocks, ...
    But timm 0.6.x uses:
      layers.0.blocks, layers.0.downsample, layers.1.blocks, ...

    Downsample shifts: checkpoint layers_N.downsample -> timm layers.(N-1).downsample
    """
    import re
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        # Remap downsample: layers_N.downsample -> layers.(N-1).downsample
        m = re.match(r'(.*\.swin\.)layers_(\d+)\.downsample\.(.*)', k)
        if m:
            prefix, stage, suffix = m.group(1), int(m.group(2)), m.group(3)
            new_k = f'{prefix}layers.{stage - 1}.downsample.{suffix}'
        else:
            # Remap blocks: layers_N.blocks -> layers.N.blocks
            m = re.match(r'(.*\.swin\.)layers_(\d+)\.(.*)', k)
            if m:
                prefix, stage, suffix = m.group(1), int(m.group(2)), m.group(3)
                new_k = f'{prefix}layers.{stage}.{suffix}'
        new_sd[new_k] = v
    return new_sd


def load_model(checkpoint_path, device):
    """Load LUNA-Net from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Remap checkpoint keys to match timm Swin naming
    state_dict = remap_luna_checkpoint(state_dict)

    model = LUNANet(
        swin_model='swin_tiny_patch4_window7_224',
        pretrained=False,
        num_classes=2,
        use_llem=True,
        use_robust_sne=False,
        use_iaf=True,
        use_naa_decoder=True,
        use_edge_head=True,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out expected missing buffers (attn_mask, relative_position_index, norm)
        real_missing = [k for k in missing
                        if 'attn_mask' not in k and 'relative_position_index' not in k]
        if real_missing:
            print(f"  WARNING: {len(real_missing)} missing keys (non-buffer):")
            for k in real_missing[:10]:
                print(f"    {k}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys:")
        for k in unexpected[:10]:
            print(f"    {k}")
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {device}")
    return model


def decode_depth(bgra):
    """CARLA depth BGRA -> meters."""
    r = bgra[:, :, 2].astype(np.float32)
    g = bgra[:, :, 1].astype(np.float32)
    b = bgra[:, :, 0].astype(np.float32)
    return (r + g * 256.0 + b * 65536.0) / (256.0**3 - 1) * 1000.0


def compute_sne_normal(depth_m, cam_param, sne_model):
    """Depth -> surface normal via SNE (CPU)."""
    depth_t = torch.from_numpy(depth_m).float()
    cam_t = torch.from_numpy(cam_param).float()
    with torch.no_grad():
        normal = sne_model(depth_t, cam_t)
    return normal.numpy()


@torch.no_grad()
def run_inference(model, rgb_bgra, normal, device):
    """Full preprocessing + LUNA-Net inference. Returns uint8 mask."""
    # RGB: BGRA -> RGB, resize, [0,1]
    rgb = cv2.cvtColor(rgb_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_W, IMG_H))
    rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    rgb_t = rgb_t.unsqueeze(0).to(device)

    # Normal: resize, to tensor
    normal_hwc = np.transpose(normal, (1, 2, 0))
    normal_resized = cv2.resize(normal_hwc, (IMG_W, IMG_H))
    normal_t = torch.from_numpy(normal_resized).float().permute(2, 0, 1)
    normal_t = normal_t.unsqueeze(0).to(device)

    # Inference
    if device.type == 'cuda':
        with torch.amp.autocast('cuda'):
            output, aux = model(rgb_t, normal_t, is_normal=True)
    else:
        output, aux = model(rgb_t, normal_t, is_normal=True)

    # Crop padding (model outputs at padded size, valid region is IMG_H x IMG_W)
    output = output[:, :, :IMG_H, :IMG_W]

    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred, aux


def main():
    print("=" * 60)
    print("  LUNA-Net Diagnostic Script")
    print("=" * 60)

    # 1. Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\nWARNING: No GPU detected, running on CPU (will be slow)")

    # 2. Load model + SNE
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return
    model = load_model(CHECKPOINT_PATH, device)
    sne_model = SNE()
    cam_param = build_cam_intrinsics(IMG_W, IMG_H, CAM_FOV)
    print(f"  SNE initialized (CPU)")
    print(f"  Camera intrinsics: fx={cam_param[0,0]:.0f}, "
          f"cx={cam_param[0,2]:.0f}, cy={cam_param[1,2]:.0f}")

    # 3. Connect to CARLA and spawn cameras
    print("\nConnecting to CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(20)
    world = client.load_world('Town05')

    # Set ClearNight weather (LUNA-Net's strength)
    weather = world.get_weather()
    weather.sun_altitude_angle = -30.0
    weather.cloudiness = 10.0
    weather.fog_density = 0.0
    world.set_weather(weather)
    print("  Weather: ClearNight")

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
    cv2.namedWindow("LUNA-Net Diag", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LUNA-Net Diag", IMG_W, IMG_H)

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

            # 1. Decode depth
            depth_m = decode_depth(depth_data)
            depth_resized = cv2.resize(depth_m, (IMG_W, IMG_H))

            # 2. SNE: depth -> surface normal
            t_sne = time.time()
            normal = compute_sne_normal(depth_resized, cam_param, sne_model)
            sne_ms = (time.time() - t_sne) * 1000.0

            # 3. Inference
            t_inf = time.time()
            mask, aux = run_inference(model, rgb_data, normal, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inf_ms = (time.time() - t_inf) * 1000.0

            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)

            # 4. Visualize: green overlay on RGB
            display = cv2.cvtColor(rgb_data[:, :, :3], cv2.COLOR_BGR2RGB)
            display = cv2.resize(display, (IMG_W, IMG_H))
            overlay = display.copy()
            # Resize mask to display size
            mask_resized = cv2.resize(mask, (IMG_W, IMG_H),
                                      interpolation=cv2.INTER_NEAREST)
            overlay[mask_resized > 0] = [0, 255, 0]
            vis = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

            # HUD
            cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"SNE: {sne_ms:.0f}ms  Inf: {inf_ms:.0f}ms",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            if device.type == 'cuda':
                mem_used = torch.cuda.memory_allocated() / 1e6
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e6
                cv2.putText(vis, f"GPU Mem: {mem_used:.0f}/{mem_total:.0f} MB",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

            road_pct = mask_resized.sum() / mask_resized.size * 100
            cv2.putText(vis, f"Road: {road_pct:.1f}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Brightness from LLEM
            brightness = aux.get("brightness", None)
            if brightness is not None:
                cv2.putText(vis, f"Brightness: {brightness.item():.3f}",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

            cv2.imshow("LUNA-Net Diag", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
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
