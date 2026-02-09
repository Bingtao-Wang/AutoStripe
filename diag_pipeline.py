"""Detailed diagnostic: analyze tag spatial distribution per row."""
import glob, os, sys, time
import cv2, numpy as np

try:
    sys.path.append(glob.glob(
        '../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major, sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla_env.setup_scene_v2 import setup_scene_v2

scene = setup_scene_v2()
world = scene['world']

for _ in range(40):
    world.tick()
    time.sleep(0.05)

with scene['_semantic_lock']:
    sem = scene['_semantic_data']['image']

tags = sem[:, :, 2]
h, w = tags.shape

# Save raw tag image (scaled for visibility)
cv2.imwrite('/tmp/diag_tags_raw.png', tags * 10)

# Analyze boundary tags per row (right side focus)
TAGS_BOUNDARY = {1, 2, 5, 8, 9, 11, 22, 27}

print(f"Image: {w}x{h}")
print(f"\nPer-row analysis (right half, every 20 rows):")
print(f"{'row':>4} | {'rightmost_bnd':>13} | {'bnd_tag':>7} | {'tag_at_799':>10} | {'tag_at_700':>10} | {'tag_at_600':>10}")
print("-" * 75)

for v in range(180, h, 20):
    # Find rightmost boundary tag in this row
    row_tags = tags[v, :]
    rightmost_bnd = -1
    bnd_tag = -1
    for u in range(w - 1, w // 2, -1):
        if row_tags[u] in TAGS_BOUNDARY:
            rightmost_bnd = u
            bnd_tag = row_tags[u]
            break

    t799 = row_tags[799] if w > 799 else -1
    t700 = row_tags[700] if w > 700 else -1
    t600 = row_tags[600] if w > 600 else -1

    print(f"{v:4d} | {rightmost_bnd:13d} | {bnd_tag:7d} | {t799:10d} | {t700:10d} | {t600:10d}")

# Also check: what tags exist in the right quarter of the image?
print(f"\nTag distribution in RIGHT QUARTER (cols 600-799):")
right_quarter = tags[:, 600:]
for t in np.unique(right_quarter):
    cnt = np.count_nonzero(right_quarter == t)
    pct = cnt / right_quarter.size * 100
    if pct > 0.1:
        print(f"  tag {t:3d}: {cnt:7d} px ({pct:.1f}%)")

from carla_env.setup_scene import cleanup
cleanup(scene['actors'])
print("Done.")
