"""Road surface segmentation from CARLA semantic camera.

V2 uses CARLA's CityScapesPalette color conversion for reliable road detection,
bypassing non-standard tag IDs in CARLA 0.9.15-dirty.
Future versions can swap in LUNA-Net by subclassing and overriding segment().
"""

import numpy as np
import cv2


# CityScapes standard road color (BGR order, as stored in CARLA BGRA array)
#   RGB (128, 64, 128) → BGR (128, 64, 128)
ROAD_COLOR_BGR = np.array([128, 64, 128], dtype=np.uint8)

# Color matching tolerance (Euclidean distance in BGR space)
COLOR_TOLERANCE = 10

# Only keep mask in lower portion of image (closer to vehicle)
MASK_TOP_RATIO = 0.35


class RoadSegmentor:
    """Extracts a binary road mask from a CityScapes-colored semantic image.

    Uses CityScapes palette color matching instead of raw tag IDs.
    """

    def segment(self, cityscapes_bgra):
        """Convert CityScapes-colored BGRA image to binary road mask.

        Args:
            cityscapes_bgra: BGRA image after cc.CityScapesPalette conversion.

        Returns:
            np.ndarray (H, W) uint8 — 255 for road pixels, 0 otherwise.
        """
        bgr = cityscapes_bgra[:, :, :3]
        h, w = bgr.shape[:2]

        # Match road color with tolerance
        diff = np.abs(bgr.astype(np.int16) - ROAD_COLOR_BGR.astype(np.int16))
        dist = np.sqrt((diff ** 2).sum(axis=2))
        road_mask = np.where(dist <= COLOR_TOLERANCE, 255, 0).astype(np.uint8)

        # Ignore upper portion (distant objects)
        cutoff = int(h * MASK_TOP_RATIO)
        road_mask[:cutoff, :] = 0

        # Morphological close to fill small gaps, then open to remove noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_open)

        return road_mask
