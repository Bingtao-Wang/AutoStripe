"""Perception accuracy metrics: AI mask vs GT mask comparison.

Two pure functions for comparing VLLiNet AI output against GT segmentation:
- compute_mask_iou: binary mask IoU
- compute_edge_deviation: per-row right-edge pixel deviation
"""

import numpy as np


def compute_mask_iou(ai_mask, gt_mask):
    """Compute IoU between AI road mask and GT road mask.

    Args:
        ai_mask: (H, W) uint8, 255=road, 0=other. May be None.
        gt_mask: (H, W) uint8, 255=road, 0=other. May be None.

    Returns:
        float: IoU in [0, 1]. Returns 0.0 if either mask is None.
    """
    if ai_mask is None or gt_mask is None:
        return 0.0

    ai_bool = ai_mask > 0
    gt_bool = gt_mask > 0

    intersection = np.count_nonzero(ai_bool & gt_bool)
    union = np.count_nonzero(ai_bool | gt_bool)

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_edge_deviation(ai_right_px, gt_right_px):
    """Compute per-row pixel deviation between AI and GT right road edges.

    Matches rows present in both edge lists and computes |u_ai - u_gt|.

    Args:
        ai_right_px: list of (u, v) tuples from AI edge extraction.
        gt_right_px: list of (u, v) tuples from GT edge extraction.

    Returns:
        dict with keys: mean_px, median_px, max_px, num_matched_rows.
        Returns None if fewer than 3 rows match.
    """
    if not ai_right_px or not gt_right_px:
        return None

    # Build row -> u mapping for GT
    gt_by_row = {}
    for u, v in gt_right_px:
        gt_by_row[v] = u

    # Match AI rows against GT
    deviations = []
    for u_ai, v in ai_right_px:
        if v in gt_by_row:
            deviations.append(abs(u_ai - gt_by_row[v]))

    if len(deviations) < 3:
        return None

    devs = np.array(deviations, dtype=np.float64)
    return {
        'mean_px': float(np.mean(devs)),
        'median_px': float(np.median(devs)),
        'max_px': float(np.max(devs)),
        'num_matched_rows': len(deviations),
    }
