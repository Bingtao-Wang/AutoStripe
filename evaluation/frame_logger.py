"""Per-frame CSV logger for paper-grade data recording.

Records 32 columns of per-frame telemetry when eval_recording is active.
Output: evaluation/framelog_YYYYMMDD_HHMMSS.csv

Usage:
    logger = FrameLogger()
    logger.start()
    for each frame:
        logger.log_frame({...})  # dict with fields
    logger.stop()
"""

import csv
import os
import time


COLUMNS = [
    'timestamp', 'frame', 'dt', 'fps',
    'veh_x', 'veh_y', 'veh_yaw', 'speed',
    'nozzle_x', 'nozzle_y', 'nozzle_edge_dist', 'poly_edge_dist',
    'driving_offset', 'steer_filter', 'steer_cmd', 'throttle_cmd', 'brake_cmd',
    'lateral_error', 'paint_state', 'painting_enabled', 'dash_phase',
    'perception_mode', 'ai_edge_pts', 'gt_edge_pts', 'road_mask_ratio',
    'poly_coeff_a', 'poly_coeff_b', 'poly_coeff_c',
    'inference_time_ms', 'sne_time_ms',
    'mask_iou', 'edge_dev_mean_px', 'edge_dev_median_px', 'edge_dev_max_px',
    'gt_nozzle_edge_dist',
]


class FrameLogger:
    """Per-frame CSV recorder for evaluation sessions."""

    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        self._output_dir = output_dir
        self._file = None
        self._writer = None
        self._path = None
        self._active = False

    @property
    def active(self):
        return self._active

    def start(self):
        """Open a new CSV file and write the header."""
        if self._active:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(
            self._output_dir, f"framelog_{timestamp}.csv")
        self._file = open(self._path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        self._active = True
        print(f"  FrameLogger: recording to {self._path}")

    def log_frame(self, data):
        """Write one row. data is a dict keyed by COLUMNS names.

        Missing keys default to empty string.
        """
        if not self._active or self._writer is None:
            return
        row = []
        for col in COLUMNS:
            val = data.get(col, '')
            if isinstance(val, float):
                row.append(f"{val:.6f}")
            else:
                row.append(val)
        self._writer.writerow(row)

    def stop(self):
        """Flush and close the CSV file."""
        if not self._active:
            return
        self._active = False
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            print(f"  FrameLogger: saved {self._path}")
