"""V4.2 Trajectory Evaluator: compare paint trail against Map API ground truth.

Generates right road edge ground truth from CARLA Map API, then computes
lateral deviation, curvature variance, and coverage metrics.
"""

import csv
import math
import os
import time
import numpy as np

import glob
import sys

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

from planning.lane_planner import generate_center_waypoints, compute_road_edges


class TrajectoryEvaluator:
    """Evaluate paint trail quality against Map API ground truth."""

    # paint 点到最近 GT 点距离超过此值视为"超出 GT 覆盖范围"
    IN_RANGE_THRESHOLD = 5.0

    def __init__(self, carla_map, num_waypoints=400, spacing=1.0,
                 coverage_threshold=2.0):
        self.carla_map = carla_map
        self.num_waypoints = num_waypoints
        self.spacing = spacing
        self.coverage_threshold = coverage_threshold

        self._eval_count = 0
        self._output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)))

    def generate_ground_truth(self, start_location):
        """Generate right road edge GT from Map API.

        Args:
            start_location: carla.Location — start of the region to evaluate

        Returns:
            list of (x, y) tuples for right road edge
        """
        wp_objs, _ = generate_center_waypoints(
            self.carla_map, start_location,
            num_waypoints=self.num_waypoints, spacing=self.spacing)

        if not wp_objs:
            return []

        _, right_edges = compute_road_edges(wp_objs)
        return [(loc.x, loc.y) for loc in right_edges]

    def compute_metrics(self, paint_trail, gt_points):
        """Compute evaluation metrics.

        Args:
            paint_trail: list of (x,y) or None (gap markers)
            gt_points: list of (x,y) from ground truth

        Returns:
            dict with metric values, or None if insufficient data
        """
        # Filter out None gap markers
        paint_pts = [p for p in paint_trail if p is not None]

        if len(paint_pts) < 2 or len(gt_points) < 2:
            return None

        paint_np = np.array(paint_pts)
        gt_np = np.array(gt_points)

        # --- Lateral deviation: each paint point to nearest GT point ---
        deviations, nearest_indices = self._compute_deviations(paint_np, gt_np)

        # --- 过滤：只保留 GT 覆盖范围内的点 ---
        in_range_mask = deviations < self.IN_RANGE_THRESHOLD
        deviations_in = deviations[in_range_mask]
        paint_in = paint_np[in_range_mask]

        if len(deviations_in) < 2:
            return None

        # --- Curvature variance (只用范围内的点) ---
        curvature_var = self._compute_curvature_variance(paint_in)

        # --- Coverage: fraction of GT points within threshold of any paint point ---
        coverage = self._compute_coverage(paint_np, gt_np)

        return {
            'num_paint_points': len(paint_pts),
            'num_paint_in_range': int(np.sum(in_range_mask)),
            'num_gt_points': len(gt_points),
            'mean_deviation': float(np.mean(deviations_in)),
            'max_deviation': float(np.max(deviations_in)),
            'std_deviation': float(np.std(deviations_in)),
            'median_deviation': float(np.median(deviations_in)),
            'curvature_variance': float(curvature_var),
            'coverage': float(coverage),
            'coverage_threshold': self.coverage_threshold,
        }

    def _compute_deviations(self, paint_np, gt_np):
        """Compute min distance from each paint point to GT edge.

        Returns:
            (deviations, nearest_indices) — distance and index of nearest GT point
        """
        deviations = []
        nearest_indices = []
        for pt in paint_np:
            dists = np.sqrt(np.sum((gt_np - pt) ** 2, axis=1))
            idx = int(np.argmin(dists))
            deviations.append(float(dists[idx]))
            nearest_indices.append(idx)
        return np.array(deviations), np.array(nearest_indices)

    def _compute_local_curvatures(self, pts):
        """Compute discrete curvature at each point using Menger curvature.

        Returns array of length len(pts), with 0.0 at endpoints.
        """
        n = len(pts)
        curvatures = np.zeros(n)
        if n < 3:
            return curvatures

        for i in range(1, n - 1):
            p0 = pts[i - 1]
            p1 = pts[i]
            p2 = pts[i + 1]

            dx1 = p1[0] - p0[0]
            dy1 = p1[1] - p0[1]
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]

            cross = abs(dx1 * dy2 - dy1 * dx2)
            d1 = math.sqrt(dx1**2 + dy1**2)
            d2 = math.sqrt(dx2**2 + dy2**2)
            d3 = math.sqrt((p2[0] - p0[0])**2 + (p2[1] - p0[1])**2)

            denom = d1 * d2 * d3
            if denom > 1e-9:
                curvatures[i] = 2.0 * cross / denom

        return curvatures

    def _compute_curvature_variance(self, pts):
        """Compute variance of discrete curvature along paint trail."""
        curvatures = self._compute_local_curvatures(pts)
        interior = curvatures[1:-1] if len(curvatures) > 2 else curvatures
        nonzero = interior[interior > 0]
        if len(nonzero) == 0:
            return 0.0
        return float(np.var(nonzero))

    def _compute_along_track_dist(self, paint_pts):
        """Compute cumulative along-track distance for each paint point.

        Returns:
            np.ndarray of shape (N,) — cumulative distance from first point
        """
        n = len(paint_pts)
        dists = np.zeros(n)
        for i in range(1, n):
            dx = paint_pts[i][0] - paint_pts[i - 1][0]
            dy = paint_pts[i][1] - paint_pts[i - 1][1]
            dists[i] = dists[i - 1] + math.sqrt(dx**2 + dy**2)
        return dists

    def _compute_coverage(self, paint_np, gt_np):
        """Fraction of GT points covered by paint trail."""
        covered = 0
        for gt_pt in gt_np:
            dists = np.sqrt(np.sum((paint_np - gt_pt) ** 2, axis=1))
            if np.min(dists) < self.coverage_threshold:
                covered += 1
        return covered / len(gt_np)

    def save_csv(self, metrics, paint_trail, gt_points):
        """Save summary and detail CSVs to evaluation/ directory.

        Detail CSV has 8 columns:
            paint_x, paint_y, nearest_gt_dist,
            gt_nearest_x, gt_nearest_y,
            along_track_dist, local_curvature, in_range
        """
        self._eval_count += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"eval_{timestamp}_{self._eval_count}"

        # Summary CSV
        summary_path = os.path.join(self._output_dir, f"{prefix}_summary.csv")
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for k, v in metrics.items():
                writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])

        # Detail CSV (8 columns)
        detail_path = os.path.join(self._output_dir, f"{prefix}_detail.csv")
        paint_pts = [p for p in paint_trail if p is not None]
        paint_np = np.array(paint_pts)
        gt_np = np.array(gt_points)

        deviations, nearest_indices = self._compute_deviations(paint_np, gt_np)
        along_track = self._compute_along_track_dist(paint_np)
        curvatures = self._compute_local_curvatures(paint_np)

        with open(detail_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'paint_x', 'paint_y', 'nearest_gt_dist',
                'gt_nearest_x', 'gt_nearest_y',
                'along_track_dist', 'local_curvature', 'in_range'])
            for i, pt in enumerate(paint_pts):
                gt_idx = nearest_indices[i]
                gt_x, gt_y = gt_points[gt_idx]
                in_range = 1 if deviations[i] < self.IN_RANGE_THRESHOLD else 0
                writer.writerow([
                    f"{pt[0]:.3f}", f"{pt[1]:.3f}",
                    f"{deviations[i]:.4f}",
                    f"{gt_x:.3f}", f"{gt_y:.3f}",
                    f"{along_track[i]:.3f}",
                    f"{curvatures[i]:.6f}",
                    in_range])

        print(f"  CSV saved: {summary_path}")
        print(f"  CSV saved: {detail_path}")
        return summary_path, detail_path

    def print_summary(self, metrics):
        """Print evaluation results to console."""
        print(f"\n{'='*60}")
        print("  TRAJECTORY EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"  Paint points:      {metrics['num_paint_points']}"
              f"  (in range: {metrics['num_paint_in_range']})")
        print(f"  GT points:         {metrics['num_gt_points']}")
        print(f"  Mean deviation:    {metrics['mean_deviation']:.3f} m")
        print(f"  Max deviation:     {metrics['max_deviation']:.3f} m")
        print(f"  Std deviation:     {metrics['std_deviation']:.3f} m")
        print(f"  Median deviation:  {metrics['median_deviation']:.3f} m")
        print(f"  Curvature var:     {metrics['curvature_variance']:.6f}")
        print(f"  Coverage:          {metrics['coverage']*100:.1f}% "
              f"(threshold={metrics['coverage_threshold']:.1f}m)")
        print(f"{'='*60}\n")

    def run_evaluation(self, paint_trail, vehicle_location):
        """One-shot evaluation: generate GT, compute metrics, print + save.

        Generates GT from the first paint point (not current vehicle position)
        so that GT covers the same road segment as the paint trail.

        Called by E key in main loop.
        """
        import carla as _carla

        paint_pts = [p for p in paint_trail if p is not None]
        if len(paint_pts) < 5:
            print("\n  Not enough paint points for evaluation "
                  f"({len(paint_pts)} points, need >= 5)")
            return None

        # Use first paint point as GT start (paint trail is behind vehicle)
        first_pt = paint_pts[0]
        last_pt = paint_pts[-1]
        gt_start = _carla.Location(x=first_pt[0], y=first_pt[1], z=0.0)

        # 计算沿轨迹累积距离（非直线距离，避免闭环时首尾重合导致长度≈0）
        trail_length = 0.0
        for i in range(1, len(paint_pts)):
            dx = paint_pts[i][0] - paint_pts[i - 1][0]
            dy = paint_pts[i][1] - paint_pts[i - 1][1]
            trail_length += math.sqrt(dx * dx + dy * dy)
        # 多生成 20% 余量，确保覆盖
        needed_wps = max(self.num_waypoints,
                         int(trail_length / self.spacing * 1.2))

        print(f"\n  Generating ground truth from first paint point "
              f"({first_pt[0]:.1f}, {first_pt[1]:.1f})...")
        print(f"  Trail length ~{trail_length:.0f}m, "
              f"generating {needed_wps} GT waypoints...")

        old_num = self.num_waypoints
        self.num_waypoints = needed_wps
        gt_points = self.generate_ground_truth(gt_start)
        self.num_waypoints = old_num
        if len(gt_points) < 5:
            print("  Failed to generate ground truth.")
            return None

        print(f"  GT generated: {len(gt_points)} edge points")
        metrics = self.compute_metrics(paint_trail, gt_points)
        if metrics is None:
            print("  Metric computation failed.")
            return None

        self.print_summary(metrics)
        self.save_csv(metrics, paint_trail, gt_points)
        return metrics
