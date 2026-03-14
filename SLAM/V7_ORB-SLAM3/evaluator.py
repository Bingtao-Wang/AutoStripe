"""SLAM evaluation: ATE and RPE metrics."""
import sys
import glob
import os

try:
    carla_egg = glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(carla_egg)
except IndexError:
    pass

import numpy as np
import csv

class SLAMEvaluator:
    def __init__(self):
        self.metrics = {}

    def compute_metrics(self, poses):
        """Compute ATE and RPE for ORB-SLAM3 and KISS-ICP."""
        if len(poses) < 2:
            return

        # Extract pose arrays
        gt_poses = []
        orb_poses = []
        kiss_poses = []

        for p in poses:
            if p['gt'] is not None:
                gt_poses.append(p['gt'])
                orb_poses.append(p['orb'])
                kiss_poses.append(p['kiss'])

        if len(gt_poses) < 2:
            return

        # Compute metrics for ORB-SLAM3
        if any(p is not None for p in orb_poses):
            self.metrics['orb'] = self._compute_slam_metrics(gt_poses, orb_poses, 'ORB-SLAM3')

        # Compute metrics for KISS-ICP
        if any(p is not None for p in kiss_poses):
            self.metrics['kiss'] = self._compute_slam_metrics(gt_poses, kiss_poses, 'KISS-ICP')

    def _compute_slam_metrics(self, gt_poses, slam_poses, name):
        """Compute ATE and RPE for a SLAM method."""
        ate_trans_errors = []
        ate_rot_errors = []
        rpe_trans_errors = []
        rpe_rot_errors = []
        valid_count = 0

        # ATE (Absolute Trajectory Error)
        for gt, slam in zip(gt_poses, slam_poses):
            if slam is None:
                continue
            valid_count += 1

            # Translation error
            trans_error = np.sqrt(
                (gt.location.x - slam.location.x)**2 +
                (gt.location.y - slam.location.y)**2 +
                (gt.location.z - slam.location.z)**2
            )
            ate_trans_errors.append(trans_error)

            # Rotation error (yaw only for simplicity)
            rot_error = abs(self._angle_diff(gt.rotation.yaw, slam.rotation.yaw))
            ate_rot_errors.append(rot_error)

        # RPE (Relative Pose Error)
        for i in range(len(gt_poses) - 1):
            if slam_poses[i] is None or slam_poses[i+1] is None:
                continue

            # GT delta
            gt_dx = gt_poses[i+1].location.x - gt_poses[i].location.x
            gt_dy = gt_poses[i+1].location.y - gt_poses[i].location.y
            gt_dz = gt_poses[i+1].location.z - gt_poses[i].location.z
            gt_dyaw = self._angle_diff(gt_poses[i+1].rotation.yaw, gt_poses[i].rotation.yaw)

            # SLAM delta
            slam_dx = slam_poses[i+1].location.x - slam_poses[i].location.x
            slam_dy = slam_poses[i+1].location.y - slam_poses[i].location.y
            slam_dz = slam_poses[i+1].location.z - slam_poses[i].location.z
            slam_dyaw = self._angle_diff(slam_poses[i+1].rotation.yaw, slam_poses[i].rotation.yaw)

            # RPE
            rpe_trans = np.sqrt((gt_dx - slam_dx)**2 + (gt_dy - slam_dy)**2 + (gt_dz - slam_dz)**2)
            rpe_rot = abs(self._angle_diff(gt_dyaw, slam_dyaw))

            rpe_trans_errors.append(rpe_trans)
            rpe_rot_errors.append(rpe_rot)

        tracking_rate = 100.0 * valid_count / len(gt_poses) if len(gt_poses) > 0 else 0.0

        return {
            'name': name,
            'ate_trans_m': np.mean(ate_trans_errors) if ate_trans_errors else 0.0,
            'ate_rot_deg': np.mean(ate_rot_errors) if ate_rot_errors else 0.0,
            'rpe_trans_m': np.mean(rpe_trans_errors) if rpe_trans_errors else 0.0,
            'rpe_rot_deg': np.mean(rpe_rot_errors) if rpe_rot_errors else 0.0,
            'tracking_rate': tracking_rate
        }

    def _angle_diff(self, a1, a2):
        """Compute smallest angle difference."""
        diff = a1 - a2
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def export_csv(self, run_dir):
        """Export metrics to CSV."""
        if not self.metrics:
            return

        # Summary CSV
        summary_path = os.path.join(run_dir, 'slam_eval_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'ate_trans_m', 'ate_rot_deg', 'rpe_trans_m', 'rpe_rot_deg', 'tracking_rate'])
            for method in ['orb', 'kiss']:
                if method in self.metrics:
                    m = self.metrics[method]
                    writer.writerow([m['name'], f"{m['ate_trans_m']:.3f}", f"{m['ate_rot_deg']:.3f}",
                                   f"{m['rpe_trans_m']:.3f}", f"{m['rpe_rot_deg']:.3f}", f"{m['tracking_rate']:.1f}"])

        print(f"[SLAM Eval] Summary saved to {summary_path}")

    def export_detail_csv(self, poses, run_dir):
        """Export detailed pose data to CSV."""
        detail_path = os.path.join(run_dir, 'slam_poses_detail.csv')
        with open(detail_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame', 'gt_x', 'gt_y', 'gt_z', 'gt_yaw',
                           'orb_x', 'orb_y', 'orb_z', 'orb_yaw', 'orb_valid',
                           'kiss_x', 'kiss_y', 'kiss_z', 'kiss_yaw'])

            for i, p in enumerate(poses):
                gt = p['gt']
                orb = p['orb']
                kiss = p['kiss']

                row = [f"{p['timestamp']:.3f}", i,
                       f"{gt.location.x:.3f}", f"{gt.location.y:.3f}", f"{gt.location.z:.3f}", f"{gt.rotation.yaw:.2f}"]

                if orb:
                    row.extend([f"{orb.location.x:.3f}", f"{orb.location.y:.3f}",
                               f"{orb.location.z:.3f}", f"{orb.rotation.yaw:.2f}", "1"])
                else:
                    row.extend(["", "", "", "", "0"])

                if kiss:
                    row.extend([f"{kiss.location.x:.3f}", f"{kiss.location.y:.3f}",
                               f"{kiss.location.z:.3f}", f"{kiss.rotation.yaw:.2f}"])
                else:
                    row.extend(["", "", "", ""])

                writer.writerow(row)

        print(f"[SLAM Eval] Detail saved to {detail_path}")
