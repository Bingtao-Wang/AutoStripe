#!/usr/bin/env python3
"""Plot loop trajectory overview for each experiment run.

For each framelog_aligned CSV found under evaluation/, generates:
  - trajectory_overview.pdf: colored by nozzle-edge distance deviation

Usage:
    python Ch4_/evaluation/plot_loop_trajectory.py
"""

import csv
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
})


def load_framelog(path):
    """Load pre-aligned framelog CSV into dict of numpy arrays."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    data = {}
    for col in ['veh_x', 'veh_y', 'nozzle_x', 'nozzle_y',
                'nozzle_edge_dist', 'speed', 'frame']:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col, 0) or 0))
            except (ValueError, TypeError):
                vals.append(0.0)
        data[col] = np.array(vals)

    # Cumulative distance
    vx, vy = data['veh_x'], data['veh_y']
    dx, dy = np.diff(vx), np.diff(vy)
    step_dist = np.sqrt(dx**2 + dy**2)
    data['cum_dist'] = np.concatenate([[0.0], np.cumsum(step_dist)])

    return data


def plot_trajectory(data, title, out_path):
    """Plot bird's-eye trajectory colored by nozzle-edge distance deviation."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    vx, vy = data['veh_x'], data['veh_y']
    ned = data['nozzle_edge_dist']
    cum_dist = data['cum_dist']

    # Downsample to max 3000 points (pgf/xelatex memory limit)
    MAX_PTS = 3000
    n_orig = len(vx)
    if n_orig > MAX_PTS:
        idx = np.linspace(0, n_orig - 1, MAX_PTS, dtype=int)
        vx, vy = vx[idx], vy[idx]
        ned = ned[idx]
        cum_dist = cum_dist[idx]
    n = len(vx)

    # Color by nozzle_edge_dist using LineCollection
    cmap = LinearSegmentedColormap.from_list('edge_dist', [
        (0.0,  '#2166ac'),
        (0.15, '#6baed6'),
        (0.35, '#1a9850'),
        (0.65, '#1a9850'),
        (0.85, '#fc8d59'),
        (1.0,  '#d73027'),
    ], N=256)
    ned_min_filtered = np.percentile(ned, 2)   # filter outlier lows
    ned_max_filtered = np.percentile(ned, 98)  # filter outlier highs
    target = 3.0
    vmin = target - (target - ned_min_filtered) * 1.1
    vmax = target + (ned_max_filtered - target) * 1.1
    norm = TwoSlopeNorm(vcenter=target, vmin=vmin, vmax=vmax)

    points = np.column_stack([vx, vy]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=5.0)
    lc.set_array(ned[:-1])
    ax.add_collection(lc)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Nozzle-Edge Dist (m)', fontsize=9)
    # Mark min / mean / max on colorbar
    ned_mean = np.mean(ned)
    for val, label in [(ned_min_filtered, f'p2={ned_min_filtered:.2f}'),
                       (ned_mean, f'avg={ned_mean:.2f}'),
                       (ned_max_filtered, f'p98={ned_max_filtered:.2f}')]:
        cbar.ax.axhline(y=val, color='black', linewidth=0.8, linestyle='--')
        cbar.ax.text(1.05, val, label, transform=cbar.ax.get_yaxis_transform(),
                     fontsize=6, va='center', ha='left')
    # Mark target 3.0m if within range
    if norm.vmin <= 3.0 <= norm.vmax:
        cbar.ax.axhline(y=3.0, color='white', linewidth=1.5)
        cbar.ax.axhline(y=3.0, color='black', linewidth=0.5, linestyle='--')

    # Direction arrows + distance labels every 200m
    arrow_interval = 200
    next_arrow = arrow_interval
    for i in range(1, n - 1):
        if cum_dist[i] >= next_arrow:
            dx = vx[i+1] - vx[i-1]
            dy = vy[i+1] - vy[i-1]
            ax.annotate('', xy=(vx[i] + dx*0.3, vy[i] + dy*0.3),
                        xytext=(vx[i], vy[i]),
                        arrowprops=dict(arrowstyle='->', color='black',
                                        lw=1.5, mutation_scale=12),
                        zorder=8)
            ax.annotate(f'{int(next_arrow)}m', (vx[i], vy[i]),
                        fontsize=6, color='#333333',
                        textcoords='offset points', xytext=(8, -8))
            next_arrow += arrow_interval
    ax.plot(vx[0], vy[0], 'o', color='#1a9850', markersize=14,
            markeredgecolor='black', markeredgewidth=0.5, zorder=10,
            label='Start')
    ax.plot(vx[-1], vy[-1], 's', color='#d73027', markersize=8,
            markeredgecolor='black', markeredgewidth=0.5, zorder=10,
            label='End')

    # Cumulative distance annotation
    total_dist = data['cum_dist'][-1]
    ax.set_title(f'{title}\nTotal: {total_dist:.0f}m',
                 fontsize=10)

    # Legend
    handles = [
        Line2D([], [], marker='o', color='#1a9850', lw=0,
               markersize=6, label='Start'),
        Line2D([], [], marker='s', color='#d73027', lw=0,
               markersize=6, label='End'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    pad = 20
    ax.set_xlim(min(vx) - pad, max(vx) + pad)
    ax.set_ylim(min(vy) - pad, max(vy) + pad)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def main():
    run_dirs = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*run_*')))
    if not run_dirs:
        print('No run_* directories found.')
        return

    for run_dir in run_dirs:
        # Prefer pre-aligned framelog
        aligned = os.path.join(run_dir, 'framelog_aligned.csv')
        if os.path.exists(aligned):
            framelog_path = aligned
        else:
            framelogs = sorted(glob.glob(os.path.join(run_dir, 'framelog_*.csv')))
            if not framelogs:
                continue
            framelog_path = framelogs[-1]
        run_name = os.path.basename(run_dir)
        print(f'\n  Processing: {run_name}')

        data = load_framelog(framelog_path)
        if data is None:
            print('    Empty framelog, skipping.')
            continue

        out_path = os.path.join(run_dir, 'trajectory_overview.pdf')
        plot_trajectory(data, run_name, out_path)


if __name__ == '__main__':
    main()
