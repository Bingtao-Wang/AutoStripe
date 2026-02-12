"""Test Nature-style colormap scheme D for map_1 trajectory."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

# --- Nature-style config ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

# Scheme D: orange-red → emerald green (3.0m) → indigo-blue
CMAP_D = LinearSegmentedColormap.from_list('scheme_d', [
    (0.0,  '#d73027'),   # red (too close, 1.5m)
    (0.25, '#fc8d59'),   # orange
    (0.5,  '#1a9850'),   # saturated emerald green (3.0m = ideal)
    (0.75, '#6baed6'),   # light blue
    (1.0,  '#2166ac'),   # indigo-blue (too far, 4.5m)
], N=256)

TARGET = 3.0
VMIN, VMAX = 1.5, 4.5


def load_framelog(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    data = {}
    for col in ['veh_x', 'veh_y', 'nozzle_x', 'nozzle_y',
                'nozzle_edge_dist', 'painting_enabled']:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (ValueError, KeyError):
                vals.append(0.0)
        data[col] = np.array(vals)
    return data


def plot_test(data, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    vx, vy = data['veh_x'], data['veh_y']
    nx, ny = data['nozzle_x'], data['nozzle_y']
    ned = data['nozzle_edge_dist']

    # Vehicle path (subtle)
    ax.plot(vx, vy, '-', color='#cccccc', alpha=0.5, lw=0.8,
            label='Vehicle path', zorder=1)

    # Nozzle trajectory colored by distance
    points = np.column_stack([nx, ny]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    lc = LineCollection(segs, cmap=CMAP_D, norm=norm, linewidths=2.5)
    lc.set_array(ned[:-1])
    ax.add_collection(lc)

    # Colorbar
    cbar = fig.colorbar(lc, ax=ax, shrink=0.7, pad=0.02, aspect=25)
    cbar.set_label('Nozzle–Edge Distance (m)', fontsize=12)
    cbar.ax.axhline(y=TARGET, color='white', linewidth=2, linestyle='-')
    cbar.ax.axhline(y=TARGET, color='black', linewidth=0.8, linestyle='--')
    # Add target annotation on colorbar
    cbar.ax.text(1.5, TARGET, '3.0 m\n(target)', va='center', fontsize=9,
                 fontweight='bold', color='#1a9850')

    # Start / End markers
    ax.plot(nx[0], ny[0], 'o', color='#1a9850', markersize=10, zorder=10,
            markeredgecolor='white', markeredgewidth=1.5, label='Start')
    ax.plot(nx[-1], ny[-1], 's', color='#d73027', markersize=10, zorder=10,
            markeredgecolor='white', markeredgewidth=1.5, label='End')

    # Direction arrows
    for frac in [0.25, 0.5, 0.75]:
        i = int(frac * len(nx))
        if i + 5 < len(nx):
            dx = nx[i + 5] - nx[i]
            dy = ny[i + 5] - ny[i]
            ax.annotate('', xy=(nx[i] + dx, ny[i] + dy),
                        xytext=(nx[i], ny[i]),
                        arrowprops=dict(arrowstyle='->', color='#333333',
                                        lw=1.5), zorder=8)

    # Axes
    pad = 20
    ax.set_xlim(min(vx) - pad, max(vx) + pad)
    ax.set_ylim(min(vy) - pad, max(vy) + pad)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Nozzle Trajectory — Edge Distance', fontsize=14,
                 fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
              edgecolor='#cccccc')
    ax.grid(True, alpha=0.15, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    fl = 'evaluation/run_20260211_022107/framelog_20260211_022107.csv'
    data = load_framelog(fl)
    out = 'evaluation/run_20260211_022107/test_scheme_d.png'
    plot_test(data, out)
