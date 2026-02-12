"""AutoStripe full-lap map-based visualizations.

Generates 6 map-based plots from framelog CSV data:
  1. Global trajectory bird's eye view (color-coded nozzle-edge distance)
  2. Paint state spatial map (CONVERGING/STABILIZED/PAINTING)
  3. Deviation heatmap (nozzle-edge distance deviation from 3.0m target)
  4. Segmented evaluation (straight vs curve statistics)
  5. Controller response spatial map (driving_offset, steer_filter)
  6. Speed-curvature spatial map

Usage:
    python evaluation/visualize_map.py
    python evaluation/visualize_map.py evaluation/framelog_20260211_011720.csv
"""

import csv
import os
import sys
import glob
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# --- Nature-style global config ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
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

# Scheme D colormap: indigo-blue (close) → emerald green (3.0m) → orange-red (far)
CMAP_EDGE_DIST = LinearSegmentedColormap.from_list('edge_dist', [
    (0.0,  '#2166ac'),   # indigo-blue (1.5m, too close)
    (0.25, '#6baed6'),   # light blue
    (0.5,  '#1a9850'),   # emerald green (3.0m, ideal)
    (0.75, '#fc8d59'),   # orange
    (1.0,  '#d73027'),   # red (4.5m, too far)
], N=256)

# Paper-safe colormaps: NO white/light/gray/pastel colors anywhere
# Diverging: dark blue → medium blue → dark teal (neutral) → medium red → dark red
CMAP_DIVERGE = LinearSegmentedColormap.from_list('diverge', [
    (0.0,  '#2166ac'),
    (0.25, '#4393c3'),
    (0.5,  '#2d6a6a'),
    (0.75, '#d6604d'),
    (1.0,  '#b2182b'),
], N=256)

# Sequential warm: dark orange → red → dark red (no light start)
CMAP_SEQ_WARM = LinearSegmentedColormap.from_list('seq_warm', [
    (0.0,  '#e6550d'),
    (0.33, '#d73027'),
    (0.66, '#b2182b'),
    (1.0,  '#a50026'),
], N=256)

# Sequential cool: green (low) → blue → purple → dark purple (high)
CMAP_SEQ_COOL = LinearSegmentedColormap.from_list('seq_cool', [
    (0.0,  '#1a9850'),
    (0.33, '#2166ac'),
    (0.66, '#762a83'),
    (1.0,  '#40004b'),
], N=256)

# Controller diverging: dense control points to avoid gray/muddy RGB interpolation
CMAP_CTRL_DIVERGE = LinearSegmentedColormap.from_list('ctrl_diverge', [
    (0.00, '#08519c'),   # dark blue (low extreme)
    (0.15, '#2171b5'),   # blue
    (0.30, '#006d2c'),   # dark forest green
    (0.42, '#238b45'),   # green
    (0.50, '#1a9850'),   # emerald green (ideal center)
    (0.58, '#7a6b00'),   # dark gold (bridge green→orange)
    (0.70, '#cc4c02'),   # dark orange
    (0.85, '#d73027'),   # red
    (1.00, '#a50026'),   # dark red (high extreme)
], N=256)

# Controller sequential: emerald green (good) → yellow-green → orange → dark red (bad)
CMAP_CTRL_SEQ = LinearSegmentedColormap.from_list('ctrl_seq', [
    (0.0,  '#1a9850'),   # emerald green (ideal/low)
    (0.33, '#66bd63'),   # medium green
    (0.55, '#d9a528'),   # gold
    (0.78, '#d73027'),   # red
    (1.0,  '#a50026'),   # dark red (extreme)
], N=256)

# Target nozzle-edge distance
TARGET_DIST = 3.0
# Curvature threshold for curve/straight classification
CURVATURE_THRESH = 0.003
# Smoothing window for curvature computation
CURV_SMOOTH_WINDOW = 5


def load_framelog(path):
    """Load framelog CSV into dict of numpy arrays."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    float_cols = [
        'timestamp', 'frame', 'dt', 'veh_x', 'veh_y', 'veh_yaw', 'speed',
        'nozzle_x', 'nozzle_y', 'nozzle_edge_dist', 'poly_edge_dist',
        'driving_offset', 'steer_filter', 'steer_cmd', 'throttle_cmd',
        'brake_cmd', 'lateral_error', 'painting_enabled',
        'ai_edge_pts', 'gt_edge_pts', 'road_mask_ratio',
        'poly_coeff_a', 'poly_coeff_b', 'poly_coeff_c', 'inference_time_ms'
    ]
    str_cols = ['paint_state', 'perception_mode', 'dash_phase']

    data = {}
    for col in float_cols:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col, 0)))
            except (ValueError, TypeError):
                vals.append(0.0)
        data[col] = np.array(vals)

    for col in str_cols:
        data[col] = [r.get(col, '') for r in rows]

    return data


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_curvature(xs, ys, yaws, window=CURV_SMOOTH_WINDOW):
    """Compute per-point curvature from heading change rate."""
    n = len(xs)
    curvatures = np.zeros(n)
    hw = window
    for i in range(hw, n - hw):
        dx = xs[i + hw] - xs[i - hw]
        dy = ys[i + hw] - ys[i - hw]
        ds = math.sqrt(dx * dx + dy * dy)
        if ds < 0.05:
            continue
        dyaw = yaws[i + hw] - yaws[i - hw]
        while dyaw > 180:
            dyaw -= 360
        while dyaw < -180:
            dyaw += 360
        curvatures[i] = abs(math.radians(dyaw) / ds)
    # Fill edges
    curvatures[:hw] = curvatures[hw]
    curvatures[-hw:] = curvatures[n - hw - 1]
    return curvatures


def classify_segments(curvatures, thresh=CURVATURE_THRESH, min_gap=80):
    """Classify trajectory into straight/curve segments.

    Returns list of (start_idx, end_idx, 'straight'|'curve').
    """
    n = len(curvatures)
    is_curve = curvatures > thresh

    # Find contiguous curve regions
    segments = []
    in_curve = False
    seg_start = 0
    for i in range(n):
        if is_curve[i] and not in_curve:
            # End straight, start curve
            if i > seg_start:
                segments.append((seg_start, i - 1, 'straight'))
            seg_start = i
            in_curve = True
        elif not is_curve[i] and in_curve:
            # End curve, start straight
            if i > seg_start:
                segments.append((seg_start, i - 1, 'curve'))
            seg_start = i
            in_curve = False
    # Last segment
    if seg_start < n - 1:
        segments.append((seg_start, n - 1, 'curve' if in_curve else 'straight'))

    # Merge short segments (< min_gap) into neighbors
    merged = []
    for seg in segments:
        s, e, typ = seg
        if e - s < min_gap and merged:
            # Absorb into previous
            ps, pe, pt = merged[-1]
            merged[-1] = (ps, e, pt)
        else:
            merged.append(seg)

    return merged


def _colored_trajectory(ax, xs, ys, values, cmap, vmin, vmax, lw=2.5):
    """Draw a trajectory colored by values using LineCollection."""
    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw)
    lc.set_array(values[:-1])
    ax.add_collection(lc)
    return lc


def _setup_map_ax(ax, xs, ys, title, pad=30):
    """Configure a map axes with equal aspect and padding."""
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.invert_yaxis()  # CARLA Y+ points south; invert for correct top-down view
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.15, linewidth=0.5)


def _add_start_end_markers(ax, xs, ys):
    """Add start/end markers and direction arrow."""
    ax.plot(xs[0], ys[0], 'o', color='#1a9850', markersize=7, zorder=10,
            markeredgecolor='black', markeredgewidth=0.8, label='Start')
    ax.plot(xs[-1], ys[-1], 's', color='#d73027', markersize=7, zorder=10,
            markeredgecolor='black', markeredgewidth=0.8, label='End')
    # Direction arrows at 25%, 50%, 75%
    for frac in [0.25, 0.5, 0.75]:
        idx = int(frac * len(xs))
        if idx + 5 < len(xs):
            dx = xs[idx + 5] - xs[idx]
            dy = ys[idx + 5] - ys[idx]
            ax.annotate('', xy=(xs[idx] + dx, ys[idx] + dy),
                        xytext=(xs[idx], ys[idx]),
                        arrowprops=dict(arrowstyle='->', color='#333333',
                                        lw=1.5, mutation_scale=15))


# ---------------------------------------------------------------------------
# Plot 1: Global trajectory bird's eye view
# ---------------------------------------------------------------------------

def plot_nozzle_trajectory(data, out_dir):
    """Nozzle trajectory colored by nozzle-edge distance (Scheme D)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    vx, vy = data['veh_x'], data['veh_y']
    nx, ny = data['nozzle_x'], data['nozzle_y']
    ned = data['nozzle_edge_dist']

    lc = _colored_trajectory(ax, nx, ny, ned, CMAP_EDGE_DIST,
                             vmin=1.5, vmax=4.5, lw=2.5)
    cbar = fig.colorbar(lc, ax=ax, shrink=0.7, pad=0.02, aspect=25)
    cbar.set_label('Nozzle\u2013Edge Distance (m)', fontsize=12)
    # 3.0m target marker on colorbar
    cbar.ax.axhline(y=TARGET_DIST, color='white', linewidth=2.0)
    cbar.ax.axhline(y=TARGET_DIST, color='black', linewidth=0.8,
                     linestyle='--')
    cbar.ax.text(-0.3, TARGET_DIST, '3.0 m\n(target)', va='center', ha='right',
                 fontsize=10, fontweight='bold', color='#1a9850',
                 transform=cbar.ax.get_yaxis_transform())
    # Min / Max marker lines + annotations (left side of colorbar)
    ned_min, ned_max = float(ned.min()), float(ned.max())
    cbar.ax.axhline(y=ned_min, color='#2166ac', linewidth=1.2, linestyle='-')
    cbar.ax.axhline(y=ned_max, color='#d73027', linewidth=1.2, linestyle='-')
    cbar.ax.text(-0.3, ned_min, 'min %.1f' % ned_min, va='center', ha='right',
                 fontsize=9, fontweight='bold', color='#2166ac',
                 transform=cbar.ax.get_yaxis_transform())
    cbar.ax.text(-0.3, ned_max, 'max %.1f' % ned_max, va='center', ha='right',
                 fontsize=9, fontweight='bold', color='#d73027',
                 transform=cbar.ax.get_yaxis_transform())
    # Legend swatches
    ax.plot([], [], '-', color='#1a9850', lw=3, label='\u22483.0 m (ideal)')
    ax.plot([], [], '-', color='#2166ac', lw=3, label='<2.0 m (too close)')
    ax.plot([], [], '-', color='#d73027', lw=3, label='>4.0 m (too far)')
    _add_start_end_markers(ax, nx, ny)
    _setup_map_ax(ax, vx, vy, 'Nozzle Trajectory \u2014 Edge Distance')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
              edgecolor='#999999')

    fig.tight_layout()
    base = os.path.join(out_dir, 'map_1a_nozzle_trajectory')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig('%s.%s' % (base, ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: %s.{png,pdf,svg}' % base)
    return base + '.png'


def plot_paint_coverage(data, out_dir):
    """Paint coverage map with solid/dashed distinction."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    vx, vy = data['veh_x'], data['veh_y']
    nx, ny = data['nozzle_x'], data['nozzle_y']
    painting = data['painting_enabled'].astype(bool)
    dash = data.get('dash_phase', np.full(len(nx), -1.0))
    if isinstance(dash, list):
        dash = np.array([float(d) if d != '' else -1.0 for d in dash])

    # Classify: solid paint / dashed paint / dashed gap / paint off
    solid_mask = painting & (dash < 0)
    dash_on_mask = painting & (dash > 0.5)
    dash_gap_mask = painting & (dash >= 0) & (dash <= 0.5)
    off_mask = ~painting

    if off_mask.any():
        ax.plot(nx[off_mask], ny[off_mask], '.', color='#999999',
                markersize=1, zorder=1)
    if solid_mask.any():
        ax.plot(nx[solid_mask], ny[solid_mask], '.', color='#2166ac',
                markersize=1.5, zorder=3)
    if dash_on_mask.any():
        ax.plot(nx[dash_on_mask], ny[dash_on_mask], '.', color='#17807e',
                markersize=1.5, zorder=3)
    if dash_gap_mask.any():
        ax.plot(nx[dash_gap_mask], ny[dash_gap_mask], '.', color='#fc8d59',
                markersize=1.5, zorder=2)
    _add_start_end_markers(ax, nx, ny)
    _setup_map_ax(ax, vx, vy, 'Paint Coverage Map')

    # Custom legend: large dots for paint categories, normal for Start/End
    handles = []
    if off_mask.any():
        handles.append(Line2D([], [], marker='o', color='w', markerfacecolor='#999999',
                              markersize=8, label='Paint OFF'))
    if solid_mask.any():
        handles.append(Line2D([], [], marker='o', color='w', markerfacecolor='#2166ac',
                              markersize=8, label='Solid (%d)' % solid_mask.sum()))
    if dash_on_mask.any():
        handles.append(Line2D([], [], marker='o', color='w', markerfacecolor='#17807e',
                              markersize=8, label='Dashed\u2013paint (%d)' % dash_on_mask.sum()))
    if dash_gap_mask.any():
        handles.append(Line2D([], [], marker='o', color='w', markerfacecolor='#fc8d59',
                              markersize=8, label='Dashed\u2013gap (%d)' % dash_gap_mask.sum()))
    handles.append(Line2D([], [], marker='o', color='w', markerfacecolor='#1a9850',
                          markersize=10, markeredgecolor='black', markeredgewidth=0.8,
                          label='Start'))
    handles.append(Line2D([], [], marker='s', color='w', markerfacecolor='#d73027',
                          markersize=10, markeredgecolor='black', markeredgewidth=0.8,
                          label='End'))
    ax.legend(handles=handles, loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='#999999')

    fig.tight_layout()
    base = os.path.join(out_dir, 'map_1b_paint_coverage')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig('%s.%s' % (base, ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: %s.{png,pdf,svg}' % base)
    return base + '.png'


# ---------------------------------------------------------------------------
# Plot 2: Paint state spatial map
# ---------------------------------------------------------------------------

def plot_paint_state_map(data, out_dir):
    """Map colored by AutoPaint state machine state."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    nx, ny = data['nozzle_x'], data['nozzle_y']
    states = data['paint_state']

    state_colors = {
        'CONVERGING': '#e41a1c',   # red
        'STABILIZED': '#ff7f00',   # orange
        'PAINTING':   '#4daf4a',   # green
    }

    # Plot each state as separate scatter
    for state, color in state_colors.items():
        mask = np.array([s == state for s in states])
        if mask.any():
            ax.plot(nx[mask], ny[mask], '.', color=color, markersize=2.5,
                    label='%s (%d frames, %.0f%%)' % (
                        state, mask.sum(), 100 * mask.sum() / len(states)),
                    zorder=3 if state == 'PAINTING' else 2)

    # Mark state transition points
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            ax.plot(nx[i], ny[i], 'D', color='black', markersize=5,
                    zorder=5, markeredgecolor='black', markeredgewidth=0.5)

    _add_start_end_markers(ax, nx, ny)
    _setup_map_ax(ax, nx, ny, 'AutoPaint State Machine — Spatial Distribution')
    # Custom legend: enlarge state dots without enlarging Start/End
    handles, labels = ax.get_legend_handles_labels()
    custom_handles = []
    for h, l in zip(handles, labels):
        if l in ('Start', 'End'):
            custom_handles.append(h)
        else:
            custom_handles.append(Line2D([], [], marker='o', color='w',
                                         markerfacecolor=h.get_color(),
                                         markersize=10, label=l))
    ax.legend(handles=custom_handles, labels=labels,
              loc='upper left', fontsize=10)

    fig.tight_layout()
    base = os.path.join(out_dir, 'map_2_paint_state')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig('%s.%s' % (base, ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: %s.{png,pdf,svg}' % base)
    return base + '.png'


# ---------------------------------------------------------------------------
# Plot 3: Deviation heatmap
# ---------------------------------------------------------------------------

def plot_deviation_heatmap(data, out_dir):
    """Nozzle-edge distance deviation from 3.0m target, on map."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    nx, ny = data['nozzle_x'], data['nozzle_y']
    ned = data['nozzle_edge_dist']
    painting = data['painting_enabled'].astype(bool)
    deviation = ned - TARGET_DIST

    # Left: all frames, signed deviation
    ax = axes[0]
    lc = _colored_trajectory(ax, nx, ny, deviation, CMAP_DIVERGE,
                             vmin=-1.5, vmax=1.5, lw=2.5)
    fig.colorbar(lc, ax=ax, label='Deviation from 3.0m (m)', shrink=0.7)
    _add_start_end_markers(ax, nx, ny)
    _setup_map_ax(ax, nx, ny, 'Signed Deviation (all frames)')
    ax.legend(loc='upper left', fontsize=9)

    # Right: PAINTING frames only, absolute deviation
    ax = axes[1]
    abs_dev = np.abs(deviation)
    p_nx, p_ny = nx[painting], ny[painting]
    p_dev = abs_dev[painting]
    sc = ax.scatter(p_nx, p_ny, c=p_dev, cmap=CMAP_SEQ_WARM, s=3,
                    vmin=0, vmax=1.0, zorder=3)
    fig.colorbar(sc, ax=ax, label='|Deviation| (m)', shrink=0.7)
    if painting.any():
        p_ned = ned[painting]
        stats = ('PAINTING stats:\n'
                 '  mean dist = %.2f m\n'
                 '  std       = %.2f m\n'
                 '  |dev| mean= %.2f m\n'
                 '  |dev| max = %.2f m' % (
                     p_ned.mean(), p_ned.std(),
                     np.abs(p_ned - TARGET_DIST).mean(),
                     np.abs(p_ned - TARGET_DIST).max()))
        ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    _add_start_end_markers(ax, nx, ny)
    _setup_map_ax(ax, nx, ny, 'Absolute Deviation (PAINTING only)')
    ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Nozzle-Edge Distance Deviation Heatmap', fontsize=15,
                 fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    base = os.path.join(out_dir, 'map_3_deviation')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig('%s.%s' % (base, ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: %s.{png,pdf,svg}' % base)
    return base + '.png'


# ---------------------------------------------------------------------------
# Plot 4: Segmented evaluation (straight vs curve)
# ---------------------------------------------------------------------------

def plot_segmented_evaluation(data, curvatures, segments, out_dir):
    """Straight vs curve segment statistics with map overlay."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    nx, ny = data['nozzle_x'], data['nozzle_y']
    ned = data['nozzle_edge_dist']
    painting = data['painting_enabled'].astype(bool)

    # --- Top-left: map with segments colored ---
    ax = fig.add_subplot(gs[0, 0])
    seg_colors = {'straight': '#2166ac', 'curve': '#b2182b'}
    for s, e, typ in segments:
        ax.plot(nx[s:e+1], ny[s:e+1], '-', color=seg_colors[typ],
                lw=2.5, alpha=0.8)
    ax.plot([], [], '-', color=seg_colors['straight'], lw=3, label='Straight')
    ax.plot([], [], '-', color=seg_colors['curve'], lw=3, label='Curve')
    _add_start_end_markers(ax, nx, ny)
    _setup_map_ax(ax, nx, ny, 'Segment Classification')
    ax.legend(loc='upper left', fontsize=10)

    # --- Top-right: curvature along trajectory ---
    ax2 = fig.add_subplot(gs[0, 1])
    lc = _colored_trajectory(ax2, nx, ny, curvatures, CMAP_SEQ_COOL,
                             vmin=0, vmax=0.01, lw=2.5)
    fig.colorbar(lc, ax=ax2, label='Curvature (1/m)', shrink=0.7)
    _setup_map_ax(ax2, nx, ny, 'Road Curvature Map')

    # --- Bottom: bar chart comparison ---
    seg_stats = []
    for s, e, typ in segments:
        mask = painting[s:e+1]
        seg_ned = ned[s:e+1][mask]
        if len(seg_ned) < 5:
            continue
        dev = np.abs(seg_ned - TARGET_DIST)
        seg_stats.append({
            'type': typ,
            'start': s, 'end': e,
            'n_frames': e - s + 1,
            'n_painting': mask.sum(),
            'mean_dist': seg_ned.mean(),
            'std_dist': seg_ned.std(),
            'mean_dev': dev.mean(),
            'max_dev': dev.max(),
        })

    # Aggregate by type
    agg = {}
    for typ in ['straight', 'curve']:
        ss = [s for s in seg_stats if s['type'] == typ]
        if not ss:
            continue
        all_ned = []
        for s_info in ss:
            s, e = s_info['start'], s_info['end']
            mask = painting[s:e+1]
            all_ned.extend(ned[s:e+1][mask].tolist())
        all_ned = np.array(all_ned)
        agg[typ] = {
            'n_segments': len(ss),
            'n_painting': len(all_ned),
            'mean_dist': all_ned.mean(),
            'std_dist': all_ned.std(),
            'mean_dev': np.abs(all_ned - TARGET_DIST).mean(),
            'max_dev': np.abs(all_ned - TARGET_DIST).max(),
            'median_dist': np.median(all_ned),
        }

    ax3 = fig.add_subplot(gs[1, 0])
    if len(agg) == 2:
        metrics = ['mean_dist', 'std_dist', 'mean_dev', 'max_dev']
        labels = ['Mean Dist', 'Std Dist', 'Mean |Dev|', 'Max |Dev|']
        x_pos = np.arange(len(metrics))
        w = 0.35
        bars_s = [agg['straight'][m] for m in metrics]
        bars_c = [agg['curve'][m] for m in metrics]
        ax3.bar(x_pos - w/2, bars_s, w, label='Straight', color='#2166ac')
        ax3.bar(x_pos + w/2, bars_c, w, label='Curve', color='#b2182b')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels)
        ax3.set_ylabel('Distance (m)')
        ax3.set_title('Straight vs Curve — Key Metrics', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        # Add value labels
        for i, (vs, vc) in enumerate(zip(bars_s, bars_c)):
            ax3.text(i - w/2, vs + 0.02, '%.2f' % vs, ha='center', fontsize=8)
            ax3.text(i + w/2, vc + 0.02, '%.2f' % vc, ha='center', fontsize=8)

    # --- Bottom-right: per-segment table ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    if seg_stats:
        col_labels = ['Segment', 'Type', 'Frames', 'Paint', 'Mean', 'Std',
                      '|Dev|', 'Max|Dev|']
        table_data = []
        for i, ss in enumerate(seg_stats):
            table_data.append([
                '#%d' % (i + 1), ss['type'].upper(),
                '%d' % ss['n_frames'], '%d' % ss['n_painting'],
                '%.2f' % ss['mean_dist'], '%.2f' % ss['std_dist'],
                '%.2f' % ss['mean_dev'], '%.2f' % ss['max_dev'],
            ])
        # Add aggregate rows
        for typ in ['straight', 'curve']:
            if typ in agg:
                a = agg[typ]
                table_data.append([
                    'ALL', typ.upper(),
                    '-', '%d' % a['n_painting'],
                    '%.2f' % a['mean_dist'], '%.2f' % a['std_dist'],
                    '%.2f' % a['mean_dev'], '%.2f' % a['max_dev'],
                ])
        tbl = ax4.table(cellText=table_data, colLabels=col_labels,
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)
        # Color header
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor('#a6bddb')
        # Color aggregate rows
        n_data = len(seg_stats)
        for k in range(len(agg)):
            for j in range(len(col_labels)):
                tbl[n_data + 1 + k, j].set_facecolor('#fdd49e')
    ax4.set_title('Per-Segment Statistics', fontweight='bold', pad=20)

    fig.suptitle('Segmented Evaluation: Straight vs Curve', fontsize=15,
                 fontweight='bold', y=0.98)
    base = os.path.join(out_dir, 'map_4_segments')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig('%s.%s' % (base, ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: %s.{png,pdf,svg}' % base)
    return base + '.png'


# ---------------------------------------------------------------------------
# Plot 5: Controller response spatial map
# ---------------------------------------------------------------------------

def plot_controller_response(data, out_dir):
    """Driving offset and steer filter on map."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    nx, ny = data['nozzle_x'], data['nozzle_y']
    vx, vy = data['veh_x'], data['veh_y']

    # Top-left: driving_offset (green at 5.0 target, diverge for deviations)
    ax = axes[0, 0]
    do = data['driving_offset']
    lc = _colored_trajectory(ax, vx, vy, do, CMAP_CTRL_DIVERGE,
                             vmin=4.5, vmax=5.5, lw=2.5)
    fig.colorbar(lc, ax=ax, label='Driving Offset (m)', shrink=0.7)
    _setup_map_ax(ax, vx, vy, 'Dynamic Driving Offset')

    # Top-right: steer_filter (green=smooth, red=aggressive)
    ax = axes[0, 1]
    sf = data['steer_filter']
    lc = _colored_trajectory(ax, vx, vy, sf, CMAP_CTRL_SEQ,
                             vmin=0.15, vmax=0.50, lw=2.5)
    fig.colorbar(lc, ax=ax, label='Steer Filter', shrink=0.7)
    _setup_map_ax(ax, vx, vy, 'Adaptive Steer Filter')

    # Bottom-left: speed (green at target ~2.3, blue=slow, red=fast)
    ax = axes[1, 0]
    spd = data['speed']
    lc = _colored_trajectory(ax, vx, vy, spd, CMAP_CTRL_DIVERGE,
                             vmin=1.5, vmax=3.0, lw=2.5)
    fig.colorbar(lc, ax=ax, label='Speed (m/s)', shrink=0.7)
    _setup_map_ax(ax, vx, vy, 'Vehicle Speed')

    # Bottom-right: lateral_error (green at 0=ideal, diverge for errors)
    ax = axes[1, 1]
    le = data['lateral_error']
    lc = _colored_trajectory(ax, vx, vy, le, CMAP_CTRL_DIVERGE,
                             vmin=-0.8, vmax=0.8, lw=2.5)
    fig.colorbar(lc, ax=ax, label='Lateral Error (m)', shrink=0.7)
    _setup_map_ax(ax, vx, vy, 'Lateral Error')

    fig.suptitle('PD Controller Response — Spatial Distribution',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    base = os.path.join(out_dir, 'map_5_controller')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig('%s.%s' % (base, ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: %s.{png,pdf,svg}' % base)
    return base + '.png'


# ---------------------------------------------------------------------------
# Plot 6: Speed-curvature spatial map
# ---------------------------------------------------------------------------

def plot_speed_curvature(data, curvatures, out_dir):
    """Speed and curvature on map, plus scatter correlation."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    vx, vy = data['veh_x'], data['veh_y']
    speed = data['speed']

    # Left: speed map — diverging around median (blue=slow, green=cruise, red=fast)
    ax = axes[0]
    spd_median = float(np.median(speed))
    spd_dev = max(spd_median - float(speed.min()),
                  float(speed.max()) - spd_median, 0.3)
    spd_vmin = spd_median - spd_dev
    spd_vmax = spd_median + spd_dev
    lc = _colored_trajectory(ax, vx, vy, speed, CMAP_CTRL_DIVERGE,
                             vmin=spd_vmin, vmax=spd_vmax, lw=2.5)
    fig.colorbar(lc, ax=ax, label='Speed (m/s)', shrink=0.7)
    _add_start_end_markers(ax, vx, vy)
    _setup_map_ax(ax, vx, vy, 'Vehicle Speed')

    # Middle: curvature map
    ax = axes[1]
    lc = _colored_trajectory(ax, vx, vy, curvatures, CMAP_SEQ_COOL,
                             vmin=0, vmax=0.01, lw=2.5)
    fig.colorbar(lc, ax=ax, label='Curvature (1/m)', shrink=0.7)
    _setup_map_ax(ax, vx, vy, 'Road Curvature')

    # Right: speed vs curvature scatter
    ax = axes[2]
    # Subsample for scatter readability
    step = max(1, len(speed) // 2000)
    s_sub = speed[::step]
    k_sub = curvatures[::step]
    ned_sub = data['nozzle_edge_dist'][::step]
    sc = ax.scatter(k_sub, s_sub, c=ned_sub, cmap=CMAP_EDGE_DIST,
                    s=8, alpha=0.6, vmin=1.5, vmax=4.5)
    fig.colorbar(sc, ax=ax, label='Nozzle-Edge Dist (m)', shrink=0.7)
    ax.set_xlabel('Curvature (1/m)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed vs Curvature', fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Speed & Curvature Analysis', fontsize=15,
                 fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    base = os.path.join(out_dir, 'map_6_speed_curvature')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig('%s.%s' % (base, ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: %s.{png,pdf,svg}' % base)
    return base + '.png'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_framelog():
    """Find the most recent framelog CSV."""
    pattern = os.path.join('evaluation', 'framelog_*.csv')
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def generate_all(framelog_path):
    """Generate all 6 map-based visualizations."""
    print('Loading framelog: %s' % framelog_path)
    data = load_framelog(framelog_path)
    if data is None:
        print('ERROR: empty framelog')
        return

    n = len(data['veh_x'])
    print('  Frames: %d' % n)
    print('  X range: [%.1f, %.1f]' % (data['veh_x'].min(), data['veh_x'].max()))
    print('  Y range: [%.1f, %.1f]' % (data['veh_y'].min(), data['veh_y'].max()))

    out_dir = os.path.join(os.path.dirname(framelog_path), 'map')
    os.makedirs(out_dir, exist_ok=True)

    # Compute curvature
    print('\nComputing curvature...')
    curvatures = compute_curvature(
        data['veh_x'], data['veh_y'], data['veh_yaw'])
    segments = classify_segments(curvatures)
    n_straight = sum(1 for _, _, t in segments if t == 'straight')
    n_curve = sum(1 for _, _, t in segments if t == 'curve')
    print('  Segments: %d straight, %d curve' % (n_straight, n_curve))

    # Generate plots
    print('\nGenerating plots...')
    plot_nozzle_trajectory(data, out_dir)
    plot_paint_coverage(data, out_dir)
    plot_paint_state_map(data, out_dir)
    plot_deviation_heatmap(data, out_dir)
    plot_segmented_evaluation(data, curvatures, segments, out_dir)
    plot_controller_response(data, out_dir)
    plot_speed_curvature(data, curvatures, out_dir)

    print('\nDone! Map plot saved to %s/' % out_dir)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_latest_framelog()
    if path is None:
        print('No framelog found. Usage: python evaluation/visualize_map.py [framelog.csv]')
        sys.exit(1)
    generate_all(path)
