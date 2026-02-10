"""Visualize AutoStripe trajectory evaluation results.

Usage:
    python evaluation/visualize_eval.py evaluation/eval_20260211_003457_4_summary.csv
    python evaluation/visualize_eval.py   # auto-detect latest eval
"""

import csv
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def load_summary(path):
    """Load summary CSV into dict."""
    metrics = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) == 2:
                key, val = row
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val
    return metrics


def load_detail(path):
    """Load detail CSV into numpy arrays. Supports both 3-col and 8-col formats.

    Returns:
        (xs, ys, dists) for 3-col format, or
        dict with all columns for 8-col format.
        Always returns (xs, ys, dists) as first 3 values for backwards compat.
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(header) >= 8:
        # New 8-column format
        data = {h: [] for h in header}
        for row in rows:
            if len(row) >= 8:
                for i, h in enumerate(header):
                    try:
                        data[h].append(float(row[i]))
                    except ValueError:
                        data[h].append(row[i])
        for h in header:
            try:
                data[h] = np.array(data[h], dtype=float)
            except (ValueError, TypeError):
                data[h] = np.array(data[h])
        return data
    else:
        # Legacy 3-column format
        xs, ys, dists = [], [], []
        for row in rows:
            if len(row) >= 3:
                xs.append(float(row[0]))
                ys.append(float(row[1]))
                dists.append(float(row[2]))
        return np.array(xs), np.array(ys), np.array(dists)


def find_latest_eval(eval_dir):
    """Find the latest summary CSV in evaluation directory."""
    pattern = os.path.join(eval_dir, "eval_*_summary.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No evaluation files found.")
        sys.exit(1)
    return files[-1]


def find_all_summaries(eval_dir):
    """Find all summary CSVs in evaluation directory."""
    pattern = os.path.join(eval_dir, "eval_*_summary.csv")
    return sorted(glob.glob(pattern))


def plot_evaluation(summary_path):
    """Main visualization: 5-panel figure for one evaluation."""
    detail_path = summary_path.replace("_summary.csv", "_detail.csv")
    if not os.path.exists(detail_path):
        print(f"Detail file not found: {detail_path}")
        return

    metrics = load_summary(summary_path)
    detail = load_detail(detail_path)

    # Handle both old (tuple) and new (dict) formats
    if isinstance(detail, dict):
        px = detail['paint_x']
        py = detail['paint_y']
        dists = detail['nearest_gt_dist']
    else:
        px, py, dists = detail

    n = len(px)

    # Identify in-range vs out-of-range points (threshold: 5m)
    RANGE_THRESH = 5.0
    in_range = dists < RANGE_THRESH

    eval_name = os.path.basename(summary_path).replace("_summary.csv", "")

    # --- Figure setup ---
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"AutoStripe Evaluation: {eval_name}", fontsize=14, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # =============================================
    # Panel 1: XY trajectory (color = deviation)
    # =============================================
    ax1 = fig.add_subplot(gs[0, 0:2])
    plot_xy_trajectory(ax1, px, py, dists, in_range, RANGE_THRESH)

    # =============================================
    # Panel 2: Summary metrics text
    # =============================================
    ax2 = fig.add_subplot(gs[0, 2])
    plot_summary_text(ax2, metrics, n, in_range, dists)

    # =============================================
    # Panel 3: Deviation along trail
    # =============================================
    ax3 = fig.add_subplot(gs[1, 0])
    plot_deviation_along_trail(ax3, dists, in_range, RANGE_THRESH)

    # =============================================
    # Panel 4: Deviation histogram (in-range only)
    # =============================================
    ax4 = fig.add_subplot(gs[1, 1])
    plot_deviation_histogram(ax4, dists, in_range)

    # =============================================
    # Panel 5: Cumulative deviation distribution
    # =============================================
    ax5 = fig.add_subplot(gs[1, 2])
    plot_cumulative_distribution(ax5, dists, in_range)

    out_path = summary_path.replace("_summary.csv", "_viz.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Auto-detect new 8-col format and generate curvature-deviation plot
    if isinstance(detail, dict) and 'local_curvature' in detail:
        plot_curvature_vs_deviation(detail_path)


def plot_xy_trajectory(ax, px, py, dists, in_range, thresh):
    """Panel 1: XY scatter colored by deviation."""
    # Clamp for colormap
    dists_clamped = np.clip(dists, 0, thresh)
    norm = Normalize(vmin=0, vmax=thresh)

    # Out-of-range points in gray
    if np.any(~in_range):
        ax.scatter(px[~in_range], py[~in_range],
                   c='lightgray', s=2, alpha=0.3, label='Out of GT range')

    # In-range points colored by deviation
    sc = ax.scatter(px[in_range], py[in_range],
                    c=dists_clamped[in_range], cmap='RdYlGn_r',
                    norm=norm, s=4, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Paint Trail (color = deviation from GT)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Deviation (m)')


def plot_summary_text(ax, metrics, n_total, in_range, dists):
    """Panel 2: Summary metrics as text."""
    ax.axis('off')

    n_in = int(np.sum(in_range))
    dists_in = dists[in_range]

    lines = [
        ("OVERALL METRICS", None),
        ("Paint points", f"{int(metrics.get('num_paint_points', 0))}"),
        ("GT points", f"{int(metrics.get('num_gt_points', 0))}"),
        ("Coverage", f"{metrics.get('coverage', 0)*100:.1f}%"),
        ("Curvature var", f"{metrics.get('curvature_variance', 0):.6f}"),
        ("", None),
        ("IN-RANGE METRICS (<5m)", None),
        ("Points in range", f"{n_in} / {n_total}"),
        ("Mean deviation", f"{np.mean(dists_in):.3f} m" if n_in else "N/A"),
        ("Median deviation", f"{np.median(dists_in):.3f} m" if n_in else "N/A"),
        ("Std deviation", f"{np.std(dists_in):.3f} m" if n_in else "N/A"),
        ("Max deviation", f"{np.max(dists_in):.3f} m" if n_in else "N/A"),
        ("Min deviation", f"{np.min(dists_in):.3f} m" if n_in else "N/A"),
        ("", None),
        ("RAW METRICS (all points)", None),
        ("Mean deviation", f"{metrics.get('mean_deviation', 0):.3f} m"),
        ("Median deviation", f"{metrics.get('median_deviation', 0):.3f} m"),
        ("Max deviation", f"{metrics.get('max_deviation', 0):.3f} m"),
    ]

    y = 0.95
    for label, value in lines:
        if value is None and label:
            ax.text(0.05, y, label, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top',
                    fontfamily='monospace')
        elif label == "":
            pass  # spacer
        else:
            ax.text(0.05, y, f"{label}:", transform=ax.transAxes,
                    fontsize=9, va='top', fontfamily='monospace')
            ax.text(0.95, y, value, transform=ax.transAxes,
                    fontsize=9, va='top', ha='right', fontfamily='monospace',
                    color='#2166ac')
        y -= 0.055


def plot_deviation_along_trail(ax, dists, in_range, thresh):
    """Panel 3: Deviation vs point index."""
    idx = np.arange(len(dists))

    ax.scatter(idx[in_range], dists[in_range],
               c='#2ca02c', s=1, alpha=0.5, label=f'In range (<{thresh}m)')
    if np.any(~in_range):
        ax.scatter(idx[~in_range], dists[~in_range],
                   c='#d62728', s=1, alpha=0.3, label=f'Out of range')

    ax.axhline(y=thresh, color='orange', linestyle='--', linewidth=1,
               label=f'Threshold ({thresh}m)')
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Deviation (m)')
    ax.set_title('Deviation Along Trail')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)


def plot_deviation_histogram(ax, dists, in_range):
    """Panel 4: Histogram of in-range deviations."""
    dists_in = dists[in_range]
    if len(dists_in) == 0:
        ax.text(0.5, 0.5, 'No in-range points', transform=ax.transAxes,
                ha='center', va='center')
        return

    ax.hist(dists_in, bins=50, color='#2ca02c', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(dists_in), color='red', linestyle='--',
               label=f'Mean: {np.mean(dists_in):.3f}m')
    ax.axvline(np.median(dists_in), color='blue', linestyle='--',
               label=f'Median: {np.median(dists_in):.3f}m')
    ax.set_xlabel('Deviation (m)')
    ax.set_ylabel('Count')
    ax.set_title('Deviation Distribution (in-range)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_cumulative_distribution(ax, dists, in_range):
    """Panel 5: CDF of in-range deviations."""
    dists_in = np.sort(dists[in_range])
    if len(dists_in) == 0:
        ax.text(0.5, 0.5, 'No in-range points', transform=ax.transAxes,
                ha='center', va='center')
        return

    cdf = np.arange(1, len(dists_in) + 1) / len(dists_in)
    ax.plot(dists_in, cdf, color='#2166ac', linewidth=2)

    # Mark percentiles
    for pct in [50, 90, 95]:
        val = np.percentile(dists_in, pct)
        ax.axhline(pct / 100, color='gray', linestyle=':', linewidth=0.5)
        ax.axvline(val, color='gray', linestyle=':', linewidth=0.5)
        ax.annotate(f'P{pct}: {val:.2f}m',
                    xy=(val, pct / 100), fontsize=7,
                    xytext=(5, -5), textcoords='offset points')

    ax.set_xlabel('Deviation (m)')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('CDF of Deviation (in-range)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)


def load_framelog(path):
    """Load framelog CSV into dict of numpy arrays.

    Args:
        path: path to framelog_*.csv

    Returns:
        dict mapping column name -> numpy array (float where possible)
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    data = {h: [] for h in header}
    for row in rows:
        for i, h in enumerate(header):
            if i < len(row):
                data[h].append(row[i])
            else:
                data[h].append('')

    # Convert to numpy, float where possible
    for h in header:
        try:
            data[h] = np.array(data[h], dtype=float)
        except (ValueError, TypeError):
            data[h] = np.array(data[h])
    return data


def plot_timeseries(framelog_path):
    """Generate timeseries figure from framelog CSV.

    6 subplots for legacy framelogs, 8 subplots when perception metrics exist.

    Subplots:
        1. nozzle_edge_dist + poly_edge_dist vs frame (with paint_state color band)
        2. driving_offset vs frame
        3. steer_cmd + steer_filter vs frame
        4. speed vs frame
        5. lateral_error vs frame
        6. inference_time_ms vs frame
        7. mask_iou vs frame (if available)
        8. edge_dev_mean_px + edge_dev_max_px vs frame (if available)
    """
    data = load_framelog(framelog_path)
    frames = data['frame']

    has_perception = 'mask_iou' in data and 'edge_dev_mean_px' in data
    n_panels = 8 if has_perception else 6
    fig_h = 24 if has_perception else 18

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, fig_h), sharex=True)
    fig.suptitle(f"Timeseries: {os.path.basename(framelog_path)}",
                 fontsize=13, fontweight='bold')

    # --- Panel 1: distances + paint state color band ---
    ax = axes[0]
    _draw_paint_state_bands(ax, data)
    ax.plot(frames, data['nozzle_edge_dist'],
            color='green', linewidth=1, label='Nozzle-Edge')
    poly = data['poly_edge_dist']
    valid = poly > 0
    if np.any(valid):
        ax.plot(frames[valid], poly[valid],
                color='magenta', linewidth=1, label='Poly-Edge')
    ax.axhline(3.0, color='gray', linestyle='--', linewidth=0.8, label='Target 3m')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Nozzle-Edge & Poly-Edge Distance')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: driving offset ---
    ax = axes[1]
    ax.plot(frames, data['driving_offset'], color='#2166ac', linewidth=1)
    ax.set_ylabel('Offset (m)')
    ax.set_title('Driving Offset')
    ax.grid(True, alpha=0.3)

    # --- Panel 3: steer cmd + filter ---
    ax = axes[2]
    ax.plot(frames, data['steer_cmd'],
            color='orange', linewidth=1, label='Steer Cmd')
    ax.plot(frames, data['steer_filter'],
            color='blue', linewidth=1, alpha=0.7, label='Steer Filter')
    ax.set_ylabel('Value')
    ax.set_title('Steering Command & Filter')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Panel 4: speed ---
    ax = axes[3]
    ax.plot(frames, data['speed'], color='#2ca02c', linewidth=1)
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Vehicle Speed')
    ax.grid(True, alpha=0.3)

    # --- Panel 5: lateral error ---
    ax = axes[4]
    ax.plot(frames, data['lateral_error'], color='#d62728', linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Error (m)')
    ax.set_title('Lateral Error')
    ax.grid(True, alpha=0.3)

    # --- Panel 6: inference time ---
    ax = axes[5]
    inf_ms = data['inference_time_ms']
    valid_inf = inf_ms >= 0
    if np.any(valid_inf):
        ax.plot(frames[valid_inf], inf_ms[valid_inf],
                color='#9467bd', linewidth=1)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Inference Time')
    ax.grid(True, alpha=0.3)

    # --- Panel 7 & 8: perception metrics (only if columns exist) ---
    if has_perception:
        # Panel 7: mask IoU
        ax = axes[6]
        iou = data['mask_iou']
        valid_iou = iou >= 0
        if np.any(valid_iou):
            ax.plot(frames[valid_iou], iou[valid_iou],
                    color='#1f77b4', linewidth=1, label='Mask IoU')
            mean_iou = float(np.mean(iou[valid_iou]))
            ax.axhline(mean_iou, color='gray', linestyle='--',
                       linewidth=0.8, label=f'Mean: {mean_iou:.3f}')
        ax.set_ylabel('IoU')
        ax.set_title('Perception: Mask IoU (AI vs GT)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

        # Panel 8: edge deviation
        ax = axes[7]
        edge_mean = data['edge_dev_mean_px']
        edge_max = data['edge_dev_max_px']
        valid_edge = edge_mean >= 0
        if np.any(valid_edge):
            ax.plot(frames[valid_edge], edge_mean[valid_edge],
                    color='#ff7f0e', linewidth=1, label='Mean dev (px)')
            ax.plot(frames[valid_edge], edge_max[valid_edge],
                    color='#d62728', linewidth=1, alpha=0.6,
                    label='Max dev (px)')
        ax.set_ylabel('Deviation (px)')
        ax.set_title('Perception: Right Edge Deviation (AI vs GT)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Set xlabel on the last panel
    axes[-1].set_xlabel('Frame')

    plt.tight_layout()
    out_path = framelog_path.replace('.csv', '_timeseries.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def _draw_paint_state_bands(ax, data):
    """Draw colored background bands for paint_state on an axis."""
    frames = data['frame']
    states = data.get('paint_state')
    if states is None or len(states) == 0:
        return

    state_colors = {
        'CONVERGING': '#ffcccc',
        'STABILIZED': '#ffffcc',
        'PAINTING': '#ccffcc',
    }

    prev_state = str(states[0])
    start_frame = frames[0]
    for i in range(1, len(frames)):
        cur = str(states[i])
        if cur != prev_state or i == len(frames) - 1:
            color = state_colors.get(prev_state, '#f0f0f0')
            ax.axvspan(start_frame, frames[i], alpha=0.2, color=color)
            prev_state = cur
            start_frame = frames[i]


def plot_curvature_vs_deviation(detail_path):
    """Scatter plot: local curvature vs deviation (8-col detail CSV)."""
    detail = load_detail(detail_path)
    if not isinstance(detail, dict):
        print("  Curvature plot requires 8-column detail CSV, skipping.")
        return
    if 'local_curvature' not in detail:
        return

    curv = detail['local_curvature']
    dists = detail['nearest_gt_dist']
    in_range = detail.get('in_range', dists < 5.0)
    if isinstance(in_range[0], float):
        in_range = in_range > 0.5

    mask = in_range & (curv > 0)
    if np.sum(mask) < 5:
        print("  Not enough in-range points with curvature > 0, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(curv[mask], dists[mask],
                    c=dists[mask], cmap='RdYlGn_r',
                    s=4, alpha=0.6)
    ax.set_xlabel('Local Curvature (1/m)')
    ax.set_ylabel('Deviation from GT (m)')
    ax.set_title('Curvature vs Deviation')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Deviation (m)')

    out_path = detail_path.replace('_detail.csv', '_curv_dev.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_multi_session_comparison(eval_dir):
    """Compare all evaluations in a single bar chart."""
    summaries = find_all_summaries(eval_dir)
    if len(summaries) < 2:
        return

    names, means, medians, coverages, counts = [], [], [], [], []
    for sp in summaries:
        m = load_summary(sp)
        dp = sp.replace("_summary.csv", "_detail.csv")
        if os.path.exists(dp):
            detail = load_detail(dp)
            if isinstance(detail, dict):
                dists = detail['nearest_gt_dist']
            else:
                _, _, dists = detail
            in_range = dists < 5.0
            dists_in = dists[in_range]
            if len(dists_in) > 0:
                names.append(os.path.basename(sp).replace("_summary.csv", "")
                             .replace("eval_", ""))
                means.append(np.mean(dists_in))
                medians.append(np.median(dists_in))
                coverages.append(m.get('coverage', 0) * 100)
                counts.append(len(dists_in))

    if len(names) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Multi-Session Comparison (in-range points only)",
                 fontsize=13, fontweight='bold')

    x = np.arange(len(names))

    # Mean deviation
    axes[0].bar(x, means, color='#2ca02c', alpha=0.8)
    axes[0].set_ylabel('Mean Deviation (m)')
    axes[0].set_title('Mean Deviation')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Median deviation
    axes[1].bar(x, medians, color='#2166ac', alpha=0.8)
    axes[1].set_ylabel('Median Deviation (m)')
    axes[1].set_title('Median Deviation')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axes[1].grid(True, alpha=0.3, axis='y')

    # In-range point count
    axes[2].bar(x, counts, color='#ff7f0e', alpha=0.8)
    axes[2].set_ylabel('In-range Points')
    axes[2].set_title('Points within 5m of GT')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(eval_dir, "comparison_all.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        summary_path = sys.argv[1]
        if not os.path.isabs(summary_path):
            summary_path = os.path.join(os.getcwd(), summary_path)
    else:
        summary_path = find_latest_eval(eval_dir)
        print(f"Auto-detected: {summary_path}")

    plot_evaluation(summary_path)
    plot_multi_session_comparison(eval_dir)

    # Auto-detect and plot latest framelog timeseries
    framelog_pattern = os.path.join(eval_dir, "framelog_*.csv")
    framelogs = sorted(glob.glob(framelog_pattern))
    if framelogs:
        latest_fl = framelogs[-1]
        print(f"Framelog detected: {latest_fl}")
        plot_timeseries(latest_fl)

    print("Done.")
