"""AutoStripe paper figure composer.

Composes screenshots into publication-ready figures:
  - weather: 2x2 LUNA-Net multi-weather front view comparison
  - mode:    1x3 GT/VLLiNet/LUNA perception mode comparison
  - overview: front + overhead + depth system panorama

Usage:
    python evaluation/image_vis/compose_figures.py --type weather
    python evaluation/image_vis/compose_figures.py --type mode --weather ClearNight
    python evaluation/image_vis/compose_figures.py --type overview --timestamp 20260227_163045
"""

import argparse
import glob
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import fontManager, FontProperties

# --- Font setup: Times New Roman (Western) + SimSun (Chinese) ---
_SIMSUN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'docs', 'Fonts', 'SIMSUN.TTC')
fontManager.addfont(_SIMSUN_PATH)
FONT_CN = FontProperties(fname=_SIMSUN_PATH, size=13)
FONT_EN = FontProperties(family='serif', size=13)

# --- Nature-style rcParams (consistent with visualize_map.py) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.unicode_minus': False,
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

# Screenshot directory (same level as this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(SCRIPT_DIR, 'snapshots')
FIGURE_DIR = os.path.join(SCRIPT_DIR, 'figures')

# Weather display order and labels
WEATHER_ORDER = ['ClearDay', 'ClearNight', 'HeavyFoggyNight', 'HeavyRainFoggyNight']
WEATHER_LABELS = {
    'ClearDay': 'Clear Day',
    'ClearNight': 'Clear Night',
    'HeavyFoggyNight': 'Heavy Foggy Night',
    'HeavyRainFoggyNight': 'Heavy Rain Foggy Night',
}

# Mode display order and labels
MODE_ORDER = ['GT', 'VLLiNet', 'LUNA']
MODE_LABELS = {'GT': 'Ground Truth', 'VLLiNet': 'VLLiNet', 'LUNA': 'LUNA-Net'}


def _add_subplot_label(ax, label, fontsize=14):
    """Add white-on-black subplot label (a)(b)(c)(d) at top-left corner."""
    ax.text(0.02, 0.95, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))


def _mixed_title(ax, parts, y=1.03):
    """Render mixed-font title centered above axes.

    Args:
        parts: [(text, FontProperties), ...] segments to concatenate.
    """
    r = ax.figure.canvas.get_renderer()
    trans = ax.transAxes
    # Measure segment widths
    tmp, widths = [], []
    for txt, fp in parts:
        t = ax.text(0, 0, txt, fontproperties=fp, transform=trans)
        widths.append(t.get_window_extent(renderer=r).width)
        tmp.append(t)
    for t in tmp:
        t.remove()
    # Place segments centered
    ax_w = ax.get_window_extent(renderer=r).width
    cur_x = (ax_w - sum(widths)) / 2.0
    for i, (txt, fp) in enumerate(parts):
        ax.text(cur_x / ax_w, y, txt, fontproperties=fp,
                transform=trans, va='bottom', ha='left')
        cur_x += widths[i]


def _find_latest(pattern):
    """Find the most recently modified file matching a glob pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_rgb(path):
    """Load image as RGB numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_figure(fig, output_dir, basename):
    """Save figure as PDF (300 DPI)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{basename}.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"  Saved: {path}")


# ============================================================
# Figure 1: Multi-weather comparison (2x2)
# ============================================================
def compose_weather(snap_dir):
    """2x2 grid: LUNA-Net front view under 4 weather conditions."""
    labels = ['(a)', '(b)', '(c)', '(d)']
    images = []

    for weather in WEATHER_ORDER:
        pattern = os.path.join(snap_dir, f"snap_LUNA_{weather}_*_front.png")
        path = _find_latest(pattern)
        if path is None:
            print(f"  WARNING: no snap found for LUNA/{weather}, skipping")
            return
        images.append((path, WEATHER_LABELS[weather]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, (path, title) in enumerate(images):
        img = _load_rgb(path)
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
        _add_subplot_label(axes[i], labels[i])

    fig.tight_layout(pad=1.0)
    _save_figure(fig, FIGURE_DIR, 'fig_weather_comparison')
    plt.close(fig)


# ============================================================
# Figure 2: Multi-mode comparison (1x3)
# ============================================================
def compose_mode(snap_dir, weather='ClearDay'):
    """1x3 grid: GT / VLLiNet / LUNA-Net front view under same weather."""
    labels = ['(a)', '(b)', '(c)']
    images = []

    for mode in MODE_ORDER:
        pattern = os.path.join(snap_dir, f"snap_{mode}_{weather}_*_front.png")
        path = _find_latest(pattern)
        if path is None:
            print(f"  WARNING: no snap found for {mode}/{weather}, skipping")
            return
        images.append((path, MODE_LABELS[mode]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for i, (path, title) in enumerate(images):
        img = _load_rgb(path)
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
        _add_subplot_label(axes[i], labels[i])

    fig.suptitle(f'Perception Mode Comparison — {WEATHER_LABELS.get(weather, weather)}',
                 fontsize=13, y=1.02)
    fig.tight_layout(pad=1.0)
    _save_figure(fig, FIGURE_DIR, f'fig_mode_comparison_{weather}')
    plt.close(fig)


# ============================================================
# Figure 3: System overview (front + overhead + depth)
# ============================================================
def compose_overview(snap_dir, timestamp):
    """Left: overhead (tall), Right: front + depth stacked."""
    prefix = f"snap_*_{timestamp}"
    front_path = _find_latest(os.path.join(snap_dir, f"{prefix}_front.png"))
    oh_path = _find_latest(os.path.join(snap_dir, f"{prefix}_overhead.png"))
    dep_path = _find_latest(os.path.join(snap_dir, f"{prefix}_depth.png"))

    if front_path is None:
        print(f"  ERROR: no front snap for timestamp {timestamp}")
        return
    if oh_path is None:
        print(f"  WARNING: no overhead snap for timestamp {timestamp}")
    if dep_path is None:
        print(f"  WARNING: no depth snap for timestamp {timestamp}")

    front_img = _load_rgb(front_path)

    # Left column: overhead (1800x1600), Right column: front + depth stacked (1248x384 each)
    # Figure height tuned so right-column images fill vertically
    fig = plt.figure(figsize=(14, 5.5))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.8, 1.0],
                  height_ratios=[1, 1], hspace=0.25, wspace=0.02)

    # Left: overhead spans both rows
    ax_oh = fig.add_subplot(gs[:, 0])
    if oh_path is not None:
        oh_img = _load_rgb(oh_path)
        ax_oh.imshow(oh_img)
    ax_oh.axis('off')

    # Right-top: front view
    ax_front = fig.add_subplot(gs[0, 1])
    ax_front.imshow(front_img)
    ax_front.axis('off')

    # Right-bottom: depth colormap
    ax_dep = fig.add_subplot(gs[1, 1])
    if dep_path is not None:
        dep_img = _load_rgb(dep_path)
        ax_dep.imshow(dep_img)
    ax_dep.axis('off')

    # Mixed-font titles: Times for (a)(b)(c), SimHei for Chinese
    fig.canvas.draw()
    _mixed_title(ax_oh,    [('(a) ', FONT_EN), ('俯视图', FONT_CN)])
    _mixed_title(ax_front, [('(b) ', FONT_EN), ('前视图', FONT_CN)])
    _mixed_title(ax_dep,   [('(c) ', FONT_EN), ('深度图', FONT_CN)])
    _save_figure(fig, FIGURE_DIR, f'fig_system_overview_{timestamp}')
    plt.close(fig)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='AutoStripe paper figure composer')
    parser.add_argument('--type', required=True,
                        choices=['weather', 'mode', 'overview'],
                        help='Figure type to compose')
    parser.add_argument('--dir', default=DEFAULT_DIR,
                        help='Screenshot directory (default: evaluation/image_vis/)')
    parser.add_argument('--weather', default='ClearDay',
                        help='Weather condition for mode comparison')
    parser.add_argument('--timestamp', default=None,
                        help='Timestamp for overview (e.g. 20260227_163045)')
    args = parser.parse_args()

    snap_dir = os.path.abspath(args.dir)
    print(f"  Snap dir: {snap_dir}")
    print(f"  Output:   {FIGURE_DIR}")

    if args.type == 'weather':
        compose_weather(snap_dir)
    elif args.type == 'mode':
        compose_mode(snap_dir, weather=args.weather)
    elif args.type == 'overview':
        if args.timestamp is None:
            print("  ERROR: --timestamp required for overview")
            sys.exit(1)
        compose_overview(snap_dir, args.timestamp)


if __name__ == '__main__':
    main()
