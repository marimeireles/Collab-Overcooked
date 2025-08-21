# Modern Cross-Play Analysis with Prettier Graphs
# Improved version of the original NumPy + matplotlib analysis
# Per-level composite figure with 3 panels:
#   (1) Self vs Cross — mean success_rate ± SEM by model size (grouped bars)
#   (2) Cross-play of Chefs — x=Assistant size, series=Chef size (grouped bars) + Self-play
#   (3) Cross-play of Assistants — x=Chef size, series=Assistant size (grouped bars) + Self-play
#
# Role mapping (agreed):
#   agent_0 = Chef (first model in 'model' string)
#   agent_1 = Assistant (second model in 'model' string)

import os
import numpy as np
import csv
import re
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib import colors as mcolors
from collections import defaultdict

# ==================== MODERN CONFIGURATION ====================
INPUT_CANDIDATES = ["statistics_data_final-data.csv"]
OUT_DIR = "./plots_modern"
VERBOSE_SUMMARY = True
SHOW_IN_NOTEBOOK = True

# ULTRA-MODERN COLOR PALETTE - Professional, accessible, and visually striking
MODERN_COLORS = {
    "self": "#667EEA",       # Modern gradient blue
    "cross": "#F093FB",      # Modern gradient pink
    "selfplay": "#4FACFE",   # Bright modern blue 
    "chef_primary": "#667EEA",     # Gradient blue
    "assistant_primary": "#F093FB", # Gradient pink
    "accent": "#FEC163",     # Warm gradient yellow
    "success": "#06D6A0",    # Modern teal
    "neutral": "#8B9DC3",    # Sophisticated blue-gray
    "background": "#FAFBFE", # Ultra-light background
    "text_primary": "#1A202C", # Rich dark text
    "text_secondary": "#4A5568", # Medium gray text
}

# Modern gradient color palettes with better visual hierarchy
CHEF_COLORS = [
    "#E8F2FF", "#C3DBFF", "#97C4FF", "#6BADFF", 
    "#3B96FF", "#0B7FFF", "#0066D9", "#004DB3"
]
ASSISTANT_COLORS = [
    "#FFF0F8", "#FFD6F0", "#FFBCE8", "#FFA2E0", 
    "#FF88D8", "#FF6ED0", "#E854C8", "#D13AC0"
]

# Ultra-modern gradient colors for special effects
GRADIENT_COLORS = {
    "blue_gradient": ["#667EEA", "#764BA2"],
    "pink_gradient": ["#F093FB", "#F5576C"], 
    "teal_gradient": ["#06D6A0", "#118AB2"],
    "purple_gradient": ["#9B59B6", "#8E44AD"]
}

# Configuration flags
ANNOTATE_COUNTS = False  # Remove n= annotations
TOP_PAIRS_COUNT = 12
INCLUDE_SELF_IN_TOP = True

# ULTRA-MODERN TYPOGRAPHY AND STYLING
plt.rcParams.update({
    # High quality rendering
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "figure.facecolor": "#FAFBFE",
    "savefig.facecolor": "#FAFBFE",
    
    # Modern typography stack
    "font.family": "sans-serif",
    "font.sans-serif": ["SF Pro Display", "Inter", "Segoe UI", "system-ui", "Roboto", "Helvetica Neue", "Arial"],
    
    # Enhanced text hierarchy
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.titlepad": 25,
    "axes.titlecolor": "#1A202C",
    "axes.labelsize": 14,
    "axes.labelweight": "600",
    "axes.labelpad": 12,
    "axes.labelcolor": "#2D3748",
    
    # Refined tick labels
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.color": "#4A5568",
    "ytick.color": "#4A5568",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    
    # Modern legend styling
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "legend.framealpha": 0.98,
    "legend.fancybox": True,
    "legend.shadow": False,
    "legend.borderpad": 0.8,
    "legend.columnspacing": 1.2,
    "legend.handlelength": 1.8,
    
    # Subtle, clean axes
    "axes.edgecolor": "#CBD5E0",
    "axes.linewidth": 1.5,
    "axes.axisbelow": True,
    
    # Minimal grid
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.6,
    "grid.linestyle": "-",
    
    # Modern spacing
    "figure.subplot.hspace": 0.3,
    "figure.subplot.wspace": 0.2,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.pad": 0.08,
})

# ==================== HELPER FUNCTIONS ====================
def find_path(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find CSV at any of: {cands}")

def parse_pair(model_str):
    parts = model_str.split("_")
    if len(parts) >= 4:
        a = parts[0] + "_" + parts[1]  # agent_0 (Chef)
        b = parts[2] + "_" + parts[3]  # agent_1 (Assistant)
    else:
        mid = len(parts) // 2
        a = "_".join(parts[:mid])
        b = "_".join(parts[mid:])
    return a, b

_size_regex = re.compile(r'(\d+(?:\.\d+)?)\s*[Bb]\b')

def extract_size(label):
    toks = label.split("-")
    for tok in reversed(toks):
        tok = tok.strip()
        if tok and (tok[-1] in "Bb"):
            m = _size_regex.match(tok)
            if m:
                num = m.group(1)
                return f"{num}B"
    m = _size_regex.search(label)
    if m:
        return f"{m.group(1)}B"
    return None

def sem(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= 1:
        return float("nan")
    return float(np.std(x, ddof=1) / np.sqrt(n))

def size_key(sz):
    try:
        return float(sz[:-1])
    except Exception:
        return float("inf")

def polish_axes_modern(ax, add_glow=True):
    """Apply ultra-modern styling to axes with optional glow effects"""
    # Ultra-subtle grid with modern styling
    ax.grid(True, axis="y", linestyle="-", linewidth=1.0, alpha=0.4, color="#E2E8F0", zorder=0)
    ax.set_axisbelow(True)
    
    # Modern background with subtle gradient effect
    ax.set_facecolor(MODERN_COLORS["background"])
    
    # Clean, minimal spines - completely remove top and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Style remaining spines with modern colors
    ax.spines['left'].set_color("#CBD5E0")
    ax.spines['bottom'].set_color("#CBD5E0")
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['bottom'].set_linewidth(1.8)
    
    # Add subtle drop shadow effect to the plot area
    if add_glow:
        # Create a subtle inner shadow effect
        shadow = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                 facecolor='none', edgecolor='#E2E8F0',
                                 linewidth=0.5, alpha=0.3, zorder=-1)
        ax.add_patch(shadow)
    
    # Modern tick styling
    ax.tick_params(axis='both', which='major', 
                   labelsize=12, colors=MODERN_COLORS["text_secondary"],
                   length=6, width=1.2, pad=8)
    ax.tick_params(axis='both', which='minor', length=3, width=0.8)
    
    # Ensure proper z-ordering
    ax.set_axisbelow(True)

def format_as_percent_modern(ax, ymax_est):
    """Modern percentage formatting"""
    if np.isfinite(ymax_est) and ymax_est <= 1.05:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
        ax.set_ylim(0, max(1.0, ymax_est * 1.05))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))

def sanitize_token(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))

def show_and_save_modern(fig, path):
    """Enhanced save with high quality"""
    if SHOW_IN_NOTEBOOK:
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            try:
                plt.show()
            except Exception:
                pass
    
    fig.savefig(path, bbox_inches="tight", facecolor='white', 
                edgecolor='none', dpi=300, format='png')
    plt.close(fig)
    print(f"✨ Saved -> {path}")

def collect_mean_sem_n(vals):
    vals = np.asarray(vals, dtype=float)
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), sem(vals), n

def get_color_from_palette(palette, index, total):
    """Get color from palette with proper indexing"""
    if total <= 1:
        return palette[0] if palette else MODERN_COLORS["neutral"]
    
    # Map index to palette range
    palette_index = int(index * (len(palette) - 1) / max(1, total - 1))
    return palette[min(palette_index, len(palette) - 1)]

def create_gradient_color(base_color, intensity=1.0, lighten=False):
    """Create a gradient-like color variation"""
    try:
        rgb = mcolors.to_rgb(base_color)
        if lighten:
            # Lighten the color
            rgb = [min(1.0, c + (1 - c) * (1 - intensity) * 0.3) for c in rgb]
        else:
            # Darken the color slightly for depth
            rgb = [c * (0.8 + 0.2 * intensity) for c in rgb]
        return mcolors.to_hex(rgb)
    except:
        return base_color

def add_subtle_glow_effect(ax, bars, glow_color=None, intensity=0.1):
    """Add a subtle glow effect behind bars"""
    if not glow_color:
        glow_color = "#E2E8F0"
    
    for bar in bars:
        # Create a slightly larger, lighter bar behind the main bar
        x = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()
        
        if np.isfinite(height) and height > 0:
            glow = patches.Rectangle((x - width*0.02, 0), width*1.04, height,
                                   facecolor=glow_color, alpha=intensity,
                                   zorder=bar.get_zorder()-1)
            ax.add_patch(glow)

def plot_modern_grouped_bars(ax, x_labels, series_keys, mean_map, sem_map, n_map,
                            label_prefix, error_kw):
    """Modern grouped bar plotting with enhanced styling"""
    x = np.arange(len(x_labels), dtype=float)
    k = max(1, len(series_keys))
    width = min(0.85 / k, 0.25)  # Slightly narrower for cleaner look
    
    # Always use blue palette for cross-play bars
    color_palette = CHEF_COLORS
    
    ymax_est = 0.0
    bars_collection = []
    
    for i, sk in enumerate(series_keys):
        heights = [mean_map[sk].get(lbl, float("nan")) for lbl in x_labels]
        errors = [sem_map[sk].get(lbl, float("nan")) for lbl in x_labels]
        counts = [n_map[sk].get(lbl, 0) for lbl in x_labels]
        
        # Bar positions
        pos = x + (i - (k - 1) / 2.0) * width
        
        # Choose color with modern gradient approach
        if sk == "Self-play":
            bar_color = MODERN_COLORS["cross"]  # Modern gradient pink for self-play
            edge_color = create_gradient_color(bar_color, intensity=0.8)
        else:
            base_color = get_color_from_palette(CHEF_COLORS, i, k)
            bar_color = create_gradient_color(base_color, intensity=0.95, lighten=True)
            edge_color = create_gradient_color(base_color, intensity=0.7)
        
        # Create bars with ultra-modern styling
        bars = ax.bar(pos, heights, width,
                     yerr=errors,
                     label=f"{label_prefix} {sk}",
                     color=bar_color,
                     edgecolor=edge_color,
                     linewidth=2.0,
                     alpha=0.92,
                     error_kw=error_kw,
                     zorder=3)
        
        # Add subtle glow effect for enhanced visual appeal
        add_subtle_glow_effect(ax, bars, glow_color=bar_color, intensity=0.08)
        
        bars_collection.extend(bars)
        
        # Track maximum y for formatting
        for h, e in zip(heights, errors):
            if np.isfinite(h):
                err_val = e if np.isfinite(e) else 0
                ymax_est = max(ymax_est, h + err_val)
        
        # Modern annotations
        if ANNOTATE_COUNTS:
            for rect, h, n, e in zip(bars, heights, counts, errors):
                if not np.isfinite(h) or n == 0:
                    continue
                
                # Calculate annotation position
                err_height = e if np.isfinite(e) else 0
                y_pos = h + err_height + ymax_est * 0.015
                
                # Styled annotation
                ax.annotate(f"n={n}", 
                           xy=(rect.get_x() + rect.get_width()/2, y_pos),
                           ha="center", va="bottom", fontsize=9,
                           color="#374151", weight="medium",
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor="white", edgecolor="#E5E7EB",
                                   alpha=0.9, linewidth=0.5))
    
    ax.set_xticks(x, x_labels)
    return ymax_est, bars_collection

# ==================== DATA LOADING ====================
print("🎨 Loading data for modern cross-play analysis...")

path = find_path(INPUT_CANDIDATES)
buckets_by_size = defaultdict(list)
cross_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
chef_sizes_by_level = defaultdict(set)
assistant_sizes_by_level = defaultdict(set)
levels_seen = set()
sizes_seen_by_level = defaultdict(set)

with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            sr = float(row["success_rate"])
            model_str = row["model"]
            level = row["level"]
        except (KeyError, ValueError):
            continue
        if not level:
            continue

        a, b = parse_pair(model_str)
        sa = extract_size(a)  # agent_0 (Chef)
        sb = extract_size(b)  # agent_1 (Assistant)
        if sa is None or sb is None:
            continue

        levels_seen.add(level)
        sizes_seen_by_level[level].update([sa, sb])

        # Self vs Cross classification
        if sa == sb:
            buckets_by_size[(level, sa, "self")].append(sr)
        else:
            buckets_by_size[(level, sa, "cross")].append(sr)
            buckets_by_size[(level, sb, "cross")].append(sr)
            cross_matrix[level][sa][sb].append(sr)
            chef_sizes_by_level[level].add(sa)
            assistant_sizes_by_level[level].add(sb)

# Sort levels
def level_key(x):
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))

levels_sorted = sorted(levels_seen, key=level_key)
print(f"✅ Loaded data for {len(levels_sorted)} levels: {levels_sorted}")

# ==================== VISUALIZATION GENERATION ====================
os.makedirs(OUT_DIR, exist_ok=True)

# Ultra-modern error bar styling with subtle aesthetics
error_kw_modern = dict(
    ecolor="#6B7280", 
    linewidth=2.0, 
    capsize=6, 
    capthick=2.0, 
    alpha=0.7,
    zorder=10
)

# Enhanced error bar styling for different contexts
error_kw_elegant = dict(
    ecolor=MODERN_COLORS["text_secondary"], 
    linewidth=1.8, 
    capsize=5, 
    capthick=1.8, 
    alpha=0.6,
    zorder=10
)

print("🎨 Generating modern visualizations...")

for lvl in levels_sorted:
    print(f"\n📈 Creating plots for Level {lvl}")
    
    sizes_sorted = sorted(sizes_seen_by_level[lvl], key=size_key)
    chef_sizes = sorted(chef_sizes_by_level[lvl], key=size_key)
    assistant_sizes = sorted(assistant_sizes_by_level[lvl], key=size_key)

    # ==================== TOP PAIRS ANALYSIS ====================
    # Build top pairs data
    pairs = []
    
    # Cross-play pairs
    for csz in chef_sizes:
        for asz in assistant_sizes:
            data = cross_matrix[lvl][csz].get(asz, [])
            if data:
                m, s, n = collect_mean_sem_n(data)
                if np.isfinite(m) and n > 0:
                    pairs.append((m, math.log(n + 1.0), n, csz, asz))
    
    # Optional self-play pairs
    if INCLUDE_SELF_IN_TOP:
        for sz in sizes_sorted:
            data = buckets_by_size.get((lvl, sz, "self"), [])
            if data:
                m, s, n = collect_mean_sem_n(data)
                if np.isfinite(m) and n > 0:
                    pairs.append((m, math.log(n + 1.0), n, sz, sz))

    if pairs:
        pairs.sort(key=lambda t: (t[0], t[1]), reverse=True)
        top = pairs[:TOP_PAIRS_COUNT]

        labels = [f"{cm} → {am}" for (_, _, _, cm, am) in top]
        means = [m for (m, _, _, _, _) in top]
        counts = [n for (_, _, n, _, _) in top]

        # Create ultra-modern top pairs figure
        fig_width = max(16, 0.8 * len(top) + 10)
        fig_tp = plt.figure(figsize=(fig_width, 9))
        fig_tp.patch.set_facecolor(MODERN_COLORS["background"])
        ax_tp = fig_tp.add_subplot(111)

        x = np.arange(len(top))
        
        # Create beautiful bars
        bars = []
        for i, (mean_val, count) in enumerate(zip(means, counts)):
            chef_size, asst_size = top[i][3], top[i][4]
            
            if chef_size == asst_size:  # Self-play
                bar_color = MODERN_COLORS["cross"]  # Modern gradient pink for self-play
                edge_color = create_gradient_color(bar_color, intensity=0.8)
            else:  # Cross-play
                # Use gradient blues for cross-play with performance-based intensity
                intensity = mean_val / max(means) if max(means) > 0 else 0
                base_color = MODERN_COLORS["chef_primary"] 
                bar_color = create_gradient_color(base_color, intensity=intensity, lighten=True)
                edge_color = create_gradient_color(base_color, intensity=0.7)
            
            bar = ax_tp.bar(x[i], mean_val, 
                           color=bar_color,
                           edgecolor=edge_color,
                           linewidth=2.5,
                           alpha=0.93,
                           zorder=3)
            bars.extend(bar)
            
            # Add glow effect to top pairs
            add_subtle_glow_effect(ax_tp, [bar[0]], glow_color=bar_color, intensity=0.12)
        
        # Modern styling
        ax_tp.set_xticks(x, labels, rotation=45, ha="right")
        ax_tp.set_ylabel("Mean Success Rate", fontweight="medium")
        ax_tp.set_title(f"🏆 Level {lvl} — Top Performing Model Pairs", 
                       fontsize=18, fontweight="bold", pad=25)
        
        polish_axes_modern(ax_tp)
        format_as_percent_modern(ax_tp, float(max(means)) if means else float("nan"))
        
        # Enhanced annotations
        if ANNOTATE_COUNTS:
            ymin, ymax = ax_tp.get_ylim()
            yrange = ymax - ymin
            for bar, n in zip(bars, counts):
                h = bar.get_height()
                if np.isfinite(h):
                    ax_tp.annotate(f"n={n}", 
                                  xy=(bar.get_x() + bar.get_width()/2, h + 0.02*yrange),
                                  ha="center", va="bottom", fontsize=10,
                                  color="#374151", weight="medium",
                                  bbox=dict(boxstyle="round,pad=0.3", 
                                          facecolor="white", edgecolor="#E5E7EB",
                                          alpha=0.95))

        plt.tight_layout()
        top_pairs_png = os.path.join(OUT_DIR, f"level_{sanitize_token(lvl)}_top_pairs_modern.png")
        show_and_save_modern(fig_tp, top_pairs_png)

    # ==================== SELF VS CROSS DATA PREPARATION ====================
    means_self_all, sems_self_all, n_self_all = [], [], []
    means_cross_all, sems_cross_all, n_cross_all = [], [], []
    
    for sz in sizes_sorted:
        xs = np.array(buckets_by_size.get((lvl, sz, "self"), []), dtype=float)
        xc = np.array(buckets_by_size.get((lvl, sz, "cross"), []), dtype=float)

        m_s, s_s, n_s = collect_mean_sem_n(xs)
        m_c, s_c, n_c = collect_mean_sem_n(xc)

        means_self_all.append(m_s)
        sems_self_all.append(s_s)
        n_self_all.append(n_s)
        means_cross_all.append(m_c)
        sems_cross_all.append(s_c)
        n_cross_all.append(n_c)

    # ==================== ROLE-AWARE MAPS ====================
    # Chef maps (series=Chef size, x=Assistant size)
    means_Chef, sems_Chef, ns_Chef = {}, {}, {}
    for csz in chef_sizes:
        means_Chef[csz], sems_Chef[csz], ns_Chef[csz] = {}, {}, {}
        for asz in assistant_sizes:
            m, s, n = collect_mean_sem_n(cross_matrix[lvl][csz].get(asz, []))
            means_Chef[csz][asz] = m
            sems_Chef[csz][asz] = s
            ns_Chef[csz][asz] = n

    # Add Self-play to Chef maps
    SELF_KEY = "Self-play"
    means_Chef[SELF_KEY], sems_Chef[SELF_KEY], ns_Chef[SELF_KEY] = {}, {}, {}
    for asz in assistant_sizes:
        m, s, n = collect_mean_sem_n(buckets_by_size.get((lvl, asz, "self"), []))
        means_Chef[SELF_KEY][asz] = m
        sems_Chef[SELF_KEY][asz] = s
        ns_Chef[SELF_KEY][asz] = n
    series_Chef = chef_sizes + [SELF_KEY]

    # Assistant maps (series=Assistant size, x=Chef size)
    means_Asst, sems_Asst, ns_Asst = {}, {}, {}
    for asz in assistant_sizes:
        means_Asst[asz], sems_Asst[asz], ns_Asst[asz] = {}, {}, {}
        for csz in chef_sizes:
            m, s, n = collect_mean_sem_n(cross_matrix[lvl][csz].get(asz, []))
            means_Asst[asz][csz] = m
            sems_Asst[asz][csz] = s
            ns_Asst[asz][csz] = n

    # Add Self-play to Assistant maps
    means_Asst[SELF_KEY], sems_Asst[SELF_KEY], ns_Asst[SELF_KEY] = {}, {}, {}
    for csz in chef_sizes:
        m, s, n = collect_mean_sem_n(buckets_by_size.get((lvl, csz, "self"), []))
        means_Asst[SELF_KEY][csz] = m
        sems_Asst[SELF_KEY][csz] = s
        ns_Asst[SELF_KEY][csz] = n
    series_Asst = assistant_sizes + [SELF_KEY]

    # ==================== INDIVIDUAL PANEL GRAPHS ====================
    
    # === PANEL 1: Self vs Cross ===
    fig_self_cross = plt.figure(figsize=(14, 9))
    fig_self_cross.patch.set_facecolor(MODERN_COLORS["background"])
    ax = fig_self_cross.add_subplot(111)
    
    x = np.arange(len(sizes_sorted))
    width = 0.42

    # Create stunning gradient bars
    bars_self = ax.bar(x - width/2, means_self_all, width, 
                      yerr=sems_self_all,
                      label="Self-play",
                      color=create_gradient_color(MODERN_COLORS["self"], intensity=0.95, lighten=True),
                      edgecolor=create_gradient_color(MODERN_COLORS["self"], intensity=0.7),
                      linewidth=2.2,
                      alpha=0.93,
                      error_kw=error_kw_elegant,
                      zorder=3)
                      
    bars_cross = ax.bar(x + width/2, means_cross_all, width, 
                       yerr=sems_cross_all,
                       label="Cross-play",
                       color=create_gradient_color(MODERN_COLORS["cross"], intensity=0.95, lighten=True),
                       edgecolor=create_gradient_color(MODERN_COLORS["cross"], intensity=0.7),
                       linewidth=2.2,
                       alpha=0.93,
                       error_kw=error_kw_elegant,
                       zorder=3)
    
    # Add subtle glow effects
    add_subtle_glow_effect(ax, bars_self, glow_color=MODERN_COLORS["self"], intensity=0.1)
    add_subtle_glow_effect(ax, bars_cross, glow_color=MODERN_COLORS["cross"], intensity=0.1)

    ax.set_xticks(x, sizes_sorted)
    ax.set_xlabel("Model Size", fontweight="medium")
    ax.set_ylabel("Mean Success Rate", fontweight="medium")
    ax.set_title(f"Level {lvl} — Self vs Cross Performance", fontsize=16, fontweight="bold", pad=20)
    
    polish_axes_modern(ax)
    
    # Calculate y-axis limits
    all_means = [v for v in means_self_all + means_cross_all if np.isfinite(v)]
    all_sems = [v for v in sems_self_all + sems_cross_all if np.isfinite(v)]
    ymax_est1 = max(all_means) + max(all_sems) if all_means and all_sems else 1.0
    format_as_percent_modern(ax, ymax_est1)
    
    # Modern legend
    legend = ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E5E7EB')
    legend.get_frame().set_alpha(0.95)

    # Annotations
    if ANNOTATE_COUNTS:
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        for bar, n in zip(bars_self, n_self_all):
            h = bar.get_height()
            if np.isfinite(h) and n > 0:
                ax.annotate(f"n={n}", 
                           xy=(bar.get_x() + bar.get_width()/2, h + 0.02*yrange),
                           ha="center", va="bottom", fontsize=9,
                           color="#374151", weight="medium",
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor="white", edgecolor="none", alpha=0.8))
        for bar, n in zip(bars_cross, n_cross_all):
            h = bar.get_height()
            if np.isfinite(h) and n > 0:
                ax.annotate(f"n={n}", 
                           xy=(bar.get_x() + bar.get_width()/2, h + 0.02*yrange),
                           ha="center", va="bottom", fontsize=9,
                           color="#374151", weight="medium",
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor="white", edgecolor="none", alpha=0.8))

    plt.tight_layout()
    self_cross_png = os.path.join(OUT_DIR, f"level_{sanitize_token(lvl)}_self_vs_cross_modern.png")
    show_and_save_modern(fig_self_cross, self_cross_png)

    # === PANEL 2: Cross-play of Chefs ===
    fig_chef = plt.figure(figsize=(14, 9))
    fig_chef.patch.set_facecolor(MODERN_COLORS["background"])
    ax = fig_chef.add_subplot(111)
    
    ymax_est2, _ = plot_modern_grouped_bars(ax,
                                           x_labels=assistant_sizes,
                                           series_keys=series_Chef,
                                           mean_map=means_Chef,
                                           sem_map=sems_Chef,
                                           n_map=ns_Chef,
                                           label_prefix="Chef",
                                           error_kw=error_kw_modern)
    
    ax.set_xlabel("Assistant Size", fontweight="medium")
    ax.set_ylabel("Mean Success Rate", fontweight="medium")
    ax.set_title(f"Level {lvl} — Chef Performance Analysis", fontsize=16, fontweight="bold", pad=20)
    
    polish_axes_modern(ax)
    format_as_percent_modern(ax, ymax_est2 if np.isfinite(ymax_est2) else np.nan)
    
    legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                      loc='upper left', fontsize=10, ncol=2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E5E7EB')
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()
    chef_png = os.path.join(OUT_DIR, f"level_{sanitize_token(lvl)}_chef_analysis_modern.png")
    show_and_save_modern(fig_chef, chef_png)

    # === PANEL 3: Cross-play of Assistants ===
    fig_asst = plt.figure(figsize=(14, 9))
    fig_asst.patch.set_facecolor(MODERN_COLORS["background"])
    ax = fig_asst.add_subplot(111)
    
    ymax_est3, _ = plot_modern_grouped_bars(ax,
                                           x_labels=chef_sizes,
                                           series_keys=series_Asst,
                                           mean_map=means_Asst,
                                           sem_map=sems_Asst,
                                           n_map=ns_Asst,
                                           label_prefix="Assistant",
                                           error_kw=error_kw_modern)
    
    ax.set_xlabel("Chef Size", fontweight="medium")
    ax.set_ylabel("Mean Success Rate", fontweight="medium")
    ax.set_title(f"Level {lvl} — Assistant Performance Analysis", fontsize=16, fontweight="bold", pad=20)
    
    polish_axes_modern(ax)
    format_as_percent_modern(ax, ymax_est3 if np.isfinite(ymax_est3) else np.nan)
    
    legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                      loc='upper left', fontsize=10, ncol=2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E5E7EB')
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()
    asst_png = os.path.join(OUT_DIR, f"level_{sanitize_token(lvl)}_assistant_analysis_modern.png")
    show_and_save_modern(fig_asst, asst_png)



    # Summary output
    if VERBOSE_SUMMARY:
        print(f"   📊 Level {lvl} Summary:")
        print(f"      • Model sizes: {sizes_sorted}")
        print(f"      • Chef sizes: {chef_sizes}")
        print(f"      • Assistant sizes: {assistant_sizes}")
        print(f"      • Generated: Top pairs + 3 individual panel graphs (Self vs Cross, Chef Analysis, Assistant Analysis)")

print(f"\n🎉 Modern cross-play analysis completed!")
print(f"📁 All plots saved to: {OUT_DIR}/")
print(f"✨ Generated {len(levels_sorted)} levels with enhanced visualizations")
