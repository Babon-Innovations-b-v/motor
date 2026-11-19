"""Figure 2 (figures/summary.pdf): MOTOR from one camera beside OpenCap's published results, drawn at the paper's text
width. Every number is typed from the evidence named beside it; nothing is computed here. Run in SCORE's Python
(matplotlib), from paper/motor:

    python figures/summary/plot.py figures/summary.pdf"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MOTOR = "#2a78d6"
OPENCAP = "#8a8984"
LINK = "#d6d5d0"
INK = "#0b0b0b"
MUTED = "#52514e"

ROWS = ["Walking", "Squats", "Sit-to-stand", "Drop jump", "Mean of 4", "Mean, no drop jumps"]
# Row positions: a gap between the activities and their means.
ROW_Y = [0, 1, 2, 3, 4.5, 5.5]

# evidence/comparison-2026-11-09.txt: MOTOR "Cam1+Cam3" (walking, squats, sit-to-stand, drop jump, mean of 4, mean no
# drop jumps; angles the matched 15) and "joint moments ... Cam1+Cam3"; OpenCap: "Uhlrich et al. 2023 (two cameras,
# Table 1)" (the mean of the first three computed); OpenCap Monocular: "Gilon, Miller, Uhlrich 2026" angles 4.8
# (walking, squats, sit-to-stand, so the no-drop-jump row only).
PANELS = [
    ("Joint angles\n(deg)", [4.48, 6.13, 5.94, 6.14, 5.67, 5.52], [4.1, 4.1, 4.7, 5.1, 4.5, 4.3], 4.8, 8),
    ("Vertical GRF\n(% body weight)", [14.32, 6.55, 10.63, 21.65, 13.29, 10.50],
     [8.2, 6.4, 5.7, 25.2, 11.4, 6.8], None, 30),
    ("Anterior GRF\n(% body weight)", [11.54, 3.86, 4.84, 5.69, 6.48, 6.75],
     [2.1, 1.3, 1.9, 8.9, 3.5, 1.8], None, 15),
    ("Medio-lateral GRF\n(% body weight)", [3.30, 3.81, 3.85, 2.86, 3.45, 3.65],
     [1.1, 5.7, 3.2, 5.3, 3.8, 3.3], None, 8),
    ("Joint moments\n(% body weight $\\times$ height)", [1.79, 1.66, 1.61, 3.61, 2.17, 1.69],
     [0.75, 0.97, 0.68, 2.50, 1.22, 0.80], None, 4),
]

# Solve time per recording, minutes. OpenCap: its recorded solves of a walk, a drop jump and a squat, 4196, 4096 and
# 2635 s (#98 comment 6696117098). MOTOR: one L4, 715 trials, median 66.214 s, range 34.843 to 788.668 s
# (evidence/labval-final-2026-11-07.txt, "NVIDIA L4 total").
OPENCAP_TIMES = [("OpenCap, walking", 4196 / 60), ("OpenCap, drop jump", 4096 / 60), ("OpenCap, squat", 2635 / 60)]
MOTOR_TIME = (66.214 / 60, 34.843 / 60, 788.668 / 60)

FONT = 9
WIDTH = 6.9
HEIGHT = 4.25


def style():
    """Computer Modern text at the caption's size, recessive axes."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "font.size": FONT, "axes.titlesize": FONT,
        "xtick.labelsize": FONT, "ytick.labelsize": FONT, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0, "axes.unicode_minus": False, "pdf.fonttype": 42,
    })


def bare(ax):
    """Only the bottom axis line and a hairline vertical grid."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.5)
    ax.set_axisbelow(True)


def dumbbell(ax, title, motor, opencap, mono, top, labels):
    """One measure's MAE per activity: OpenCap and MOTOR joined by a light link."""
    bare(ax)
    for y_pos, ours, theirs in zip(ROW_Y, motor, opencap):
        ax.plot([ours, theirs], [y_pos, y_pos], color=LINK, linewidth=2, zorder=1, solid_capstyle="round")
        ax.plot(theirs, y_pos, "o", color=OPENCAP, markersize=5.5, zorder=2)
        ax.plot(ours, y_pos, "o", color=MOTOR, markersize=5.5, zorder=3)
    if mono is not None:
        ax.plot(mono, ROW_Y[-1], "D", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1,
                markersize=5.5, zorder=4)
    ax.axhline(3.75, color="#c8c7c1", linewidth=0.5)
    ax.set_xlim(0, top)
    ax.set_ylim(ROW_Y[-1] + 0.6, -0.6)
    ax.set_yticks(ROW_Y)
    ax.set_yticklabels(ROWS if labels else [])
    ax.set_title(title, loc="left", fontsize=FONT, color=INK, pad=4)
    ax.locator_params(axis="x", nbins=4)


def times(ax):
    """Solve time per recording on a log axis in minutes, each mark labelled with its value."""
    bare(ax)
    names = [name for name, _ in OPENCAP_TIMES] + ["MOTOR, median of 715"]
    for y_pos, (_, minutes) in enumerate(OPENCAP_TIMES):
        ax.plot(minutes, y_pos, "o", color=OPENCAP, markersize=5.5)
        ax.text(minutes * 1.18, y_pos, f"{minutes:.0f} min", va="center", fontsize=FONT, color=INK)
    median, low, high = MOTOR_TIME
    ax.plot([low, high], [3, 3], color=MOTOR, linewidth=1.2, alpha=0.6)
    for end in (low, high):
        ax.plot([end, end], [2.8, 3.2], color=MOTOR, linewidth=1.2, alpha=0.6)
    ax.plot(median, 3, "o", color=MOTOR, markersize=5.5)
    ax.text(high * 1.18, 3, f"{median:.1f} min ({low:.1f} to {high:.0f})", va="center", fontsize=FONT,
            color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 700)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.set_ylim(3.6, -0.6)
    ax.set_yticks(range(4))
    ax.set_yticklabels(names)
    ax.set_title("Solve time per recording (min, log scale)", loc="left", fontsize=FONT, color=INK, pad=4)


def draw(out):
    """Lay out the legend, four force-and-angle panels, the moments and the solve time; write the PDF."""
    style()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    left, gap, right = 1.36, 0.2, 0.04
    width = (WIDTH - left - right - 3 * gap) / 4
    row_h = 1.3
    top_row = HEIGHT - 0.32 - 0.42 - row_h
    bottom_row = 0.25
    for col, panel in enumerate(PANELS[:4]):
        ax = fig.add_axes([(left + col * (width + gap)) / WIDTH, top_row / HEIGHT, width / WIDTH, row_h / HEIGHT])
        dumbbell(ax, *panel[:3], panel[3], panel[4], col == 0)
    ax = fig.add_axes([left / WIDTH, bottom_row / HEIGHT, width / WIDTH, row_h / HEIGHT])
    dumbbell(ax, *PANELS[4][:3], PANELS[4][3], PANELS[4][4], True)
    time_left = left + width + gap + 1.3
    ax = fig.add_axes([time_left / WIDTH, (bottom_row + 0.2) / HEIGHT, (WIDTH - right - time_left) / WIDTH,
                       (row_h - 0.4) / HEIGHT])
    times(ax)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=MOTOR, markersize=5.5, label="MOTOR, one camera"),
        Line2D([], [], marker="o", linestyle="", color=OPENCAP, markersize=5.5,
               label="OpenCap, two cameras (published)"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="white", markeredgecolor=INK, markersize=5.5,
               label="OpenCap Monocular, one camera (published)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
               fontsize=FONT, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1)
    fig.savefig(out)


if __name__ == "__main__":
    draw(sys.argv[1])
