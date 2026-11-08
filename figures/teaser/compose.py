"""The MOTOR paper's teaser (Figure 1, figures/teaser.pdf): render.py's two body panels, cut to one box so they keep
one scale, the activation colour bar, and the run's force and knee traces as vector plots, at the paper's text width
in its font (Computer Modern, matplotlib's cmr10).

    python compose.py <renders folder> <teaser json> <out.pdf>

<teaser json> is paper/motor/evidence/teaser-2026-11-07.json: the plotted series and the frame the renders show."""
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# render.RAMP's stops (linear RGB); the bar shows them in sRGB, as the renders do.
RAMP = ((0.0, (0.50, 0.58, 0.74)), (0.35, (0.85, 0.60, 0.18)), (0.7, (0.92, 0.22, 0.04)), (1.0, (0.62, 0.0, 0.02)))
WIDTH = 6.5  # inches: the paper's \linewidth (letter, 2.5 cm margins)
HEIGHT = 2.0  # inches: the main body stays within 4 pages
FONT = 8  # pt
PAD = 0.03  # of the box's height, kept around what the renders show
SHADOW = 0.25  # the floor shadow's darkest
GAP = 0.012  # of the width, between the body panels
RIGHT, LEFT = "#1f4e79", "#7fa7cf"  # right and left foot / knee

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": FONT, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "xtick.labelsize": FONT - 1, "ytick.labelsize": FONT - 1,
                     "pdf.fonttype": 42, "legend.fontsize": FONT - 1})


def srgb(linear):
    linear = np.asarray(linear)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def on_floor(image, shadow):
    """A render over white with the floor's shadow under it, the shadow at SHADOW strength and faded out toward the
    panel's sides so it ends softly, not at the cut."""
    across = np.linspace(-1, 1, shadow.shape[1])
    fade = np.clip((1 - np.abs(across)) / 0.35, 0, 1) ** 2
    floor = 1 - SHADOW * shadow[..., 3:] * fade[None, :, None]
    return np.clip(image[..., :3] * image[..., 3:] + floor * (1 - image[..., 3:]), 0, 1)


def panels(renders):
    """Both renders on the floor's shadow, cut to the one box that holds the body in either plus PAD (more below,
    for the shadow), so they keep one scale."""
    images = [plt.imread(renders / f"{name}.png") for name in ("skin", "inside")]
    shadow = plt.imread(renders / "shadow.png")
    rows, columns = np.nonzero(np.logical_or(*[image[..., 3] > 0.03 for image in images]))
    pad = int(PAD * (rows.max() - rows.min()))
    top, bottom = max(rows.min() - pad, 0), min(rows.max() + 2 * pad, images[0].shape[0])
    left, right = max(columns.min() - pad, 0), min(columns.max() + pad, images[0].shape[1])
    return [on_floor(image[top:bottom, left:right], shadow[top:bottom, left:right]) for image in images]


def traces(fig, box, record):
    """(c): each foot's vertical ground reaction force and both knees' flexion, the shown frame dashed."""
    left, bottom, width, height = box
    time = np.arange(len(record["knee angle deg"]["right"])) / record["rate"]
    gap = 0.07
    force = fig.add_axes([left, bottom + (height + gap) / 2, width, (height - gap) / 2])
    knee = fig.add_axes([left, bottom, width, (height - gap) / 2], sharex=force)
    for side, colour in (("right", RIGHT), ("left", LEFT)):
        force.plot(time, record["vertical force %BW"][side], color=colour, lw=0.9, label=side)
        knee.plot(time, record["knee angle deg"][side], color=colour, lw=0.9)
    force.set_ylabel("vertical GRF\n(% body weight)")
    knee.set_ylabel("knee flexion\n(deg)")
    knee.set_xlabel("time (s)", labelpad=1)
    force.set_ylim(0, 150)
    force.set_yticks([0, 50, 100, 150])
    knee.set_ylim(-5, 90)
    knee.set_yticks([0, 40, 80])
    knee.set_xlim(time[0], time[-1])
    force.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.0,
                 bbox_to_anchor=(0.0, 1.18), borderaxespad=0)
    plt.setp(force.get_xticklabels(), visible=False)
    shown = record["frame shown"] / record["rate"]
    for axis in (force, knee):
        axis.axvline(shown, color="0.35", lw=0.6, ls=(0, (3, 2)))
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_label_coords(-0.13, 0.5)
    return force


def colour_bar(fig, box):
    colours = LinearSegmentedColormap.from_list("activation", [(stop, srgb(colour)) for stop, colour in RAMP])
    bar = fig.add_axes(box)
    bar.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto", cmap=colours, extent=(0, 1, 0, 1))
    bar.set_xticks([])
    bar.yaxis.tick_right()
    bar.set_yticks([0, 1])
    bar.tick_params(length=2, pad=1.5)
    for spine in bar.spines.values():
        spine.set_linewidth(0.4)
    bar.set_title("activation", fontsize=FONT - 1, pad=3)


def fitted(fig, text, room):
    """Raise when a panel's label is wider than its panel (a clipped or overlapping label is a broken figure)."""
    fig.canvas.draw()
    wide = text.get_window_extent().width / fig.dpi / WIDTH
    if wide > room:
        raise SystemExit(f"label {text.get_text()!r} is {wide:.3f} of the width, its panel {room:.3f}")


def figure(renders, record, out):
    skin, inside = panels(renders)
    top, bottom = 0.90, 0.0
    tall = (top - bottom) * HEIGHT
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    start = 0.0
    for image, label in ((skin, "(a) Input body (SOMA)"), (inside, "(b) MyoFullBody in its skin")):
        wide = tall * image.shape[1] / image.shape[0] / WIDTH
        axis = fig.add_axes([start, bottom, wide, top - bottom])
        axis.imshow(image, interpolation="lanczos")
        axis.set_axis_off()
        fitted(fig, fig.text(start + wide / 2, 0.93, label, ha="center", va="bottom"), wide + 0.02)
        start += wide + GAP
    colour_bar(fig, [start - GAP + 0.004, 0.42, 0.012, 0.36])
    left = start + 0.135
    traces(fig, (left, 0.17, 0.995 - left, 0.69), record)
    fitted(fig, fig.text(left - 0.075, 0.93, "(c) Outputs of the warm start", ha="left", va="bottom"),
           1.0 - left + 0.075)
    fig.savefig(out, dpi=600)  # the panels embedded at 600 dpi
    print(f"teaser: {out}")


if __name__ == "__main__":
    figure(pathlib.Path(sys.argv[1]), json.loads(pathlib.Path(sys.argv[2]).read_text()), sys.argv[3])
