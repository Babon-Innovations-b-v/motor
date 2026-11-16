"""A MOTOR demo clip: render_clip.py's frames of one trial as one horizontal H.264 video and a looping GIF, three
compact windows side by side in time: the input body (SOMA), MyoFullBody in its skin with the teaser's activation
colour bar, and MOTOR's vertical ground reaction force per foot (% body weight) with a moving time marker. The
teaser's look (figures/teaser/compose.py: its floor shadow, colours and font). Both body windows are cut to the one box
that holds the body in every frame of the clip, so they keep one scale and a fixed view. The force is the run's own:
out.json's frames[].touches, each foot's y (up) in N, over the subject's weight (record.json's kg times 9.81, as
LabValidation's scorer weighs it), unfiltered; a push whose centre of pressure stands over OFF_FLOOR (a drop jump's
box) is left out, as the scorer leaves it out, so the curve is the floor's force the lab's plates measure.

    python compose_clip.py <renders folder> <trial folder> <out.mp4> <out.gif>"""
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "teaser"))
import compose as teaser  # noqa: E402  (sets matplotlib's Agg backend and the paper's font)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HEIGHT = 440  # pixels, the whole clip's
MARGIN, GAP, INSET = 8, 10, 10 # pixels: around the clip, between windows, inside a window's border
PLOT_WIDTH = 820  # pixels, the force window's
DPI = 100
G = 9.81  # m/s2, LabValidation's scorer's
OFF_FLOOR = 0.015  # m: a push's centre of pressure higher than this is a box's, not the floor's (the scorer's)
FEET = (("right", "RightFoot", teaser.RIGHT), ("left", "LeftFoot", teaser.LEFT))
FONT = 15
RATE = 30.0  # frames a second, the run's
GIF_RATE = 12  # frames a second, the GIF's
BORDER = "0.78"


def vertical_force(folder):
    """{side: % body weight per frame} from the run's out.json and the subject's weight, the floor's pushes only."""
    name = folder.name
    frames = json.loads((folder / "out.json").read_text())["people"][name]["frames"]
    weight = json.loads((folder / "record.json").read_text())["measured"]["kg"] * G
    found = {}
    for side, foot, _ in FEET:
        rows = [frame["touches"].get(foot) for frame in frames]
        found[side] = np.array([0.0 if row is None or row[4] > OFF_FLOOR else row[1] / weight * 100 for row in rows])
    return found


def frame_image(renders, index, panel):
    """One pass's frame on the floor's shadow over white (teaser.on_floor)."""
    image = plt.imread(renders / f"{panel}_{index:04d}.png")
    return teaser.on_floor(image, plt.imread(renders / f"shadow_{index:04d}.png"))


def body_box(renders, count):
    """(top, bottom, left, right) pixel rows and columns holding the body in either pass of every frame, with a
    margin (more below, for the shadow)."""
    seen = None
    for index in range(count):
        for panel in ("skin", "inside"):
            alpha = plt.imread(renders / f"{panel}_{index:04d}.png")[..., 3] > 0.03
            seen = alpha if seen is None else seen | alpha
    rows, columns = np.nonzero(seen)
    pad = int(0.04 * (rows.max() - rows.min()))
    return (max(rows.min() - pad, 0), min(rows.max() + 2 * pad, seen.shape[0]),
            max(columns.min() - pad, 0), min(columns.max() + pad, seen.shape[1]))


def window(figure, left, width, label, size):
    """A rounded, thin-bordered window `width` pixels wide from `left`, its label top-left; its inner box in pixels."""
    width_px, height_px = size
    figure.patches.append(FancyBboxPatch((left / width_px, MARGIN / height_px), width / width_px,
                                         (HEIGHT - 2 * MARGIN) / height_px, boxstyle="round,pad=0,rounding_size=0.02",
                                         transform=figure.transFigure, facecolor="none", edgecolor=BORDER, lw=1.0, zorder=5,
                                         mutation_aspect=width_px / height_px))
    figure.text((left + 12) / width_px, 1 - (MARGIN + 10) / height_px, label, ha="left", va="top", fontsize=FONT,
                color="0.2")
    return left + INSET, MARGIN + INSET, width - 2 * INSET, HEIGHT - 2 * (MARGIN + INSET)


def axes_at(figure, inner, size):
    """Axes over a pixel box (left, bottom, width, height)."""
    left, bottom, width, height = inner
    return figure.add_axes([left / size[0], bottom / size[1], width / size[0], height / size[1]])


def layout(renders, box, force):
    """The figure, its two image artists and its time marker, at frame 0."""
    top, bottom, left, right = box
    inner_height = HEIGHT - 2 * (MARGIN + INSET)
    body_width = int(round((right - left) / (bottom - top) * inner_height)) + 2 * INSET
    width = 2 * MARGIN + 2 * body_width + PLOT_WIDTH + 2 * GAP
    width += width % 2  # H.264's yuv420p wants an even width
    size = (width, HEIGHT)
    plt.rcParams.update({"font.size": FONT, "xtick.labelsize": FONT - 3, "ytick.labelsize": FONT - 3,
                         "legend.fontsize": FONT - 2, "axes.linewidth": 0.8})
    figure = plt.figure(figsize=(width / DPI, HEIGHT / DPI), dpi=DPI, facecolor="white")
    images = []
    places = (MARGIN, MARGIN + body_width + GAP)
    for place, (panel, label) in zip(places, (("skin", "Input body (SOMA)"), ("inside", "MyoFullBody"))):
        inner = window(figure, place, body_width, label, size)
        axis = axes_at(figure, inner, size)
        images.append(axis.imshow(frame_image(renders, 0, panel)[top:bottom, left:right], interpolation="antialiased",
                                  aspect="auto"))
        axis.set_axis_off()
    bar_left = places[1] + body_width - INSET - 40
    teaser.colour_bar(figure, [bar_left / width, 0.30, 12 / width, 0.36])
    figure.axes[-1].set_title("activation", fontsize=FONT - 4, pad=6)
    figure.axes[-1].tick_params(labelsize=FONT - 4)
    plot_left = MARGIN + 2 * (body_width + GAP)
    window(figure, plot_left, PLOT_WIDTH, "Vertical GRF (% body weight)", size)
    plot = axes_at(figure, (plot_left + 70, MARGIN + 52, PLOT_WIDTH - 96, HEIGHT - 2 * MARGIN - 112), size)
    time = np.arange(len(force["right"])) / RATE
    for side, _, colour in FEET:
        plot.plot(time, force[side], color=colour, lw=2.0, label=side)
    plot.set_ylim(0, max(150.0, np.ceil(max(force["right"].max(), force["left"].max()) / 50) * 50))
    plot.set_xlim(time[0], time[-1])
    plot.set_xlabel("time (s)", fontsize=FONT - 2, labelpad=2)
    plot.spines[["top", "right"]].set_visible(False)
    plot.legend(frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.0), borderaxespad=0)
    marker = plot.axvline(0, color="0.35", lw=1.2, ls=(0, (4, 3)))
    return figure, images, marker, size


def frames_of(renders):
    """How many frames render_clip wrote all three passes of, in order from 0."""
    count = 0
    while all((renders / f"{panel}_{count:04d}.png").exists() for panel in ("skin", "inside", "shadow")):
        count += 1
    return count


def gif_of(video, out):
    """The video as a looping GIF at GIF_RATE, one palette over the whole clip."""
    palette = f"fps={GIF_RATE},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];" \
              "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex", palette, "-loop", "0",
                    str(out)], check=True)


def main(renders, folder, out, gif):
    force = vertical_force(folder)
    count = frames_of(renders)
    if count != len(force["right"]):
        raise SystemExit(f"{renders}: {count} rendered frames of the run's {len(force['right'])}")
    box = body_box(renders, count)
    figure, images, marker, (width, height) = layout(renders, box, force)
    top, bottom, left, right = box
    encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                                "-s", f"{width}x{height}", "-r", str(RATE), "-i", "-", "-c:v", "libx264", "-crf", "23",
                                "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                               stdin=subprocess.PIPE)
    for index in range(count):
        for image, panel in zip(images, ("skin", "inside")):
            image.set_data(frame_image(renders, index, panel)[top:bottom, left:right])
        marker.set_xdata([index / RATE, index / RATE])
        figure.canvas.draw()
        encoder.stdin.write(np.asarray(figure.canvas.buffer_rgba()).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit(f"ffmpeg failed on {out}")
    gif_of(out, gif)
    print(f"demo: {out} and {gif} ({width} x {height}, {count} frames, {count / RATE:.1f} s)")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4])
