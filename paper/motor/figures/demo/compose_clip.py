"""A MOTOR demo clip: render_clip.py's frames of one trial as one H.264 video, 1080 wide, three panels stacked top to
bottom and in time: (a) the input body (SOMA), (b) MyoFullBody in its skin with the teaser's activation colour bar,
(c) MOTOR's vertical ground reaction force per foot (% body weight) with a moving time marker. The teaser's look
(figures/teaser/compose.py: its floor shadow, colours and font). The force is the run's own: out.json's
frames[].touches, each foot's y (up) in N, over the subject's weight (record.json's kg times 9.81, as LabValidation's
scorer weighs it), unfiltered, as the teaser plots it.

    python compose_clip.py <renders folder> <trial folder> <out.mp4> <title>"""
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "teaser"))
import compose as teaser  # noqa: E402  (sets matplotlib's Agg backend and the paper's font)
import matplotlib.pyplot as plt  # noqa: E402

WIDTH, HEIGHT = 1080, 1920  # pixels
DPI = 100
HEADER, PANEL = 80, 720  # pixels: the title strip, each body panel (render_clip's 1080 x 720); the plot takes the rest
G = 9.81  # m/s2, LabValidation's scorer's
FEET = (("right", "RightFoot", teaser.RIGHT), ("left", "LeftFoot", teaser.LEFT))
FONT = 22
RATE = 30.0  # frames a second, the run's


def vertical_force(folder):
    """{side: % body weight per frame} from the run's out.json and the subject's weight."""
    name = folder.name
    frames = json.loads((folder / "out.json").read_text())["people"][name]["frames"]
    weight = json.loads((folder / "record.json").read_text())["measured"]["kg"] * G
    return {side: np.array([frame["touches"].get(foot, [0, 0])[1] / weight * 100 for frame in frames])
            for side, foot, _ in FEET}


def frame_image(renders, index, panel):
    """One pass's frame on the floor's shadow over white (teaser.on_floor)."""
    image = plt.imread(renders / f"{panel}_{index:04d}.png")
    return teaser.on_floor(image, plt.imread(renders / f"shadow_{index:04d}.png"))


def box(top, tall):
    """Axes box [left, bottom, width, height] for a strip `tall` pixels high whose top is `top` pixels from the top."""
    return [0, 1 - (top + tall) / HEIGHT, 1, tall / HEIGHT]


def layout(renders, force, title):
    """The figure, its two image artists and its time markers, at frame 0."""
    plt.rcParams.update({"font.size": FONT, "xtick.labelsize": FONT - 3, "ytick.labelsize": FONT - 3,
                         "legend.fontsize": FONT - 3, "axes.linewidth": 1.0})
    figure = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor="white")
    figure.text(0.5, 1 - HEADER / 2 / HEIGHT, title, ha="center", va="center", fontsize=FONT + 2)
    images = []
    for row, (panel, label) in enumerate((("skin", "(a) Input body (SOMA)"), ("inside", "(b) MyoFullBody in its skin"))):
        axis = figure.add_axes(box(HEADER + row * PANEL, PANEL))
        images.append(axis.imshow(frame_image(renders, 0, panel), interpolation="none"))
        axis.set_axis_off()
        axis.text(0.03, 0.95, label, transform=axis.transAxes, ha="left", va="top")
    bottom = HEADER + 2 * PANEL
    teaser.colour_bar(figure, [0.90, 1 - (HEADER + PANEL + 0.62 * PANEL) / HEIGHT, 0.022, 0.42 * PANEL / HEIGHT])
    figure.axes[-1].set_title("activation", fontsize=FONT - 3, pad=14)
    plot = figure.add_axes([0.13, 0.055, 0.83, (HEIGHT - bottom) / HEIGHT - 0.1])
    time = np.arange(len(force["right"])) / RATE
    for side, _, colour in FEET:
        plot.plot(time, force[side], color=colour, lw=2.0, label=side)
    top = max(150.0, np.ceil(max(force["right"].max(), force["left"].max()) / 50) * 50)
    plot.set_ylim(0, top)
    plot.set_xlim(time[0], time[-1])
    plot.set_ylabel("vertical GRF\n(% body weight)")
    plot.set_xlabel("time (s)")
    plot.spines[["top", "right"]].set_visible(False)
    plot.legend(frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 0.99), borderaxespad=0)
    plot.set_title("(c) vertical GRF", loc="left", fontsize=FONT, pad=10)
    marker = plot.axvline(0, color="0.35", lw=1.4, ls=(0, (4, 3)))
    return figure, images, marker


def frames_of(renders):
    """How many frames render_clip wrote all three passes of, in order from 0."""
    count = 0
    while all((renders / f"{panel}_{count:04d}.png").exists() for panel in ("skin", "inside", "shadow")):
        count += 1
    return count


def main(renders, folder, out, title):
    force = vertical_force(folder)
    count = frames_of(renders)
    if count != len(force["right"]):
        raise SystemExit(f"{renders}: {count} rendered frames of the run's {len(force['right'])}")
    figure, images, marker = layout(renders, force, title)
    encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                                "-s", f"{WIDTH}x{HEIGHT}", "-r", str(RATE), "-i", "-", "-c:v", "libx264", "-crf", "18",
                                "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                               stdin=subprocess.PIPE)
    for index in range(count):
        for image, panel in zip(images, ("skin", "inside")):
            image.set_data(frame_image(renders, index, panel))
        marker.set_xdata([index / RATE, index / RATE])
        figure.canvas.draw()
        encoder.stdin.write(np.asarray(figure.canvas.buffer_rgba()).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit(f"ffmpeg failed on {out}")
    print(f"demo: {out} ({count} frames, {count / RATE:.1f} s)")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4])
