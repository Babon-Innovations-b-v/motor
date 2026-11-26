"""A walking clip's two body windows (compose_fit.py's layout) with a view that follows the body: each frame cut to
a window CROP_SHARE of the clip's body box wide, centred on the body's columns in that frame (the skin's and the
inside's alpha), the centres smoothed over SMOOTH frames, so a walk across the fixed camera fills its window.

    MOTOR_REPO=<motor> python compose_follow.py <renders folder> <trial folder> - <out.mp4> <out.gif>"""
import os, pathlib, subprocess, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compose_fit as base  # noqa: E402
plt = base.plt
CROP_SHARE = 0.8
base.HEIGHT = int(os.environ.get("FOLLOW_HEIGHT", 1080))  # pixels: the walk renders are 1200 high, so the clip keeps their detail
GIF_HEIGHT = 540  # pixels: the GIF, small enough to show inline in a comment
SMOOTH = 15


def centres(renders, count):
    """Each frame's body centre column, smoothed over SMOOTH frames."""
    found = []
    for index in range(count):
        alpha = np.maximum(plt.imread(renders / f"skin_{index:04d}.png")[..., 3],
                           plt.imread(renders / f"inside_{index:04d}.png")[..., 3])
        columns = np.nonzero(alpha.max(axis=0) > 0.05)[0]
        found.append((columns.min() + columns.max()) / 2)
    padded = np.pad(np.array(found), SMOOTH // 2, mode="edge")
    return np.convolve(padded, np.ones(SMOOTH) / SMOOTH, mode="valid")


def main(renders, out, gif):
    count = base.frames_of(renders)
    top, bottom, _, _ = base.body_box(renders, count)
    full = plt.imread(renders / "skin_0000.png").shape[1]
    crop = int(round((bottom - top) * CROP_SHARE))
    lefts = [int(np.clip(round(centre - crop / 2), 0, full - crop)) for centre in centres(renders, count)]
    figure, images, _, (width, height) = base.layout(renders, (top, bottom, lefts[0], lefts[0] + crop), None, None)
    encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                                "-s", f"{width}x{height}", "-r", str(base.RATE), "-i", "-", "-c:v", "libx264",
                                "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(out)], stdin=subprocess.PIPE)
    for index in range(count):
        left = lefts[index]
        for image, panel in zip(images, ("skin", "inside")):
            image.set_data(base.frame_image(renders, index, panel)[top:bottom, left:left + crop])
        figure.canvas.draw()
        encoder.stdin.write(np.asarray(figure.canvas.buffer_rgba()).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit(f"ffmpeg failed on {out}")
    palette = (f"fps={base.GIF_RATE},scale=-2:{GIF_HEIGHT}:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:"
               "stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-filter_complex", palette, "-loop", "0",
                    str(gif)], check=True)
    print(f"demo: {out} and {gif} ({width} x {height}, {count} frames)")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), sys.argv[4], sys.argv[5])
