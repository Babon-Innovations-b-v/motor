"""Level 3 on the card, every frame at once (#70 speed, 2026-11-06): mink's frames (`../_frames.Frames.fitted`) are a
recurrence, each frame's fit without the skin from the frame before's fit with it, then its fit with the skin from
that; this solves the same recurrence in sweeps of the card loop (`_clip.Clip.angles`), every frame not yet final one
row, so its answer is the recurrence's, not each frame's own minimum.

A sweep seeds each frame's fit without the skin from the frame before's last answer (the first frame from mink's own
`start`), then its fit with the skin from that. A frame is final once the frame before is final and moved under
SETTLED_TURN / SETTLED_PLACE between the sweep that seeded it and the one after (its seed is then the final one, to
that tolerance); the first frame not final is final after the next sweep by construction (its seed final), so the
sweeps end in at most one per frame, the later ones over the frames not final only. The first sweep's seeds are a
chain without the skin every STRIDE-th frame (each from the one before, one row a solve), each frame from the chain's
frame at or before it: never a cold rest pose, which may land in another minimum. Every solve's residuals are on
MuJoCo's own float64 frames (`Clip.angles(exact=True)`, `_forward.Exact`): on Newton's float32-built joint places the
card minimised another model and stopped up to 0.0073 deg off mink in the shoulder girdle (mean's jump, 2026-11-06)."""
import numpy as np

STRIDE = 5  # the first sweep's seeding chain: every STRIDE-th frame, each from the one before
# A frame's answer moved under these between two sweeps is settled (rad on the joints and the root's turn, m on its
# place): under the gate's 0.01 deg and 0.01 mm, over the card's own stop in the shoulder's flat elevation-plane valley
# (0.0025 deg from mink's on the proto clip), where a tighter one would settle one frame a sweep
SETTLED_TURN, SETTLED_PLACE = 1e-4, 2e-6


def chained(clip, turns, points, start, stride=STRIDE):
    """The first sweep's answers [frames, nq]: every `stride`-th frame fitted without the skin, each from the one
    before (the first from `start`), every frame given the chain's frame at or before it."""
    qpos, found = np.asarray(start, dtype=float), []
    for frame in range(0, len(turns), stride):
        qpos = clip.angles(turns[frame:frame + 1], points[frame:frame + 1], qpos[None],
                           exact=True)[0][0].astype(float)
        found.append(qpos)
    return np.array(found)[np.arange(len(turns)) // stride]


def settled(now, before):
    """Per frame whether its answer moved under SETTLED_TURN on every coordinate but the root's place and under
    SETTLED_PLACE there."""
    moved = np.abs(np.asarray(now, dtype=float) - np.asarray(before, dtype=float))
    return (moved[:, :3].max(axis=1) <= SETTLED_PLACE) & (moved[:, 3:].max(axis=1) <= SETTLED_TURN)


def sweep(clip, turns, points, skins, seeds):
    """One sweep over some frames: each fitted without the skin from its seed, then with the skin (`skins` their
    posed skins, or None) from that. Returns their answers [frames, nq]."""
    free = clip.angles(turns, points, seeds, exact=True)[0].astype(float)
    if skins is None:
        return free
    return clip.angles(turns, points, free, skins, exact=True)[0].astype(float)


def angles(clip, turns, points, skins, start):
    """Every frame's angles [frames, nq] (MuJoCo's layout) as mink's `Frames.fitted` makes them, by sweeps on the card:
    `clip` the fitted model's `_clip.Clip`, `turns` frames x groups x 3 x 3, `points` frames x markers x 3, `skins`
    each frame's posed skin (`fit._skin.PosedSkins`, or None: no skin rows), `start` the first frame's seed
    (`Frames.start`). Returns (angles, sweeps)."""
    turns, points = np.asarray(turns), np.asarray(points)
    start = np.asarray(start, dtype=float)
    found = chained(clip, turns, points, start)
    final, sweeps = 0, 0
    while final < len(turns):
        frames = np.arange(final, len(turns))
        seeds = np.vstack([start[None] if final == 0 else found[final - 1:final], found[frames[:-1]]])
        answers = sweep(clip, turns[frames], points[frames], None if skins is None else skins[frames], seeds)
        still = settled(answers, found[frames])
        found[frames] = answers
        final += 1  # its seed was final
        while final < len(turns) and still[final - 1 - frames[0]]:
            final += 1
        sweeps += 1
    return found, sweeps
