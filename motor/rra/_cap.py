"""RRA's caps (owner, 2026-10-28): each frame's cap linearised
as balls (`cap_rows`), the adjusted path held within the clip's `follows` limit (`within_follows`) and within the
per-frame cap as measured (`within_frames`), and the measures they read (`turned_off`, `off_clip`)."""
import mujoco
import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.spatial.transform import Rotation

from motor.model import posed_path
from motor.rra._linear import adjusted_leads, moved
from motor.solvers import CUTOFF, smoothed

CAP_STEP = 1e-3  # metres or radians: `cap_rows`' difference step
FRAME_ROUNDS = 20  # `within_frames`' rounds of lowering the frames over their cap
FRAME_MARGIN = 0.95  # a frame over its cap lowered this much further than its excess, so the rounds end
# The most RRA's move may raise a region's share of bone vertices outside SOMA's skin on a frame over the fitted
# path's (a share: one point of the region), and how far a frame's share of the move is lowered when it does: halved,
# so twenty rounds take it to a millionth, the fitted path's own pose (#70, 2026-11-05).
SKIN_SLACK = 0.01
SKIN_LOWER = 0.5


def turns_off(model, path, fitted, riders):
    """Each segment's turn of `path` off the fitted poses, as world rotation vectors (frames x segments x 3, rad)."""
    posed = posed_path(model, path, *riders[:3])[..., :3, :3]
    relative = posed @ np.swapaxes(fitted[..., :3, :3], -1, -2)
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(*relative.shape[:2], 3)


def cap_rows(physics, path, total, land, fitted, riders, most):
    """The per-frame cap linearised at the path moved by `total` (frames x adjusted): each frame's balls, every
    segment's turn off the fitted poses (`most[1]`), the lumbar leaders' own turn (`most[1]`: RRA's per-coordinate
    bound, so a trunk turned one way and a pelvis the other cannot bend the lumbar twice the cap) and the pelvis'
    place off the fitted path (`most[0]`), as rows on that frame's move scaled to radius one (frames x balls x 3 x
    adjusted) and their values now (frames x balls x 3); balls that are another's or never move left out (every
    segment riding the chest turns as it does; the held feet do not). A segment's change with the move is measured
    through `land` (the legs follow the feet), by forward differences: the path moved by `total` and by each column's
    nudge of it moved, landed and posed as one stacked path (`land` takes whole copies of the clip; on the card every
    copy's frames in one solve)."""
    model, adjusted = physics.model, adjusted_leads(physics)
    count, width = total.shape
    tried = np.stack([total] + [total + CAP_STEP * np.eye(width)[column] for column in range(width)])
    found = turns_off(model, land(moved(physics, path, tried, adjusted)), np.concatenate([fitted] * len(tried)),
                      riders).reshape(len(tried), count, -1, 3)
    now = found[0]
    turning = np.zeros((count, now.shape[1], 3, width))
    for column in range(width):
        turning[..., column] = (found[1 + column] - now) / (CAP_STEP * most[1])
    kept = []
    for ball in range(now.shape[1]):
        mine = np.concatenate([turning[:, ball].reshape(count, -1), now[:, ball] / most[1]], axis=1)
        if np.abs(mine).max() > CAP_STEP and not any(np.allclose(mine, other, atol=CAP_STEP) for other in kept):
            kept.append(mine)
    pelvis, lumbar = np.zeros((count, 1, 3, width)), np.zeros((count, 1, 3, width))
    pelvis[:, 0, :, :3] = np.eye(3) / most[0]
    lumbar[:, 0, :, 6:9] = np.eye(3) / most[1]
    rows = np.concatenate([np.stack([mine[:, :3 * width].reshape(count, 3, width) for mine in kept], axis=1), pelvis,
                           lumbar], axis=1)
    values = np.concatenate([np.stack([mine[:, 3 * width:] for mine in kept], axis=1), total[:, None, :3] / most[0],
                             total[:, None, 6:9] / most[1]], axis=1)
    return rows, values


def turned_off(before, after):
    """Each segment's turn between two posed paths (frames x segments x 4 x 4), degrees (frames x segments)."""
    return np.degrees(Rotation.from_matrix((np.swapaxes(after[..., :3, :3], -1, -2) @ before[..., :3, :3])
                                           .reshape(-1, 3, 3)).magnitude()).reshape(before.shape[:2])


def off_clip(model, path, worlds, segments, bodies, riding, parents):
    """The `follows` gate's measure on a path: the mean angle (degrees) between each segment's turn against its parent
    (riding its bone) and the clip's, over segments and frames."""
    found = []
    for posed, world in zip(posed_path(model, path, segments, bodies, riding), worlds):
        found += [np.degrees(Rotation.from_matrix((world[parent][:3, :3].T @ world[index][:3, :3])
                                                   @ (posed[index][:3, :3].T @ posed[parent][:3, :3])).magnitude())
                  for index, parent in enumerate(parents) if parent >= 0]
    return float(np.mean(found))


def within_follows(physics, path, adjusted, moves, land, worlds, riders, follows):
    """The adjusted path kept within the clip's `follows` limit (the owner's cap: never further off the clip than the
    `follows` gate allows, degrees on the mean): the whole move, else the largest of its quarters that is, else none;
    and the share kept. A fitted path already further off its clip than the limit keeps the whole move, held only by
    the per-frame cap (`within_frames`: every segment within its turn and the pelvis within its move of the fitted
    path on every frame): no move meets a limit the fit itself is past, and its excess is the `follows` gate's to fail
    (#98, 2026-11-07: most video fits' paths 5 to 9 degrees off, every one's share 0; a limit at the fit's own measure kept none either, RRA's whole
    move adding 0.36 to 1.04 degrees on the mean)."""
    model = physics.model
    if off_clip(model, path, worlds, *riders) > follows:
        return adjusted, 1.0
    for share in (1.0, 0.75, 0.5, 0.25):
        found = adjusted if share == 1.0 else land(moved(physics, path, share * moves, adjusted_leads(physics)))
        if off_clip(model, found, worlds, *riders) <= follows:
            return found, share
    return path, 0.0


def _frame_excess(model, path, found, fitted, riders, most):
    """Each frame's largest move of `found` off the fitted `path` over its cap: its segments' turns (riding their bones,
    against `fitted`'s poses) over `most[1]` and the pelvis' place over `most[0]`, the larger."""
    turned = turned_off(fitted, posed_path(model, found, *riders[:3]))
    shift = np.linalg.norm(found[:, :3] - path[:, :3], axis=1)
    return np.maximum(turned.max(axis=1) / np.degrees(most[1]), shift / most[0])


def held_share(lowered, rate):
    """`lowered` (a share per frame) eroded over a CUTOFF period and averaged under a Hann window no wider: smooth, no
    corner a frame wide (the share multiplies the move, so a corner in it is a jerk in the path; the min of the share
    and its low-pass, before 2026-11-05, kept every corner the lowering made: the pickup's shins 30 times the clip's
    jerk), and nowhere above `lowered`, since each erosion the average takes on a frame is at most that frame's."""
    radius = max(int(rate // CUTOFF), 1)
    window = np.hanning(2 * radius + 3)[1:-1]
    eroded = minimum_filter1d(np.asarray(lowered, dtype=float), 2 * radius + 1, mode="nearest")
    padded = np.pad(eroded, radius, mode="edge")
    return np.convolve(padded, window / window.sum(), mode="valid")


def smoothed_off(physics, path, found, rate):
    """`found` with only its difference from `path` low-passed (`smoothed`, CUTOFF, zero lag), the followers set
    again: what RRA and the legs moved off the fitted path is made smooth, the fitted path's own motion (low-passed
    once already, `model.filtered`) kept as it is."""
    model, couplings = physics.model, physics.couplings
    offsets = np.zeros((len(path), model.nv))
    for frame, (before, after) in enumerate(zip(path, found)):
        mujoco.mj_differentiatePos(model, offsets[frame], 1.0, before, after)
    offsets = smoothed(offsets, rate)
    smooth = []
    for before, offset in zip(path, offsets):
        qpos = before.copy()
        mujoco.mj_integratePos(model, qpos, offset, 1.0)
        smooth.append(couplings.apply(qpos))
    return np.array(smooth)


def skin_excess(found, fitted, labels):
    """Each frame's largest rise, over the regions (`labels`, one a bone vertex), in the share of a region's bone
    vertices outside SOMA's skin from the fitted path (`fitted`) to `found` (both frames x vertices, outside or not)."""
    rises = [found[:, labels == label].mean(axis=1) - fitted[:, labels == label].mean(axis=1)
             for label in np.unique(labels)]
    return np.max(rises, axis=0)


def within_frames(physics, path, moves, land, fitted, riders, most, rate, skins):
    """The adjusted path held within `most` of the fitted path on every frame, all rounds together (owner, 2026-10-28:
    each segment's turn at most 5 degrees and the pelvis at most 2 cm off the fitted motion), and with no region's
    bones further out of SOMA's skin than the fitted path holds them on that frame (`skins`, `fit.FrameSkins`: by
    over SKIN_SLACK of the region; #70, 2026-11-05: the trunk turned by RRA carries the arms and head with it, and the
    pickup's hands stood 37.7% outside the skin against the fit's 1.6%): each frame's share of the move (`moves`,
    frames x adjusted) lowered by how far it is over its cap (`_frame_excess`), halved where it puts bones out, the
    shares held smooth (`held_share`), its feet kept (`land`), the difference from the fitted path low-passed and the
    feet kept again (`smoothed_off`: the per-frame leg solve otherwise shakes the shins), until no frame is over; the
    path and the shares."""
    adjusted, share = adjusted_leads(physics), np.ones(len(path))
    kept_in = skins.outside(path)
    for _ in range(FRAME_ROUNDS):
        found = land(smoothed_off(physics, path, land(moved(physics, path, share[:, None] * moves, adjusted)), rate))
        over = _frame_excess(physics.model, path, found, fitted, riders, most)
        out = skin_excess(skins.outside(found), kept_in, skins.labels) > SKIN_SLACK
        if over.max() <= 1.0 and not out.any():
            return found, share
        lowered = np.where(over > 1.0, share * FRAME_MARGIN / np.maximum(over, 1.0), share)
        share = np.clip(held_share(np.where(out, lowered * SKIN_LOWER, lowered), rate), 0.0, None)
    raise ValueError(f"the adjusted path is still {over.max():.2f} times its per-frame cap, {int(out.sum())} frames "
                     f"with bones put out of the skin, after {FRAME_ROUNDS} rounds")
