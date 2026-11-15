"""Where every row of the card loop starts (#70, the card build plan's step 2): the frames solve side by side, so no
frame starts from the one before as mink's do. Iteration 0 starts each frame from mink's own angles for it, made once
on the CPU (`evaluation._case` writes them as angles.npz beside a clip's measures); every later iteration starts each
frame from its own last accepted solve. A cold start (the rest pose) may land in another minimum and break the
equality with mink, so it is never used here."""
import pathlib

import numpy as np

SEEDS = "angles.npz"  # as evaluation.SEEDS: a clip's mink angles per frame, qpos [frames, nq]


def first(folder, case):
    """Iteration 0's seeds of `case`: mink's angles per frame (qpos [frames, nq], MuJoCo's layout) from
    folder/case/angles.npz."""
    return np.load(pathlib.Path(folder) / case / SEEDS)["qpos"]


def kept(seeds, case, accepted):
    """`seeds` ({case: qpos}) with `case`'s rows replaced by its last accepted solve (qpos [frames, nq]); the shape
    must not change."""
    accepted = np.asarray(accepted, dtype=float)
    if accepted.shape != seeds[case].shape:
        raise ValueError(f"{case}: accepted {accepted.shape} against seeds {seeds[case].shape}")
    return {**seeds, case: accepted}


def newton_order(qpos):
    """qpos [rows, nq] with the free root's quaternion from MuJoCo's order (w, x, y, z) to Newton's (x, y, z, w): the
    card loop's joint_q."""
    joint_q = np.array(qpos, dtype=float)
    joint_q[:, 3:7] = joint_q[:, [4, 5, 6, 3]]
    return joint_q
