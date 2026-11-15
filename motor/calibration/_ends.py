"""Level 1's segment ends (#70; owner, 2026-11-03: "what i didnt want anymore was joint centre to joint centre fitting,
but segment ends ... is better"): every bone group's start that is a SOMA rig joint (`fit.group_ends`' ends) has its
end marker shifted on its holder body by a level 1 unknown, never MyoFullBody's joint centre moved (the sizing, the
measure's joints and the calibration keep the model's own). A shift is `calibration.END_SHIFTS`' kind (m, the person's
frame at rest, Y up, x sideways; `fit.end_marker` moves the marker by -shift in its holder body's frame,
`fit._model.shifted_end`), so the evaluation's worker lays it in that table. A left and right end share one unknown,
the right's x mirrored (`SIDE_MIRROR`); a middle end never moves sideways. Each starts at its shift read at rest in
SOMA's skin (`calibration.END_READS`), mirrored. Numpy only."""
import numpy as np

SIDEWAYS = 0  # the person's frame's sideways axis: a middle end's shift never moves along it
SIDE_MIRROR = np.array([-1.0, 1.0, 1.0])  # a right end's shift as its left one's (the person's frame, x sideways)


def segment_ends(groups, soma_middles):
    """{MyoFullBody end: [the segments it starts]} in `fit.group_ends`' rule: every group's start, a SOMA middle whose
    rig joints are all other groups' starts (the hips' middle: the two hip ends) left out, so none is counted twice.
    `groups` is `bone_groups.groups()`, `soma_middles` `landmarks.SOMA_MIDDLES`."""
    starts = {soma[0] for soma, _, _, _ in groups.values()}
    found = {}
    for segment, (soma, myo, _, _) in groups.items():
        if soma[0] in soma_middles and set(soma_middles[soma[0]]) <= starts:
            continue
        found.setdefault(myo[0], []).append(segment)
    return {end: sorted(segments) for end, segments in found.items()}


def end_members(pool):
    """The ends one shift unknown moves and each one's factor ([(end, [3])]): a left end and its right twin mirrored,
    a middle end alone."""
    if pool.endswith("_l"):
        return [(pool, np.ones(3)), (pool[:-2] + "_r", SIDE_MIRROR)]
    return [(pool, np.ones(3))]


def mirrored_starts(reads, ends):
    """Each end's start shift ({end: (3,)} m): a left and right pair the mean of the left's read and the right's
    mirrored (one read alone when the other is missing), the right that mirrored back; a middle end its read with no
    sideways part; an end with no read 0."""
    found = {}
    for end in ends:
        if end.endswith("_r"):
            continue
        if end.endswith("_l"):
            twin = end[:-2] + "_r"
            sides = [np.asarray(reads[end], dtype=float)] if end in reads else []
            sides += [np.asarray(reads[twin], dtype=float) * SIDE_MIRROR] if twin in reads else []
            left = np.mean(sides, axis=0) if sides else np.zeros(3)
            found[end], found[twin] = left, left * SIDE_MIRROR
        else:
            middle = np.array(reads.get(end, (0.0, 0.0, 0.0)), dtype=float)
            middle[SIDEWAYS] = 0.0
            found[end] = middle
    return found


def shifts(unknowns, entries, starts, unit):
    """Every end's shift ({end: [3]} m, the person's frame) at `unknowns`: its start (`starts`) plus `unit` times its
    "end" unknowns, mirrored on the right."""
    found = {end: np.array(start, dtype=float) for end, start in starts.items()}
    for value, (pool, kind, axis, _) in zip(unknowns, entries):
        if kind == "end":
            for end, factor in end_members(pool):
                found[end][axis] += unit * float(value) * float(factor[axis])
    return {end: [float(part) for part in shift] for end, shift in found.items()}
