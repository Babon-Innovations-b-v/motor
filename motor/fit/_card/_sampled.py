"""Level 2's sampled frames posed on the card (#70, option A, 2026-10-30): one launch per frame for every set of a
body (`_sets.Sets`), each frame seeded by that set's previous frame, each recording restarted from its own start
(`Frames.start`). Chosen over mink because mink stops on its ConfigurationLimit bounds where the card does not
(heavy_short_legged frame 16: cost 0.307 against 0.0094, the card inside every range), and that holds level 2's
right-arm constants on MARK_MOST."""
import numpy as np


def path_across(sets, turns, points, firsts, start_of):
    """Every set's sampled frames' angles ([frames x nq] per set): `turns`, `points` one array per set, `firsts` the
    frame indices where a recording begins, `start_of(set, index)` that set's start seed there."""
    count, found, seeds = len(turns[0]), [[] for _ in turns], None
    for index in range(count):
        if index in firsts:
            seeds = [np.array([start_of(number, index)]) for number in range(len(turns))]
        solved = sets.angles([rows[index:index + 1] for rows in turns], [rows[index:index + 1] for rows in points],
                             seeds)
        seeds = [qpos for qpos, _ in solved]
        for number, qpos in enumerate(seeds):
            found[number].append(qpos[0])
    return [np.array(rows) for rows in found]


def recording_firsts(lengths):
    """The indices where each recording's frames begin, the recordings' `lengths` laid end to end."""
    return set(np.cumsum([0] + list(lengths))[:-1].tolist())
