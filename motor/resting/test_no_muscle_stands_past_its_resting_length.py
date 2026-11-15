"""resting: a muscle that stands stretched past REST_MOST of its optimal fibre length has its tendon slack length
lengthened so it stands at REST_MOST, its optimal fibre length kept; a muscle standing shorter is not moved. It runs
in Newton's own Python (MuJoCo's, MOTOR's own Python, the pins in pyproject.toml).

Breaks (each applied once with the env linked into a sandbox home, seen red, reverted), 2026-10-29:
- slack_moves' `np.clip(standing(model) - REST_MOST, 0.0, None)` made `standing(model) - REST_MOST`: the short
  muscle is moved too.
- slack_moves' `* optimal` dropped: the long muscle's move is in normalised lengths, not metres (0.3 m, not 0.15),
  and it stands at 0.8, not 1.1.

Run: .venv/bin/python motor/resting/test_no_muscle_stands_past_its_resting_length.py
"""
import pathlib
import sys

INSIDE = pathlib.Path(__file__).resolve().parents[1]
# Two muscles on slides, each path length its slide's position: optimal fibre length 0.5 m and tendon slack length
# 0.25 m (lengthrange 0.5 to 1.0 over range 0.5 to 1.5), standing at 1.4 and at 0.9 of their optimal length.
BODY = """<mujoco model="two">
  <worldbody>
    <body name="long"><joint name="long" type="slide" ref="0.95"/><geom size="0.1" mass="1"/></body>
    <body name="short"><joint name="short" type="slide" ref="0.7"/><geom size="0.1" mass="1"/></body>
  </worldbody>
  <actuator>
    <muscle name="long" joint="long" range="0.5 1.5" lengthrange="0.5 1.0" force="100"/>
    <muscle name="short" joint="short" range="0.5 1.5" lengthrange="0.5 1.0" force="100"/>
  </actuator>
</mujoco>
"""


def a_stretched_muscle_stands_at_rest_most():
    """The long muscle's slack moved 0.15 m ((1.4 - 1.1) x 0.5 m) and it stands at 1.1, its optimal length still
    0.5 m and its passive force lower; the short one moved 0."""
    import mujoco
    import numpy as np
    from motor import resting
    model = mujoco.MjModel.from_xml_string(BODY)
    problems = []
    if not np.allclose(resting.standing(model), [1.4, 0.9]):
        problems.append(f"the toy stands at {resting.standing(model)}, not [1.4, 0.9]")
    moves = resting.slack_moves(model)
    if not np.allclose(moves, [0.15, 0.0]):
        problems.append(f"slack moved {moves} m, not [0.15, 0]")
    before = resting.passive(model)
    model.actuator_lengthrange[:] += moves[:, None]
    optimal, _ = resting.lengths(model)
    if not np.allclose(resting.standing(model), [1.1, 0.9]):
        problems.append(f"after it stands at {resting.standing(model)}, not [1.1, 0.9]")
    if not np.allclose(optimal, 0.5):
        problems.append(f"optimal fibre lengths {optimal}, not 0.5 m")
    after = resting.passive(model)
    if not (after[0] < before[0] and np.isclose(after[1], before[1])):
        problems.append(f"passive force {before} N before, {after} N after")
    return problems


CHECKS = (a_stretched_muscle_stands_at_rest_most,)


def in_newton():
    """The checks, in Newton's own Python."""
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(in_newton())
