"""solvers: the bounded least squares, the least squares over a ball, the low-pass and the central differences each
give the answer worked out by hand (pure numpy and scipy: the score venv runs it).

Breaks (each applied once, seen red, reverted), 2026-10-28:
- least's line search `np.clip(x + size * step, lowest, highest)` made `x + size * step`: a bound is crossed.
- within_balls' projection `np.minimum(1.0, CAP_INSIDE / ...)` made `1.0`: the answer leaves its ball.
- smoothed's `filtfilt(...) if ... else moves` made `moves`: a 15 Hz shake at 30 frames a second goes through.
- stencils' `rate / (after - before)` made `rate`: a line's speed is doubled between its ends.

Run: .venv/bin/python motor/solvers/test_each_solver_meets_a_hand_worked_answer.py
"""
import pathlib
import sys

import numpy as np

INSIDE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSIDE))
from motor import solvers  # noqa: E402


def the_bounded_least_squares_keeps_its_bounds():
    """|x - (1, -1)|^2 + 0.01 |x|^2 over x >= 0: x1 = 1 / 1.01 and x2 held at 0 (its bound); with x1 at most 0.5 too,
    x1 = 0.5."""
    system, wanted, costs = np.eye(2), np.array([1.0, -1.0]), np.full(2, 0.1)
    free = solvers.least(system, wanted, costs, np.full(2, np.inf))
    capped = solvers.least(system, wanted, costs, np.array([0.5, np.inf]))
    problems = [] if np.allclose(free, [1 / 1.01, 0.0], atol=1e-8) else [f"unbounded above: {free}"]
    return problems + ([] if np.allclose(capped, [0.5, 0.0], atol=1e-8) else [f"x1 at most 0.5: {capped}"])


def the_ball_solve_stops_at_its_ball():
    """|y - (2, 0, 0)|^2 with |y| at most CAP_INSIDE (0.95): y = (0.95, 0, 0), its ball at 0.95 (ADMM's tolerance)."""
    found, furthest = solvers.within_balls(np.eye(3), np.array([2.0, 0.0, 0.0]), np.eye(3), np.zeros(3))
    wanted = [solvers.CAP_INSIDE, 0.0, 0.0]
    problems = [] if np.linalg.norm(found - wanted) < 5e-3 else [f"y {found}, not {wanted}"]
    return problems + ([] if abs(furthest - solvers.CAP_INSIDE) < 5e-3 else [f"furthest ball at {furthest}"])


def the_low_pass_keeps_a_constant_and_drops_a_shake():
    """At 30 frames a second, the low-pass (6 Hz) as a matrix keeps a constant (its rows sum to one) and takes a
    15 Hz shake (+1, -1, ...) out of the middle frames."""
    matrix = solvers.filter_matrix(60, 30.0)
    shake = matrix @ np.array([(-1.0) ** frame for frame in range(60)])
    problems = [] if np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9) else ["a constant is not kept"]
    return problems + ([] if np.abs(shake[10:50]).max() < 0.01 else [f"the shake left: {np.abs(shake[10:50]).max()}"])


def a_lines_speed_is_its_slope():
    """A value rising 0.1 a frame at 30 frames a second moves at 3 a second, its ends included; its acceleration 0."""
    speeds, accelerations = solvers.stencils(6, 30.0)
    line = 0.1 * np.arange(6)
    problems = [] if np.allclose(speeds @ line, 3.0) else [f"speeds {speeds @ line}"]
    return problems + ([] if np.allclose(accelerations @ line, 0.0, atol=1e-9) else
                       [f"accelerations {accelerations @ line}"])


CHECKS = (the_bounded_least_squares_keeps_its_bounds, the_ball_solve_stops_at_its_ball,
          the_low_pass_keeps_a_constant_and_drops_a_shake, a_lines_speed_is_its_slope)


if __name__ == "__main__":
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    sys.exit(1 if failures else 0)
