"""ground: a point stands on the highest drawn triangle under it at most REACH_UP over it, a body's
points over its lowest point, with HEADROOM free over it, at the height worked out by hand, and every triangle's
normal points up (pure numpy: the score venv runs it).

Breaks (each applied once, seen red, reverted), 2026-10-28, 2026-10-30, 2026-10-31 and 2026-11-07:
- standing's `free = (above - heights >= HEADROOM) & ...` made `>= 0`: a point 1.1 m over a floor, over a counter
  top at 1.0 m with a cupboard 0.5 m over it, stands on the counter top, under the cupboard.
- standing's `top = which != np.append(which[1:], -1)` made the first of each point (the lowest, as before
  2026-10-30): a point on a deck 2.958 m over a floor stands on the floor, the deck read as its roof (greenhouse's
  astronauts on Structure/deck_2).
- standing's `& (heights <= points[which, 1] + REACH_UP)` dropped: a point on the floor under the deck stands on the
  deck.
- standing's `reach = points[:, 1] + REACH_UP if reach is None else ...` made to ignore `reach` (each point its own
  reach, as 75903a1a left it, 2026-10-31): a 1.76 m body under a roof at 1.84 m reads 0.24 m into its ground, its
  head grounded on the roof (tube's person_mid_left_2, greenhouse's astronauts 1, 4 and 5).
- body_reach's `max(..., float(np.median(points[:, 1])))` dropped (its lowest point plus REACH_UP alone, as before
  2026-11-07): a body seated on a 0.40 m seat, its feet on the floor, has no ground under its seat (nan), the seat
  past its reach and the floor under it covered (#98's sit-to-stand chair).
- normals' `found *= np.sign(found[:, 1:2]) + (found[:, 1:2] == 0)` dropped: a triangle wound the other way points
  down.

Run: .venv/bin/python motor/ground/test_a_point_stands_on_the_highest_ground_in_reach.py
"""
import pathlib
import sys

import numpy as np

INSIDE = pathlib.Path(__file__).resolve().parents[1]
REPO = pathlib.Path(__file__).resolve().parents[5]
sys.path[:0] = [str(INSIDE), str(REPO)]
from motor import ground  # noqa: E402


def square(height, low=-5.0, high=5.0):
    """A flat square at `height` (Y up) as two triangles."""
    return [[[low, height, low], [high, height, high], [high, height, low]],
            [[low, height, low], [low, height, high], [high, height, high]]]


def a_point_over_a_slope_stands_at_its_height():
    """The slope through (0, 0, 0), (2, 1, 0) and (0, 0, 2) is y = x / 2: a point at x 1, z 0.5, 5 m up, stands on
    it at 0.5 m, 4.5 m over it; a point off every triangle has no ground (nan, -1)."""
    slope = np.array([[[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 2.0]]])
    points = np.array([[1.0, 5.0, 0.5], [100.0, 0.0, 100.0]])
    heights, triangles = ground.standing(points, slope)
    problems = [] if abs(heights[0] - 0.5) < 1e-12 and triangles[0] == 0 else [
        f"over the slope: {heights[0]} on triangle {triangles[0]}, not 0.5 on 0"]
    problems += [] if np.isnan(heights[1]) and triangles[1] == -1 else [f"off the ground: {heights[1]}, {triangles[1]}"]
    over = ground.heights_over(points[:1], slope)[0]
    return problems + ([] if abs(over - 4.5) < 1e-12 else [f"height over the slope {over}, not 4.5"])


def a_low_top_is_a_block_not_a_floor():
    """A floor at 0, a counter top at 1.0 m and a cupboard's bottom at 1.5 m over it: the top has 0.5 m free, under
    HEADROOM (1 m), so a point 1.1 m up, the top within its reach, stands on the floor (1.0 m free); a point 3 m up
    stands on the cupboard."""
    under = np.array(square(0.0) + square(1.0, -1.0, 1.0) + square(1.5, -1.0, 1.0))
    on_floor = ground.standing(np.array([[0.5, 1.1, 0.5]]), under)[0][0]
    on_cupboard = ground.standing(np.array([[0.5, 3.0, 0.5]]), under)[0][0]
    return ([] if on_floor == 0.0 else [f"over the counter top under the cupboard: stands at {on_floor}, not 0"]) + (
        [] if on_cupboard == 1.5 else [f"over the cupboard: stands at {on_cupboard}, not 1.5"])


def a_point_on_a_deck_stands_on_the_deck():
    """A floor at 0 and a deck at 2.958 m (greenhouse's Structure/deck_2): a point on the deck stands on it, not on
    the floor 2.958 m under it; a point on the floor under the deck, the deck out of its reach, stands on the floor."""
    decked = np.array(square(0.0) + square(2.958, -1.0, 1.0))
    on_deck = ground.standing(np.array([[0.5, 2.958, 0.5]]), decked)[0][0]
    on_floor = ground.standing(np.array([[0.5, 0.0, 0.5]]), decked)[0][0]
    return ([] if on_deck == 2.958 else [f"on the deck: stands at {on_deck}, not 2.958"]) + (
        [] if on_floor == 0.0 else [f"under the deck: stands at {on_floor}, not 0"])


def a_body_under_a_roof_stands_on_its_floor():
    """A floor at 0 and a single-sided roof at 1.84 m: a 1.76 m body standing on the floor (points from its feet to its
    head) reads 0 m over its ground, its head not grounded on the roof 0.08 m over it; the same body on a deck at
    2.958 m stands on the deck, and one on the floor under the deck on the floor. A seated body sits on its seat
    (`seated_reach_problems`)."""
    body = np.array([[0.5, height, 0.5] for height in np.linspace(0.0, 1.76, 12)])
    roofed = np.array(square(0.0) + square(1.84, -1.0, 1.0))
    decked = np.array(square(0.0) + square(2.958, -1.0, 1.0))
    found = {name: np.nanmin(ground.heights_over(points, under, ground.body_reach(points)))
             for name, points, under in (("under the roof", body, roofed), ("on the deck", body + [0, 2.958, 0], decked),
                                         ("under the deck", body, decked))}
    return [f"{name}: {over} over its ground, not 0" for name, over in found.items() if abs(over) > 1e-9] + \
        seated_reach_problems()


def seated_reach_problems():
    """A floor at 0 and a chair's seat 0.40 m up under a seated body's thighs, its feet on the floor before it and its
    trunk up to 1.3 m: its thighs stand 0 m over the seat, its feet 0 m over the floor."""
    feet = [[0.5, height, 0.0] for height in np.linspace(0.0, 0.4, 5)]
    thighs = [[x, 0.4, 0.0] for x in np.linspace(-0.3, 0.1, 5)]
    trunk = [[-0.3, height, 0.0] for height in np.linspace(0.5, 1.3, 9)]
    body = np.array(feet + thighs + trunk)
    seated = np.array(square(0.0) + square(0.4, -0.4, 0.2))
    over = ground.heights_over(body, seated, ground.body_reach(body))
    return [f"{name}: {found} over its ground, not 0" for name, found in
            (("the thighs", np.nanmax(np.abs(over[5:10])) if not np.isnan(over[5:10]).any() else np.nan),
             ("the feet", over[0])) if not abs(found) < 1e-9]


def every_normal_points_up():
    """A triangle wound either way has the upward unit normal (0, 1, 0)."""
    both = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                     [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    found = ground.normals(both)
    return [] if np.allclose(found, [[0.0, 1.0, 0.0]] * 2) else [f"normals {found.tolist()}"]


CHECKS = (a_point_over_a_slope_stands_at_its_height, a_low_top_is_a_block_not_a_floor,
          a_point_on_a_deck_stands_on_the_deck, a_body_under_a_roof_stands_on_its_floor,
          every_normal_points_up)


if __name__ == "__main__":
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    sys.exit(1 if failures else 0)
