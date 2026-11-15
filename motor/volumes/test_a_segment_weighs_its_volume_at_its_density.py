"""volumes: a closed box 0.2 x 1 x 0.1 m is filled to its volume (0.02 m3), cut between two bone lines into halves
of 0.01 m3 centred a quarter up and down, each with a solid's spread V (a2 + b2 + c2) / 12, and the halves weighed
at their densities share the mass as density times volume, each inertia's trace m (a2 + b2 + c2) / 6, and each half's
girth for its length is its 0.2 x 0.1 m section's width, sqrt(0.02) m, over 0.5 m (all worked out by hand). Pure: the
score venv runs it.

Breaks (each applied once, seen red, reverted), 2026-10-28:
- filled's `np.searchsorted(crossings, ys) % 2 == 1` made `== 0`: the outside filled, not the inside.
- shares' `nearer = away < best` made `nearer = away > best`: every cell to the farthest line.
- measured's `+ volume * cell ** 2 / 4` dropped: the cells' own spread lost (6.25e-8 m5 under a solid's).
- weighed's `2.0 * density(group)` made `density(group)`: half the inertia.
- (2026-11-01) slenderness's `/ length` at its end dropped: the width, not the width over the length.
- (2026-11-01) slenderness's `(GIRTH_MIDDLE[1] - GIRTH_MIDDLE[0]) * length` made `length`: the section read over the
  whole stretch while the cells are counted in its middle.

Run: .venv/bin/python motor/volumes/test_a_segment_weighs_its_volume_at_its_density.py
"""
import pathlib
import sys

import numpy as np

INSIDE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSIDE))
from motor import volumes  # noqa: E402

# A box from (0, 0, 0) to (0.2, 1, 0.1), its 12 faces.
CORNERS = np.array([[x, y, z] for x in (0, 0.2) for y in (0, 1.0) for z in (0, 0.1)])
FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
                  [0, 6, 4], [1, 5, 7], [1, 7, 3]])
# Two bone lines up the box's middle: the lower half's (y 0 to 0.5) and the upper half's (0.5 to 1).
LINES = [[((0.1, 0.0, 0.05), (0.1, 0.5, 0.05))], [((0.1, 0.5, 0.05), (0.1, 1.0, 0.05))]]


def halves():
    cells = volumes.filled(CORNERS, FACES)
    return cells, volumes.measured(cells, volumes.shares(cells, LINES), 2)


def the_box_fills_to_its_volume_in_two_halves():
    """0.02 m3 whole within one cell layer (1%); each half 0.01 m3 centred at (0.1, 0.25 or 0.75, 0.05), its spread
    0.01 (0.04 + 0.25 + 0.01) / 12 = 2.5e-4 m5 within 1e-9 (the cells fill the box exactly)."""
    cells, found = halves()
    problems = [] if abs(len(cells) * volumes.CELL ** 3 - 0.02) < 2e-4 else [f"{len(cells)} cells"]
    for (volume, centre, spread), height in zip(found, (0.25, 0.75)):
        if abs(volume - 0.01) > 1e-4 or not np.allclose(centre, [0.1, height, 0.05], atol=1e-3):
            problems.append(f"half at {height}: {volume} m3 at {centre}")
        if abs(spread - 2.5e-4) > 1e-9:
            problems.append(f"half at {height}: spread {spread}")
    return problems


def the_halves_share_the_mass_by_density():
    """As a thigh (1.05 kg/l) and a shin (1.09 kg/l), the halves of 21.4 kg: 10.5 and 10.9 kg (their volumes the
    same), each trace m (0.04 + 0.25 + 0.01) / 6 = 0.05 m."""
    _, found = halves()
    weighed = volumes.weighed(["LeftLeg", "LeftShin"], found, 21.4)
    problems = []
    for group, kg in (("LeftLeg", 10.5), ("LeftShin", 10.9)):
        mass, _, trace = weighed[group]
        if abs(mass - kg) > 0.01 * kg or abs(trace - 0.05 * mass) > 0.002 * 0.05 * mass:
            problems.append(f"{group}: {mass:.3f} kg, trace {trace:.5f}")
    return problems


def each_half_is_as_wide_for_its_length_as_its_section():
    """Each half's girth for its length sqrt(0.2 x 0.1) / 0.5 = 0.2828 within 1% (its cells fill the box exactly)."""
    cells, _ = halves()
    owner = volumes.shares(cells, LINES)
    problems = []
    for group, [(start, end)] in enumerate(LINES):
        found = volumes.slenderness(cells, owner, group, start, end)
        if abs(found - np.sqrt(0.02) / 0.5) > 0.01 * np.sqrt(0.02) / 0.5:
            problems.append(f"half {group}: {found:.4f}")
    return problems


CHECKS = (the_box_fills_to_its_volume_in_two_halves, the_halves_share_the_mass_by_density,
          each_half_is_as_wide_for_its_length_as_its_section)


if __name__ == "__main__":
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    sys.exit(1 if failures else 0)
