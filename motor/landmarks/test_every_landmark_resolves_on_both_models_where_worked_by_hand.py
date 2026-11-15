"""landmarks: every landmark of SCORE's set resolves on both models (a face of SOMA's template topology with
barycentric weights, on a SOMA segment; a MyoFullBody body of that segment's bone group), each left landmark is its
right one mirrored; a landmark is its face's barycentric point on a person's mesh, and on the clip's skin its corners'
skinned moves' (each corner its joints' weighted moves from the bind); a SOMA joint end is its joint, the hip joints'
middle or the hand's tip, and a segment's frame stands at its start; a knuckle's skin point stands within KNUCKLE_SKIN of its bone point,
a fingertip's within FINGERTIP_SKIN (all worked out by hand; pure: the score venv runs it).

Breaks (each applied once, seen red, reverted):
- (2026-10-28) on_skin's `np.asarray(row[1]) @ points[faces[row[0]]]` made `points[faces[row[0]]].mean(axis=0)`:
  every landmark at its face's middle.
- (2026-10-28) soma_point's `1.0 - TIP_SHARE` made `0.5`: the hand's tip is its far half's middle.
- (2026-10-28) posed's `np.asarray(worlds, dtype=float) @ np.linalg.inv(bind)` made `np.asarray(worlds,
  dtype=float)`: a joint's move is its frame, not its move from the bind.
- (2026-10-28) posed's `np.einsum("lcs,flcsi->flci", weights, moved)` made `moved[..., 0, :]`: every corner moves
  with its first segment only.
- (2026-10-29) _table's ASIS_r bone point x `0` made `0.01`: the right ASIS is 1 cm off the left's mirror.
- (2026-10-29) _table's spinous_L4 body `lumbar4` made `lumbar3`: it no longer resolves in its segment's group.
- (2026-10-29) _table's MT1_head_l face `3570` made `9100`: past the template's faces.
- (2026-10-29) _table's MC3_head_r trust `0.098` made `0.099` (the study's own side): its trust is not the left's.
- (2026-11-01) _table's MC5_head_l link `(-0.01197, -0.00386, -0.00927)` made the study's `(0.01192, 0.00289,
  -0.02595)`: its skin point 29 mm off the bone, on the ring finger's knuckle.
- (2026-11-04) HELD's `"head_side_l"` made `"head_side"`: a held name that is no landmark, so nothing is held there.
- (2026-11-04) at_starts' `frame[:3, 3] = soma_point(start, rig)` dropped: the Hips' frame stands at its own joint,
  8 cm over the hip joints' middle where fit.riding stands it, and a seated skin sinks into its seat.
- (2026-11-06) _table's thumb_tip_l row made the study's (face 2750, link (0.00155, 0.00787, -0.04418)): its skin
  point 26 mm off the thumb's tip, 45 mm off its bone ("thumb_tip_l: link 44.9 mm").

Run: .venv/bin/python motor/landmarks/test_every_landmark_resolves_on_both_models_where_worked_by_hand.py
"""
import pathlib
import sys

import numpy as np

INSIDE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSIDE))
from motor import landmarks  # noqa: E402
from motor.bone_groups import groups  # noqa: E402

TEMPLATE_FACES = 9006  # SOMA's template mesh (4505 vertices), whose topology every person's mesh shares


def a_landmark_is_its_faces_barycentric_point():
    """A mesh whose every face is the triangle (0, 0, 0), (1, 0, 0), (0, 1, 0): ASIS_l (weights 0.1252, 0.15773,
    0.71708) stands at (0.15773, 0.71708, 0)."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.zeros((10000, 3), dtype=int) + [0, 1, 2]
    found = landmarks.on_skin(points, faces)["ASIS_l"]
    return [] if np.allclose(found, [0.15773, 0.71708, 0.0]) else [f"ASIS_l at {found}"]


def a_skinned_landmark_is_its_corners_blended_moves():
    """Every face the triangle (0, 0, 0), (1, 0, 0), (0, 1, 0); corner 0 all Hips', corner 1 half Hips' half
    LeftLeg's, corner 2 all LeftLeg's (Root weighs nothing); LeftLeg's bind stands at (0, 1, 0). In a frame where Hips
    moves 1 along x and LeftLeg turns 90 degrees about z at its bind place, corner 0 is at (1, 0, 0), corner 1 at
    ((2, 0, 0) + (1, 2, 0)) / 2 and corner 2 stays: ASIS_l (weights 0.1252, 0.15773, 0.71708) at (0.361795, 0.87481,
    0)."""
    corners = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    bind = np.tile(np.eye(4), (3, 1, 1))
    bind[2, 1, 3] = 1.0
    rig = {"names": ["Root", "Hips", "LeftLeg"], "bind": bind, "points": corners,
           "faces": np.zeros((10000, 3), dtype=int) + [0, 1, 2],
           "weights": np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])}
    hips, leg = np.eye(4), np.eye(4)
    hips[0, 3] = 1.0
    leg[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    leg[1, 3] = 1.0
    found = landmarks.posed(rig, ["Hips", "LeftLeg"], np.array([[hips, leg]]))
    first = landmarks.solver_marks().index("ASIS_l")
    wanted = [0.361795, 0.87481, 0.0]
    return [] if np.allclose(found[0, first], wanted, atol=1e-6) else [f"ASIS_l skinned at {found[0, first]}"]








def every_landmark_resolves_on_both_models():
    """Each landmark's face is one of the template's, its weights none negative and summing to 1, its segment a bone
    group's and its MyoFullBody body one of that group's bodies; every rule reads landmarks the solver takes, every
    one trusted above 0; every held landmark (`HELD`) is one the solver takes."""
    table, problems = groups(), []
    for name, (face, weights, segment, body, _, _, trust, solver) in landmarks.LANDMARKS.items():
        if not 0 <= face < TEMPLATE_FACES or min(weights) < 0 or abs(sum(weights) - 1) > 1e-3:
            problems.append(f"{name}: face {face}, weights {weights}")
        if segment not in table or body not in table[segment][3]:
            problems.append(f"{name}: {body} is not in {segment}'s bone group")
        if solver and not 0 < trust <= 1:
            problems.append(f"{name}: trust {trust}")
    problems += [f"{name}: held but no solver landmark" for name in landmarks.HELD
                 if name not in landmarks.LANDMARKS or not landmarks.LANDMARKS[name][7]]
    return problems


def a_stretch_end_is_a_joint_the_hips_middle_or_the_hands_tip():
    """A rig with LeftLeg at (0.1, 1, 0), RightLeg at (-0.1, 1, 0.02), LeftHand at 0 and Hips at (0, 1.08, -0.02)
    turned 90 degrees about x, its 100 own points along x at 0.01 to 1.00 m: hipmid (0, 1, 0.01); the hand's tip the
    2% furthest, 0.99 and 1.00: (0.995, 0, 0); at their starts (hipmid, LeftHand) the Hips' frame keeps its turn and
    stands at (0, 1, 0.01), the hand's is its bind's."""
    bind = np.tile(np.eye(4), (4, 1, 1))
    bind[0, :3, 3], bind[1, :3, 3], bind[3, :3, 3] = [0.1, 1, 0], [-0.1, 1, 0.02], [0, 1.08, -0.02]
    bind[3, 1:3, 1:3] = [[0, -1], [1, 0]]
    points = np.column_stack([np.arange(1, 101) / 100.0, np.zeros(100), np.zeros(100)])
    weights = np.zeros((100, 4))
    weights[:, 2] = 1.0
    rig = {"names": ["LeftLeg", "RightLeg", "LeftHand", "Hips"], "bind": bind, "points": points, "weights": weights}
    problems = [] if np.allclose(landmarks.soma_point("hipmid", rig), [0, 1, 0.01]) else ["hipmid"]
    tip = landmarks.soma_point("LeftHand tip", rig)
    problems += [] if np.allclose(tip, [0.995, 0, 0]) else [f"the hand's tip at {tip}"]
    started = landmarks.at_starts(rig, ["Hips", "LeftHand"], ["hipmid", "LeftHand"])
    hips = np.array(bind[3])
    hips[:3, 3] = [0, 1, 0.01]
    return problems + ([] if np.allclose(started, [hips, bind[2]]) else [f"the frames at their starts {started}"])


def each_left_point_is_its_right_ones_mirror():
    """Every `_l` landmark and centre has its `_r` one on the other side's body (or the same middle body), its bone
    point and link mirrored across the body's z (MyoFullBody's sides mirror in z) within 2 mm, its trust and solver
    the same; hipmid is half of each hip."""
    problems = []
    for table, points, body_at in ((landmarks.LANDMARKS, (4, 5), 3), (landmarks.CENTRES, (1,), 0)):
        for name, row in table.items():
            if not name.endswith("_l"):
                continue
            other = table.get(name[:-2] + "_r")
            twin = row[body_at][:-2] + "_r" if row[body_at].endswith("_l") else row[body_at]
            if other is None or other[body_at] != twin:
                problems.append(f"{name} has no right twin")
                continue
            for at in points:
                if np.abs(np.asarray(row[at]) * [1, 1, -1] - other[at]).max() > 0.002:
                    problems.append(f"{name} {row[at]} against {other[at]}")
            if table is landmarks.LANDMARKS and row[6:] != other[6:]:
                problems.append(f"{name} trust {row[6:]} against {other[6:]}")
    weights = [part[2] for part in landmarks.myo_parts("hipmid")]
    return problems + ([] if weights == [0.5, 0.5] else [f"hipmid weights {weights}"])




# The back of a hand over a metacarpal head: skin, the extensor tendon and its hood, at most this thick on SOMA's
# template (the knuckles' links measured on it 2026-11-01: 15.6 to 23.0 mm). The study's MC5_head stood 28.7 mm off
# its bone point, on the face over the ring finger's knuckle (35 mm from the little finger's), and turned the hand's
# calibration 15 degrees about its long axis.
KNUCKLE_SKIN = 0.025


def a_knuckles_skin_point_is_over_its_bone():
    """Each MC head's link (its skin point off its bone point) is at most KNUCKLE_SKIN long."""
    return [f"{name}: link {1000 * np.linalg.norm(row[5]):.1f} mm" for name, row in landmarks.LANDMARKS.items()
            if name.startswith("MC") and "_head_" in name and np.linalg.norm(row[5]) > KNUCKLE_SKIN]


# A fingertip's pulp past the distal phalanx's tip: at most this thick on SOMA's template (the tips' links measured on
# it 2026-11-06, the hand at SOMA's Hand joint with the fingers at `calibration.FINGERS`: 5.6 to 8.7 mm). The study's
# thumb and little finger tips stood 26 and 24 mm off their fingers' ends (the skin nearest a bone tip of MyoFullBody's
# straight fingers and opposed thumb), the index's and middle's on their pads, 6 to 9 mm short of the ends.
FINGERTIP_SKIN = 0.010


def a_fingertips_skin_point_is_past_its_bone():
    """Each fingertip's link (its skin point off its bone point) is at most FINGERTIP_SKIN long."""
    return [f"{name}: link {1000 * np.linalg.norm(row[5]):.1f} mm" for name, row in landmarks.LANDMARKS.items()
            if name.split("_")[1:2] == ["tip"] and row[2].endswith("Hand") and np.linalg.norm(row[5]) > FINGERTIP_SKIN]


CHECKS = (a_knuckles_skin_point_is_over_its_bone, a_fingertips_skin_point_is_past_its_bone, every_landmark_resolves_on_both_models, a_landmark_is_its_faces_barycentric_point,
          a_skinned_landmark_is_its_corners_blended_moves, a_stretch_end_is_a_joint_the_hips_middle_or_the_hands_tip,
          each_left_point_is_its_right_ones_mirror)


if __name__ == "__main__":
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    sys.exit(1 if failures else 0)
