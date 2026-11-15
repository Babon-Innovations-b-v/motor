"""calibration: `aligned` lays a MyoFullBody bone along its SOMA segment's (worked by hand), and the stored table keeps
the two models' symmetry: a middle group turns no body sideways, a right group is its left one with the body's half
turn about its z (MyoFullBody's right bodies are its left ones mirrored in z; measured on the table within 0.03
degrees, 2026-10-28); the held fingers' rest turns are written alike for both hands, each a finger or thumb joint
of MyoFullBody's; the wrists' end holds the hand at SOMA's Hand joint; a levelled foot's row rises as MyoFullBody's
default foot about its hinge. Pure: the score venv runs it.

Breaks (each applied once, seen red, reverted), 2026-10-28:
- aligned's `soma_turn.T @ frame(...)` made `frame(...)` (the segment's own turn dropped).
- frame's `np.cross(across, along)` made `np.cross(along, across)`: a left-handed frame.
- TABLE's RightLeg `(0.707106, 0.707106, ...)` made `(-0.707106, 0.707106, ...)` (the left's).
- TABLE's Spine2 `0.00165502` made `0.1`: the lumbar turned sideways.

Break (applied once, seen red, reverted), 2026-10-29:
- REFERENCES' `{Side}Arm` made `("z of {Side}ForeArm", "axis of elbow_flex_{side}")`: a twist from a rig joint's
  bind and a model joint's axis, not the same landmarks on both models.

Break (applied once, seen red, reverted), 2026-11-01:
- fingers' `for side in "lr"` made `for side in "l"`: the right hand's fingers stay straight.

Breaks (each applied once, seen red, reverted), 2026-11-01 (the rows read in the skin):
- remeasured's `TABLE[group] if group in INSIDE_GROUPS else tuple(quaternion)` made `tuple(quaternion)`: level 1's
  re-read replaced the skin-read rows ("Head: its skin-read row was replaced"). Since 2026-11-06 the head's row is laid
  on its landmarks (LAID_GROUPS): `INSIDE_GROUPS + LAID_GROUPS` made `INSIDE_GROUPS` gives "Head: its kept row was
  replaced".
- turned's `MIRROR @ np.asarray(rotvec)` for a right group made `np.asarray(rotvec)`: the right side turned unmirrored
  ("RightShin's turn is not LeftShin's, mirrored").

Break (applied once, seen red, reverted), 2026-11-01 (the head's end read in the skin):
- END_SHIFTS' C0_C1 `(0.0, 0.006, 0.008)` made `(0.004, 0.006, 0.008)` (since 2026-11-04 level 1's `(0.0, 0.006094,
  0.008121)`, since 2026-11-06 the landmark lay's `(0.0, 0.000417, 0.000517)`, the same break): the head held off its
  joint sideways
  ("C0_C1: a middle end shifted sideways").

Breaks (each applied once, seen red, reverted), 2026-11-04 (the hinges' rows):
- hinged's `along - (along @ across) * across` made `along` (the long axis not made square): "hinged does not keep its
  reference exact (x, y, z)".
- REFERENCES' `{Side}Foot` made `("own x", "own x")`: "{Side}ToeBase: a hinge group's reference is not its pair's".

Break (applied once, seen red, reverted), 2026-11-06 (the hands at their joints):
- END_SHIFTS' wrist_l made level 1's `(-0.001684, 0.018127, -0.007317)`: the hand held 18 mm up the forearm, every
  fingertip bone 20 to 22 mm short of SOMA's ("wrist_l: shifted 19.6 mm off SOMA's Hand joint").

Break (applied once, seen red, reverted), 2026-11-06 (the shins back on their line):
- INSIDE_GROUPS given "LeftShin" and "RightShin" again: "LeftShin: a leg group's row read in the skin".

Breaks (each applied once, seen red, reverted), 2026-11-06 (#98, the feet levelled):
- levelled's `row @ Rotation.from_rotvec(... * hinge)` made `Rotation.from_rotvec(... * hinge).as_matrix() @ row`
  (turned about the hinge in the segment's frame, not the body's): "the levelled foot's long axis rises 2.339
  degrees, not 2" and "levelling moved the hinge's line".
- levelled's `- rise` dropped from its miss: "the levelled foot's long axis rises 0.000 degrees, not 2"; LEVELLED made ():
  "the groups levelled are (), not the feet".

Run: .venv/bin/python motor/calibration/test_the_calibration_lays_each_bone_along_its_segment.py
"""
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation

INSIDE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSIDE))
from motor import calibration  # noqa: E402
from motor import landmarks  # noqa: E402
from motor.calibration import _ends as tuning  # noqa: E402
from motor.bone_groups import groups  # noqa: E402

TURN_Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 degrees about z


def a_hanging_bone_is_laid_along_its_segment():
    """A SOMA segment unturned, its bone along x, its reference z, and a body unturned, its bone hanging along -y,
    reference z: R_cal is 90 degrees about z ([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), which lays the body's -y on the
    segment's x. The same bone hanging (along -y) in a segment turned -90 degrees about z, whose own x hangs: R_cal is
    the same, read in the segment's own frame. A bone along x with reference z has the axes x, y, z (right-handed)."""
    problems = [] if np.allclose(calibration.frame([2, 0, 0], [0.3, 0, 1]), np.eye(3)) else ["frame is not x, y, z"]
    if not np.allclose(calibration.hinged([2, 0, 0.3], [0, 0, 1]), np.eye(3)):
        problems.append("hinged does not keep its reference exact (x, y, z)")
    found = calibration.aligned((np.eye(3), [1, 0, 0], [0, 0, 1]), (np.eye(3), [0, -1, 0], [0, 0, 1]))
    if not np.allclose(found, TURN_Z):
        problems.append(f"unturned: {np.round(found, 3).tolist()}")
    found = calibration.aligned((TURN_Z.T, [0, -1, 0], [0, 0, 1]), (np.eye(3), [0, -1, 0], [0, 0, 1]))
    if not np.allclose(found, TURN_Z):
        problems.append(f"turned: {np.round(found, 3).tolist()}")
    return problems


def the_table_keeps_both_sides_and_the_middle():
    """Every stored turn is a rotation; each Right group's is its Left one's times diag(-1, -1, 1) within 0.1
    degrees; each middle group (Hips to Head) keeps the sideways axis (|R[2, 2]| over cos 0.5 degrees)."""
    problems = []
    for group in calibration.TABLE:
        turn = calibration.turn(group)
        if group.startswith("Right"):
            mirrored = calibration.turn("Left" + group[len("Right"):]) @ np.diag([-1.0, -1.0, 1.0])
            angle = np.degrees(np.arccos(np.clip((np.trace(mirrored.T @ turn) - 1) / 2, -1, 1)))
            if angle > 0.1:
                problems.append(f"{group} is {angle:.2f} degrees off its left one mirrored")
        elif not group.startswith("Left") and abs(turn[2, 2]) < np.cos(np.radians(0.5)):
            problems.append(f"{group} turns its body sideways: R[2, 2] {turn[2, 2]:.4f}")
    return problems


def each_twist_is_read_from_the_same_landmarks_on_both_models():
    """Every reference is the person's own axis or one landmark line, the same on SOMA and on MyoFullBody, its two
    landmarks in the landmark set on both sides; a hinge's groups (HINGED) come as parent and child sharing one
    reference, so both rest frames hold the hinge alike."""
    from motor.landmarks import LANDMARKS
    hinge_groups = set(calibration.HINGED)
    problems = [f"{child}: hinged without its parent {parent}" for parent, child in
                (("{Side}Arm", "{Side}ForeArm"), ("{Side}Foot", "{Side}ToeBase"))
                if (parent in hinge_groups) != (child in hinge_groups)]
    problems += [f"{group}: a hinge group's reference is not its pair's" for group, pair in
                 (("{Side}ForeArm", "{Side}Arm"), ("{Side}ToeBase", "{Side}Foot"))
                 if group in hinge_groups and calibration.REFERENCES[group] != calibration.REFERENCES[pair]]
    for group, (soma, myo) in calibration.REFERENCES.items():
        if soma != myo:
            problems.append(f"{group}: SOMA's reference {soma!r} is not MyoFullBody's {myo!r}")
        elif soma.startswith("line "):
            missing = [name.format(side=side) for name in soma.split()[1:] for side in "lr"
                       if name.format(side=side) not in LANDMARKS]
            problems += [f"{group}: {name} is no landmark" for name in missing]
        elif soma not in ("own x", "own y"):
            problems.append(f"{group}: {soma!r} is neither an own axis nor a landmark line")
    return problems


def the_rows_read_in_the_skin_stay_and_turn_both_sides_alike():
    """`remeasured` takes a landmark-line row read again for every group but INSIDE_GROUPS' and LAID_GROUPS', whose
    row stays (the head's among them, laid on its landmarks);
    `turned` turns a right group by its left one's rotvec mirrored (its move off its row is the left one's, mirrored),
    so the pair keeps the table's mirror, and a middle
    group turned about its body's z alone keeps the sideways axis."""
    problems = []
    lines = {group: (0.0, 0.0, 0.0, 1.0) for group in calibration.TABLE}
    kept = calibration.remeasured(lines)
    own = calibration.INSIDE_GROUPS + calibration.LAID_GROUPS
    problems += [] if "Head" in calibration.LAID_GROUPS else ["Head: its row is not laid on its landmarks"]
    problems += [f"{group}: its kept row was replaced" for group in own if kept[group] != calibration.TABLE[group]]
    problems += [f"{group}: its landmark-line row was not taken" for group in calibration.TABLE
                 if group not in own and kept[group] != lines[group]]
    rotvec = np.radians([3.0, -5.0, 7.0])
    left = Rotation.from_quat(calibration.turned("LeftShin", rotvec)).as_matrix()
    right = Rotation.from_quat(calibration.turned("RightShin", rotvec)).as_matrix()
    left_move = calibration.turn("LeftShin").T @ left
    right_move = calibration.turn("RightShin").T @ right
    if not np.allclose(calibration.MIRROR @ left_move @ calibration.MIRROR, right_move, atol=1e-9):
        problems.append("RightShin's turn is not LeftShin's, mirrored")
    head = Rotation.from_quat(calibration.turned("Head", [0.0, 0.0, np.radians(9.0)])).as_matrix()
    if abs(head[2, 2]) < np.cos(np.radians(0.5)):
        problems.append(f"Head turned about z turns sideways: R[2, 2] {head[2, 2]:.4f}")
    return problems


FINGER_JOINTS = ("cmc_flexion", "cmc_abduction", "mp_flexion", "ip_flexion",
                 *(f"{joint}{finger}_{turn}" for finger in "2345"
                   for joint, turn in (("mcp", "flexion"), ("mcp", "abduction"), ("pm", "flexion"), ("md", "flexion"))))


def both_hands_rest_their_fingers_alike():
    """`fingers()` gives every FINGERS stem on both sides with the same turn; every stem is one of MyoFullBody's
    finger and thumb joints (the four thumb joints and each finger's four, as the published model names them)."""
    written = calibration.fingers()
    problems = [f"{stem}: not a finger or thumb joint" for stem in calibration.FINGERS if stem not in FINGER_JOINTS]
    return problems + [f"{stem}_{side}: {written.get(f'{stem}_{side}')} against {turn}"
                       for stem, turn in calibration.FINGERS.items() for side in "lr"
                       if written.get(f"{stem}_{side}") != turn]


def a_middle_end_shifts_in_the_middle_plane():
    """A middle end's shift (`END_SHIFTS`, the person's frame, x sideways; an end with no side suffix, on the body's
    middle plane) moves its bones up, down, forward or back, never sideways; every shifted end is a segment end
    level 1 tunes (`tuning.segment_ends`)."""
    known = tuning.segment_ends(groups(), landmarks.SOMA_MIDDLES)
    return [f"{end}: not a known end" if end not in known else f"{end}: a middle end shifted sideways ({shift[0]} m)"
            for end, shift in calibration.END_SHIFTS.items()
            if end not in known or (not end.endswith(("_l", "_r")) and shift[0] != 0.0)]


# How far a wrist's end may hold the hand off SOMA's Hand joint (#70, 2026-11-06): the rule centre stands 4.5 mm off
# the joint on the template; the shift that buries the fingers moves every fingertip bone that far short of SOMA's.
WRIST_SHIFT_MOST = 0.002


def a_wrists_end_holds_the_hand_at_its_joint():
    """Each wrist's end shift (`END_SHIFTS`) is at most WRIST_SHIFT_MOST long."""
    return [f"{end}: shifted {1000 * np.linalg.norm(shift):.1f} mm off SOMA's Hand joint"
            for end, shift in calibration.END_SHIFTS.items()
            if end.startswith("wrist_") and np.linalg.norm(shift) > WRIST_SHIFT_MOST]


def the_legs_rows_are_their_landmark_lines():
    """No leg group (thigh, shin, foot, toes) is read in the skin (`INSIDE_GROUPS`): each lies on the line between
    its two end joints, the ends the fit holds on SOMA's rig joints. A skin-read shin row stood 5.5 degrees off its
    knee to ankle line, its ankle 25 mm forward of SOMA's on the mean body (#70, 2026-11-06). A foot's row levelled
    rises as its default about its hinge (`levelled_foot_problems`)."""
    legs = [group for group in calibration.INSIDE_GROUPS
            if group.removeprefix("Left").removeprefix("Right") in ("Leg", "Shin", "Foot", "ToeBase")]
    return [f"{group}: a leg group's row read in the skin" for group in legs] + levelled_foot_problems()


def levelled_foot_problems():
    """A foot's row levelled (`calibration.levelled`, #98): a row turning the body 10 degrees toes-out and 5
    toes-down, a hinge 20 degrees off the across axis (the MT heads' oblique line), SOMA's rest frame the identity and
    the default body's long axis rising 2 degrees: it comes back turned about that hinge in the body's frame alone (the
    hinge's line laid where the row laid it) so the long axis rises 2 degrees; and the feet are the groups levelled."""
    hinge = np.array([np.sin(np.radians(20.0)), 0.0, np.cos(np.radians(20.0))])
    row = Rotation.from_euler("yz", [10.0, -5.0], degrees=True).as_matrix()
    default = Rotation.from_euler("z", 2.0, degrees=True).as_matrix()
    flat = calibration.levelled(row, np.eye(3), default, hinge)
    problems = []
    if abs(flat[1, 0] - default[1, 0]) > 1e-9:
        problems.append(f"the levelled foot's long axis rises {np.degrees(np.arcsin(flat[1, 0])):.3f} degrees, not 2")
    if np.abs(flat @ hinge - row @ hinge).max() > 1e-9:
        problems.append("levelling moved the hinge's line")
    if calibration.LEVELLED != ("{Side}Foot",):
        problems.append(f"the groups levelled are {calibration.LEVELLED}, not the feet")
    return problems


CHECKS = (a_wrists_end_holds_the_hand_at_its_joint, the_legs_rows_are_their_landmark_lines, a_middle_end_shifts_in_the_middle_plane, both_hands_rest_their_fingers_alike, a_hanging_bone_is_laid_along_its_segment, the_table_keeps_both_sides_and_the_middle,
          each_twist_is_read_from_the_same_landmarks_on_both_models, the_rows_read_in_the_skin_stay_and_turn_both_sides_alike)


if __name__ == "__main__":
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    sys.exit(1 if failures else 0)
