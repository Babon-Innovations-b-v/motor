"""The static calibration table (#70; pure numpy): for each bone group, the fixed turn between SOMA's segment frame
and MyoFullBody's group frame (`turn`), so the group's body turns as R_body = R_segment @ R_cal. It is read once from
the two models' rest poses, SOMA's rest skeleton (its template rig's bind) against MyoFullBody's default pose (its
couplings applied: MyoSuite's `stand`), the same for every person and every clip, never fitted, so it cannot absorb a
person's posture such as a curved spine. `_measure.py` (Newton's Python) measures it again and prints `TABLE`.

`aligned` is the maths: each rest's anatomical axes (columns: the long axis, from the stretch's start to its end, the
reference axis crossed with it, and the reference axis made square to it; `frame`), in SOMA's segment frame (a_s)
and in the group body's frame (a_m), R_cal = a_s a_m^T: the body frame in the segment frame when the two rests' bones
lie along each other. The long axis runs between the group's two end joints, the same ends sizing measures (SOMA's
rig joints, MyoFullBody's joint centres). The reference fixes the twist about the bone (`REFERENCES`), from the same
landmarks on both models (since 2026-10-29, #70; before, SOMA's bind z of the forearm and shin and MyoFullBody's elbow
and knee axes, which left the upper arm turned about 9 degrees on every body): the humerus' epicondyles for upper arm
and forearm (the ulna's flexion axis: SOMA's forearm does not pronate), the femur's for the thigh, the malleoli for
the shin, MC5 head to MC2 head for the hand, MT1 head to MT5 head for the foot and toes (since 2026-11-04); the
person's own x (left) for the trunk, own y (up) for the shoulder. The toes' hinge's two groups (`HINGED`) keep the reference
exact and square their long axis to it; the lumbar pair takes the whole lumbar's line (`LONG_LINES`); the feet are
then levelled about their hinge so a flat rest foot is a flat rest foot (`LEVELLED`).

Against taking SOMA's rest as MyoFullBody's default literally, it differs by the two rests' posture (measured
2026-10-28): the arms 48 degrees (SOMA's A-pose), toes 26, shin 12.6, shoulder 12, Spine2 12, Spine1 9.6, head 6.7,
hips 6.1, thigh 5.3, foot 7.2, chest 3.6. SOMA's bind has its knees bent about 15 degrees."""
import numpy as np
from scipy.spatial.transform import Rotation

# Each group's reference axis on SOMA and on MyoFullBody: "own x"/"own y" the person's own axis, "line <a> <b>" the
# line from landmark a to landmark b (`landmarks.LANDMARKS`: on SOMA its point on the template's skin, on MyoFullBody
# its bone point), the same two landmarks on both models.
_ELBOW = "line humerus_medial_epicondyle_{side} humerus_lateral_epicondyle_{side}"
_KNEE = "line femur_medial_epicondyle_{side} femur_lateral_epicondyle_{side}"
_TOES = "line MT1_head_{side} MT5_head_{side}"
REFERENCES = {"Hips": ("own x", "own x"), "Spine1": ("own x", "own x"), "Spine2": ("own x", "own x"),
              "Chest": ("own x", "own x"), "Neck1": ("own x", "own x"), "Neck2": ("own x", "own x"),
              "Head": ("own x", "own x"), "{Side}Shoulder": ("own y", "own y"),
              "{Side}Arm": (_ELBOW, _ELBOW), "{Side}ForeArm": (_ELBOW, _ELBOW),
              "{Side}Hand": ("line MC5_head_{side} MC2_head_{side}",) * 2, "{Side}Leg": (_KNEE, _KNEE),
              "{Side}Shin": ("line medial_malleolus_{side} lateral_malleolus_{side}",) * 2,
              "{Side}Foot": (_TOES, _TOES), "{Side}ToeBase": (_TOES, _TOES)}
# The groups on each side of a hinge (the toes' mtp) take their reference exact, the line the hinge turns about, and
# their long axis made square to it (`hinged`), so the parent's and the child's rest frames share the hinge on both
# models and the two rests differ only by a turn the hinge makes (#70, 2026-11-04). Long axis first and own x, the
# toes' pitch against the foot differed between the two models by a turn the oblique mtp hinge does not make: a fixed
# 15 degrees on each toe, every frame. Not the elbow: hinged there, the forearm's bones lay along MyoFullBody's carrying
# angle, 12 degrees off SOMA's forearm (env1 standing: its turn 12.2 -> 0.7 degrees, but the forearm's bones outside
# SOMA's skin 1.9 -> 12.6 %), so it stays on its line, its fixed 12 degrees the hinge's (`LACKS`).
HINGED = ("{Side}Foot", "{Side}ToeBase")
# The lumbar pair's long axis: the whole lumbar's line on both models (SOMA's Spine1 to Chest joints, MyoFullBody's
# L5/S1 to T12/L1 centres), not each level's own (#70, 2026-11-04). MyoFullBody's lumbar is one joint spread over five
# levels by fixed shares (`LACKS`), so its rest curvature between the levels is not a turn any joint makes: each level
# on its own line left Spine1 a fixed 15 degrees and Spine2 18 off SOMA's in flexion, every frame.
LONG_LINES = {"Spine1": (("Spine1", "Chest"), ("L5_S1", "T12_L1")), "Spine2": (("Spine1", "Chest"), ("L5_S1", "T12_L1"))}
# The feet's rows turned about their hinge line (their reference, which the toes share, so the hinge stays exact) until
# SOMA's bind foot, which stands flat on its floor, is MyoFullBody's calcaneus standing flat on its floor at its default
# pose: the long axis rising as the default's does (`levelled`; #98, 2026-11-06: "a foot the floor carries lies flat").
# On the landmark lines alone SOMA's flat bind foot was a calcaneus 3.0 degrees toes-down (the two models' feet differ
# in shape: the lines' rise over the sole is each model's own), so the sim body's feet stood toes-down on every frame
# where SOMA's sole lay flat.
LEVELLED = ("{Side}Foot",)
# R_cal per bone group as a quaternion (x, y, z, w), printed by `_measure.py` on 2026-10-29 (MyoFullBody of
# MuscleMimic 1.0.6, SOMA's template rig of people-clips-make; the shoulders again that day, their start the
# sternoclavicular centre, no longer the landmark of that name; the hands again on 2026-11-01, MC5_head moved onto the
# little finger's knuckle: the hand turned 14.9 degrees about its long axis). INSIDE_GROUPS' rows read in the skin on
# 2026-11-01 (`_measure.table_rows`): Head 6.6 degrees off its line, Shins 5.5 (on their line since 2026-11-06), ForeArms 0.1, Hips 0.2. The lumbar
# pair, feet and toes printed again on 2026-11-04 (`HINGED`, `LONG_LINES`, the MT heads line): Spine1 13.5 and Spine2
# 8.3 degrees off their rows before, the toes 13.5, the feet 2.2. The feet levelled on 2026-11-06 (`LEVELLED`): 3.0
# degrees about their hinge.
TABLE = {
    "Hips": (0.664729, 0.747084, 0.000268241, -0.000291347),
    "Spine1": (0.675027, 0.737792, 0.00121172, 0.000810537),
    "Spine2": (0.652946, 0.757403, 0.00123513, 0.000774402),
    "Chest": (0.683552, 0.729902, 0.000118136, -0.000455094),
    "Neck1": (0.770799, 0.637077, -0.000728835, -0.00110972),
    "Neck2": (0.770799, 0.637077, -0.000728836, -0.00110972),
    "Head": (0.6876, 0.72609, 0.000171, -0.000396),
    "LeftShoulder": (-0.446904, -0.498353, 0.576873, 0.468122),
    "RightShoulder": (0.498428, -0.446843, -0.46818, 0.576809),
    "LeftArm": (-0.00349838, 0.0330295, 0.698869, 0.714478),
    "RightArm": (0.0330002, 0.00346837, 0.714478, -0.698871),
    "LeftForeArm": (-0.0687563, -0.0554584, 0.679663, 0.728186),
    "RightForeArm": (-0.0555178, 0.0686927, 0.728193, -0.679658),
    "LeftHand": (-0.0829827, -0.0998119, 0.680886, 0.720795),
    "RightHand": (-0.0999123, 0.0829961, 0.720768, -0.680899),
    "LeftLeg": (-0.70709, 0.707102, 0.00266821, 0.00488039),
    "RightLeg": (0.707102, 0.70709, 0.00483335, -0.00262117),
    "LeftShin": (-0.698192, 0.715865, -0.005736, -0.005594),
    "RightShin": (0.715866, 0.698193, -0.005569, 0.00571),
    "LeftFoot": (0.98282, -0.182204, -0.00590453, -0.0288245),
    "RightFoot": (0.182209, 0.982812, 0.0290654, -0.00597333),
    "LeftToeBase": (0.999263, -0.0345712, -0.0128052, 0.0106455),
    "RightToeBase": (0.0345713, 0.999265, -0.0104016, -0.0128581),
}

# The held fingers' rest turns (radians, each joint by its name without its side, both sides alike): MyoFullBody's
# finger and thumb joints turned so its hand at rest has SOMA's template hand's shape (SOMA has no finger joints, its
# hand one rigid shape; MyoFullBody's fingers are held where they stand, `fit`), each finger bone as deep in SOMA's
# skin as its joints' ranges let it with the hand laid by TABLE at SOMA's Hand joint, as the fit holds it
# (`_measure.fingers_measured`); laid in the person's file by `myo._rested`. Never per
# person or per clip. Printed by `_measure.py` on 2026-11-01 (the hands' turns of that day): on the template, 67.6 %
# of the finger bones' vertices outside SOMA's skin at the published zero pose, 24.7 % at the first read (the hand at
# SOMA's Hand joint; the thumb 25 to 45 mm out before, inside after; the index stays out: SOMA's stands dorsal of where
# its metacarpal puts it, and its knuckle then did not bend back, range 0 to 90 degrees). Read again with the hand placed
# (#70, 2026-11-01): the fingers bend 6 to 13 degrees more at the knuckles, the thumb's CMC 11 less; on the mean body's
# 15 clips the hand 9.43 -> 5.42 % outside (forearm 2.87 -> 4.27, shoulder 2.07 -> 1.51), all 6.95 -> 6.59 %, the turns
# 4.603 -> 4.587 degrees, joints 14.14 -> 13.43 mm; env1 standing all 13.48 -> 13.43 %, seated 13.82 -> 13.82 %.
# Read again on 2026-11-01 with the finger knuckles bending back 45 degrees (`myo._movable.KNUCKLES`).
# Read again on 2026-11-06 (#70, the owner: "hands just about missing the finger tips") on the mean body's rig with
# TABLE as it stands and the hand no longer moved as one body: read for a hand moved off the joint, the 11-01 turns left
# the thumb, index and little finger out at SOMA's Hand joint (19.4 % of the hand's bone vertices), so the wrist's end
# shift read 18 mm up the forearm to bury them and every fingertip bone stood 20 to 22 mm short of SOMA's fingertip;
# with these the hand stands 1.5 % outside at the joint unshifted, the tips 4.6 to 9.0 mm inside SOMA's (thumb to
# little finger; the little's 3.7 past). The fit on 12 frames of the mean body's pickup and jump and env1's standing
# (mink, this PC's processor): the thumb to ring fingertips 19.4 to 28.9 mm off SOMA's -> 6.5 to 13.4,
# the little's 10.8 to 13.1 -> 2.5 to 3.5, the wrists 16.8 to 19.5 mm off
# SOMA's Hand joint -> 4.0 to 4.4, the hand 1.76 / 1.66 / 0.94 % outside -> 1.10 / 1.08 / 1.03.
FINGERS = {
    "cmc_abduction": -0.39,
    "cmc_flexion": 0.4976,
    "ip_flexion": -0.0759,
    "mcp2_abduction": 0.0428,
    "mcp2_flexion": -0.4287,
    "mcp3_abduction": -0.0606,
    "mcp3_flexion": -0.2007,
    "mcp4_abduction": -0.0175,
    "mcp4_flexion": -0.1252,
    "mcp5_abduction": -0.1793,
    "mcp5_flexion": -0.1193,
    "md2_flexion": 0.1374,
    "md3_flexion": 0.1268,
    "md4_flexion": 0.1785,
    "md5_flexion": 0.0286,
    "mp_flexion": -0.2805,
    "pm2_flexion": 0.5017,
    "pm3_flexion": 0.3977,
    "pm4_flexion": 0.3817,
    "pm5_flexion": 0.3606,
}


def fingers():
    """FINGERS written out for both sides ({joint: rest turn})."""
    return {f"{stem}_{side}": angle for stem, angle in FINGERS.items() for side in "lr"}


def turn(group):
    """The group's calibration turn R_cal (3 x 3): the group body's frame in its SOMA segment's frame."""
    return Rotation.from_quat(TABLE[group]).as_matrix()


# The groups whose row is read by laying the group's bones in SOMA's skin at rest (`_measure.inside_measured`: its
# bones moved rigidly to their least depth outside SOMA's template skin, the turn kept; the owner and the coordinator,
# 2026-11-01, for the skull and jaw, then every group where it lowers the bones outside at rest without worsening the
# turns check on the mean body's 15 clips) instead of the landmark lines (#70). Each side named, both sides alike.
# Level 1's worker reads the other rows again from its centres (`remeasured`); these stay as written.
# The feet left it on 2026-11-04: their row is their hinge's line (`HINGED`), whose shared turn a read in the skin
# would break. The shins left it on 2026-11-06 (#70, the owner: "the shin being slightly to much forward and out of the
# skin"): their skin-read row stood 5.5 degrees off their knee to ankle line, so a shin laid from SOMA's knee put its
# ankle 25 mm forward of SOMA's ankle joint and its malleoli 22 to 26 mm in front of SOMA's (mean body's bind), and the
# knee's end read in the skin with it (0.004, 0.006, -0.024) lay on the read's 24 mm reach, holding the tibia's top
# 24 mm behind SOMA's knee. On its line (`_measure.measured`, the template rig of l1n), every leg row is its line's.
INSIDE_GROUPS = ("Hips", "LeftForeArm", "RightForeArm")

# The groups whose row is laid on their landmarks at rest (`_measure.laid_measured`: the head's turn about its body's z
# and its C0_C1's shift solved so its landmarks' markers, the face's nasion and menton among them, meet SOMA's template
# skin; #70, 2026-11-06). Read in the skin before (INSIDE_GROUPS), the least depth outside turned the skull 9.8 degrees
# chin up against SOMA's face: its nasion, chin and menton stood 31, 34 and 30 mm off SOMA's at rest, 25 mm high and
# the same on every clip (the owner: "nose isnt at nose, chin isnt at chin").
LAID_GROUPS = ("Head",)

# A group end's shift read in SOMA's skin (`_measure.end_shift_measured`; m, the person's frame, Y up): the marker at
# that MyoFullBody centre is moved by -shift (`fit.end_marker`), so SOMA's rig joint holds the group's bones that far off
# it (#70, 2026-11-01). The head's: the skull and jaw laid at the Head row on SOMA's template, 20.6 % of their vertices
# outside its skin at its Head joint, 2.5 % at this shift; on the mean body's 15 clips the head 17.4 -> 13.0 %, env1's
# standing 6.7 -> 3.0 %, seated 5.4 -> 2.7 %, the turns within 0.01 degree. Since 2026-11-04 every end's shift is level
# 1's (run l1n, its 21 tuned ends to the micrometre, an end it left at zero not written; END_READS its starts): the
# region average of bones outside SOMA's skin 3.97 -> 3.55 % on env1 standing, 10.85 -> 9.73 % seated, 6.41 -> 6.04 %
# on the mean body's 15 clips. The wrists' unshifted since 2026-11-06 (#70): level 1's 18 mm up the forearm buried the
# stale FINGERS' thumb and index and held every fingertip bone 20 to 22 mm short of SOMA's; read again with the fingers
# of that day it is 6.6 mm, which keeps 1.1 % more of the hand's bones inside at the cost of 3 mm of every fingertip's
# reach, so the wrist stays at SOMA's Hand joint (the rule centre stands 4.5 mm off it on the template). Static like the
# table: never per person.
END_SHIFTS = {"L5_S1": (0.0, 6e-06, 1e-05), "T12_L1": (0.0, 0.00403, -0.007995), "C7_T1": (0.0, -0.00197, 0.006004),
              "mid_cervical": (0.0, 0.000142, 6e-05), "C0_C1": (0.0, 0.000417, 0.000517),
              "sternoclavicular_l": (1.2e-05, -3e-05, 0.00013), "sternoclavicular_r": (-1.2e-05, -3e-05, 0.00013),
              "glenohumeral_l": (-7.2e-05, 3.7e-05, 1e-06), "glenohumeral_r": (7.2e-05, 3.7e-05, 1e-06),
              "elbow_l": (-7.3e-05, 2.2e-05, 0.002024), "elbow_r": (7.3e-05, 2.2e-05, 0.002024),
              "wrist_l": (0.0, 0.0, 0.0), "wrist_r": (0.0, 0.0, 0.0),
              "hip_l": (7.6e-05, 0.000168, -0.000406), "hip_r": (-7.6e-05, 0.000168, -0.000406),
              "knee_l": (-0.002, 0.004, -0.006), "knee_r": (0.002, 0.004, -0.006),
              "ankle_l": (0.0, -0.006, 0.002), "ankle_r": (0.0, -0.006, 0.002),
              "mtp_l": (-0.006, 0.002, -0.012), "mtp_r": (0.006, 0.002, -0.012)}
# The leg's three ends (knee, ankle, mtp) are their reads with the shins on their line (2026-11-06, #70, the left's
# read mirrored; END_READS), not level 1's l1n values, which were tuned on the skin-read shin row: knee (4, 5, -24.5)
# mm, ankle (-1, -6, 1.5), mtp (-21, 4.8, -18). With them the calcaneus sat 11 to 15 mm off its place on the talus
# (back and up: the heel high and out of the skin's back) and the talus 7 mm forward of SOMA's ankle. Measured on the
# mean body's pickup and jump and env1's standing (18 frames each, level 3 by mink): knee joint 13 to 17 -> 4 mm, ankle
# 7 to 9 -> 1.4 to 2.9, the calcaneus' move 12 to 15 -> 6 to 7 mm, the calcaneus' bones outside 7.5 to 10 -> 4 to 7.5 %,
# the tibia's 2.5 to 5 -> 2 to 4.4 %; the toes' 5 to 12 -> 12 to 15 %. Level 1's to tune again from here.
# Every group's start end's shift read the same way (`_measure.end_shift_measured` of the group it starts, SOMA's
# template rig, MyoFullBody sized to it at its default pose; #70, 2026-11-02 and 11-03): level 1's starts for its
# segment end unknowns (`tuning._ends.mirrored_starts`, a pair's mean mirrored, a middle's sideways part dropped:
# T12_L1 read 10 mm sideways), never read by the stage. Neck2 starts no bones of its own (mid_cervical: none read).
END_READS = {"L5_S1": (0.0, 0.0, 0.0), "L3_L4": (0.0, 0.0, 0.0), "T12_L1": (0.01, 0.004, -0.008),
             "C7_T1": (0.0, -0.002, 0.006), "C0_C1": (0.0, 0.0, 0.001),
             "sternoclavicular_l": (0.0, 0.0, 0.0), "sternoclavicular_r": (0.0, 0.0, 0.0),
             "glenohumeral_l": (0.0, 0.0, 0.0), "glenohumeral_r": (0.0, 0.0, 0.0),
             "elbow_l": (0.0, 0.0, 0.002), "elbow_r": (0.0, 0.0, 0.002),
             "wrist_l": (-0.002, 0.006, -0.002), "wrist_r": (0.002, 0.006, -0.002),
             "hip_l": (0.0, 0.0, 0.0), "hip_r": (0.0, 0.0, 0.0),
             "knee_l": (-0.002, 0.004, -0.006), "knee_r": (0.002, 0.004, -0.006),
             "ankle_l": (0.0, -0.006, 0.002), "ankle_r": (0.0, -0.006, 0.002),
             "mtp_l": (-0.006, 0.002, -0.012), "mtp_r": (0.006, 0.002, -0.012)}

MIRROR = np.diag([-1.0, -1.0, 1.0])  # a right body is its left one mirrored in z: R_right = R_left @ MIRROR


def remeasured(measured):
    """TABLE with `measured`'s rows ({group: quaternion}, the landmark lines read again) for every group but
    INSIDE_GROUPS' and LAID_GROUPS', whose rows are read in the skin or laid on their landmarks and kept."""
    return {group: TABLE[group] if group in INSIDE_GROUPS + LAID_GROUPS else tuple(quaternion)
            for group, quaternion in {**TABLE, **measured}.items()}


def turned(group, rotvec):
    """R_cal of `group` turned by `rotvec` (radians, about the group body's own axes), as a quaternion (x, y, z, w):
    TABLE's turn times exp(rotvec), a right group's rotvec its left one's mirrored (MIRROR @ rotvec), so one rotvec
    turns both sides alike and the table keeps its mirror."""
    rotvec = MIRROR @ np.asarray(rotvec, dtype=float) if group.startswith("Right") else np.asarray(rotvec, dtype=float)
    return Rotation.from_matrix(turn(group) @ Rotation.from_rotvec(rotvec).as_matrix()).as_quat()


def frame(long, reference):
    """Anatomical axes (3 x 3, columns): the long axis, the reference crossed with it, the reference made square."""
    along = np.asarray(long, dtype=float) / np.linalg.norm(long)
    across = np.asarray(reference, dtype=float) - (np.asarray(reference, dtype=float) @ along) * along
    across = across / np.linalg.norm(across)
    return np.column_stack([along, np.cross(across, along), across])


def hinged(long, reference):
    """A hinge group's anatomical axes (3 x 3, columns, as `frame`'s): the long axis made square to the reference,
    the reference crossed with it, the reference itself (the line the hinge turns about)."""
    across = np.asarray(reference, dtype=float) / np.linalg.norm(reference)
    along = np.asarray(long, dtype=float) - (np.asarray(long, dtype=float) @ across) * across
    along = along / np.linalg.norm(along)
    return np.column_stack([along, np.cross(across, along), across])


def levelled(row, segment_turn, body_turn, hinge):
    """R_cal `row` (3 x 3) turned about `hinge` (the reference line, in the body's frame) so that the body turned by it
    at SOMA's rest (`segment_turn` R_cal) has its long axis (its own x) rising as the body at its default pose
    (`body_turn`, both in one shared frame, y up) does; the least such turn within 30 degrees."""
    from scipy.optimize import brentq

    hinge = np.asarray(hinge, dtype=float) / np.linalg.norm(hinge)
    rise = body_turn[1, 0]

    def miss(angle):
        return (segment_turn @ row @ Rotation.from_rotvec(angle * hinge).as_matrix())[1, 0] - rise

    angles = np.radians(np.linspace(-30.0, 30.0, 121))
    signs = np.sign([miss(angle) for angle in angles])
    roots = [brentq(miss, angles[index], angles[index + 1]) for index in np.flatnonzero(signs[:-1] * signs[1:] <= 0)]
    if not roots:
        raise ValueError("no turn about the hinge within 30 degrees levels the foot")
    return row @ Rotation.from_rotvec(min(roots, key=abs) * hinge).as_matrix()


def aligned(soma, myo, axes=frame):
    """R_cal from each rest's (frame turn, long axis, reference axis), all in one shared frame: a_s a_m^T, a_s and
    a_m the anatomical axes (`axes`: `frame`, else `hinged` for a hinge group) in the segment's and the body's frame."""
    (soma_turn, soma_long, soma_reference), (body_turn, myo_long, myo_reference) = soma, myo
    return (soma_turn.T @ axes(soma_long, soma_reference)) @ (body_turn.T @ axes(myo_long, myo_reference)).T
