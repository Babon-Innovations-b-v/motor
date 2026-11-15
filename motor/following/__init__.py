"""The honest measures of following SOMA (#70), from one fitted clip: each SOMA joint's position error (MyoFullBody's
joint centre against SOMA's joint, mm), each bone group's rotation error on the calibration table as it stands
(`judged_turn`: `calibration.TABLE`, which level 1's worker reads again from each evaluation's centres, the table the
stage builds its follows targets from; deg) and beside it on main's FROZEN copy (fit._frozen, never tuned on), the share of bone vertices outside SOMA's posed skin by region (every EVERY-th frame) and its smooth twin, the
depth outside (`fit._skin.depth_mm`, level 1's least-squares residual only), and beside them tip,
length and floor readings. Never a landmark distance the solver pulls on. The skin, the winding test, the regions and
the tables are the stage's own (fit._skin, calibration, fit._frozen), so the record and level 1's score are one code. Every
value is unrounded (the solver's); `evaluation.rounded` makes the copy people read (m.json).
Newton's Python (mujoco, warp)."""
import mujoco
import numpy as np
from scipy.spatial import cKDTree

from motor import calibration
from motor.contact import surface
from motor.fit._frozen import frozen_turn
from motor.fit._skin import (EVERY, HAND, bone_meshes, bone_points, depth_mm, envelope, marks, region, skin as skinned,
                       unfolded)
from motor.landmarks import soma_point
from motor.model import TO_OWN


# frozen from origin/main 5a46a96e landmarks._table.CENTRES and bone_groups: SOMA joint -> MyoFullBody centre parts;
# level 1's worker moves each group end's by the evaluation's offset (`_case.load_centres`), as the fit's own
CENTRES = {
    "hip_l": ("femur_l", (0, 0, 0)), "knee_l": ("tibia_l", (0, 0, 0)), "ankle_l": ("talus_l", (0, 0, 0)),
    "mtp_l": ("toes_l", (0, 0, 0)), "sternoclavicular_l": ("clavicle_l", (0, 0, 0)),
    "glenohumeral_l": ("humerus_l", (0, 0, 0)), "elbow_l": ("ulna_l", (0, 0, 0)), "wrist_l": ("lunate_l", (0, 0, 0)),
    "hip_r": ("femur_r", (0, 0, 0)), "knee_r": ("tibia_r", (0, 0, 0)), "ankle_r": ("talus_r", (0, 0, 0)),
    "mtp_r": ("toes_r", (0, 0, 0)), "sternoclavicular_r": ("clavicle_r", (0, 0, 0)),
    "glenohumeral_r": ("humerus_r", (0, 0, 0)), "elbow_r": ("ulna_r", (0, 0, 0)), "wrist_r": ("lunate_r", (0, 0, 0)),
    "L5_S1": ("lumbar5", (0, 0, 0)), "L3_L4": ("lumbar3", (0, 0, 0)),
    "C7_T1": ("torso", (-0.03192, 0.39675, 0.00031)), "C0_C1": ("head", (-0.00763, 0.00738, 0.00005)),
    "T12_L1": ("torso", (-0.01368, 0.11893, 0.00045)),
    "finger_tip_l": ("3distph_l", (0.00112, -0.01854, -0.00137)),
    "finger_tip_r": ("3distph_r", (0.00112, -0.01854, 0.00137)),
    "toe_tip_l": ("toes_l", (0.07692, -0.003, 0.0155)), "toe_tip_r": ("toes_r", (0.07692, -0.003, -0.0155)),
    "head_vertex": ("head", (-0.00767, 0.15057, 0)),
}
MIDDLE = {"mid_cervical": ("C7_T1", "C0_C1")}
JOINTS = {"Spine1": "L5_S1", "Spine2": "L3_L4", "Chest": "T12_L1", "Neck1": "C7_T1", "Neck2": "mid_cervical",
          "Head": "C0_C1"}
for side, letter in (("Left", "l"), ("Right", "r")):
    JOINTS.update({f"{side}Shoulder": f"sternoclavicular_{letter}", f"{side}Arm": f"glenohumeral_{letter}",
                   f"{side}ForeArm": f"elbow_{letter}", f"{side}Hand": f"wrist_{letter}",
                   f"{side}Leg": f"hip_{letter}", f"{side}Shin": f"knee_{letter}", f"{side}Foot": f"ankle_{letter}",
                   f"{side}ToeBase": f"mtp_{letter}"})
# tips: SOMA end joint carried rigidly by its segment, MyoFullBody point
TIPS = {"LeftToeEnd": ("LeftToeBase", "toe_tip_l"), "RightToeEnd": ("RightToeBase", "toe_tip_r"),
        "HeadEnd": ("Head", "head_vertex")}
FRAME_BODY = {"Hips": "pelvis", "Spine1": "lumbar4", "Spine2": "lumbar1", "Chest": "torso", "Neck1": "neck",
              "Neck2": "neck", "Head": "head"}
for side, letter in (("Left", "l"), ("Right", "r")):
    FRAME_BODY.update({f"{side}Shoulder": f"scapula_{letter}", f"{side}Arm": f"humerus_{letter}",
                       f"{side}ForeArm": f"ulna_{letter}", f"{side}Hand": f"lunate_{letter}",
                       f"{side}Leg": f"femur_{letter}", f"{side}Shin": f"tibia_{letter}",
                       f"{side}Foot": f"calcn_{letter}", f"{side}ToeBase": f"toes_{letter}"})


def body(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name.replace("-", "_").replace(" ", "_"))


def centre(model, data, name, scales):
    if name in MIDDLE:
        return sum(centre(model, data, part, scales) for part in MIDDLE[name]) / 2
    bone, local = CENTRES[name]
    index = body(model, bone)
    return data.xpos[index] + data.xmat[index].reshape(3, 3) @ (np.asarray(local) * scales[bone])


def judged_turn(segment):
    """The R_cal a group's turn is judged on (3 x 3): the calibration table as it stands, read again from the
    evaluation's centres (`_case.recalibrated`), as the stage's follows targets ride it (2026-11-01: on the frozen
    table level 1 never saw a centre move turn Spine1's calibration 6 degrees, Spine2's 10, and worsened follows)."""
    return calibration.turn(segment)


def segment_turn(world):
    """A SOMA segment's turn (..., 3, 3) as the nearest rotation to its world's 3 x 3 (the polar factor): the played
    worlds sit up to 3.5e-6 off orthonormal (heavy_kimodo, 2026-11-01), and `turn_angle`'s arccos of the trace on
    such a target read a 0.06 degree turn as 0 (the taped measure, which turns it through a quaternion, read 0.06)."""
    left, _, right = np.linalg.svd(np.asarray(world, dtype=float)[..., :3, :3])
    return left @ right


def turn_angle(seen, wanted):
    """The angle (degrees) between two turns."""
    return np.degrees(np.arccos(np.clip((np.trace(seen.T @ wanted) - 1) / 2, -1, 1)))


def summary(values):
    values = np.asarray(values, dtype=float)
    return [float(np.median(values)), float(np.percentile(values, 95)), float(values.max())]


def root_mean_square(values):
    """The per-frame values' root mean square: level 1's tuned summary, smooth where the median's middle frames swap
    (#70, 2026-10-30); the median stays the reported measure."""
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))


def measures(model, angles, rig, segments, worlds, scales):
    data = mujoco.MjData(model)
    names = [str(joint) for joint in rig["names"]]
    bind = np.asarray(rig["bind"], dtype=float)
    colliders = [geom for geom in range(model.ngeom) if model.geom_contype[geom] or model.geom_conaffinity[geom]]
    joints = {segment: [] for segment in JOINTS if segment in segments}
    tips = {tip: [] for tip in TIPS}
    hand = {side: [] for side in ("Left", "Right")}
    turns = {segment: [] for segment in segments if segment in FRAME_BODY}
    frozen = {segment: [] for segment in turns}
    lowest, skin_lowest, outside, depth, deep, by_bone, hand_parts, no_fingers = [], [], {}, {}, {}, {}, {}, {}
    folded = []  # the bone vertices' share outside on each skin frame whose skin folds (`fit._skin.unfolded`)
    faces, meshes = envelope(rig), bone_meshes(model)
    hand_tips = {side: soma_point(f"{side}Hand tip", rig) for side in hand}
    for frame, (qpos, world) in enumerate(zip(angles, worlds)):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        for segment in joints:
            placed = TO_OWN @ centre(model, data, JOINTS[segment], scales)
            joints[segment].append(1000 * np.linalg.norm(placed - world[segments.index(segment)][:3, 3]))
        for tip, (segment, myo) in TIPS.items():
            seg = world[segments.index(segment)] @ np.linalg.inv(bind[names.index(segment)])
            soma = seg[:3, :3] @ bind[names.index(tip), :3, 3] + seg[:3, 3]
            tips[tip].append(1000 * np.linalg.norm(TO_OWN @ centre(model, data, myo, scales) - soma))
        for side in hand:
            seg = world[segments.index(f"{side}Hand")] @ np.linalg.inv(bind[names.index(f"{side}Hand")])
            soma = seg[:3, :3] @ hand_tips[side] + seg[:3, 3]
            hand[side].append(1000 * np.linalg.norm(
                TO_OWN @ centre(model, data, f"finger_tip_{side[0].lower()}", scales) - soma))
        for segment in turns:
            held = TO_OWN @ data.xmat[body(model, FRAME_BODY[segment])].reshape(3, 3)
            wanted = segment_turn(world[segments.index(segment)])
            turns[segment].append(turn_angle(held @ judged_turn(segment).T, wanted))
            frozen[segment].append(turn_angle(held @ frozen_turn(segment).T, wanted))
        low = float((surface(model, data, colliders) @ TO_OWN.T)[:, 1].min())
        skin = skinned(rig, segments, world)
        lowest.append(1000 * low)
        skin_lowest.append(1000 * float(skin[:, 1].min()))
        if frame % EVERY == 0:
            own, bones = bone_points(model, data, meshes)
            parts = np.array([region(bone) for bone in bones])
            labels = np.where(np.isin(parts, HAND), "hand", parts)  # the score's regions: the hand as one
            unbent = labels != "hand"
            unbent |= np.isin(parts, ("carpals", "metacarpals"))
            out, depths = marks(own, skin, faces)
            if not unfolded(skin, rig):
                folded.append(float(out.mean()))
                continue
            tree = cKDTree(skin)
            far, _ = tree.query(own[out]) if out.any() else (np.zeros(0), None)
            for label in set(labels):
                mine = labels == label
                outside.setdefault(label, []).append(float(out[mine].mean()))
                deep.setdefault(label, []).extend(depths[mine].tolist())
                depth.setdefault(label, []).extend((1000 * far[labels[out] == label]).tolist())
            for part in HAND:
                hand_parts.setdefault(part, []).append(float(out[parts == part].mean()))
            for label in set(labels):
                mine = (labels == label) & unbent
                no_fingers.setdefault(label, []).append(float(out[mine].mean()))
            for bone in set(bones):
                by_bone.setdefault(bone, []).append(float(out[bones == bone].mean()))
    segs = {"LeftLeg": ("hip_l", "knee_l", "LeftShin"), "LeftShin": ("knee_l", "ankle_l", "LeftFoot"),
            "LeftArm": ("glenohumeral_l", "elbow_l", "LeftForeArm"), "LeftForeArm": ("elbow_l", "wrist_l", "LeftHand"),
            "LeftFoot": ("ankle_l", "mtp_l", "LeftToeBase")}
    segs.update({key.replace("Left", "Right"): (a[:-1] + "r", b[:-1] + "r", c.replace("Left", "Right"))
                 for key, (a, b, c) in list(segs.items())})
    data.qpos[:] = angles[0]
    mujoco.mj_kinematics(model, data)
    long = {key: 1000 * float(np.linalg.norm(centre(model, data, b, scales) - centre(model, data, a, scales))
                              - np.linalg.norm(bind[names.index(c), :3, 3] - bind[names.index(key), :3, 3]))
            for key, (a, b, c) in segs.items()}
    free = sum(joints.values(), [])
    return {"joints mm": summary(free), "per joint mm": {key: summary(value)[:2] for key, value in joints.items()},
            "turns deg": summary(sum(turns.values(), [])),
            "per group deg": {key: summary(value)[:2] for key, value in turns.items()},
            "per joint rms mm": {key: root_mean_square(value) for key, value in joints.items()},
            "per group rms deg": {key: root_mean_square(value) for key, value in turns.items()},
            "turns deg, main's frozen table": summary(sum(frozen.values(), [])),
            "per group deg, main's frozen table": {key: summary(value)[:2] for key, value in frozen.items()},
            "joint length over SOMA mm": long, "tips mm": {key: summary(value)[:2] for key, value in tips.items()},
            "hand tips mm": {key: summary(value)[:2] for key, value in hand.items()},
            "lowest collider mm": [float(np.median(lowest)), float(np.min(lowest))],
            "skin lowest mm": [float(np.median(skin_lowest)), float(np.min(skin_lowest))],
            "collider under skin mm": float(np.median(np.array(skin_lowest) - np.array(lowest))),
            "bones outside skin %": {key: 100 * float(np.mean(value)) for key, value in outside.items()},
            "bones outside depth mm": {key: depth_mm(value) for key, value in deep.items()},
            "outside depth p95 mm": {key: float(np.percentile(value, 95)) if value else 0.0
                                     for key, value in depth.items()},
            "bones outside all %": 100 * float(np.mean(sum(outside.values(), []))) if outside else None,
            "skin folded frames": {"frames": len(folded), "all": 100 * float(np.mean(folded)) if folded else 0.0},
            "hand parts outside %": {key: 100 * float(np.mean(value)) for key, value in hand_parts.items()},
            "bones outside no fingers %": 100 * float(np.mean(sum(no_fingers.values(), []))) if outside else None,
            "bones outside by bone %": {key: 100 * float(np.mean(value)) for key, value in by_bone.items()}}


