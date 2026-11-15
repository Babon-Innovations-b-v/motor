"""How closely a path follows SOMA (#70's judgement, `following`), the honest measures first: the bones outside SOMA's
skin by region (`_skin`) and each bone group's turn against its SOMA segment's through main's frozen calibration table
(`_frozen`, degrees), then beside them each joint SOMA places against MyoFullBody's centre of the same joint (mm, partly
what the solver pulls on: the group ends are its targets) and the turns on the tree's own table, every frame, split by
whether a leading joint above it stands at an end of its range then; and where each SOMA segment
rides its group's frame body (`riding`: the calibration turn, its start joint on MyoFullBody's centre)."""
import mujoco
import numpy as np

from motor.bone_groups import groups
from motor.calibration import turn
from motor.fit._frozen import frozen_turn
from motor.fit._skin import bones_outside
from motor.landmarks import myo_parts
from motor.model import TO_OWN, Couplings, body_index, default_pose

AT_LIMIT = 1e-3  # radians (metres for a slide): a leading joint this near an end of its range stands at it
# SOMA joints judged: every group's start and end but the hip joints' middle (both hips are judged), the hand's tip
# and the head's and toes' ends (skin points, not joints).
NOT_JOINTS = ("hipmid", " tip", "End")


def judged():
    """[(SOMA joint, MyoFullBody centre)] once each."""
    found = {}
    for (soma_start, soma_end), (myo_start, myo_end), _, _ in groups().values():
        for soma, myo in ((soma_start, myo_start), (soma_end, myo_end)):
            if not any(part in soma for part in NOT_JOINTS):
                found[soma] = myo
    return list(found.items())


def centre(model, data, name, scales):
    """A MyoFullBody centre in MuJoCo's frame, its body-local point scaled with its body (`scales` by source name)."""
    found = np.zeros(3)
    for body, local, weight in myo_parts(name):
        index = body_index(model, body)
        found += weight * (data.xpos[index] + data.xmat[index].reshape(3, 3) @ (np.asarray(local) * scales[body]))
    return found


def _limit_joints(model, bodies, own):
    """The leading joints whose range end may stop a point on `bodies`' last: every joint on the path from the root
    (its own body's too when `own`), a follower by its leader."""
    leader = {int(model.eq_obj1id[equality]): int(model.eq_obj2id[equality]) for equality in range(model.neq)
              if model.eq_type[equality] == mujoco.mjtEq.mjEQ_JOINT}
    found, body = set(), bodies[-1]
    if not own:
        body = model.body_parentid[body]
    while body > 0:
        for joint in range(model.njnt):
            if model.jnt_bodyid[joint] == body:
                lead = leader.get(joint, joint)
                if lead >= 0 and model.jnt_limited[lead]:
                    found.add(lead)
        body = model.body_parentid[body]
    return sorted(found)


def _at_limit(model, qpos, joints):
    """Whether any of `joints` stands within AT_LIMIT of an end of its range."""
    values = qpos[model.jnt_qposadr[joints]] if joints else np.zeros(0)
    ranges = model.jnt_range[joints] if joints else np.zeros((0, 2))
    return bool(np.any((values - ranges[:, 0] < AT_LIMIT) | (ranges[:, 1] - values < AT_LIMIT)))


def _summary(values, limited):
    """[median, max, count] of the values not at a limit and of those at one."""
    values, limited = np.asarray(values), np.asarray(limited, dtype=bool)
    parts = (("free", values[~limited]), ("at a limit", values[limited]))
    return {label: ([round(float(np.median(part)), 2), round(float(part.max()), 2), int(part.size)] if part.size
                    else [None, None, 0]) for label, part in parts}


def _angle(seen, wanted):
    """The angle (degrees) between two turns."""
    return np.degrees(np.arccos(np.clip((np.trace(seen.T @ wanted) - 1.0) / 2.0, -1.0, 1.0)))


def following(model, path, worlds, segments, scales, rig):
    """The path's following of SOMA: {"bones outside skin %" by region, "turns deg, main's frozen table", "joints mm",
    "turns deg"}, the last three each {"free", "at a limit"}: [median, max, count] over every frame and joint (group),
    and per joint and per group [median, max] over the frames. `worlds` frames x segments x 4 x 4 (the person's
    frame), `scales` each body's size by its source name, `rig` SOMA's (names, bind, points, faces, weights)."""
    data, table = mujoco.MjData(model), groups()
    joints = [(soma, myo, _limit_joints(model, [body_index(model, part[0]) for part in myo_parts(myo)],
                                        any(np.any(part[1]) for part in myo_parts(myo))))
              for soma, myo in judged() if soma in segments]
    frames = [(group, body_index(model, table[group][2]), turn(group), frozen_turn(group))
              for group in table if group in segments]
    limits = {group: _limit_joints(model, [body], True) for group, body, _, _ in frames}
    misses, turns = {soma: [] for soma, _, _ in joints}, {group: [] for group, _, _, _ in frames}
    frozen = {group: [] for group, _, _, _ in frames}
    limited_joints, limited_turns = {soma: [] for soma, _, _ in joints}, {group: [] for group, _, _, _ in frames}
    for qpos, world in zip(path, worlds):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        for soma, myo, held in joints:
            placed = TO_OWN @ centre(model, data, myo, scales)
            misses[soma].append(1000 * np.linalg.norm(placed - world[segments.index(soma)][:3, 3]))
            limited_joints[soma].append(_at_limit(model, qpos, held))
        for group, body, calibration, main in frames:
            wanted = world[segments.index(group)][:3, :3]
            turns[group].append(_angle(TO_OWN @ data.xmat[body].reshape(3, 3) @ calibration.T, wanted))
            frozen[group].append(_angle(TO_OWN @ data.xmat[body].reshape(3, 3) @ main.T, wanted))
            limited_turns[group].append(_at_limit(model, qpos, limits[group]))
    shares, folded = bones_outside(model, path, worlds, segments, rig)
    return {"bones outside skin %": shares, "skin folded frames": folded,
            "turns deg, main's frozen table": _summary(sum(frozen.values(), []), sum(limited_turns.values(), [])),
            "per group deg, main's frozen table": {group: [round(float(np.median(found)), 2),
                                                           round(float(np.max(found)), 2)]
                                                   for group, found in frozen.items()},
            "joints mm": _summary(sum(misses.values(), []), sum(limited_joints.values(), [])),
            "turns deg": _summary(sum(turns.values(), []), sum(limited_turns.values(), [])),
            "per joint mm": {soma: [round(float(np.median(found)), 1), round(float(np.max(found)), 1)]
                             for soma, found in misses.items()},
            "per group deg": {group: [round(float(np.median(found)), 2), round(float(np.max(found)), 2)]
                              for group, found in turns.items()}}


def riding(model, scales, segments):
    """Each SOMA segment's frame in its group's frame body ({segment: 4 x 4}): turned by the calibration table
    (R_body = R_segment R_cal), standing at its start joint's MyoFullBody centre at the model's default pose."""
    data, table, found = default_pose(model, Couplings(model)), groups(), {}
    for segment in segments:
        (_, (start, _), frame_body, _) = table[segment]
        body = body_index(model, frame_body)
        place = data.xmat[body].reshape(3, 3).T @ (centre(model, data, start, scales) - data.xpos[body])
        found[segment] = np.eye(4)
        found[segment][:3, :3], found[segment][:3, 3] = turn(segment).T, place
    return found
