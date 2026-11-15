"""The honest measures level 1 scores on, on the card (#70, the card build plan's step 3): each SOMA joint's position
error (mm), each bone group's rotation error on the calibration table as it stands (`evaluation._measure.judged_turn`,
deg; beside it on main's frozen table) and the share of bone vertices outside SOMA's
posed skin by region, as `evaluation._measure.measures` gives them ("joints mm", "per joint mm", "turns deg",
"per group deg", "bones outside skin %", "bones outside depth mm", "bones outside all %"); its other readings (tips, lengths, floor, depths)
stay on the CPU. The frames' poses by MuJoCo's float64 kinematics (`Posed`), the joint and turn errors one Warp
kernel each in float64, the winding test Warp's mesh query on the card over every EVERY-th frame against the skin posed once per case
(`Skins`: identical each evaluation); medians on the host from the per-frame arrays, as the CPU's `summary`.
Newton's Python (mujoco, warp) and a CUDA card."""
import hashlib

import mujoco
import numpy as np
import warp

from motor import following as cpu
from motor.fit._frozen import frozen_turn
from motor.fit._skin import (EVERY, HAND, marks_kernel, bone_meshes, bone_points, depth_mm, envelope, region, skin as skinned,
                       unfolded)
from motor.model import TO_OWN

DEVICE = "cuda:0"


@warp.kernel
def _joint_errors(xpos: warp.array2d(dtype=warp.vec3d), xmat: warp.array2d(dtype=warp.mat33d),
                  bodies: warp.array2d(dtype=int), locals_: warp.array2d(dtype=warp.vec3d),
                  wanted: warp.array2d(dtype=warp.vec3d), out: warp.array2d(dtype=warp.float64)):
    frame, joint = warp.tid()
    first, second = bodies[joint, 0], bodies[joint, 1]
    placed = warp.float64(0.5) * (xpos[frame, first] + xmat[frame, first] @ locals_[joint, 0]
                    + xpos[frame, second] + xmat[frame, second] @ locals_[joint, 1])
    out[frame, joint] = warp.float64(1000.0) * warp.length(placed - wanted[frame, joint])


@warp.kernel
def _turn_errors(xmat: warp.array2d(dtype=warp.mat33d), bodies: warp.array(dtype=int),
                 wanted: warp.array2d(dtype=warp.mat33d), out: warp.array2d(dtype=warp.float64)):
    frame, group = warp.tid()
    rel = warp.transpose(xmat[frame, bodies[group]]) @ wanted[frame, group]
    cosine = warp.clamp(warp.float64(0.5) * (warp.trace(rel) - warp.float64(1.0)), warp.float64(-1.0), warp.float64(1.0))
    out[frame, group] = warp.degrees(warp.acos(cosine))


POSED = {}  # {`posed_key`: float32 skin points per skin frame}, a clip's own, shared by every nudged set


def posed_key(rig, segments, worlds):
    """What `skinned` reads: the rig's arrays, the segments and the clip's worlds (a set changes none of them)."""
    digest = hashlib.sha256(np.ascontiguousarray(worlds, dtype=float).tobytes())
    for key in ("names", "weights", "bind", "points"):
        digest.update(key.encode() + np.ascontiguousarray(np.asarray(rig[key])).tobytes())
    return digest.hexdigest(), tuple(segments)


def posed_skins(rig, segments, worlds, frames):
    """SOMA's skin points (float32) on each of `frames`, posed once per `posed_key` and read-only after."""
    key = posed_key(rig, segments, worlds)
    if key not in POSED:
        POSED[key] = [skinned(rig, segments, worlds[frame]).astype(np.float32) for frame in frames]
        for points in POSED[key]:
            points.flags.writeable = False  # shared by later sets: a write in place would reach them
    return POSED[key]


KEPT = {}  # {`posed_key`: whether each skin frame's skin is kept (`fit._skin.unfolded`)}, a clip's own


def kept_skins(rig, segments, worlds, frames):
    """Whether each of `frames`' posed skins is kept in the skin rows and the share (`fit._skin.unfolded`), once per
    `posed_key`."""
    key = posed_key(rig, segments, worlds)
    if key not in KEPT:
        KEPT[key] = np.array([unfolded(points, rig) for points in posed_skins(rig, segments, worlds, frames)])
    return KEPT[key]


class Skins:
    """SOMA's skin posed on every EVERY-th frame of one case, each a Warp mesh on the card (posed once per clip), and
    which are `kept` (`kept_skins`)."""

    def __init__(self, rig, segments, worlds):
        self.frames = np.arange(0, len(worlds), EVERY)
        faces = warp.array(envelope(rig).ravel(), dtype=int, device=DEVICE)
        self.meshes = [warp.Mesh(points=warp.array(points, dtype=warp.vec3, device=DEVICE), indices=faces,
                                 support_winding_number=True)
                       for points in posed_skins(rig, segments, worlds, self.frames)]
        self.kept = kept_skins(rig, segments, worlds, self.frames)
        self.ids = warp.array(np.array([mesh.id for mesh in self.meshes], dtype=np.uint64), dtype=warp.uint64,
                              device=DEVICE)


class Posed:
    """Every frame's body and geom poses by MuJoCo's own float64 kinematics on the host (the CPU measure's; with MuJoCo
    Warp's float32 poses one bone vertex fell across the skin on heavy_kimodo, 2026-10-29), the body poses
    on the card in float64, and the kept bone vertices of the skin frames as `fit._skin.bone_points` places them."""

    def __init__(self, model, angles, frames):
        data, meshes = mujoco.MjData(model), bone_meshes(model)
        xpos, xmat, points = [], [], []
        for frame, qpos in enumerate(angles):
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            xpos.append(data.xpos.copy())
            xmat.append(data.xmat.reshape(-1, 3, 3).copy())
            if frame in frames:
                own, self.bones = bone_points(model, data, meshes)
                points.append(own)
        self.xpos = warp.array(np.array(xpos), dtype=warp.vec3d, device=DEVICE)
        self.xmat = warp.array(np.array(xmat), dtype=warp.mat33d, device=DEVICE)
        self.points = warp.array(np.array(points, dtype=np.float32), dtype=warp.vec3, device=DEVICE)


def joint_table(model, joints, scales):
    """Each measured joint's two (body, local offset) parts in MuJoCo's frame (a single centre twice, a middle its
    two ends), as `evaluation._measure.centre` reads them."""
    def part(name):
        bone, local = cpu.CENTRES[name]
        return cpu.body(model, bone), np.asarray(local, dtype=float) * scales[bone]
    pairs = [[part(name) for name in cpu.MIDDLE.get(cpu.JOINTS[segment], (cpu.JOINTS[segment],) * 2)]
             for segment in joints]
    return (np.array([[body for body, _ in pair] for pair in pairs], dtype=np.int32),
            np.array([[local for _, local in pair] for pair in pairs], dtype=float))


def joint_errors(model, data, worlds, segments, scales):
    """{segment: [mm per frame]}: each SOMA joint's miss, MyoFullBody's centre against SOMA's joint."""
    joints = [segment for segment in cpu.JOINTS if segment in segments]
    bodies, locals_ = joint_table(model, joints, scales)
    wanted = np.asarray(worlds)[:, [segments.index(segment) for segment in joints]][..., :3, 3] @ TO_OWN
    out = warp.zeros((len(worlds), len(joints)), dtype=warp.float64, device=DEVICE)
    warp.launch(_joint_errors, dim=out.shape, device=DEVICE,
                inputs=[data.xpos, data.xmat, warp.array(bodies, dtype=int, device=DEVICE),
                        warp.array(locals_, dtype=warp.vec3d, device=DEVICE),
                        warp.array(wanted, dtype=warp.vec3d, device=DEVICE), out])
    return dict(zip(joints, out.numpy().T.astype(float)))


def turn_errors(model, data, worlds, segments, table=cpu.judged_turn):
    """{segment: [deg per frame]}: each bone group's rotation off SOMA's segment on `table` (a group's R_cal, 3 x 3:
    the table as it stands, else main's frozen one)."""
    groups = [segment for segment in segments if segment in cpu.FRAME_BODY]
    bodies = np.array([cpu.body(model, cpu.FRAME_BODY[segment]) for segment in groups], dtype=np.int32)
    turned = np.stack([TO_OWN.T @ cpu.segment_turn(np.asarray(worlds)[:, segments.index(segment)]) @ table(segment)
                       for segment in groups], axis=1)
    out = warp.zeros((len(worlds), len(groups)), dtype=warp.float64, device=DEVICE)
    warp.launch(_turn_errors, dim=out.shape, device=DEVICE,
                inputs=[data.xmat, warp.array(bodies, dtype=int, device=DEVICE),
                        warp.array(turned, dtype=warp.mat33d, device=DEVICE), out])
    return dict(zip(groups, out.numpy().T.astype(float)))


def outside_marks(data, skins):
    """([skin frames x points] whether each kept bone vertex is outside the skin posed on that frame, its depth
    outside in m), by `fit._skin`'s own kernel."""
    signs, depths = (warp.zeros(data.points.shape, dtype=float, device=DEVICE) for _ in range(2))
    warp.launch(marks_kernel, dim=signs.shape, device=DEVICE, inputs=[skins.ids, data.points, signs, depths])
    return signs.numpy() >= 0, depths.numpy().astype(float)


def scored_labels(bones):
    """Each kept bone vertex's scored region, the hand's parts as one ("hand")."""
    parts = np.array([region(bone) for bone in bones])
    return np.where(np.isin(parts, HAND), "hand", parts)


def skin_shares(marks, bones):
    """{label: [share per skin frame]} of the bone vertices outside, the hand's parts as one ("hand")."""
    labels = scored_labels(bones)
    return {label: [float(shot[labels == label].mean()) for shot in marks]
            for label in sorted(set(labels))}  # sorted: a set of strings orders by the process's hash seed


def skin_depths(depths, bones):
    """{label: mm} level 1's skin residual per region (`fit._skin.depth_mm` over every skin frame's points)."""
    labels = scored_labels(bones)
    return {label: depth_mm(depths[:, labels == label]) for label in sorted(set(labels))}


def measures(model, angles, rig, segments, worlds, scales, skins=None):
    """The score's measures of one fitted clip, keyed and summarised as `evaluation._measure.measures`."""
    skins = skins or Skins(rig, segments, worlds)
    data = Posed(model, angles, set(skins.frames.tolist()))
    joints = joint_errors(model, data, worlds, segments, scales)
    turns = turn_errors(model, data, worlds, segments)
    frozen = turn_errors(model, data, worlds, segments, frozen_turn)
    marks, depths = outside_marks(data, skins)
    unfolded_any = bool(skins.kept.any())  # none: the clip's skin shares are no measure (`evaluation.case_score`)
    outside = skin_shares(marks[skins.kept], data.bones) if unfolded_any else {}
    folded = marks[~skins.kept]
    return {"joints mm": cpu.summary(np.concatenate(list(joints.values()))),
            "per joint mm": {key: cpu.summary(value)[:2] for key, value in joints.items()},
            "turns deg": cpu.summary(np.concatenate(list(turns.values()))),
            "per group deg": {key: cpu.summary(value)[:2] for key, value in turns.items()},
            "per joint rms mm": {key: cpu.root_mean_square(value) for key, value in joints.items()},
            "per group rms deg": {key: cpu.root_mean_square(value) for key, value in turns.items()},
            "turns deg, main's frozen table": cpu.summary(np.concatenate(list(frozen.values()))),
            "per group deg, main's frozen table": {key: cpu.summary(value)[:2] for key, value in frozen.items()},
            "bones outside skin %": {key: 100 * float(np.mean(value)) for key, value in outside.items()},
            "bones outside depth mm": skin_depths(depths[skins.kept], data.bones) if unfolded_any else {},
            "bones outside all %": 100 * float(np.mean(sum(outside.values(), []))) if unfolded_any else None,
            "skin folded frames": {"frames": len(folded),
                                   "all": 100 * float(folded.mean()) if len(folded) else 0.0}}

