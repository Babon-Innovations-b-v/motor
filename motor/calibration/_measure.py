"""Measures the static calibration table again and prints it as `TABLE` (Newton's own Python: MuJoCo and Newton):

    <env>/bin/python motor/calibration/_measure.py <model folder> <SOMA template rig.npz>

<model folder>: the MyoFullBody folder (vendor/myofullbody); <rig.npz>: people-clips-make's template rig. Each
bone group's stretch on SOMA's bind (`landmarks.soma_point`) and on MyoFullBody at its default pose
(`landmarks.myo_parts`), each reference axis (`REFERENCES`: a landmark line on SOMA's template skin and between
MyoFullBody's bone points), then `aligned` (a hinge group's axes `hinged`, the lumbar pair's long axis
`LONG_LINES`'), the feet `levelled` about their hinge. The MyoFullBody reference is turned to agree in sign with
SOMA's. Then `FINGERS` (`fingers_measured`, on the table as it stands in `TABLE`: paste the turns first, then measure
again). `calibration.INSIDE_GROUPS`' rows are
read in SOMA's skin instead (`inside_measured`, from their rows as they stand; `table_rows`)."""
import pathlib
import sys

import mujoco
import numpy as np
import warp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motor.bone_groups import groups, sided_map  # noqa: E402
from motor import calibration  # noqa: E402
from motor.calibration import REFERENCES, aligned  # noqa: E402
from motor.landmarks import LANDMARKS, myo_parts, on_skin, soma_point  # noqa: E402
from motor.model import TO_OWN, Couplings, default_pose  # noqa: E402
from motor.myo import SOURCE  # noqa: E402

OWN = {"own x": np.array([1.0, 0.0, 0.0]), "own y": np.array([0.0, 1.0, 0.0])}


def myo_point(model, data, parts):
    """Weighted MyoFullBody body points (`myo_parts`' form) in the person's frame at the default pose."""
    found = np.zeros(3)
    for body, local, weight in parts:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        found += weight * (data.xpos[index] + data.xmat[index].reshape(3, 3) @ np.asarray(local))
    return TO_OWN @ found


def bone_point(landmark):
    """A landmark's bone point in `myo_parts`' form (its MyoFullBody body, the body-local point, weight 1)."""
    return [(LANDMARKS[landmark][3], LANDMARKS[landmark][4], 1.0)]


def soma_reference(name, rig):
    """A reference axis on SOMA's bind: an own axis, or a landmark line on the template's skin."""
    if name in OWN:
        return OWN[name]
    start, end = name.split()[1:]
    skin = on_skin(rig["points"], rig["faces"])
    return skin[end] - skin[start]


def myo_reference(name, model, data):
    """A reference axis on MyoFullBody at its default pose: an own axis, or a landmark line between bone points."""
    if name in OWN:
        return OWN[name]
    start, end = name.split()[1:]
    return myo_point(model, data, bone_point(end)) - myo_point(model, data, bone_point(start))


def measured(folder, rig_path):
    """{group: R_cal quaternion (x, y, z, w)}."""
    model = mujoco.MjModel.from_xml_path(f"{folder}/{SOURCE}")
    data = default_pose(model, Couplings(model))
    rig = dict(np.load(rig_path))
    names = [str(joint) for joint in rig["names"]]
    references = sided_map(REFERENCES)
    hinge_groups = set(sided_map({group: "" for group in calibration.HINGED}))
    levelled = set(sided_map({group: "" for group in calibration.LEVELLED}))
    found = {}
    for segment, ((start, end), (myo_start, myo_end), frame_body, _) in groups().items():
        (start, end), (myo_start, myo_end) = calibration.LONG_LINES.get(segment, ((start, end), (myo_start, myo_end)))
        soma_ref, myo_ref = references[segment]
        soma_axis = soma_reference(soma_ref, rig)
        myo_axis = myo_reference(myo_ref, model, data)
        myo_axis = myo_axis * (1.0 if myo_axis @ soma_axis >= 0 else -1.0)
        segment_turn = rig["bind"][names.index(segment), :3, :3]
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_body)
        body_turn = TO_OWN @ data.xmat[body].reshape(3, 3)
        found[segment] = aligned(
            (segment_turn, soma_point(end, rig) - soma_point(start, rig), soma_axis),
            (body_turn, myo_point(model, data, myo_parts(myo_end)) - myo_point(model, data, myo_parts(myo_start)), myo_axis),
            calibration.hinged if segment in hinge_groups else calibration.frame)
        if segment in levelled:
            found[segment] = calibration.levelled(found[segment], segment_turn, body_turn, body_turn.T @ myo_axis)
    from scipy.spatial.transform import Rotation
    return {segment: Rotation.from_matrix(turn).as_quat() for segment, turn in found.items()}


DEEP = 0.010  # metres: deeper than any finger's flesh holds its bone, so every finger bone vertex pulls (centred)


def fingers_measured(folder, rig_path, kg=70.0):
    """{joint stem: rest turn}: MyoFullBody sized to SOMA's template (its file at FINGERS as they stand), each hand laid
    on the template's bind by `calibration.turn` at its Hand joint, where the fit holds it (the wrist's end unshifted,
    END_SHIFTS), never moved (#70, 2026-11-06: moved as one body to where its palm stood least outside, the fingers were
    read for a hand that is not where the fit holds it; their thumb and index then stood out at the joint, the wrist's
    end shift read 18 mm up the forearm to bury them and every fingertip bone stood 20 to 22 mm short of SOMA's); the
    finger and thumb joints of both hands (one turn a stem, both sides alike) solved by scipy's bounded `least_squares`
    within their ranges so each finger bone vertex (`_skin.bone_meshes`' kept ones) stands DEEP inside SOMA's skin
    (the shortfall its residual: the bones as deep in the flesh as they go) and the carpals and metacarpals inside it;
    the found turns added to FINGERS'."""
    import tempfile

    from scipy.optimize import least_squares

    from motor import myo
    from motor.fit import _skin
    from motor.sizing import Size
    rig = dict(np.load(rig_path))
    path = pathlib.Path(tempfile.mkdtemp()) / "template.xml"
    myo.person_file(folder, Size(rig), kg, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    hands = [_hand(model, rig, side) for side in ("Left", "Right")]
    stems = sorted({stem for hand in hands for stem in hand["stems"]})
    lower = np.max([[hand["range"][stem][0] for stem in stems] for hand in hands], axis=0)
    upper = np.min([[hand["range"][stem][1] for stem in stems] for hand in hands], axis=0)
    mesh = warp.Mesh(points=warp.array(rig["points"].astype(np.float32), dtype=warp.vec3, device="cpu"),
                     indices=warp.array(rig["faces"].astype(np.int32).ravel(), dtype=int, device="cpu"),
                     support_winding_number=True)
    data, meshes = mujoco.MjData(model), _skin.bone_meshes(model)
    rest = default_pose(model, Couplings(model)).qpos.copy()

    def shortfall(turns):
        found = []
        for hand in hands:
            data.qpos[:] = rest
            data.qpos[[hand["address"][stem] for stem in stems]] = turns
            mujoco.mj_kinematics(model, data)
            points, bones = _skin.bone_points(model, data, meshes)
            laid = _laid(points, data, hand)
            fingers, palm = np.isin(bones, hand["fingers"]), np.isin(bones, hand["palm"])
            found += [DEEP - _depth_inside(mesh, laid[fingers]), -_depth_inside(mesh, laid[palm])]
        return np.maximum(0.0, np.concatenate(found))
    solved = least_squares(shortfall, np.clip(np.zeros(len(stems)), lower + 0.02, upper - 0.02), bounds=(lower, upper),
                           diff_step=1e-3, x_scale=0.1)
    return {stem: round(calibration.FINGERS.get(stem, 0.0) + float(turn), 4)
            for stem, turn in zip(stems, solved.x[:len(stems)])}


def inside_measured(folder, rig_path, groups_of, kg=70.0):
    """{group: (R_cal quaternion (x, y, z, w), its turn off TABLE's in degrees, the share of its bone vertices outside
    SOMA's skin at TABLE's and at the found turn, the shift in mm)} for `groups_of` (TABLE's names; a left group brings
    its right one): MyoFullBody sized to SOMA's template at its default pose, each group's bone vertices
    (`_skin.bone_meshes`' kept ones on the group's bodies) laid on SOMA's bind with their start centre (`myo_parts`,
    scaled with its body) at the group's start joint and turned by TABLE, then turned (a rotvec about the body's own
    axes, `calibration.turned`) and shifted as one rigid body by scipy's `least_squares` to their least depth outside
    SOMA's skin (`_skin.envelope`: the head's pockets closed). A middle group turns about its body's z alone, so it
    turns no body sideways; a side's two groups share one turn, mirrored, each its own shift. The shift is not kept:
    the row is the turn."""
    import tempfile

    from motor import myo
    from motor.fit import _skin
    from motor.sizing import Size
    rig = dict(np.load(rig_path))
    path = pathlib.Path(tempfile.mkdtemp()) / "template.xml"
    _, scales = myo.person_file(folder, Size(rig), kg, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = default_pose(model, Couplings(model))
    mesh = warp.Mesh(points=warp.array(rig["points"].astype(np.float32), dtype=warp.vec3, device="cpu"),
                     indices=warp.array(_skin.envelope(rig).ravel(), dtype=int, device="cpu"),
                     support_winding_number=True)
    points, bones = _skin.bone_points(model, data, _skin.bone_meshes(model))
    found = {}
    for unit in _units(groups_of):
        laid = [_bones_at_rest(model, data, scales, rig, points, bones, group) for group in unit]
        if all(len(local) for local, _, _ in laid):  # Neck2 has no body of its own
            found.update(_fitted_in_skin(mesh, unit, laid))
    return found


SHIFT_REACH = 0.024  # m: the head's shift read within this on each axis
SHIFT_STEP = 0.002  # m


def end_shift_measured(folder, rig_path, group="Head", kg=70.0):
    """The shift (m, the person's frame, Y up) of `group`'s bone vertices off SOMA's start joint at their least share
    outside SOMA's skin (`_skin.envelope`: the head's pockets closed): MyoFullBody sized to SOMA's template at its
    default pose, the bones laid as `inside_measured` lays them at TABLE's turn, every shift on a SHIFT_STEP grid within
    SHIFT_REACH tried (the share is a step, which a least squares' differences do not see), a tie the shortest."""
    import itertools
    import tempfile

    from motor import myo
    from motor.fit import _skin
    from motor.sizing import Size
    rig = dict(np.load(rig_path))
    path = pathlib.Path(tempfile.mkdtemp()) / "template.xml"
    _, scales = myo.person_file(folder, Size(rig), kg, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = default_pose(model, Couplings(model))
    mesh = warp.Mesh(points=warp.array(rig["points"].astype(np.float32), dtype=warp.vec3, device="cpu"),
                     indices=warp.array(_skin.envelope(rig).ravel(), dtype=int, device="cpu"),
                     support_winding_number=True)
    points, bones = _skin.bone_points(model, data, _skin.bone_meshes(model))
    local, segment_turn, start = _bones_at_rest(model, data, scales, rig, points, bones, group)
    laid = start + local @ (segment_turn @ calibration.turn(group)).T
    steps = np.arange(-SHIFT_REACH, SHIFT_REACH + SHIFT_STEP / 2, SHIFT_STEP)
    tried = [(float((_depth_inside(mesh, laid + np.array(shift)) < 0).mean()), float(np.linalg.norm(shift)), shift)
             for shift in itertools.product(steps, repeat=3)]
    return tuple(round(float(value), 4) for value in min(tried)[2])


def _units(groups_of):
    """The groups fitted together: a middle group alone, a side's two groups as one pair (left first)."""
    found = []
    for group in groups_of:
        unit = (group,) if not group.startswith(("Left", "Right")) else (
            "Left" + group.removeprefix("Left").removeprefix("Right"), "Right" + group.removeprefix("Left").removeprefix("Right"))
        if unit not in found:
            found.append(unit)
    return found


def _bones_at_rest(model, data, scales, rig, points, bones, group):
    """(its bone vertices in the group body's own frame from its start centre, its segment's bind turn, SOMA's start
    joint): what `_fitted_in_skin` lays."""
    from motor.fit._follow import centre
    from motor.fit._model import own_name
    (start, _), (myo_start, _), frame_body, bodies = groups()[group]
    mine = np.isin([own_name(bone) for bone in bones], bodies)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_body)
    local = (points[mine] - TO_OWN @ centre(model, data, myo_start, scales)) @ (TO_OWN @ data.xmat[body].reshape(3, 3))
    names = [str(joint) for joint in rig["names"]]
    return local, rig["bind"][names.index(group), :3, :3], soma_point(start, rig)


def _fitted_in_skin(mesh, unit, laid):
    """{group: (quaternion, degrees, outside at TABLE, outside found, shift mm)} for one unit: its shared turn (a
    middle group's about z alone) and each group's shift solved to the least depth outside the skin."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    middle = len(unit) == 1

    def rotvec(unknowns):
        return np.array([0.0, 0.0, unknowns[0]]) if middle else unknowns[:3]

    def placed(unknowns):
        turns = 1 if middle else 3
        shifts = unknowns[turns:].reshape(len(unit), 3)
        return [start + shift + local @ (segment_turn @ Rotation.from_quat(calibration.turned(group, rotvec(unknowns))).as_matrix()).T
                for group, (local, segment_turn, start), shift in zip(unit, laid, shifts)]

    def depths(unknowns):
        return np.maximum(0.0, -_depth_inside(mesh, np.concatenate(placed(unknowns))))
    unknowns = np.zeros((1 if middle else 3) + 3 * len(unit))
    solved = least_squares(depths, unknowns, x_scale=0.01, diff_step=1e-4)
    sizes = np.cumsum([0] + [len(local) for local, _, _ in laid])
    before, after = depths(unknowns) > 0, depths(solved.x) > 0
    shifts = solved.x[(1 if middle else 3):].reshape(len(unit), 3)
    return {group: (calibration.turned(group, rotvec(solved.x)), float(np.degrees(np.linalg.norm(rotvec(solved.x)))),
                    float(before[sizes[index]:sizes[index + 1]].mean()), float(after[sizes[index]:sizes[index + 1]].mean()),
                    1000 * shifts[index])
            for index, group in enumerate(unit)}


def _hand(model, rig, side):
    """One hand's finger bodies (the Hand group's bodies but its frame body and the carpals), its palm's (the rest),
    their hinges by stem (address, range), its frame body and its lay on SOMA's bind (turn, SOMA's Hand joint)."""
    segment = f"{side}Hand"
    (start, _), _, frame_body, bodies = groups()[segment]
    fingers = [body for body in bodies if "ph" in body or "thumb" in body or body.startswith("firstmc")]
    palm = [body for body in bodies if body not in fingers]
    stems, address, ranges = [], {}, {}
    for joint in range(model.njnt):
        name = model.joint(joint).name
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE and any(
                model.body(model.jnt_bodyid[joint]).name.startswith(body) for body in fingers):
            stem = name[:-2]
            stems.append(stem)
            address[stem], ranges[stem] = model.jnt_qposadr[joint], model.jnt_range[joint]
    names = [str(joint) for joint in rig["names"]]
    return {"stems": stems, "address": address, "range": ranges, "fingers": fingers, "palm": palm,
            "frame": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_body),
            "turn": rig["bind"][names.index(segment), :3, :3] @ calibration.turn(segment),
            "start": soma_point(start, rig)}


def _laid(points, data, hand):
    """Bone points (the person's frame) laid on SOMA's bind: the hand's frame body at SOMA's Hand joint, turned by the
    table."""
    body_turn = TO_OWN @ data.xmat[hand["frame"]].reshape(3, 3)
    return hand["start"] + (points - TO_OWN @ data.xpos[hand["frame"]]) @ body_turn @ hand["turn"].T


@warp.kernel
def _inside_kernel(mesh: warp.uint64, points: warp.array(dtype=warp.vec3), depths: warp.array(dtype=float)):
    """Each point's distance to the mesh's closest point, inside positive (the exact winding number's side)."""
    index = warp.tid()
    found = warp.mesh_query_point_sign_winding_number(mesh, points[index], 1.0e6, 2.0, 0.5)
    closest = warp.mesh_eval_position(mesh, found.face, found.u, found.v)
    depths[index] = -found.sign * warp.length(points[index] - closest)


def _depth_inside(mesh, points):
    """Each point's depth inside the closed mesh (m, outside negative)."""
    depths = warp.zeros(len(points), dtype=float, device="cpu")
    warp.launch(_inside_kernel, dim=len(points), device="cpu",
                inputs=[mesh.id, warp.array(np.asarray(points, dtype=np.float32), dtype=warp.vec3, device="cpu"), depths])
    return depths.numpy().astype(float)


# A face landmark's assumed soft tissue over its bone point when the head is laid (m, along the body's outward axis):
# typical adult depths, only a start for the lay; the links written are then SOMA's skin off the laid bone point.
FACE_TISSUE = {"nasion": ((1.0, 0.0, 0.0), 0.0065), "menton": ((0.0, -1.0, 0.0), 0.0075)}


def laid_measured(folder, rig_path, kg=70.0):
    """The head laid on SOMA's template by its landmarks (`calibration.LAID_GROUPS`): MyoFullBody sized to it at its
    default pose, the skull's turn about its body's z (TABLE's row turned, the sideways axis kept) and its C0_C1's
    shift in the middle plane solved by least squares so each head landmark's marker (its bone point scaled, then its
    link; a FACE_TISSUE landmark its assumed depth instead) meets its SOMA point at the bind. Returns (R_cal quaternion,
    C0_C1's shift m, {landmark: its link, SOMA's point off the laid bone point in the body's frame})."""
    import tempfile

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from motor import myo
    from motor.fit._follow import centre
    from motor.sizing import Size
    rig = dict(np.load(rig_path))
    path = pathlib.Path(tempfile.mkdtemp()) / "template.xml"
    _, scales = myo.person_file(folder, Size(rig), kg, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = default_pose(model, Couplings(model))
    (start, _), (myo_start, _), frame_body, _ = groups()["Head"]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_body)
    axes = TO_OWN @ data.xmat[body].reshape(3, 3)
    origin = axes.T @ (TO_OWN @ (centre(model, data, myo_start, scales) - data.xpos[body]))  # C0_C1, body frame
    names = [str(joint) for joint in rig["names"]]
    segment_turn, joint = rig["bind"][names.index("Head"), :3, :3], soma_point(start, rig)
    marks = [name for name, row in LANDMARKS.items() if row[3] == frame_body]
    bones = {name: np.asarray(LANDMARKS[name][4]) * scales[frame_body] - origin for name in marks}
    links = {name: np.asarray(FACE_TISSUE[name][0]) * FACE_TISSUE[name][1] if name in FACE_TISSUE
             else np.asarray(LANDMARKS[name][5], dtype=float) for name in marks}
    skin = on_skin(rig["points"], rig["faces"], marks)

    def placed(unknowns, name, link):
        turn = segment_turn @ Rotation.from_quat(calibration.turned("Head", [0.0, 0.0, unknowns[0]])).as_matrix()
        return joint + np.array([0.0, unknowns[1], unknowns[2]]) + turn @ (bones[name] + link)

    def gaps(unknowns):
        return np.concatenate([skin[name] - placed(unknowns, name, links[name]) for name in marks])
    solved = least_squares(gaps, np.zeros(3), x_scale=0.01).x
    turn = segment_turn @ Rotation.from_quat(calibration.turned("Head", [0.0, 0.0, solved[0]])).as_matrix()
    return (calibration.turned("Head", [0.0, 0.0, solved[0]]), (0.0, float(solved[1]), float(solved[2])),
            {name: turn.T @ (skin[name] - placed(solved, name, np.zeros(3))) for name in marks})


def table_rows(folder, rig_path):
    """{group: R_cal quaternion}: the landmark lines' rows, INSIDE_GROUPS' read in the skin, LAID_GROUPS' (the head)
    laid on its landmarks (`laid_measured`)."""
    inside = inside_measured(folder, rig_path, calibration.INSIDE_GROUPS)
    inside.update({group: (laid_measured(folder, rig_path)[0],) for group in calibration.LAID_GROUPS})
    return {group: inside[group][0] if group in inside else quaternion
            for group, quaternion in measured(folder, rig_path).items()}


if __name__ == "__main__":
    for group, quaternion in table_rows(*sys.argv[1:3]).items():
        print(f'    "{group}": ({", ".join(f"{value:.6g}" for value in quaternion)}),')
    for stem, turn in fingers_measured(*sys.argv[1:3]).items():
        print(f'    "{stem}": {turn},')
    print(f'    "C0_C1": {laid_measured(*sys.argv[1:3])[1]},')
