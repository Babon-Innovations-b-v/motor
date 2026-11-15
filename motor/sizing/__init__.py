"""MyoFullBody sized to one person's SOMA body (#70: SOMA is the truth, its size included), `Size`:
- each bone group (`bone_groups.GROUPS`, trunk and pelvis included; a group with no body of its own, Neck2, is
  sized with the one before it on its chain: `stretches`) scaled evenly so its stretch (its two end joints: SOMA's
  rig joints at its bind, MyoFullBody's child bodies' joint anchors at its default pose) is SOMA's
  (`soma_lengths`, `myo_lengths`); all scales solved together (`Size.scales`), as a group's end may sit on another
  group's body; the hip joints stand in the pelvis by its scale, as its muscles' attachments do (#98: placed at
  SOMA's hip joints apart, 213 to 227 mm against 136 at the pelvis' scale, they cut the hip muscles' arms 25 to
  38% and saturated the abductors walking);
- each group's mass, centre of mass and inertia's trace from SOMA's segment volumes (`volumes`: Dempster's densities),
  laid on the group's bodies (frames under FRAME_KG left out) in the model's own proportions, the centre carried
  into the group's frame by the calibration table and the inertias scaled together to the trace (`weigh`); the
  bodies of no group (the kneecaps, the abdominal wrapper) keep the model's share;
- each limb group's width (GIRTH_GROUPS; the two axes across its stretch) its length's scale times the person's girth
  for its length over SOMA's mean shape's (`volumes.slenderness`, MEAN_SLENDERNESS: MyoFullBody sized evenly is
  taken to fit the mean shape's girth, so a slimmer limb gets slimmer bones; `Size.widths`);
- each foot's width (FOOT_WIDTH) across its forefoot: the forefoot bone's axis across the foot scaled so its
  metatarsal heads stand as far apart across the foot as SOMA's on its skin (`forefoot_width`, `Size.fixed`), never
  by its length or girth;
- each kneecap left on MyoFullBody's own couplings to its knee, sized with its femur: refitted to lie inside SOMA's
  skin at every bend (#70, 2026-11-04 to 2026-11-07), it followed the linear-blend skin's cave at a bent knee and
  took the quadriceps' lever with it (#98: knee extension arm 1.1 to 1.4 cm at 60 to 75 degrees, MyoFullBody's own
  2.7 to 3.5; 13 cm at 120);
- what SOMA does not give marked as assumed with a plausible range (`assumed`).
A body's scale is three, one per axis of its own frame: its group's length scale along the axis nearest the group's
stretch at the default pose (`long_axes`), its width scale across (the same as the length's outside GIRTH_GROUPS); a
body of no group takes its parent's. A body's place in its parent scales with its parent, but OWN_PLACED's with its
own and JOINT_PLACED's about the centre it turns about on its parent (`joint_placed`). Positions, geoms, sites, joint anchors, meshes, inertias and a slide joint's coupling and range scale with it
(`sized`; a shape's or an inertia's axes, and a slide's, by the stretch along each, `_stretched`); each muscle's path length at the zero pose is read for
OpenSim's muscle scaling (`path_lengths`). The masses' centres stand at SOMA's own joints (`soma_point`,
`myo_point`): where a group's mass sits is not its length."""
import mujoco
import numpy as np

from motor import volumes
from motor.bone_groups import groups, sided
from motor.calibration import turn
from motor.landmarks import LANDMARKS, myo_parts, on_skin, soma_point
from motor.model import Couplings, default_pose

WRAPPER_KG = 0.001  # a split body's wrapper (MuJoCo's boundmass for MyoFullBody, which Newton does not apply)
# Bodies placed in their parent by their own scale: the torso, welded on lumbar1 with its origin 8.9 cm below
# lumbar1's (inside its own bone, the source model): by lumbar1's scale a longer lumbar stretch moved T12/L1 down.
OWN_PLACED = {"torso"}
# Bodies placed in their parent by the centre they turn about on it (#70, 2026-11-05; the owner: "the neck is
# dettached"): {body: that centre}. The centre stays where its parent's scale puts it, the body's own bones scaled about
# it by the body's own scale (`joint_placed`). The neck's origin is MyoHead's, at the torso's 40 cm under its own bones:
# placed by its parent's scale, the torso's 0.917 on the mean body dropped C7/T1 33 mm under the neck's 0.998, its
# cervical bones stood 35 mm over the torso's top at rest and the neck's length scale was never asked to reach SOMA's.
JOINT_PLACED = {"neck": "C7_T1"}
SCALE_RANGE = (0.5, 2.0)  # a bone group's scale's bounds
# More SOMA bone lines a group's volume is given by, beyond its stretch: the hips' line for the pelvis.
MORE_LINES = {"Hips": (("LeftLeg", "RightLeg"),)}
# The limb groups whose bones' width is sized from SOMA's girth (#70, 2026-11-01: the coordinator's decision; env1's
# forearm bones, scaled evenly by length, stood 58 to 63 mm across in a 55 to 57 mm wrist). The toes are left out:
# their short stretch's girth reads 0.79 to 1.43 times the mean shape's over level 1's 41 bodies, with no tie to build.
GIRTH_GROUPS = tuple(sided(("{Side}Arm", "{Side}ForeArm", "{Side}Hand", "{Side}Leg", "{Side}Shin", "{Side}Foot")))
# The foot's width (#70, 2026-11-02): {group: (the heel's landmark, the two metatarsal heads' across the foot)}, each
# head's bone point on the forefoot's body. MyoFullBody's metatarsal heads stand 85.8 mm apart across the foot, SOMA's
# skin's 74.1 on env1, 73.4 on the mean shape, 79.0 on g01, so no length or girth scale fits them (the girth rule gave
# env1's foot 1.00, its bones 1.0275 times as wide: 9.8 and 11.3 % of the calcanei outside standing, 25.7 and 22.2 % of
# the toes; across alone at 0.85, 3.7 and 6.3 %; their height changed nothing). The bone's heads then stand where the
# skin's do, MyoFullBody's bone points of the landmark study, SOMA's skin points.
FOOT_WIDTH = {f"{side}Foot": (f"heel_back_{letter}", f"MT1_head_{letter}", f"MT5_head_{letter}")
              for side, letter in (("Left", "l"), ("Right", "r"))}
# SOMA's mean shape's girth for its length per limb group (`volumes.slenderness`; its coefficients zero, level 1's
# `mean` body, 1.684 m; both sides' mean; `Size(rig).slender` on its rig, measured 2026-11-01). Its limbs are the
# reference MyoFullBody's bones, sized evenly by length, are taken to fit (env1's forearm bones' best even shrink in
# its skin 0.85, the mean body's 1.00; env1's forearm reads 0.89 of the mean's).
MEAN_SLENDERNESS = {"Arm": 0.3668, "ForeArm": 0.2481, "Hand": 0.246, "Leg": 0.3572, "Shin": 0.2222, "Foot": 0.4672}
# What SOMA does not give, as assumed, with a plausible range (`assumed` adds the bone shape measured on the person).
ASSUMED = {
    "muscle strength": {
        "value": "MyoFullBody's own peak isometric forces, not scaled with the person",
        "plausible": "0.25 to 1.7 times: specific tension reported from 15 to over 100 N/cm2 (Rajagopal et al. 2016, "
                     "doi:10.1109/TBME.2016.2586891, whose 60 N/cm2 the model's legs take)"},
    "joint ranges": {
        "value": "MyoFullBody's own ranges (MyoSuite; the legs as Rajagopal et al. 2016)",
        "plausible": "a person's own differ by sex and age from the textbook norms (Soucie et al. 2011, Haemophilia "
                     "17:500, 674 healthy people); no spread read per joint, so none is given"},
    "segment density": {
        "value": "Dempster 1955 as Winter 2009 Table 4.1 gives it, per bone group (volumes.DENSITY)",
        "plausible": f"{volumes.DENSITY_RANGE[0]} to {volumes.DENSITY_RANGE[1]} times it (Chandler et al. 1975)"}}


def stretches(table=None):
    """The sizing stretches ({group: ((SOMA start, end), (MyoFullBody start, end), bodies)}): each bone group with
    bodies, a body-less one after it on its chain merged into it."""
    table = table or groups()
    found = {group: (soma, myo, bodies) for group, (soma, myo, _, bodies) in table.items() if bodies}
    for group, ((start, end), (_, myo_end), _, bodies) in table.items():
        if not bodies:
            host = next(name for name, ((_, last), _, _) in found.items() if last == start)
            (first, _), (myo_first, _), held = found[host]
            found[host] = ((first, end), (myo_first, myo_end), held)
    return found


def soma_lengths(rig, table=None):
    """Each stretch's length on SOMA at its bind (metres): its two ends' rig joints apart (`soma_point`)."""
    return {group: float(np.linalg.norm(soma_point(end, rig) - soma_point(start, rig)))
            for group, ((start, end), _, _) in stretches(table).items()}


def myo_point(model, data, name, scales):
    """A MyoFullBody centre in MuJoCo's frame, its body-local point scaled with its body."""
    found = np.zeros(3)
    for body, local, weight in myo_parts(name):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        found += weight * (data.xpos[index] + data.xmat[index].reshape(3, 3) @ (np.asarray(local) * scales[body]))
    return found


def myo_lengths(model, scales, table=None):
    """Each stretch's length on the model at its default pose (metres): its two ends' joint centres apart (each a
    child body's joint anchor, `myo_point`)."""
    data = default_pose(model, Couplings(model))
    return {group: float(np.linalg.norm(myo_point(model, data, end, scales) - myo_point(model, data, start, scales)))
            for group, (_, (start, end), _) in stretches(table).items()}


def posed_lengths(model, scales, table=None):
    """`myo_lengths` of the model as `sized` would place it (a body's place by its parent's scale, OWN_PLACED's by
    its own, joint anchors by their body's, a slide joint's coupling by its body's parent's), the model left as it
    was."""
    kept = model.body_pos.copy(), model.jnt_pos.copy(), model.eq_data.copy()
    name = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) for body in range(model.nbody)]
    even = np.ones(3)
    above = np.array([even] + [scales.get(name[model.body_parentid[body]], even) for body in range(1, model.nbody)])
    placed = np.array([even] + [scales[name[body]] if name[body] in OWN_PLACED else above[body]
                                for body in range(1, model.nbody)])
    anchored = joint_placed(model, scales)
    model.body_pos[:] = kept[0] * placed
    for body, place in anchored.items():
        model.body_pos[_body_id(model, body)] = place
    model.jnt_pos[:] = kept[1] * np.array([scales.get(name[model.jnt_bodyid[joint]], even)
                                          for joint in range(model.njnt)])
    for equality in range(model.neq):
        joint = model.eq_obj1id[equality]
        if model.eq_type[equality] == mujoco.mjtEq.mjEQ_JOINT and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE:
            body = model.jnt_bodyid[joint]
            model.eq_data[equality, :5] = kept[2][equality, :5] * _slide_scale(above[body], model.body_quat[body],
                                                                               model.jnt_axis[joint])
    try:
        return myo_lengths(model, scales, table)
    finally:
        model.body_pos[:], model.jnt_pos[:], model.eq_data[:] = kept


def body_scales(model, scales, table=None, spreads=None, fixed=None):
    """Each body's scale ({body name: [3], one per axis of its frame}): its group's length scale, across its long
    axis times its group's width over length (`spreads`: {body: (long axis, width over length)}, `Size.spreads`;
    none: even), along an axis of `fixed` ({body: (axis, scale)}, `Size.fixed`) that scale alone; a body of no group
    its parent's."""
    owner = {body: group for group, (_, _, bodies) in stretches(table).items() for body in bodies}
    spreads = spreads or {}
    found = {}
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[body])
        if name not in owner:
            found[name] = found.get(parent, np.ones(3))
            continue
        axis, across = spreads.get(name, (0, 1.0))
        found[name] = np.full(3, scales[owner[name]] * across)
        found[name][axis] = scales[owner[name]]
        if name in (fixed or {}):
            found[name][fixed[name][0]] = fixed[name][1]
    return found


def long_axes(model, groups_found, table=None):
    """Each body of `groups_found` ({body name: 0, 1 or 2}): the axis of its frame nearest its group's stretch (start
    to end centre) at the default pose."""
    data = default_pose(model, Couplings(model))
    even = body_scales(model, dict.fromkeys(stretches(table), 1.0), table)
    found = {}
    for group in groups_found:
        _, (start, end), bodies = stretches(table)[group]
        along = myo_point(model, data, end, even) - myo_point(model, data, start, even)
        for name in bodies:
            local = data.xmat[_body_id(model, name)].reshape(3, 3).T @ along
            found[name] = int(np.argmax(np.abs(local)))
    return found


def forefoot_width(heel, first, fifth):
    """How far apart the two metatarsal heads stand across the foot (m) and that across's direction: their apart less
    its share along the heel to their middle (the same rule on SOMA's skin points and MyoFullBody's bone points)."""
    heel, first, fifth = (np.asarray(point, dtype=float) for point in (heel, first, fifth))
    along = (first + fifth) / 2 - heel
    along /= np.linalg.norm(along)
    apart = first - fifth
    across = apart - along * (apart @ along)
    return float(np.linalg.norm(across)), across / np.linalg.norm(across)


def _stretched(scale, quat):
    """How far each axis of a frame turned by `quat` in a body stretches when the body scales by `scale` ([3])."""
    turn = np.zeros(9)
    mujoco.mju_quat2Mat(turn, np.asarray(quat, dtype=float))
    return np.linalg.norm(np.asarray(scale)[:, None] * turn.reshape(3, 3), axis=0)


def _shape_size(kind, size, stretch):
    """A geom's or site's size under its axes' stretches: a sphere's radius by their geometric mean, a capsule's or
    cylinder's radius by its two cross axes' and half length by its own, any other shape axis by axis."""
    size = np.asarray(size, dtype=float).copy()
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        size[0] *= float(np.prod(stretch)) ** (1 / 3)
    elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        size[0] *= float(np.sqrt(stretch[0] * stretch[1]))
        size[1] *= stretch[2]
    else:
        size *= stretch
    return size


def _inertia(inertia, quat, scale, mass_share):
    """A body's principal inertias under its scale: each second moment along an inertial axis by that axis's squared
    stretch, times the mass's share."""
    inertia = np.asarray(inertia, dtype=float)
    moments = inertia.sum() / 2 - inertia
    moments = moments * _stretched(scale, quat) ** 2 * mass_share
    return moments.sum() - moments


BALANCE_MARGIN = 1e-5  # the share each body's largest principal inertia is kept under the sum of the other two
BALANCE_MOST = 1e-4  # the most a balance may lower it (more is an inertia sizing broke, not a rounding)


def balanced(spec):
    """Every body's principal inertias kept off the triangle's edge: its largest at most (1 - BALANCE_MARGIN) of the
    other two's sum. MyoFullBody's lumbar vertebrae sit exactly on it (lumbar2 0.0104155 = 0.00365967 + 0.00675583,
    a zero second moment), sizing and weigh keep it there, and the 6 digits written tipped it to -1e-8 on some
    people's scales; Newton's add_mjcf corrects only breaches beyond float32's eps times the sum, and SolverMuJoCo's
    compile has no balanceinertia, so a clip failed at model build ('inertia must satisfy A + B >= C') by how its
    scales rounded (2026-11-04). Raises when a body's largest is over the sum by more than BALANCE_MOST."""
    for body in spec.bodies[1:]:
        inertia = np.asarray(body.inertia, dtype=float).copy()
        largest = int(np.argmax(inertia))
        most = (inertia.sum() - inertia[largest]) * (1 - BALANCE_MARGIN)
        if inertia[largest] <= most:
            continue
        if inertia[largest] > most * (1 + BALANCE_MOST):
            raise RuntimeError(f"{body.name}'s inertia {inertia.tolist()} breaks A + B >= C beyond a rounding")
        inertia[largest] = most
        body.inertia = inertia


def _slide_scale(scale, quat, axis):
    """How far a slide joint's travel stretches: its axis (in its body's frame, the body turned by `quat` in its
    parent) under its parent's `scale`."""
    turn = np.zeros(9)
    mujoco.mju_quat2Mat(turn, np.asarray(quat, dtype=float))
    direction = turn.reshape(3, 3) @ np.asarray(axis, dtype=float)
    return float(np.linalg.norm(np.asarray(scale) * direction) / np.linalg.norm(direction))


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_placed(model, scales):
    """{body: its place in its parent} for JOINT_PLACED's bodies of the source `model`: pos' = above (pos + R q) -
    R (scale q), q its centre in its own frame at the model's rest, R its turn in its parent, `above` its parent's
    scale and `scale` its own (`scales`), so the centre scales with the parent and the body's own bones about it."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    found = {}
    for name, centre in JOINT_PLACED.items():
        body = _body_id(model, name)
        if body < 0:  # a model without it (a toy)
            continue
        parent = model.body_parentid[body]
        point = sum(weight * (data.xpos[_body_id(model, holder)] + data.xmat[_body_id(model, holder)].reshape(3, 3) @ np.asarray(local))
                    for holder, local, weight in myo_parts(centre))
        own, above = data.xmat[body].reshape(3, 3), data.xmat[parent].reshape(3, 3)
        turn = above.T @ own
        in_body = own.T @ (point - data.xpos[body])
        in_parent = above.T @ (point - data.xpos[parent])
        parent_scale = np.asarray(scales.get(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent), np.ones(3)))
        found[name] = parent_scale * in_parent - turn @ (np.asarray(scales[name]) * in_body)
    return found


def sized(spec, model, scales, kg, wrappers):
    """The spec sized body by body (see the module's docstring), its mass brought to `kg` with `wrappers` wrapper
    bodies' mass left for them, in the model's own proportions."""
    mujoco.mj_forward(model, mujoco.MjData(model))
    grown = sum(model.body_mass[_body_id(model, body.name)] * np.prod(scales[body.name]) for body in spec.bodies[1:])
    share = (kg - wrappers * WRAPPER_KG) / grown
    meshes = {mesh.name: mesh for mesh in spec.meshes}
    placed = joint_placed(model, scales)
    for body in spec.bodies[1:]:
        scale = np.asarray(scales[body.name], dtype=float)
        above = np.asarray(scales.get(body.parent.name, scale) if body.parent.name != "world" else np.ones(3))
        body.pos = placed[body.name] if body.name in placed else np.asarray(body.pos) * (scale if body.name in OWN_PLACED else above)
        index = _body_id(model, body.name)
        mass = model.body_mass[index] * np.prod(scale) * share
        body.explicitinertial = True
        body.ipos = model.body_ipos[index] * scale
        body.iquat = model.body_iquat[index]
        body.inertia = _inertia(model.body_inertia[index], model.body_iquat[index], scale,
                                mass / max(model.body_mass[index], 1e-12))
        body.fullinertia = [float("nan")] * 6
        body.mass = mass
        for geom in body.geoms:
            stretch = _stretched(scale, geom.quat)
            geom.pos = np.asarray(geom.pos) * scale
            geom.size = _shape_size(geom.type, geom.size, stretch)
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                meshes[geom.meshname].scale = np.asarray(meshes[geom.meshname].scale) * stretch
        for site in body.sites:
            site.pos = np.asarray(site.pos) * scale
            site.size = _shape_size(site.type, site.size, _stretched(scale, site.quat))
        for joint in body.joints:
            joint.pos = np.asarray(joint.pos) * scale
            if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                joint.range = np.asarray(joint.range) * _slide_scale(above, body.quat, joint.axis)
    joints = {joint.name: joint for joint in spec.joints}
    for equality in spec.equalities:
        joint = joints.get(equality.name1)
        if joint is not None and joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            data = np.asarray(equality.data)
            holder = joint.parent
            data[:5] *= _slide_scale(scales.get(holder.parent.name, np.ones(3)), holder.quat, joint.axis)
            equality.data = data


FILLED = {}  # {the mesh's bytes' digest: `volumes.filled`'s cells, read-only}: a person's mesh is filled once


def filled(points, faces):
    """`volumes.filled` of SOMA's mesh, made once per mesh (a level 1 centre set sizes the person anew, its mesh the
    same: 0.47 s a set, 2026-10-31)."""
    import hashlib
    key = hashlib.sha256(np.ascontiguousarray(points, dtype=float).tobytes()
                         + np.ascontiguousarray(faces, dtype=np.int64).tobytes()).hexdigest()
    if key not in FILLED:
        FILLED[key] = volumes.filled(points, faces)
        FILLED[key].flags.writeable = False
    return FILLED[key]


class Size:
    """One person's sizing from its SOMA rig ({"names", "bind", "points", "faces", "weights"}, its bind in its own
    frame): `scales` (each body's), `weigh` (the masses from SOMA's volumes), `record`."""

    def __init__(self, rig, table=None, turns=None):
        self.rig, self.table = rig, table or groups()
        self.lengths = soma_lengths(rig, self.table)
        self.turns = turns or {group: turn(group) for group in stretches(self.table)}
        cells = filled(rig["points"], rig["faces"])
        names = list(stretches(self.table))
        lines = [[(soma_point(start, rig), soma_point(end, rig)) for start, end in self._pieces(group)]
                 for group in names]
        owner = volumes.shares(cells, lines)
        self.found = volumes.measured(cells, owner, len(names))
        self.slender = {group: volumes.slenderness(cells, owner, names.index(group), *lines[names.index(group)][0])
                        for group in GIRTH_GROUPS if group in names}
        self.names = names
        self.feet = {group: forefoot_width(*on_skin(rig["points"], rig["faces"], marks).values())
                     for group, marks in FOOT_WIDTH.items() if group in names}
        self.moved = {}  # each group's centre of mass moved by `weigh` (cm)

    def _pieces(self, group):
        """The SOMA bone lines a group's volume is given by: its own stretch, a body-less group's after it on its
        chain, and MORE_LINES."""
        (start, end), _, _, _ = self.table[group]
        found = [(start, end)]
        for (later_start, later_end), _, _, bodies in self.table.values():
            if not bodies and later_start == found[-1][1]:
                found.append((later_start, later_end))
        return found + list(MORE_LINES.get(group, ()))

    def widths(self):
        """Each GIRTH_GROUPS group's width over its length scale: the person's girth for its length over SOMA's mean
        shape's (MEAN_SLENDERNESS)."""
        return {group: found / MEAN_SLENDERNESS[group.removeprefix("Left").removeprefix("Right")]
                for group, found in self.slender.items()}

    def spreads(self, source):
        """{body: (its long axis, its group's width over its length scale)} for the GIRTH_GROUPS groups' bodies."""
        widths = self.widths()
        axes = long_axes(source, widths, self.table)
        return {name: (axis, widths[group]) for group in widths
                for name, axis in axes.items() if name in stretches(self.table)[group][2]}

    def fixed(self):
        """{body: (its axis nearest the foot's across, SOMA's forefoot width over the bone's)} for each FOOT_WIDTH
        group, the body its metatarsal heads' bone points ride (`forefoot_width`)."""
        found = {}
        for group, (soma, _) in self.feet.items():
            marks = FOOT_WIDTH[group]
            bone, across = forefoot_width(*(LANDMARKS[name][4] for name in marks))
            found[LANDMARKS[marks[1]][3]] = (int(np.argmax(np.abs(across))), soma / bone)
        return found

    def scales(self, source):
        """Each body's scale ([3]): the group scales whose stretches at the default pose are SOMA's lengths, one
        bounded least squares (scipy's `least_squares`, a scale within SCALE_RANGE) on the source model posed at them
        (`posed_lengths`), as a group's end may sit on another group's body; the limbs' widths with them (`spreads`)
        and the feet's (`fixed`)."""
        from scipy.optimize import least_squares
        groups_found = list(self.lengths)
        wanted = np.array([self.lengths[group] for group in groups_found])
        own = myo_lengths(source, body_scales(source, dict.fromkeys(groups_found, 1.0), self.table), self.table)
        spreads, fixed = self.spreads(source), self.fixed()

        def misses(logs):
            scales = body_scales(source, dict(zip(groups_found, np.exp(logs))), self.table, spreads, fixed)
            found = posed_lengths(source, scales, self.table)
            return np.array([found[group] for group in groups_found]) - wanted
        start = np.log(np.clip(wanted / [own[group] for group in groups_found], *SCALE_RANGE))
        solved = least_squares(misses, start, bounds=(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1])), xtol=1e-10)
        return body_scales(source, dict(zip(groups_found, np.exp(solved.x))), self.table, spreads, fixed)

    def weigh(self, spec, grown, scales, kg):
        """Each group's bodies given its mass, centre of mass and inertia's trace from SOMA's volumes (`kg` the
        groups' share of the person's mass); returns {group: inertia scale} (the factor on the bodies' own
        inertias, its floor INERTIA_FLOOR)."""
        data = default_pose(grown, Couplings(grown))
        inertials = volumes.weighed(self.names, self.found, kg)
        bodies = {body.name: body for body in spec.bodies[1:]}
        factors = {}
        for group, ((start, _), (myo_start, _), held) in stretches(self.table).items():
            mass, centre, trace = inertials[group]
            held = [name for name in held if grown.body_mass[_body_id(grown, name)] >= FRAME_KG]
            index = [_body_id(grown, name) for name in held]
            masses = grown.body_mass[index] * mass / max(grown.body_mass[index].sum(), 1e-12)
            names = [str(name) for name in self.rig["names"]]
            segment = np.asarray(self.rig["bind"])[names.index(group), :3, :3]
            frame = _body_id(grown, self.table[group][2])
            offset = self.turns[group].T @ segment.T @ (centre - soma_point(start, self.rig))
            target = myo_point(grown, data, myo_start, scales) + data.xmat[frame].reshape(3, 3) @ offset
            shift = target - masses @ data.xipos[index] / mass
            self.moved[group] = round(float(np.linalg.norm(shift)) * 100, 2)
            spread = sum(2.0 * part * np.sum((data.xipos[body] + shift - target) ** 2)
                         for part, body in zip(masses, index))
            own = sum(float(grown.body_inertia[body].sum()) * part / max(grown.body_mass[body], 1e-12)
                      for part, body in zip(masses, index))
            factors[group] = max((trace - spread) / max(own, 1e-12), INERTIA_FLOOR)
            for name, part, body in zip(held, masses, index):
                bodies[name].mass = float(part)
                bodies[name].ipos = grown.body_ipos[body] + data.xmat[body].reshape(3, 3).T @ shift
                bodies[name].inertia = (grown.body_inertia[body] * part / max(grown.body_mass[body], 1e-12)
                                        * factors[group])
        return factors

    def record(self, model, scales):
        """The sized model's lengths against SOMA's ({group: [SOMA m, sized m]})."""
        found = myo_lengths(model, scales, self.table)
        return {group: [round(self.lengths[group], 4), round(found[group], 4)] for group in self.lengths}


# A body under this mass is a frame, not a bone (MyoFullBody's phantoms and attachments, 1 g, some with inertias of
# several kg m2): `weigh` leaves it as sized.
FRAME_KG = 0.01
INERTIA_FLOOR = 0.1  # the least factor on a group's own inertias (a group whose bodies' spread alone passes SOMA's)


def ungrouped_kg(model, table=None):
    """The mass of the model's bodies `weigh` leaves as sized (kg): those in no bone group (the kneecaps, the
    abdominal wrapper) and the frames (under FRAME_KG)."""
    held = {body for _, _, bodies in stretches(table).values() for body in bodies}
    return float(sum(model.body_mass[body] for body in range(1, model.nbody)
                     if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) not in held
                     or model.body_mass[body] < FRAME_KG))


def assumed(model, scales, rig):
    """ASSUMED with the bone shape: each group scaled by its length, the limbs' widths by SOMA's girth, its widths
    measured against SOMA's (the hip joints and the shoulder joints apart, SOMA's over the sized model's)."""
    data = default_pose(model, Couplings(model))
    widths = {}
    for label, (soma_pair, myo_pair) in {"hips": (("LeftLeg", "RightLeg"), ("hip_l", "hip_r")),
                                         "shoulders": (("LeftArm", "RightArm"),
                                                       ("glenohumeral_l", "glenohumeral_r"))}.items():
        soma = np.linalg.norm(soma_point(soma_pair[0], rig) - soma_point(soma_pair[1], rig))
        myo = np.linalg.norm(myo_point(model, data, myo_pair[0], scales) - myo_point(model, data, myo_pair[1], scales))
        widths[label] = round(float(soma / myo), 3)
    return {**ASSUMED, "bone shape": {
        "value": "MyoFullBody's own bones, each bone group scaled by its stretch's length, a limb's width by the "
                 "person's girth for its length over SOMA's mean shape's, a foot's across its forefoot by SOMA's "
                 "metatarsal heads apart",
        "plausible": f"widths {min(widths.values())} to {max(widths.values())} times the sized ones (SOMA's joints "
                     f"apart over the sized body's: {widths})"}}


def path_lengths(model):
    """Each actuator's path length at the zero pose (metres)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data.actuator_length.copy()


def kept_bodies(table=None):
    """The bodies named in a scale record: each stretch's frame body."""
    table = table or groups()
    return {table[group][2] for group in stretches(table)}
