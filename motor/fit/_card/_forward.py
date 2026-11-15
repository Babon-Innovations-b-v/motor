"""Every row's link frames in float64 on the card (the residuals' precision; Newton's own FK is float32, and its
float32 noise is what held the shoulder's elevation plane 0.02 deg off mink's float64 minimum).

The same maths as Newton's eval_single_articulation_fk for the joints a person model has: free (position, then
the turn x, y, z, w), revolute about its axis, prismatic along it, fixed; a joint's parent link is laid before it (MJCF order).
Each row reads its joint places (joint_X_p) from its set's row of a [sets, joints] array (level 3 across sets: rows =
set x frame, each set's inside-joint moves its own places); one set by default, the model's own.

`Exact` gives the same link frames from MuJoCo's own compiled model in float64 instead (`_level2_wide._kinematics`,
mj_kinematics' steps): Newton's add_mjcf builds its joint places and axes in float32, so its float64 frames of the same
pose stood up to 7.3 um off MuJoCo's at a hand (mean's jump, 2026-11-06), and level 3's card fit, minimising on
them, stopped 0.002 to 0.007 deg off mink's in the shoulder girdle's weak direction (sites 0.024 mm)."""
import mujoco
import numpy as np
import warp as wp
from newton import JointType

FREE = int(JointType.FREE)
REVOLUTE = int(JointType.REVOLUTE)
FIXED = int(JointType.FIXED)
PRISMATIC = int(JointType.PRISMATIC)


@wp.func
def _still():
    return wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))


@wp.kernel(enable_backward=False)
def _frames(joint_q: wp.array2d[wp.float32], joint_type: wp.array[wp.int32], joint_parent: wp.array[wp.int32],
            joint_child: wp.array[wp.int32], q_start: wp.array[wp.int32], qd_start: wp.array[wp.int32],
            joint_x_p: wp.array2d[wp.transformd], row_set: wp.array[wp.int32], joint_x_c: wp.array[wp.transformd], joint_axis: wp.array[wp.vec3d],
            body_q: wp.array2d[wp.transformd]):
    row = wp.tid()
    for joint in range(joint_type.shape[0]):
        kind = joint_type[joint]
        start = q_start[joint]
        across = wp.transformd(wp.vec3d(), _still())
        if kind == REVOLUTE:
            angle = wp.float64(joint_q[row, start])
            across = wp.transformd(wp.vec3d(), wp.quat_from_axis_angle(joint_axis[qd_start[joint]], angle))
        if kind == PRISMATIC:
            across = wp.transformd(joint_axis[qd_start[joint]] * wp.float64(joint_q[row, start]), _still())
        if kind == FREE:
            place = wp.vec3d(wp.float64(joint_q[row, start]), wp.float64(joint_q[row, start + 1]),
                             wp.float64(joint_q[row, start + 2]))
            turn = wp.quatd(wp.float64(joint_q[row, start + 3]), wp.float64(joint_q[row, start + 4]),
                            wp.float64(joint_q[row, start + 5]), wp.float64(joint_q[row, start + 6]))
            across = wp.transformd(place, turn)
        parent_joint = joint_x_p[row_set[row], joint]
        parent = joint_parent[joint]
        if parent >= 0:
            parent_joint = body_q[row, parent] * parent_joint
        body_q[row, joint_child[joint]] = parent_joint * across * wp.transform_inverse(joint_x_c[joint])


class Forward:
    """The model's joint tree in float64 on the card; raises ValueError on a joint kind it does not hold. set_places
    [sets, joints, 7] (Newton's transform layout) and row_set [rows] give each row its set's joint places; neither, the
    model's own for every row."""

    def __init__(self, model, set_places=None, row_set=None):
        kinds = model.joint_type.numpy()
        unknown = sorted(set(int(kind) for kind in kinds) - {FREE, REVOLUTE, PRISMATIC, FIXED})
        if unknown:
            raise ValueError(f"joint kinds {unknown} have no float64 frames here")
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        if any(parent >= 0 and parent not in children[:joint] for joint, parent in enumerate(parents)):
            raise ValueError("a joint's parent link is laid after it")
        device = model.device
        self.bodies = model.body_count
        self.joint_type = wp.array(kinds.astype(np.int32), dtype=wp.int32, device=device)
        self.joint_parent = wp.array(parents.astype(np.int32), dtype=wp.int32, device=device)
        self.joint_child = wp.array(children.astype(np.int32), dtype=wp.int32, device=device)
        self.q_start = wp.array(model.joint_q_start.numpy()[:-1].astype(np.int32), dtype=wp.int32, device=device)
        self.qd_start = wp.array(model.joint_qd_start.numpy()[:-1].astype(np.int32), dtype=wp.int32, device=device)
        places = model.joint_X_p.numpy()[None] if set_places is None else np.asarray(set_places)
        if places.shape[1:] != (len(kinds), 7):
            raise ValueError(f"set places {places.shape} are not [sets, {len(kinds)} joints, 7]")
        self.joint_x_p = wp.array(places.astype(np.float64), dtype=wp.transformd, device=device)
        if row_set is None and places.shape[0] > 1:
            raise ValueError(f"{places.shape[0]} sets of joint places and no row_set")
        self.row_set = None if row_set is None else wp.array(np.asarray(row_set, dtype=np.int32), dtype=wp.int32,
                                                                  device=device)
        self.joint_x_c = wp.array(model.joint_X_c.numpy().astype(np.float64), dtype=wp.transformd, device=device)
        self.joint_axis = wp.array(model.joint_axis.numpy().astype(np.float64), dtype=wp.vec3d, device=device)


@wp.kernel(enable_backward=False)
def _wide_qpos(joint_q: wp.array2d[wp.float32], qpos: wp.array2d[wp.float64]):
    """Newton's coordinates (the free root's turn x, y, z, w at 3) as MuJoCo's qpos (w, x, y, z) in float64."""
    row, coord = wp.tid()
    source = coord
    if coord == 3:
        source = 6
    elif coord > 3 and coord < 7:
        source = coord - 1
    qpos[row, coord] = wp.float64(joint_q[row, source])


@wp.kernel(enable_backward=False)
def _links(xpos: wp.array2d[wp.vec3d], xquat: wp.array2d[wp.quatd], body_of: wp.array[wp.int32],
           body_q: wp.array2d[wp.transformd]):
    """Each Newton link's frame: its MuJoCo body's."""
    row, link = wp.tid()
    body = body_of[link]
    body_q[row, link] = wp.transformd(xpos[row, body], xquat[row, body])


@wp.kernel(enable_backward=False)
def _narrowed_links(wide: wp.array2d[wp.transformd], body_q: wp.array2d[wp.transform]):
    """The link frames in float32 (Newton's analytic Jacobian reads its own float32 layout)."""
    row, link = wp.tid()
    frame = wide[row, link]
    place, turn = wp.transform_get_translation(frame), wp.transform_get_rotation(frame)
    body_q[row, link] = wp.transform(wp.vec3(wp.float32(place[0]), wp.float32(place[1]), wp.float32(place[2])),
                                     wp.quat(wp.float32(turn[0]), wp.float32(turn[1]), wp.float32(turn[2]),
                                             wp.float32(turn[3])))


@wp.kernel(enable_backward=False)
def _subspace(xanchor: wp.array2d[wp.vec3d], xaxis: wp.array2d[wp.vec3d], dof_joint: wp.array[wp.int32],
              dof_turns: wp.array[wp.int32], joint_s: wp.array2d[wp.spatial_vector]):
    """Each hinge or slide dof's world twist (Newton's (linear, angular) layout) from MuJoCo's float64 anchor and axis:
    a hinge (anchor x axis, axis), a slide (axis, 0); the free root's dofs (dof_joint -1) left as Newton made them."""
    row, dof = wp.tid()
    joint = dof_joint[dof]
    if joint < 0:
        return
    axis = xaxis[row, joint]
    if dof_turns[dof] != 0:
        moved = wp.cross(xanchor[row, joint], axis)
        joint_s[row, dof] = wp.spatial_vector(wp.float32(moved[0]), wp.float32(moved[1]), wp.float32(moved[2]),
                                              wp.float32(axis[0]), wp.float32(axis[1]), wp.float32(axis[2]))
    else:
        joint_s[row, dof] = wp.spatial_vector(wp.float32(axis[0]), wp.float32(axis[1]), wp.float32(axis[2]),
                                              wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))


class Exact:
    """`Forward`'s link frames made from MuJoCo's compiled model `mjm` (float64 geometry, mj_kinematics' steps) for
    Newton's model of the same file `model` (its links by body name, its coordinates MuJoCo's with the free root's turn
    reordered); one set only. `drive(optimizer)` has Newton's IKOptimizerLM read the same geometry in its analytic
    Jacobian (its link frames and its hinges' and slides' twists, float32 only at the end)."""

    def __init__(self, model, mjm):
        if int(mjm.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE) or model.joint_coord_count != mjm.nq:
            raise ValueError("MuJoCo's frames need a free root first and Newton's coordinates MuJoCo's")
        from ._level2_wide import BALL
        if np.any(mjm.jnt_type == BALL):
            raise ValueError("a ball joint: the frames hold free, hinge and slide joints")
        device = model.device
        array = lambda values, kind: wp.array(np.ascontiguousarray(values), dtype=kind, device=device)  # noqa: E731
        self.bodies, self.nbody, self.njnt, self.nq = model.body_count, mjm.nbody, max(mjm.njnt, 1), mjm.nq
        self.body_of = array([mjm.body(label.split("/")[-1]).id for label in model.body_label], wp.int32)
        self.qpos0 = array(mjm.qpos0.astype(float), wp.float64)
        self.tree = [array(mjm.body_parentid.astype(np.int32), wp.int32),
                     array(mjm.body_jntadr.astype(np.int32), wp.int32),
                     array(mjm.body_jntnum.astype(np.int32), wp.int32),
                     array(mjm.body_pos.astype(float), wp.vec3d), array(mjm.body_quat.astype(float), wp.vec4d),
                     array(mjm.jnt_type.astype(np.int32), wp.int32), array(mjm.jnt_qposadr.astype(np.int32), wp.int32),
                     array(mjm.jnt_pos.astype(float), wp.vec3d), array(mjm.jnt_axis.astype(float), wp.vec3d)]
        hinge, slide = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        dof_joint = np.full(model.joint_dof_count, -1, dtype=np.int32)
        dof_turns = np.zeros(model.joint_dof_count, dtype=np.int32)
        kinds, children, starts = model.joint_type.numpy(), model.joint_child.numpy(), model.joint_qd_start.numpy()
        for joint, (kind, child) in enumerate(zip(kinds, children)):
            dofs = int(starts[joint + 1] - starts[joint])
            if int(kind) in (FREE, FIXED) or dofs == 0:
                continue
            body = mjm.body(model.body_label[child].split("/")[-1]).id
            own = int(mjm.body_jntadr[body])
            if int(kind) not in (REVOLUTE, PRISMATIC) or dofs != 1 or int(mjm.body_jntnum[body]) != 1 or \
                    int(mjm.jnt_type[own]) != (hinge if int(kind) == REVOLUTE else slide):
                raise ValueError(f"Newton's joint {joint} into {mjm.body(body).name} is not its body's one MuJoCo"
                                 " joint")
            dof_joint[starts[joint]], dof_turns[starts[joint]] = own, int(int(kind) == REVOLUTE)
        self.dof_joint, self.dof_turns = array(dof_joint, wp.int32), array(dof_turns, wp.int32)
        self.device, self.held, self.wide = device, None, None

    def frames(self, joint_q, body_q):
        """body_q [rows, bodies] transformd: every row's link frames from joint_q [rows, coords] (Newton's layout)."""
        from ._level2_wide import _kinematics
        rows, device = joint_q.shape[0], self.device
        if self.held is None or self.held[0].shape[0] != rows:
            self.held = (wp.zeros((rows, self.nq), dtype=wp.float64, device=device),
                         wp.zeros((rows, self.nbody), dtype=wp.vec3d, device=device),
                         wp.zeros((rows, self.nbody), dtype=wp.quatd, device=device),
                         wp.zeros((rows, self.njnt), dtype=wp.vec3d, device=device),
                         wp.zeros((rows, self.njnt), dtype=wp.vec3d, device=device))
        qpos, xpos, xquat, xanchor, xaxis = self.held
        wp.launch(_wide_qpos, dim=qpos.shape, inputs=[joint_q], outputs=[qpos], device=device)
        wp.launch(_kinematics, dim=rows, inputs=[qpos, self.qpos0, *self.tree], outputs=[xpos, xquat, xanchor, xaxis],
                  device=device)
        wp.launch(_links, dim=(rows, self.bodies), inputs=[xpos, xquat, self.body_of], outputs=[body_q], device=device)

    def drive(self, optimizer):
        """Have `optimizer` (Newton's IKOptimizerLM) read MuJoCo's geometry in its analytic Jacobian: its FK's link
        frames these frames in float32 (`_fk_two_pass`), its hinges' and slides' twists from MuJoCo's anchors and axes
        (`_compute_motion_subspace`; the free root's its own, exact: its joint place is the identity). Instance
        overrides, as `_per_row.rowwise`'s."""
        own_fk, own_subspace = optimizer._fk_two_pass, optimizer._compute_motion_subspace

        def fk_two_pass(model, joint_q, body_q, X_local, n_batch):
            own_fk(model, joint_q, body_q, X_local, n_batch)
            if self.wide is None or self.wide.shape[0] != n_batch:
                self.wide = wp.zeros((n_batch, self.bodies), dtype=wp.transformd, device=self.device)
            self.frames(joint_q, self.wide)
            wp.launch(_narrowed_links, dim=self.wide.shape, inputs=[self.wide], outputs=[body_q], device=self.device)

        def motion_subspace(*, joint_q_in, body_q, joint_S_s_out):
            own_subspace(joint_q_in=joint_q_in, body_q=body_q, joint_S_s_out=joint_S_s_out)
            _, _, _, xanchor, xaxis = self.held
            wp.launch(_subspace, dim=joint_S_s_out.shape, inputs=[xanchor, xaxis, self.dof_joint, self.dof_turns],
                      outputs=[joint_S_s_out], device=self.device)

        optimizer._fk_two_pass = fk_two_pass
        optimizer._compute_motion_subspace = motion_subspace
        return optimizer


def frames(forward, joint_q, body_q):
    """body_q [rows, bodies] transformd: every row's link frames from joint_q [rows, coords] (Newton's layout), by
    `forward` (a `Forward`, or an `Exact`: MuJoCo's own geometry)."""
    if isinstance(forward, Exact):
        forward.frames(joint_q, body_q)
        return
    rows = joint_q.shape[0]
    if forward.row_set is None:
        forward.row_set = wp.zeros(rows, dtype=wp.int32, device=joint_q.device)
    if forward.row_set.shape[0] != rows:
        raise ValueError(f"row_set holds {forward.row_set.shape[0]} rows, joint_q {rows}")
    wp.launch(_frames, dim=joint_q.shape[0],
              inputs=[joint_q, forward.joint_type, forward.joint_parent, forward.joint_child, forward.q_start,
                      forward.qd_start, forward.joint_x_p, forward.row_set, forward.joint_x_c, forward.joint_axis],
              outputs=[body_q], device=joint_q.device)
