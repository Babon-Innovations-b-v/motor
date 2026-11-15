"""Level 3 across sets: Newton's batched forward kinematics with each row's own joint places (joint_X_p), so one card
batch holds every set's frames (rows = set x frame) though each set's inside-joint moves change its body places.
The kernel is Newton's _eval_fk_articulation_batched (ik_common) with joint_X_p read as the row's set's row of a
[sets, joints] array; the arithmetic is Newton's own eval_single_articulation_fk, so all rows on one set equal
Newton's bit for bit."""
import warp as wp
from newton._src.sim.articulation import eval_single_articulation_fk
from newton._src.sim.enums import BodyFlags, JointType


@wp.kernel
def _per_row_fk(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_articulation: wp.array[int],
    joint_q: wp.array2d[wp.float32],
    joint_qd: wp.array2d[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    set_places: wp.array2d[wp.transform],
    row_set: wp.array[wp.int32],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    body_q: wp.array2d[wp.transform],
    body_qd: wp.array2d[wp.spatial_vector],
):
    problem_idx, articulation_idx = wp.tid()
    eval_single_articulation_fk(
        articulation_start[articulation_idx], articulation_end[articulation_idx], joint_articulation,
        joint_q[problem_idx], joint_qd[problem_idx], joint_q_start, joint_qd_start, joint_type, joint_parent,
        joint_child, set_places[row_set[problem_idx]], joint_X_c, joint_axis, joint_dof_dim, body_com, body_flags,
        int(BodyFlags.ALL), body_q[problem_idx], body_qd[problem_idx])


def per_row_fk(model, set_places, row_set, joint_q, joint_qd, body_q, body_qd):
    """Body frames of every row, row r with joint places set_places[row_set[r]] ([sets, joints] transforms)."""
    wp.launch(
        kernel=_per_row_fk, dim=[joint_q.shape[0], model.articulation_count],
        inputs=[model.articulation_start, model.articulation_end, model.joint_articulation, joint_q, joint_qd,
                model.joint_q_start, model.joint_qd_start, model.joint_type, model.joint_parent, model.joint_child,
                set_places, row_set, model.joint_X_c, model.joint_axis, model.joint_dof_dim, model.body_com,
                model.body_flags],
        outputs=[body_q, body_qd], device=model.device)


def _late_jcalc():
    """Newton's jcalc helpers, imported late as Newton's own optimizer does (a circular import otherwise)."""
    from newton._src.sim.articulation import jcalc_motion_subspace, write_free_distance_motion_subspace
    from newton._src.solvers.featherstone.kernels import jcalc_transform
    return jcalc_motion_subspace, write_free_distance_motion_subspace, jcalc_transform


_motion_subspace_at, _free_subspace_at, _transform_at = _late_jcalc()


@wp.kernel
def _fk_local_rows(joint_type: wp.array[wp.int32], joint_q: wp.array2d[wp.float32],
                   joint_q_start: wp.array[wp.int32], joint_qd_start: wp.array[wp.int32],
                   joint_axis: wp.array[wp.vec3], joint_dof_dim: wp.array2d[wp.int32],
                   set_places: wp.array2d[wp.transform], row_set: wp.array[wp.int32],
                   joint_X_c: wp.array[wp.transform], X_local_out: wp.array2d[wp.transform]):
    """IKOptimizerLM's _fk_local with the row's set's joint place."""
    row, joint = wp.tid()
    moved = _transform_at(joint_type[joint], joint_axis, joint_qd_start[joint], joint_dof_dim[joint, 0],
                          joint_dof_dim[joint, 1], joint_q[row], joint_q_start[joint])
    X_local_out[row, joint] = set_places[row_set[row], joint] * moved * wp.transform_inverse(joint_X_c[joint])


@wp.kernel
def _motion_subspace_rows(joint_type: wp.array[wp.int32], joint_parent: wp.array[wp.int32],
                          joint_q_start: wp.array[wp.int32], joint_qd_start: wp.array[wp.int32],
                          joint_q: wp.array2d[wp.float32], joint_axis: wp.array[wp.vec3],
                          joint_dof_dim: wp.array2d[wp.int32], body_q: wp.array2d[wp.transform],
                          set_places: wp.array2d[wp.transform], row_set: wp.array[wp.int32],
                          joint_S_s: wp.array2d[wp.spatial_vector]):
    """IKOptimizerLM's _compute_motion_subspace_2d with the row's set's joint place."""
    row, joint = wp.tid()
    parent = joint_parent[joint]
    X_wpj = set_places[row_set[row], joint]
    if parent >= 0:
        X_wpj = body_q[row, parent] * X_wpj
    kind = joint_type[joint]
    if kind == int(JointType.FREE) or kind == int(JointType.DISTANCE):
        _free_subspace_at(X_wpj, wp.transform_get_translation(X_wpj), joint_qd_start[joint], joint_S_s[row])
    else:
        _motion_subspace_at(kind, joint_axis, joint_q[row], joint_dof_dim[joint, 0], joint_dof_dim[joint, 1], X_wpj,
                            wp.transform_identity(), wp.vec3(), joint_q_start[joint], joint_qd_start[joint],
                            joint_S_s[row])


def rowwise(optimizer, set_places, row_set):
    """Make Newton's IKOptimizerLM read each row's joint places as set_places[row_set[row]] in its analytic FK and
    its Jacobian's motion subspace (instance overrides of _fk_two_pass and _compute_motion_subspace)."""
    from newton._src.sim.ik.ik_common import fk_accum
    model = optimizer.model

    def fk_two_pass(model, joint_q, body_q, X_local, n_batch):
        wp.launch(_fk_local_rows, dim=[n_batch, model.joint_count],
                  inputs=[model.joint_type, joint_q, model.joint_q_start, model.joint_qd_start, model.joint_axis,
                          model.joint_dof_dim, set_places, row_set, model.joint_X_c], outputs=[X_local],
                  device=model.device)
        wp.launch(fk_accum, dim=[n_batch, model.joint_count], inputs=[model.joint_parent, X_local],
                  outputs=[body_q], device=model.device)

    def motion_subspace(*, joint_q_in, body_q, joint_S_s_out):
        wp.launch(_motion_subspace_rows, dim=[body_q.shape[0], model.joint_count],
                  inputs=[model.joint_type, model.joint_parent, model.joint_q_start, model.joint_qd_start, joint_q_in,
                          model.joint_axis, model.joint_dof_dim, body_q, set_places, row_set],
                  outputs=[joint_S_s_out], device=model.device)

    optimizer._fk_two_pass = fk_two_pass
    optimizer._compute_motion_subspace = motion_subspace
    return optimizer
