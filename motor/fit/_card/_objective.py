"""A site's place and a link frame's turn on the card with their own offset and weight per problem (rows from several
bodies or level-1 sets in one batch): Newton's IKObjectivePosition and IKObjectiveRotation with the offset and weight
read per problem instead of one for all."""
import warp as wp
from newton import ik


@wp.kernel(enable_backward=False)
def _place_residuals(body_q: wp.array2d[wp.transform], target: wp.array[wp.vec3], link_index: int,
                     offset: wp.array[wp.vec3], weight: wp.array[wp.float32], start_idx: int,
                     problem_of: wp.array[wp.int32], residuals: wp.array2d[wp.float32]):
    row = wp.tid()
    problem = problem_of[row]
    error = target[problem] - wp.transform_point(body_q[row, link_index], offset[problem])
    for axis in range(3):
        residuals[row, start_idx + axis] = weight[problem] * error[axis]


@wp.kernel(enable_backward=False)
def _place_jacobian(link_index: int, offset: wp.array[wp.vec3], weight: wp.array[wp.float32],
                    affects_dof: wp.array[wp.uint8], body_q: wp.array2d[wp.transform],
                    joint_s: wp.array2d[wp.spatial_vector], start_idx: int, jacobian: wp.array3d[wp.float32]):
    problem, dof = wp.tid()
    if affects_dof[dof] == 0:
        return
    site = wp.transform_point(body_q[problem, link_index], offset[problem])
    motion = joint_s[problem, dof]
    velocity = wp.vec3(motion[0], motion[1], motion[2]) + wp.cross(wp.vec3(motion[3], motion[4], motion[5]), site)
    for axis in range(3):
        jacobian[problem, start_idx + axis, dof] = -weight[problem] * velocity[axis]


class PlacePerProblem(ik.IKObjectivePosition):
    """target [problems] vec3, offset [problems] vec3 in the link's frame, weight [problems] float32; analytic only."""

    def __init__(self, link_index, offsets, targets, weights):
        super().__init__(link_index, wp.vec3(0.0, 0.0, 0.0), targets, weight=1.0)
        self.offsets = offsets
        self.weights = weights

    def supports_analytic(self):
        return True

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_place_residuals, dim=body_q.shape[0],
                  inputs=[body_q, self.target_positions, self.link_index, self.offsets, self.weights, start_idx,
                          problem_idx], outputs=[residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_place_jacobian, dim=[body_q.shape[0], model.joint_dof_count],
                  inputs=[self.link_index, self.offsets, self.weights, self.affects_dof, body_q, joint_S_s,
                          start_idx], outputs=[jacobian], device=self.device)


@wp.func
def _turn_error(transform: wp.transform, target: wp.vec4, offset: wp.quat):
    """The world-frame log of the link's turn (with its offset) times the target's inverse, canonicalised."""
    actual = wp.quat(transform[3], transform[4], transform[5], transform[6]) * offset
    wanted = wp.quat(target[0], target[1], target[2], target[3])
    error = actual * wp.quat_inverse(wanted)
    if wp.dot(actual, wanted) < 0.0:
        error = -error
    length = wp.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])
    turn = wp.vec3(2.0 * error[0], 2.0 * error[1], 2.0 * error[2])
    if length > 1e-8:
        turn = wp.vec3(error[0], error[1], error[2]) * (2.0 * wp.atan2(length, error[3]) / length)
    return turn


@wp.func
def _log_rate(turn: wp.vec3, spin: wp.vec3):
    """How fast the log turn moves under a world spin: SO(3)'s inverse left Jacobian at turn, times spin."""
    angle = wp.length(turn)
    bend = 1.0 / 12.0
    if angle > 1e-4:
        bend = 1.0 / (angle * angle) - (1.0 + wp.cos(angle)) / (2.0 * angle * wp.sin(angle))
    across = wp.cross(turn, spin)
    return spin - 0.5 * across + bend * wp.cross(turn, across)


@wp.kernel(enable_backward=False)
def _turn_residuals(body_q: wp.array2d[wp.transform], target: wp.array[wp.vec4], link_index: int,
                    offset: wp.array[wp.quat], weight: wp.array[wp.float32], start_idx: int,
                    problem_of: wp.array[wp.int32], residuals: wp.array2d[wp.float32]):
    row = wp.tid()
    problem = problem_of[row]
    turn = _turn_error(body_q[row, link_index], target[problem], offset[problem])
    for axis in range(3):
        residuals[row, start_idx + axis] = weight[problem] * turn[axis]


@wp.kernel(enable_backward=False)
def _turn_jacobian(body_q: wp.array2d[wp.transform], target: wp.array[wp.vec4], link_index: int,
                   offset: wp.array[wp.quat], weight: wp.array[wp.float32], affects_dof: wp.array[wp.uint8],
                   joint_s: wp.array2d[wp.spatial_vector], start_idx: int, jacobian: wp.array3d[wp.float32]):
    """The log turn's exact Jacobian, as mink's (Newton's own is the spin alone, off by the log's bend away from the
    target: a toy row 0.1 rad off its turn then cycled between two poses and never stopped)."""
    problem, dof = wp.tid()
    if affects_dof[dof] == 0:
        return
    motion = joint_s[problem, dof]
    turn = _turn_error(body_q[problem, link_index], target[problem], offset[problem])
    rate = _log_rate(turn, wp.vec3(motion[3], motion[4], motion[5]))
    for axis in range(3):
        jacobian[problem, start_idx + axis, dof] = weight[problem] * rate[axis]


class TurnPerProblem(ik.IKObjectiveRotation):
    """A link frame's turn: target [problems] vec4 (x, y, z, w), offset [problems] quat in the link's frame,
    weight [problems] float32; Newton's IKObjectiveRotation (canonicalised error) per problem, analytic only, its
    Jacobian the log's exact one (mink's), not Newton's spin alone."""

    def __init__(self, link_index, offsets, targets, weights):
        super().__init__(link_index, wp.quat_identity(), targets, canonicalize_quat_err=True, weight=1.0)
        self.offsets = offsets
        self.weights = weights

    def supports_analytic(self):
        return True

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_turn_residuals, dim=body_q.shape[0],
                  inputs=[body_q, self.target_rotations, self.link_index, self.offsets, self.weights, start_idx,
                          problem_idx], outputs=[residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_turn_jacobian, dim=[body_q.shape[0], model.joint_dof_count],
                  inputs=[body_q, self.target_rotations, self.link_index, self.offsets, self.weights,
                          self.affects_dof, joint_S_s, start_idx], outputs=[jacobian],
                  device=self.device)


@wp.func
def _wide(value: wp.vec3):
    return wp.vec3d(wp.float64(value[0]), wp.float64(value[1]), wp.float64(value[2]))


@wp.func
def _wide_turn(value: wp.vec4):
    return wp.quatd(wp.float64(value[0]), wp.float64(value[1]), wp.float64(value[2]), wp.float64(value[3]))


@wp.kernel(enable_backward=False)
def _place_residuals_wide(body_q: wp.array2d[wp.transformd], target: wp.array[wp.vec3], link_index: int,
                          offset: wp.array[wp.vec3], weight: wp.array[wp.float32], start_idx: int,
                          problem_of: wp.array[wp.int32], residuals: wp.array2d[wp.float64]):
    row = wp.tid()
    problem = problem_of[row]
    error = _wide(target[problem]) - wp.transform_point(body_q[row, link_index], _wide(offset[problem]))
    for axis in range(3):
        residuals[row, start_idx + axis] = wp.float64(weight[problem]) * error[axis]


@wp.kernel(enable_backward=False)
def _turn_residuals_wide(body_q: wp.array2d[wp.transformd], target: wp.array[wp.vec4], link_index: int,
                         offset: wp.array[wp.quat], weight: wp.array[wp.float32], start_idx: int,
                         problem_of: wp.array[wp.int32], residuals: wp.array2d[wp.float64]):
    row = wp.tid()
    problem = problem_of[row]
    turn_of = offset[problem]
    actual = wp.transform_get_rotation(body_q[row, link_index]) * _wide_turn(
        wp.vec4(turn_of[0], turn_of[1], turn_of[2], turn_of[3]))
    wanted = _wide_turn(target[problem])
    error = actual * wp.quat_inverse(wanted)
    if wp.dot(actual, wanted) < wp.float64(0.0):
        error = -error
    length = wp.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])
    turn = wp.vec3d(wp.float64(2.0) * error[0], wp.float64(2.0) * error[1], wp.float64(2.0) * error[2])
    if length > wp.float64(1e-12):
        turn = wp.vec3d(error[0], error[1], error[2]) * (wp.float64(2.0) * wp.atan2(length, error[3]) / length)
    for axis in range(3):
        residuals[row, start_idx + axis] = wp.float64(weight[problem]) * turn[axis]


def wide_residuals(objective, body_q, start_idx, problem_of, residuals):
    """One objective's residuals in float64 into residuals [rows, count] from body_q [rows, bodies] transformd."""
    if isinstance(objective, PlacePerProblem):
        kernel, target = _place_residuals_wide, objective.target_positions
    elif isinstance(objective, TurnPerProblem):
        kernel, target = _turn_residuals_wide, objective.target_rotations
    else:
        raise TypeError(f"{type(objective).__name__} has no float64 residuals")
    wp.launch(kernel, dim=body_q.shape[0], inputs=[body_q, target, objective.link_index, objective.offsets,
                                                   objective.weights, start_idx, problem_of],
              outputs=[residuals], device=body_q.device)
