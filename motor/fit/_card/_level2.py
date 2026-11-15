"""Level 2's exact Jacobian (plan step 4): the Jacobian of `fit._constants.Problem.residuals` in its own sparse
pattern (`Problem.sparsity`), analytic, for scipy's `least_squares(jac=...)` in place of its 2-point differences, the
optimiser and its stopping unchanged. Per frame, the unknowns reach the dofs by `dof_map` (the root's place one for one,
its rotation vector off mink's turn through SO(3)'s right Jacobian, MuJoCo's free joint turning in its body's frame; a
leading joint its own dof and each follower's by its polynomial's slope); a place row is its site's place Jacobian
through that, a turn row (the log of R_site^T R_target) its site's turn Jacobian through -J_l^-1(log) R_site^T; a
marker's move reaches its site through its body's turn, an inside joint's move every site it carries through its
parent's turn; a prior row its constant's weight. The frames' kinematics come from MuJoCo in float64 on the host
(`Kinematics`, the reference); `_level2_wide` gives the same in float64 with them on the card."""
import mujoco
import numpy as np
from scipy.sparse import coo_matrix


def skew(vector):
    """The cross-product matrix of a 3-vector."""
    return np.array([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]])


def right_jacobian(rotation):
    """SO(3)'s right Jacobian at a rotation vector: R(v + dv) = R(v) exp(J_r(v) dv)."""
    angle, cross = np.linalg.norm(rotation), skew(rotation)
    if angle < 1e-8:
        return np.eye(3) - 0.5 * cross
    return (np.eye(3) - (1 - np.cos(angle)) / angle ** 2 * cross
            + (angle - np.sin(angle)) / angle ** 3 * cross @ cross)


def left_jacobian_inverse(rotation):
    """SO(3)'s inverse left Jacobian at a rotation vector: log(exp(d) R) = log(R) + J_l^-1 d."""
    angle, cross = np.linalg.norm(rotation), skew(rotation)
    if angle < 1e-8:
        return np.eye(3) - 0.5 * cross
    factor = 1 / angle ** 2 - (1 + np.cos(angle)) / (2 * angle * np.sin(angle))
    return np.eye(3) - 0.5 * cross + factor * cross @ cross


def couplings(model):
    """Each joint coupling as (follower joint, leader joint, polynomial coefficients); a lock (no leader) left out."""
    return [(int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq]), model.eq_data[eq, :5].copy())
            for eq in range(model.neq) if model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[eq] >= 0]


def dof_map(problem, qpos, rotation, coupled):
    """How a frame's leading coordinates move the dofs (nv x width): the root's place one for one, its rotation vector
    by J_r, a leader its own dof, a follower by d polyval / d leader at the frame's `qpos` (`coupled`: `couplings`)."""
    model = problem.fitting.model
    found = np.zeros((model.nv, problem.width))
    start = 0
    if problem.root is not None:
        dof = model.jnt_dofadr[0]
        found[dof:dof + 3, :3] = np.eye(3)
        found[dof + 3:dof + 6, 3:6] = right_jacobian(rotation)
        start = 6
    column = {int(joint): start + index for index, joint in enumerate(problem.leading)}
    for joint, at in column.items():
        found[model.jnt_dofadr[joint], at] = 1.0
    for follower, leader, coefficients in coupled:
        if leader in column:
            moved = qpos[model.jnt_qposadr[leader]] - model.qpos0[model.jnt_qposadr[leader]]
            slope = sum(power * coefficients[power] * moved ** (power - 1) for power in range(1, 5))
            found[model.jnt_dofadr[follower], column[leader]] += slope
    return found


class Kinematics:
    """One frame's kinematics on the fitting model (MuJoCo, float64): the sites' places and turns, their place and
    turn Jacobians (world frame, dofs), each body's turn."""

    def __init__(self, fitting, qpos):
        model = fitting.model
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        self.site_turns = data.site_xmat[fitting.frame_sites].reshape(-1, 3, 3).copy()
        self.body_turns = data.xmat.reshape(-1, 3, 3).copy()
        self.places = np.zeros((len(fitting.place_sites), 3, model.nv))
        self.spins = np.zeros((len(fitting.frame_sites), 3, model.nv))
        for index, site in enumerate(fitting.place_sites):
            mujoco.mj_jacSite(model, data, self.places[index], None, site)
        for index, site in enumerate(fitting.frame_sites):
            mujoco.mj_jacSite(model, data, None, self.spins[index], site)


def frame_rows(problem, frame, kinematics, bends, values, pose, shared):
    """A frame's rows of the Jacobian as COO triplets (rows counted from the frame's first, 3 (groups + markers) of
    them; columns of the whole unknowns; values): its coordinates' columns, the free markers' and the inside joints'
    moves, each block only where it reads. `bends`: the frame's -J_l^-1(log) R_site^T per group (`turn_bends`, made
    for every frame at once). `pose`: the frame's qpos at `values` (`problem.qpos`, made once by the
    caller for the kinematics too). `shared`: the model's `couplings` and the problem's `moved_blocks`, the same in
    every frame."""
    fitting = problem.fitting
    turn_cost, place_costs = problem.costs
    held = len(problem.slots)
    rotation = values[3:6] if problem.root is not None else np.zeros(3)
    coupled, (marks, starts, bodies) = shared
    mapping = dof_map(problem, pose, rotation, coupled)
    groups = len(fitting.frame_sites)
    turned = -turn_cost * bends @ (kinematics.spins @ mapping)
    placed = place_costs[:, None, None] * (kinematics.places @ mapping)
    own = np.concatenate([turned, placed]).reshape(-1, problem.width)
    row, column = np.nonzero(own)
    rows, columns, entries = [row], [held + frame * problem.width + column], [own[row, column]]
    across, block = np.arange(3), (len(marks), 3, 3)
    rows.append(np.broadcast_to(3 * (groups + marks)[:, None, None] + across[None, :, None], block).ravel())
    columns.append(np.broadcast_to(starts[:, None, None] + across[None, None, :], block).ravel())
    entries.append((place_costs[marks][:, None, None] * kinematics.body_turns[bodies]).ravel())
    return np.concatenate(rows), np.concatenate(columns), np.concatenate(entries)


def moved_blocks(problem):
    """Each 3 x 3 block a constant's move puts in a frame's place rows: its marker, its first column and the body whose
    turn it is (a free marker's own body; an inside joint's parent, on every marker it carries), as arrays."""
    fitting, model = problem.fitting, problem.fitting.model
    found = [(mark, 3 * index, model.site_bodyid[fitting.place_sites[mark]]) for index, mark in enumerate(problem.free)]
    found += [(mark, 3 * (len(problem.free) + joint), model.body_parentid[fitting.inside[joint]])
              for joint, riders in enumerate(fitting.riding) for mark in riders]
    return tuple(np.array([block[part] for block in found], dtype=int) for part in range(3))


def left_jacobian_inverses(rotations):
    """`left_jacobian_inverse` of each rotation vector (n x 3 -> n x 3 x 3), every row in one array pass."""
    rotations = np.asarray(rotations, dtype=float)
    angles = np.linalg.norm(rotations, axis=1)
    cross = np.zeros((len(rotations), 3, 3))
    cross[:, 0, 1], cross[:, 0, 2], cross[:, 1, 2] = -rotations[:, 2], rotations[:, 1], -rotations[:, 0]
    cross[:, 1, 0], cross[:, 2, 0], cross[:, 2, 1] = rotations[:, 2], -rotations[:, 1], rotations[:, 0]
    bent = angles >= 1e-8
    safe = np.where(bent, angles, 1.0)
    factor = np.where(bent, 1 / safe ** 2 - (1 + np.cos(safe)) / (2 * safe * np.sin(safe)), 0.0)
    return np.eye(3) - 0.5 * cross + factor[:, None, None] * cross @ cross


def turn_bends(kinematics, targets):
    """Every frame's -J_l^-1(log(R_site^T R_target)) R_site^T per group (frames x groups x 3 x 3; the turn cost left
    to `frame_rows`), the logs and inverses made over every frame and group in one pass."""
    seen = np.transpose(np.array([frame.site_turns for frame in kinematics]), (0, 1, 3, 2))
    shape = seen.shape
    offs = _logs((seen @ np.asarray(targets)[:shape[0]]).reshape(-1, 3, 3))
    return left_jacobian_inverses(offs).reshape(shape) @ seen


def _logs(turns):
    """Rotation matrices' rotation vectors (scipy's, as the residuals take them)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(turns).as_rotvec()


def jacobian(problem, unknowns):
    """The whole Jacobian of `problem.residuals` at `unknowns` (sparse, rows and columns as the residuals and
    `problem.sparsity`), the constants laid on the model as the residuals lay them; the frames' kinematics MuJoCo's on
    the host."""
    fitting = problem.fitting
    fitting.apply(problem.constants(unknowns))
    held = len(problem.slots)
    values = [unknowns[held + frame * problem.width:held + (frame + 1) * problem.width]
              for frame in range(len(problem.path))]
    poses = [problem.qpos(frame, value) for frame, value in enumerate(values)]
    seen = [Kinematics(fitting, pose) for pose in poses]
    per = 3 * (len(fitting.frame_sites) + len(fitting.place_sites))
    triplets, shared = [], (couplings(fitting.model), moved_blocks(problem))
    bends = turn_bends(seen, problem.turns)
    for frame, value in enumerate(values):
        rows, columns, entries = frame_rows(problem, frame, seen[frame], bends[frame], value, poses[frame], shared)
        triplets.append((rows + frame * per, columns, entries))
    triplets.append(_priors(problem, len(problem.path) * per))
    rows, columns, entries = (np.concatenate(part) for part in zip(*triplets))
    shape = (len(problem.path) * per + held, held + len(problem.path) * problem.width)
    return coo_matrix((entries, (rows, columns)), shape=shape).tocsr()


def _priors(problem, first):
    """The constants' prior rows as COO triplets from row `first`: each its constant's weight on its own column."""
    turn_cost, place_costs = problem.costs
    held = len(problem.slots)
    moved = 3 * len(problem.free)
    weights = np.concatenate([np.repeat(place_costs[problem.free], 3),
                              np.full(held - moved, float(np.mean(place_costs)))])
    return first + np.arange(held), np.arange(held), weights
