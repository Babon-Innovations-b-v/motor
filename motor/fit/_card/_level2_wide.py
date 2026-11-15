"""Level 2 in float64 with every frame's kinematics on the card (#70, 2026-10-31: level 1's card process sat at ~1.3
cores and the card near 0 %, level 2's per-frame host loops its largest share). `Wide(problem)` stands in for a
`fit._constants.Problem` in `staged`: `residuals` equal to `Problem.residuals` and `exact` to `_level2.jacobian` on the
host's MuJoCo kinematics, both to float64's rounding. Per call every frame's pose is made in one array pass on the
host (`poses`: `Problem.qpos` and `Fitting.coupled` over all frames at once), every frame's MuJoCo kinematics run in
one card launch (`_kinematics`: mj_kinematics' own steps, a thread a frame, in float64), its sites and their Jacobians
(mj_jacSite's columns: a hinge's axis, a slide's, the free joint's world places and body-frame turns) in one launch
each, and the rows assembled over all frames at once in the order `_level2.jacobian` lays them. Ball joints are
refused (MyoFullBody has none). The `TOLD` rule reads this Jacobian: it is MuJoCo's double, not the card's float32."""
import mujoco
import numpy as np
import warp as wp
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from . import _level2

FREE, BALL, SLIDE, HINGE = (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL),
                            int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE))


@wp.func
def _quat(wxyz: wp.vec4d):
    """MuJoCo's (w, x, y, z) as Warp's quaternion (x, y, z, w)."""
    return wp.quatd(wxyz[1], wxyz[2], wxyz[3], wxyz[0])


@wp.kernel(enable_backward=False)
def _kinematics(qpos: wp.array2d[wp.float64], qpos0: wp.array[wp.float64], body_parentid: wp.array[wp.int32],
                body_jntadr: wp.array[wp.int32], body_jntnum: wp.array[wp.int32], body_pos: wp.array[wp.vec3d],
                body_quat: wp.array[wp.vec4d], jnt_type: wp.array[wp.int32], jnt_qposadr: wp.array[wp.int32],
                jnt_pos: wp.array[wp.vec3d], jnt_axis: wp.array[wp.vec3d], xpos: wp.array2d[wp.vec3d],
                xquat: wp.array2d[wp.quatd], xanchor: wp.array2d[wp.vec3d], xaxis: wp.array2d[wp.vec3d]):
    """One frame's bodies, in MuJoCo's order and steps (engine_core_smooth.c mj_kinematics)."""
    frame = wp.tid()
    xpos[frame, 0] = wp.vec3d()
    xquat[frame, 0] = wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    for body in range(1, body_parentid.shape[0]):
        first = body_jntadr[body]
        count = body_jntnum[body]
        place = wp.vec3d()
        turn = wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        if count == 1 and jnt_type[first] == FREE:
            at = jnt_qposadr[first]
            place = wp.vec3d(qpos[frame, at], qpos[frame, at + 1], qpos[frame, at + 2])
            turn = wp.quatd(qpos[frame, at + 4], qpos[frame, at + 5], qpos[frame, at + 6], qpos[frame, at + 3])
            xanchor[frame, first] = place
            xaxis[frame, first] = jnt_axis[first]
        else:
            parent = body_parentid[body]
            if parent > 0:
                place = wp.quat_rotate(xquat[frame, parent], body_pos[body]) + xpos[frame, parent]
                turn = xquat[frame, parent] * _quat(body_quat[body])
            else:
                place = body_pos[body]
                turn = _quat(body_quat[body])
            for joint in range(first, first + count):
                at = jnt_qposadr[joint]
                axis = wp.quat_rotate(turn, jnt_axis[joint])
                anchor = wp.quat_rotate(turn, jnt_pos[joint]) + place
                if jnt_type[joint] == SLIDE:
                    place = place + axis * (qpos[frame, at] - qpos0[at])
                if jnt_type[joint] == HINGE:
                    turn = turn * wp.quat_from_axis_angle(jnt_axis[joint], qpos[frame, at] - qpos0[at])
                    place = anchor - wp.quat_rotate(turn, jnt_pos[joint])
                xanchor[frame, joint] = anchor
                xaxis[frame, joint] = axis
        xpos[frame, body] = place
        xquat[frame, body] = wp.normalize(turn)


@wp.kernel(enable_backward=False)
def _sites(xpos: wp.array2d[wp.vec3d], xquat: wp.array2d[wp.quatd], sites: wp.array[wp.int32],
           site_bodyid: wp.array[wp.int32], site_pos: wp.array[wp.vec3d], site_quat: wp.array[wp.vec4d],
           site_xpos: wp.array2d[wp.vec3d], site_xmat: wp.array2d[wp.mat33d]):
    """Each picked site's place and turn on one frame (mj_local2Global)."""
    frame, index = wp.tid()
    site = sites[index]
    body = site_bodyid[site]
    site_xpos[frame, index] = xpos[frame, body] + wp.quat_rotate(xquat[frame, body], site_pos[site])
    site_xmat[frame, index] = wp.quat_to_matrix(wp.normalize(xquat[frame, body] * _quat(site_quat[site])))


@wp.kernel(enable_backward=False)
def _jacobians(xpos: wp.array2d[wp.vec3d], xquat: wp.array2d[wp.quatd], xanchor: wp.array2d[wp.vec3d],
               xaxis: wp.array2d[wp.vec3d], site_xpos: wp.array2d[wp.vec3d], ancestor: wp.array2d[wp.int32],
               dof_jntid: wp.array[wp.int32], jnt_type: wp.array[wp.int32], jnt_dofadr: wp.array[wp.int32],
               jnt_bodyid: wp.array[wp.int32], places: wp.array4d[wp.float64], spins: wp.array4d[wp.float64]):
    """One picked site's place and turn Jacobian column for one dof on one frame (mj_jacSite)."""
    frame, index, dof = wp.tid()
    linear = wp.vec3d()
    angular = wp.vec3d()
    if ancestor[index, dof] != 0:
        joint = dof_jntid[dof]
        point = site_xpos[frame, index]
        kind = jnt_type[joint]
        if kind == FREE:
            local = dof - jnt_dofadr[joint]
            if local < 3:
                linear[local] = wp.float64(1.0)
            else:
                body = jnt_bodyid[joint]
                angular = wp.quat_rotate(xquat[frame, body], wp.vec3d(wp.float64(wp.where(local == 3, 1.0, 0.0)),
                                                                       wp.float64(wp.where(local == 4, 1.0, 0.0)),
                                                                       wp.float64(wp.where(local == 5, 1.0, 0.0))))
                linear = wp.cross(angular, point - xpos[frame, body])
        if kind == HINGE:
            angular = xaxis[frame, joint]
            linear = wp.cross(angular, point - xanchor[frame, joint])
        if kind == SLIDE:
            linear = xaxis[frame, joint]
    for axis in range(3):
        places[frame, index, axis, dof] = linear[axis]
        spins[frame, index, axis, dof] = angular[axis]


class Seen:
    """One frame's calibration sites' turns as `_level2.turn_bends` reads a frame's kinematics."""

    def __init__(self, site_turns):
        self.site_turns = site_turns


def ancestors(model, sites):
    """[picked site, dof] 1 where the dof moves the site (its body is the site's body or one above it)."""
    found = np.zeros((len(sites), model.nv), dtype=np.int32)
    for index, site in enumerate(sites):
        body = model.site_bodyid[site]
        chain = set()
        while body > 0:
            chain.add(body)
            body = model.body_parentid[body]
        found[index] = [model.dof_bodyid[dof] in chain for dof in range(model.nv)]
    return found


def polar(turns):
    """The targets as scipy's `from_matrix` reads R_site^T R_target: a target off orthogonal (Gram matrix not within
    1e-12 of I) replaced by its nearest rotation (U V^T), which R_site^T, a rotation, carries through unchanged."""
    flat = np.array(turns, dtype=float).reshape(-1, 3, 3)
    orthogonal = np.all(np.isclose(flat @ np.transpose(flat, (0, 2, 1)), np.eye(3), atol=1e-12), axis=(1, 2))
    if not orthogonal.all():
        left, _, right = np.linalg.svd(flat[~orthogonal], full_matrices=False)
        flat[~orthogonal] = left @ right
    return flat.reshape(np.shape(turns))


class Wide:
    """`problem` (a `fit._constants.Problem`) with `residuals` and `exact` (its Jacobian) in float64, every frame's
    kinematics on the card; every other attribute is the problem's, but `costs`, which a caller may set on the Wide
    itself (the adjoint's `sited_problem_at` and `_cost_gradients` do) and both then read."""

    def __init__(self, problem):
        model, fitting = problem.fitting.model, problem.fitting
        if np.any(model.jnt_type == BALL):
            raise ValueError("a ball joint: level 2's card kinematics hold free, hinge and slide joints")
        self.problem, self.frames = problem, len(problem.path)
        self.sites = list(fitting.place_sites) + list(fitting.frame_sites)
        self.turns = polar(problem.turns)
        self.couplings = [(int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq]), model.eq_data[eq, :5].copy())
                          for eq in range(model.neq) if model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT]
        array = lambda values, kind: wp.array(np.ascontiguousarray(values), dtype=kind)
        self.qpos0 = array(model.qpos0.astype(float), wp.float64)
        self.tree = [array(model.body_parentid.astype(np.int32), wp.int32),
                     array(model.body_jntadr.astype(np.int32), wp.int32),
                     array(model.body_jntnum.astype(np.int32), wp.int32)]
        self.body_quat = array(model.body_quat.astype(float), wp.vec4d)
        self.joints = [array(model.jnt_type.astype(np.int32), wp.int32),
                       array(model.jnt_qposadr.astype(np.int32), wp.int32),
                       array(model.jnt_pos.astype(float), wp.vec3d), array(model.jnt_axis.astype(float), wp.vec3d)]
        self.site_parts = [array(np.array(self.sites, dtype=np.int32), wp.int32),
                           array(model.site_bodyid.astype(np.int32), wp.int32)]
        self.dofs = [array(ancestors(model, self.sites), wp.int32), array(model.dof_jntid.astype(np.int32), wp.int32),
                     array(model.jnt_type.astype(np.int32), wp.int32), array(model.jnt_dofadr.astype(np.int32), wp.int32),
                     array(model.jnt_bodyid.astype(np.int32), wp.int32)]
        frames, sites = self.frames, len(self.sites)
        self.xpos = wp.zeros((frames, model.nbody), dtype=wp.vec3d)
        self.xquat = wp.zeros((frames, model.nbody), dtype=wp.quatd)
        self.xanchor = wp.zeros((frames, max(model.njnt, 1)), dtype=wp.vec3d)
        self.xaxis = wp.zeros((frames, max(model.njnt, 1)), dtype=wp.vec3d)
        self.site_xpos = wp.zeros((frames, sites), dtype=wp.vec3d)
        self.site_xmat = wp.zeros((frames, sites), dtype=wp.mat33d)
        self.places = wp.zeros((frames, sites, 3, model.nv), dtype=wp.float64)
        self.spins = wp.zeros((frames, sites, 3, model.nv), dtype=wp.float64)
        self.shared = (_level2.couplings(model), _level2.moved_blocks(problem))

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def laid(self, constants):
        """(body_pos, site_pos) with the constants laid on as `Fitting.apply` lays them, on the fitting's base as it
        stands now (the adjoint's `sited_problem_at` moves the marker sites' base between calls)."""
        fitting = self.problem.fitting
        marks, joints = fitting.unpack(constants)
        body_pos, site_pos = (np.array(part, dtype=float) for part in fitting.base)
        body_pos[fitting.inside] += joints
        site_pos[fitting.mark_sites] += marks
        return body_pos, site_pos

    def poses(self, unknowns):
        """Every frame's qpos (frames x nq) at `unknowns`, `Problem.qpos` over all frames in one pass."""
        problem, model = self.problem, self.problem.fitting.model
        values = unknowns[len(problem.slots):].reshape(self.frames, problem.width)
        qpos = problem.path.copy()
        start = 0
        if problem.root is not None:
            at = problem.root
            qpos[:, at:at + 3] = values[:, :3]
            turn = Rotation.from_quat(problem.path[:, at + 3:at + 7], scalar_first=True) * Rotation.from_rotvec(
                values[:, 3:6])
            qpos[:, at + 3:at + 7] = turn.as_quat(scalar_first=True)
            start = 6
        qpos[:, model.jnt_qposadr[problem.leading]] = values[:, start:]
        for follower, leader, coefficients in self.couplings:
            moved = qpos[:, model.jnt_qposadr[leader]] - model.qpos0[model.jnt_qposadr[leader]] if leader >= 0 else 0.0
            qpos[:, model.jnt_qposadr[follower]] = model.qpos0[model.jnt_qposadr[follower]] + sum(
                coefficients[power] * moved ** power for power in range(5))
        return qpos

    def kinematics(self, unknowns, jacobians):
        """Every frame's bodies and picked sites (and their Jacobians when `jacobians`) on the card at `unknowns`;
        the poses (frames x nq)."""
        poses = self.poses(unknowns)
        body_pos, site_pos = self.laid(self.problem.constants(unknowns))
        model, frames = self.problem.fitting.model, self.frames
        wp.launch(_kinematics, dim=frames,
                  inputs=[wp.array(poses, dtype=wp.float64), self.qpos0, *self.tree, wp.array(body_pos, dtype=wp.vec3d),
                          self.body_quat, *self.joints],
                  outputs=[self.xpos, self.xquat, self.xanchor, self.xaxis])
        wp.launch(_sites, dim=(frames, len(self.sites)),
                  inputs=[self.xpos, self.xquat, *self.site_parts, wp.array(site_pos, dtype=wp.vec3d),
                          wp.array(model.site_quat.astype(float), dtype=wp.vec4d)],
                  outputs=[self.site_xpos, self.site_xmat])
        if jacobians:
            wp.launch(_jacobians, dim=(frames, len(self.sites), model.nv),
                      inputs=[self.xpos, self.xquat, self.xanchor, self.xaxis, self.site_xpos, *self.dofs],
                      outputs=[self.places, self.spins])
        return poses

    def picked(self):
        """(site places frames x marks x 3, site turns frames x groups x 3 x 3) from the last `kinematics`."""
        marks = len(self.problem.fitting.place_sites)
        return self.site_xpos.numpy()[:, :marks], self.site_xmat.numpy()[:, marks:]

    def residuals(self, unknowns):
        """`Problem.residuals` at `unknowns`."""
        problem = self.problem
        self.kinematics(unknowns, False)
        places, seen = self.picked()
        turn_cost, place_costs = self.costs
        off = Rotation.from_matrix((np.transpose(seen, (0, 1, 3, 2)) @ self.turns).reshape(-1, 3, 3)).as_rotvec()
        framed = np.concatenate([turn_cost * off.reshape(self.frames, -1),
                                 (place_costs[None, :, None] * (places - problem.points)).reshape(self.frames, -1)],
                                axis=1)
        held, moved = len(problem.slots), 3 * len(problem.free)
        return np.concatenate([framed.ravel(), (place_costs[problem.free][:, None] * unknowns[:moved].reshape(-1, 3)).ravel(),
                               float(np.mean(place_costs)) * unknowns[moved:held]])

    def exact(self, unknowns):
        """`_level2.jacobian(problem, unknowns)` (host kinematics) with the frames' kinematics on the card and every
        frame's rows made in one array pass (`_level2.frame_rows`' maths, its entries in its order)."""
        problem = self.problem
        poses = self.kinematics(unknowns, True)
        marks, groups = len(problem.fitting.place_sites), len(problem.fitting.frame_sites)
        _, seen = self.picked()
        places, spins = self.places.numpy(), self.spins.numpy()
        turn_cost, place_costs = self.costs
        held, frames, width = len(problem.slots), self.frames, problem.width
        values = unknowns[held:].reshape(frames, width)
        mapping = self.mappings(poses, values)[:, None]
        bends = _level2.turn_bends([Seen(turns) for turns in seen], self.turns)
        turned = -turn_cost * bends @ (spins[:, marks:] @ mapping)
        placed = place_costs[None, :, None, None] * (places[:, :marks] @ mapping)
        own = np.concatenate([turned, placed], axis=1).reshape(frames, -1, width)
        frame, row, column = np.nonzero(own)
        per = 3 * (groups + marks)
        moved, starts, bodies = self.shared[1]
        across, block = np.arange(3), (len(moved), 3, 3)
        move_rows = np.broadcast_to(3 * (groups + moved)[:, None, None] + across[None, :, None], block).ravel()
        move_columns = np.broadcast_to(starts[:, None, None] + across[None, None, :], block).ravel()
        move_entries = (place_costs[moved][None, :, None, None] * self.body_turns()[:, bodies]).reshape(frames, -1)
        keys = np.concatenate([2 * frame, np.repeat(2 * np.arange(frames) + 1, move_rows.size)])
        order = np.argsort(keys, kind="stable")
        rows = np.concatenate([frame * per + row, (np.arange(frames)[:, None] * per + move_rows).ravel()])[order]
        columns = np.concatenate([held + frame * width + column, np.tile(move_columns, frames)])[order]
        entries = np.concatenate([own[frame, row, column], move_entries.ravel()])[order]
        prior_rows, prior_columns, prior_entries = _level2._priors(self, frames * per)
        shape = (frames * per + held, held + frames * width)
        return coo_matrix((np.concatenate([entries, prior_entries]),
                           (np.concatenate([rows, prior_rows]), np.concatenate([columns, prior_columns]))),
                          shape=shape).tocsr()

    def mappings(self, poses, values):
        """Every frame's `_level2.dof_map` (frames x nv x width) in one pass: the root's place one for one, its rotation
        vector by J_r, a leader its dof, a follower by its polynomial's slope at the frame's pose."""
        problem, model = self.problem, self.problem.fitting.model
        found = np.zeros((self.frames, model.nv, problem.width))
        start = 0
        if problem.root is not None:
            dof = model.jnt_dofadr[0]
            found[:, dof:dof + 3, :3] = np.eye(3)
            found[:, dof + 3:dof + 6, 3:6] = [_level2.right_jacobian(rotation) for rotation in values[:, 3:6]]
            start = 6
        column = {int(joint): start + index for index, joint in enumerate(problem.leading)}
        for joint, at in column.items():
            found[:, model.jnt_dofadr[joint], at] = 1.0
        for follower, leader, coefficients in self.shared[0]:
            if leader in column:
                moved = poses[:, model.jnt_qposadr[leader]] - model.qpos0[model.jnt_qposadr[leader]]
                found[:, model.jnt_dofadr[follower], column[leader]] += sum(
                    power * coefficients[power] * moved ** (power - 1) for power in range(1, 5))
        return found

    def body_turns(self):
        """Every frame's body turns (frames x nbody x 3 x 3) from the last `kinematics`."""
        quats = self.xquat.numpy()  # Warp's (x, y, z, w)
        return Rotation.from_quat(quats.reshape(-1, 4)).as_matrix().reshape(*quats.shape[:2], 3, 3)
