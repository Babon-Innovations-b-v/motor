"""`Footing` on the card (#70, 2026-11-05; the owner: "should prob just fix that now aswell, so we know what were
working with"): every frame's legs solved at once, one row a frame, by the card loop the fit's level 3 runs on
(`fit._card._solve.solved`: Newton's IK Jacobians, the damped step in the reduced dofs, the exact projection) with
mink's tasks of `_footed.Footing`: each foot body's place and turn (weight 1, as its `FrameTask`), the pelvis' place
(FOOTED_HEIGHT; its place and turn frozen but its height, so its height alone), every leading dof but the legs' and
the root's height frozen, the couplings exact, the legs' leading joints within their ranges or within where `before`
stands them (each row's own bounds), and a leg's free directions held where the frame stands them (`Held`, the card's
`FreeHold`: the projector onto the null space of the tasks' reduced Jacobian at the frame's own pose, by
Gram-Schmidt on the card), each step judged as mink's are (`_footed.FOOTED_LAM_FIRST`). What the
card gives back is float32: the frozen coordinates are the frame's own float64 ones and the followers set again from
their leaders in float64 (`Couplings.apply`), so only the legs' leading joints carry the card's rounding. A frame
the card does not stop in MOST_STEPS (its float32 pose cycling at its floor: 9 of the jump's 180 frames on one call,
2026-11-05), or that it moved a leg joint further than LEAP from the frame (its step run off), is handed back to
`Footing`, which solves it by mink, the reference; never kept half solved."""
import mujoco
import numpy as np
import warp as wp

from motor.fit._card._clip import couplings as coupled
from motor.fit._card._objective import PlacePerProblem, TurnPerProblem
from motor.fit._card._project import Projection
from motor.fit._card._seeds import newton_order
from motor.fit._card._solve import MOST_STEPS, solved
from motor.fit._card._step import Reduction
from motor.rra._footed import FOOTED_LAM_FIRST

DEVICE = "cuda:0"
BIG = 1e30  # a bound that never holds
# A task row whose part not already spanned by the rows before it is under this share of the longest row's length is
# taken as seen already (numpy's rank rule in `_footed.FreeHold` is float64's eps; the card's Jacobian is float32, so a
# row dependent on the others, the pelvis' three place rows on its one height dof, leaves 1e-7 of itself)
HOLD_RANK = 1e-5
# A leg joint (or the pelvis' height) the card moved further than this from where the frame stands it is no footing of
# a move capped at 2 cm and 5 degrees: its float32 step ran off (a near-singular leg's Cholesky, 2026-11-05: the
# standing clip's rows at 1e30), and the frame goes to mink
LEAP = 0.5  # rad or m
# m: Newton's body places off MuJoCo's at the first frame, else the card's model is not the file's. Newton's float32
# forward kinematics of the right file stands MyoFullBody's bodies 0.0087 mm off MuJoCo's float64 at any pose and
# place (#98, 2026-11-07: on a sized person's file with its lumbar wrappers, 0.0086 to 0.0111 mm over 80 poses
# on Warp's CPU, the root at the origin or 4 m out), so 1e-5 refused the right model; a wrong file stands a body off by its sizing, millimetres
FK_MATCHED = 1e-4
LIMITED_TYPES = (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE))


@wp.kernel(enable_backward=False)
def _held(reduced: wp.array3d[wp.float32], qpos: wp.array2d[wp.float32], target: wp.array2d[wp.float32],
          free_coord: wp.array[wp.int32], weight: wp.float64, basis: wp.array3d[wp.float64],
          hessian: wp.array3d[wp.float32], gradient: wp.array2d[wp.float32]):
    """One row's held directions (`_footed.FreeHold` on the card): an orthonormal basis of the rows of its tasks'
    reduced Jacobian (modified Gram-Schmidt, twice, a row under HOLD_RANK of the longest dropped), the projector P onto
    what they leave, and the hold's J^T J = weight P and J^T r = weight P (q - target) on the reduced dofs."""
    row = wp.tid()
    count = reduced.shape[1]
    size = reduced.shape[2]
    longest = wp.float64(0.0)
    for residual in range(count):
        length = wp.float64(0.0)
        for column in range(size):
            value = wp.float64(reduced[row, residual, column])
            length += value * value
        longest = wp.max(longest, wp.sqrt(length))
    kept = int(0)
    for residual in range(count):
        for column in range(size):
            basis[row, kept, column] = wp.float64(reduced[row, residual, column])
        for _ in range(2):
            for earlier in range(kept):
                dot = wp.float64(0.0)
                for column in range(size):
                    dot += basis[row, earlier, column] * basis[row, kept, column]
                for column in range(size):
                    basis[row, kept, column] = basis[row, kept, column] - dot * basis[row, earlier, column]
        length = wp.float64(0.0)
        for column in range(size):
            length += basis[row, kept, column] * basis[row, kept, column]
        length = wp.sqrt(length)
        if length > wp.float64(HOLD_RANK) * longest:
            for column in range(size):
                basis[row, kept, column] = basis[row, kept, column] / length
            kept += 1
    for first in range(size):
        pulled = wp.float64(0.0)
        for second in range(size):
            value = wp.float64(0.0)
            if first == second:
                value = wp.float64(1.0)
            for vector in range(kept):
                value -= basis[row, vector, first] * basis[row, vector, second]
            hessian[row, first, second] = wp.float32(weight * value)
            offset = wp.float64(qpos[row, free_coord[second]]) - wp.float64(target[row, free_coord[second]])
            pulled += value * offset
        gradient[row, first] = wp.float32(weight * pulled)


@wp.kernel(enable_backward=False)
def _pulled(qpos: wp.array2d[wp.float32], target: wp.array2d[wp.float32], free_coord: wp.array[wp.int32],
            hessian: wp.array3d[wp.float32], gradient: wp.array2d[wp.float32]):
    """One row's hold J^T r = weight P (q - target) by the projector `_held` found at the solve's first pose (its
    `hessian`, weight P)."""
    row, first = wp.tid()
    pulled = wp.float64(0.0)
    for second in range(hessian.shape[2]):
        offset = wp.float64(qpos[row, free_coord[second]]) - wp.float64(target[row, free_coord[second]])
        pulled += wp.float64(hessian[row, first, second]) * offset
    gradient[row, first] = wp.float32(pulled)


@wp.kernel(enable_backward=False)
def _held_cost(qpos: wp.array2d[wp.float32], target: wp.array2d[wp.float32], free_coord: wp.array[wp.int32],
               hessian: wp.array3d[wp.float32], cost: wp.array[wp.float64]):
    """One row's hold cost at `qpos`, |r|^2 = weight (q - target)^T P (q - target) by the solve's projector (`hessian`,
    weight P; zero at the solve's first pose, the target, whatever the buffer holds then)."""
    row = wp.tid()
    total = wp.float64(0.0)
    for first in range(hessian.shape[1]):
        ahead = wp.float64(qpos[row, free_coord[first]]) - wp.float64(target[row, free_coord[first]])
        for second in range(hessian.shape[2]):
            offset = wp.float64(qpos[row, free_coord[second]]) - wp.float64(target[row, free_coord[second]])
            total += ahead * wp.float64(hessian[row, first, second]) * offset
    cost[row] = total


class Held:
    """The hold's rows for `solved` (its `hold`): each row's free directions held at `target` (the rows' poses,
    MuJoCo's layout), weight FOOTED_HOLD squared, none of it in the damping (mink's FreeHold has no LM damping). The
    directions are the solve's first pose's (`begin` before each solve; the first step's `normal` finds them, the
    later ones only pull along them), as `_footed.FreeHold` fixes its own at the frame's pose: found again at every
    step, a straight leg's bend leaves the feet's rank and comes back a hair off it, held on one step and let go on
    the next (#98, 2026-11-07)."""

    def __init__(self, rows, residuals, free_coord, weight, target):
        size = free_coord.shape[0]
        self.free_coord, self.weight, self.target = free_coord, float(weight), target
        self.basis = wp.zeros((rows, residuals, size), dtype=wp.float64, device=DEVICE)
        self.hessian = wp.zeros((rows, size, size), dtype=wp.float32, device=DEVICE)
        self.gradient = wp.zeros((rows, size), dtype=wp.float32, device=DEVICE)
        self.squared = wp.zeros(rows, dtype=wp.float32, device=DEVICE)
        self.measured = wp.zeros(rows, dtype=wp.float64, device=DEVICE)
        self.found = False

    def begin(self):
        """A new solve: its first step finds the directions again, at its first pose."""
        self.found = False

    def normal(self, slopes, reduced, qpos):
        """(J^T J, J^T r, 0) of the hold at `qpos`, its directions found from the tasks' reduced Jacobian `reduced`
        on the solve's first step and kept for the rest."""
        if not self.found:
            wp.launch(_held, dim=reduced.shape[0], inputs=[reduced, qpos, self.target, self.free_coord,
                                                           wp.float64(self.weight), self.basis],
                      outputs=[self.hessian, self.gradient], device=DEVICE)
            self.found = True
        else:
            wp.launch(_pulled, dim=self.gradient.shape, inputs=[qpos, self.target, self.free_coord, self.hessian],
                      outputs=[self.gradient], device=DEVICE)
        return self.hessian, self.gradient, self.squared

    def cost(self, qpos):
        """Each row's hold cost at `qpos` (MuJoCo's layout), float64: its part of the judged steps' cost, as mink's
        FreeHold is one of the judged tasks (`fit.judged_solve`)."""
        wp.launch(_held_cost, dim=qpos.shape[0], inputs=[qpos, self.target, self.free_coord, self.hessian],
                  outputs=[self.measured], device=DEVICE)
        return self.measured


def newton_bodies(model, built):
    """Each MuJoCo body of `model` (Newton's build, names by path) as Newton's body index in `built`: the label whose
    last part names it, the longest when several do; -1 for the world."""
    labels = [label.split("/")[-1] for label in built.body_label]
    found = [-1]
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        matched = [index for index, label in enumerate(labels) if name == label or name.endswith("_" + label)]
        if not matched:
            raise ValueError(f"no Newton body for {name}")
        found.append(max(matched, key=lambda index: len(labels[index])))
    return found


def coordinate_of(model, dof):
    """A dof's own coordinate: a free joint's translation its own, a hinge's or slide's its qpos address."""
    joint = model.dof_jntid[dof]
    if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
        offset = dof - model.jnt_dofadr[joint]
        if offset >= 3:
            raise ValueError("a free joint's turn has no coordinate of its own")
        return int(model.jnt_qposadr[joint] + offset)
    return int(model.jnt_qposadr[joint])


def frozen_coordinates(model, frozen):
    """Every coordinate the frozen dofs hold: a free joint's turn its whole quaternion."""
    found = set()
    for dof in frozen:
        joint = model.dof_jntid[dof]
        if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE) and dof - model.jnt_dofadr[joint] >= 3:
            found.update(range(model.jnt_qposadr[joint] + 3, model.jnt_qposadr[joint] + 7))
        else:
            found.add(coordinate_of(model, dof))
    return sorted(found)


class CardFooting:
    """`Footing`'s solve on the card for `model` (Newton's MuJoCo model of the person's file at `path`): `feet` the foot
    bodies, `moving` the legs' leading dofs and the root's height, `frozen` every other leading dof, `couplings`
    the model's (`model.Couplings`), `costs` (foot, height, hold) the tasks' weights."""

    def __init__(self, model, path, couplings, feet, moving, frozen, costs):
        import newton
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_mjcf(str(path), convert_mjc_equality_constraints=False, parse_meshes=False)
        self.built = builder.finalize(device=DEVICE)
        if self.built.joint_coord_count != model.nq or self.built.joint_dof_count != model.nv:
            raise ValueError(f"Newton's {self.built.joint_coord_count} coords / {self.built.joint_dof_count} dofs "
                             f"against the model's {model.nq} / {model.nv}")
        self.model, self.couplings, self.feet, self.costs = model, couplings, list(feet), costs
        self.root = int(model.jnt_bodyid[0])
        self.links = newton_bodies(model, self.built)
        self.moving = [int(dof) for dof in sorted(moving)]
        self.moving_coord = [coordinate_of(model, dof) for dof in self.moving]
        self.frozen_coord = frozen_coordinates(model, frozen)
        ranged = np.array([bool(model.jnt_limited[model.dof_jntid[dof]])
                           and int(model.jnt_type[model.dof_jntid[dof]]) in LIMITED_TYPES for dof in self.moving])
        self.lower = np.where(ranged, [model.jnt_range[model.dof_jntid[dof], 0] for dof in self.moving], -BIG)
        self.upper = np.where(ranged, [model.jnt_range[model.dof_jntid[dof], 1] for dof in self.moving], BIG)
        pairs = coupled(model)
        dof_of = {coordinate_of(model, dof): dof for dof in range(model.nv)
                  if not (int(model.jnt_type[model.dof_jntid[dof]]) == int(mujoco.mjtJoint.mjJNT_FREE)
                          and dof - model.jnt_dofadr[model.dof_jntid[dof]] >= 3)}
        poly = np.array([values for _, _, values in pairs]).reshape(-1, 5)
        self.reduction = Reduction(self.moving, self.moving_coord, list(self.lower), list(self.upper),
                                   [dof_of[follower] for follower, _, _ in pairs],
                                   [dof_of[leader] if leader >= 0 else -1 for _, leader, _ in pairs],
                                   [leader for _, leader, _ in pairs],
                                   [model.qpos0[leader] if leader >= 0 else 0.0 for _, leader, _ in pairs], poly,
                                   DEVICE)
        self.projection = Projection([-BIG] * model.nq, [BIG] * model.nq, [follower for follower, _, _ in pairs],
                                     [leader for _, leader, _ in pairs],
                                     [model.qpos0[follower] for follower, _, _ in pairs],
                                     [model.qpos0[leader] if leader >= 0 else 0.0 for _, leader, _ in pairs], poly,
                                     self.frozen_coord, DEVICE)
        self.data = mujoco.MjData(model)
        self.solvers = {}
        self.checked = False
        self.most_steps = MOST_STEPS

    def _check(self, qpos):
        """Refuse a Newton model whose bodies stand off MuJoCo's at `qpos` by over FK_MATCHED."""
        import newton
        state = self.built.state()
        self.built.joint_q.assign(newton_order(qpos[None])[0].astype(np.float32))
        newton.eval_fk(self.built, self.built.joint_q, self.built.joint_qd, state)
        placed = state.body_q.numpy()[:, :3].astype(float)
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        miss = max(float(np.linalg.norm(placed[self.links[body]] - self.data.xpos[body]))
                   for body in range(1, self.model.nbody))
        if miss > FK_MATCHED:
            raise ValueError(f"Newton's bodies {miss * 1000:.4f} mm off MuJoCo's: the card's model is not the file's")
        self.checked = True

    def _solver(self, rows):
        """The batch's IKOptimizerLM for `rows` rows, its objectives' target arrays to fill (made once a row count)."""
        from newton import ik
        if rows not in self.solvers:
            zeros = lambda kind, width: wp.zeros(rows, dtype=kind, device=DEVICE) if width == 1 else wp.array(
                np.zeros((rows, width), dtype=np.float32), dtype=kind, device=DEVICE)
            weight = lambda value: wp.array(np.full(rows, value, dtype=np.float32), dtype=wp.float32, device=DEVICE)
            identity = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (rows, 1)).astype(np.float32), dtype=wp.quat,
                                device=DEVICE)
            places = [zeros(wp.vec3, 3) for _ in self.feet]
            turns = [zeros(wp.vec4, 4) for _ in self.feet]
            height = zeros(wp.vec3, 3)
            objectives = []
            for body, place, turn in zip(self.feet, places, turns):
                objectives.append(PlacePerProblem(self.links[body], zeros(wp.vec3, 3), place, weight(self.costs[0])))
                objectives.append(TurnPerProblem(self.links[body], identity, turn, weight(self.costs[0])))
            objectives.append(PlacePerProblem(self.links[self.root], zeros(wp.vec3, 3), height, weight(self.costs[1])))
            optimizer = ik.IKSolver(self.built, rows, objectives, optimizer="lm",
                                    jacobian_mode=ik.IKJacobianType.ANALYTIC)._impl
            seeds = wp.zeros((rows, self.model.nq), dtype=wp.float32, device=DEVICE)
            hold = Held(rows, optimizer.n_residuals, self.reduction.free_coord, self.costs[2] ** 2, seeds)
            self.solvers[rows] = (optimizer, places, turns, height, seeds, hold)
        return self.solvers[rows]

    def _targets(self, path, before):
        """Each frame's feet in `before` (place, turn x, y, z, w) and the pelvis' place in `path`."""
        places, turns, heights = [], [], []
        for qpos, still in zip(path, before):
            self.data.qpos[:] = still
            mujoco.mj_kinematics(self.model, self.data)
            places.append(self.data.xpos[self.feet].copy())
            turns.append(self.data.xquat[self.feet][:, [1, 2, 3, 0]].copy())
            self.data.qpos[:] = qpos
            mujoco.mj_kinematics(self.model, self.data)
            heights.append(self.data.xpos[self.root].copy())
        return np.array(places), np.array(turns), np.array(heights)

    def _bounds(self, before):
        """Each row's ranges: the legs' leading joints within their range, or within where `before` stands them
        (reduced [rows, size] and per coordinate [rows, coords]), every other coordinate unbounded."""
        still = before[:, self.moving_coord]
        lower, upper = np.minimum(self.lower, still), np.maximum(self.upper, still)
        lower_all, upper_all = np.full(before.shape, -BIG), np.full(before.shape, BIG)
        lower_all[:, self.moving_coord], upper_all[:, self.moving_coord] = lower, upper
        return [wp.array(np.ascontiguousarray(values, dtype=np.float32), dtype=wp.float32, device=DEVICE)
                for values in (lower, upper, lower_all, upper_all)]

    def footed(self, path, before):
        """`Footing.footed` on the card: `path` with each foot put back where it stands in `before`, every frame at
        once, and the frames the card did not stop in MOST_STEPS (its float32 pose cycling, 2026-11-05: 9 of the
        jump's 180 frames), for `Footing` to solve by mink."""
        if not self.checked:
            self._check(np.asarray(before[0], dtype=float))
        rows = len(path)
        optimizer, places, turns, height, seeds, hold = self._solver(rows)
        found_places, found_turns, found_heights = self._targets(path, before)
        for foot in range(len(self.feet)):
            places[foot].assign(found_places[:, foot].astype(np.float32))
            turns[foot].assign(found_turns[:, foot].astype(np.float32))
        height.assign(found_heights.astype(np.float32))
        seeds.assign(np.asarray(path, dtype=np.float32))
        joint_q = wp.array(newton_order(path).astype(np.float32), dtype=wp.float32, device=DEVICE)
        hold.begin()
        qpos, steps = solved(optimizer, self.reduction, self.projection, joint_q, seeds, self.most_steps, hold=hold,
                             bounds=self._bounds(np.asarray(before, dtype=float)), marked=True,
                             judged_from=FOOTED_LAM_FIRST)
        found = qpos.numpy().astype(float)
        moved = np.abs(found[:, self.moving_coord] - np.asarray(path, dtype=float)[:, self.moving_coord])
        unstopped = (steps.numpy() < 0) | ~np.isfinite(moved).all(axis=1) | (moved.max(axis=1) > LEAP)
        found[:, self.frozen_coord] = np.asarray(path, dtype=float)[:, self.frozen_coord]
        found[unstopped] = np.asarray(path, dtype=float)[unstopped]  # mink's to fill
        return np.array([self.couplings.apply(qpos) for qpos in found]), np.flatnonzero(unstopped)
