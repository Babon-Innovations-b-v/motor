"""MOTOR (#98), step 1: the whole trial's movement as one solve on the card, warm started from the fit.

The fit (`fit.angles`) solves every frame alone and the chain then low-passes the path (`model.filtered`, CUTOFF);
here every leading coordinate's path over the whole trial is one uniform cubic B-spline (`_spline`), and one
Levenberg-Marquardt solve over all their control points at once minimises the fit's own cost summed over the frames
(the groups' turns, the landmarks' markers, the bones held inside SOMA's posed skin: `_rows`, level 3's residuals on
the card) plus the spline's jerk, the leading joints inside their ranges by box bounds on their control points (a
B-spline stays inside the hull of its control points). The root's place is a spline like a joint's; its turn is the
fit's turn low-passed (`smooth_turns`) times the exponential of a spline of rotation vectors in the body's frame.

The jerk's weight a coordinate is not tuned: it is the one that puts the solve's own smoothing at the chain's cutoff
(`jerk_weights`): a node's data stiffness a (its Gauss-Newton diagonal at the warm start, the frames' median) against
a jerk weight w smooths a path as 1 / (1 + (w step / a) (2 pi f)^6), half at w = a / (step (2 pi CUTOFF)^6), where
the chain's Butterworth run forward and back is half too.

`solved(fitting, turns, points, skins, angles, rate)` gives (every frame's angles [N, nq] in the fitting model's
layout, the record, the motion: its problem and control points, step 2's warm start). `dynamics_solved` is step 2's
stage A on it (`_stage_a`: inverse dynamics, contact and the pelvis' residual as rows of the same solve);
`muscles_solved` step 4's stage B on stage A's answer (`_stage_b`: the muscles' activations solved with the movement). Card only: it raises where Warp sees no CUDA card."""
import time

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from motor.trial import _spline
from motor.solvers import CUTOFF

MOST_ROUNDS = 60
RELATIVE_DONE = 1e-7  # a kept round lowering the whole cost by under this share of it ends the solve
DAMPING_FIRST, DAMPING_RAISE, DAMPING_MOST = 1e-4, 10.0, 1e6
HOLD_PASSES = 3  # solves after the first, each holding the bounded unknowns the last one pushed out
STIFFNESS_FLOOR = 1e-6  # a coordinate's data stiffness at least this share of the largest (one no frame sees)
METHOD = ("MOTOR step 1 (#98): every leading coordinate's whole-trial path one uniform cubic B-spline (the root's "
          "turn a body-frame rotation-vector spline on the fit's low-passed turn), one Levenberg-Marquardt solve of "
          "all control points against the fit's own cost summed over the frames (turns, markers, bones inside the "
          "skin; level 3's residuals on the card) plus the jerk at the chain's cutoff, ranges as bounds; warm "
          "started from the fit")


def low_passed(values, rate):
    """`values` [N, m] through the chain's 2nd-order Butterworth at CUTOFF run forward and back (`model.filtered`'s)."""
    low, high = butter(2, CUTOFF / (rate / 2))
    if len(values) <= 3 * max(len(low), len(high)):
        return np.array(values, dtype=float)
    return filtfilt(low, high, values, axis=0)


def smooth_turns(quaternions, rate):
    """The root's turns [N, 4] (MuJoCo's w, x, y, z) low-passed as `model.filtered` does: sign kept continuous, run
    through `low_passed`, length one."""
    turns = np.array(quaternions, dtype=float)
    for frame in range(1, len(turns)):
        if turns[frame] @ turns[frame - 1] < 0:
            turns[frame] *= -1
    turns = low_passed(turns, rate)
    return turns / np.linalg.norm(turns, axis=1, keepdims=True)


def skew(vector):
    """[v]x."""
    return np.array([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]])


def right_jacobian(vector):
    """SO(3)'s right Jacobian at a rotation vector: d exp(v + d) = exp(v) exp(J_r(v) d)."""
    angle = float(np.linalg.norm(vector))
    cross = skew(vector)
    if angle < 1e-8:
        return np.eye(3) - 0.5 * cross
    return (np.eye(3) - (1.0 - np.cos(angle)) / angle ** 2 * cross
            + (angle - np.sin(angle)) / angle ** 3 * cross @ cross)


class Layout:
    """The spline's unknowns a reduced column (`clip.reduction`): a leading joint's coordinate, the root's place
    (columns of dofs 0-2) and its turn's rotation vector off `reference` (dofs 3-5, the body's frame)."""

    def __init__(self, reduction, seeds, reference):
        self.free_dof = reduction.free_dof.numpy().astype(int)
        self.free_coord = reduction.free_coord.numpy().astype(int)
        self.root = np.flatnonzero(self.free_dof < 6)
        if list(self.free_dof[self.root]) != list(range(6)):
            raise ValueError(f"the root's reduced columns hold dofs {self.free_dof[self.root]}, not 0-5")
        self.joints = np.flatnonzero(self.free_dof >= 6)
        self.lower = reduction.lower.numpy().astype(float)
        self.upper = reduction.upper.numpy().astype(float)
        self.lower[self.root], self.upper[self.root] = -np.inf, np.inf
        self.seeds = np.asarray(seeds, dtype=float)
        self.reference = Rotation.from_quat(np.asarray(reference)[:, [1, 2, 3, 0]])

    def values(self, qpos):
        """The unknowns' values [N, size] of poses qpos [N, nq]."""
        found = np.zeros((len(qpos), len(self.free_dof)))
        found[:, self.joints] = qpos[:, self.free_coord[self.joints]]
        found[:, self.root[:3]] = qpos[:, :3]
        turns = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
        found[:, self.root[3:]] = (self.reference.inv() * turns).as_rotvec()
        return found

    def poses(self, values):
        """The poses [N, nq] of the unknowns' values (frozen coordinates and followers the seeds', for `project`)."""
        qpos = self.seeds.copy()
        qpos[:, self.free_coord[self.joints]] = values[:, self.joints]
        qpos[:, :3] = values[:, self.root[:3]]
        qpos[:, 3:7] = (self.reference * Rotation.from_rotvec(values[:, self.root[3:]])).as_quat()[:, [3, 0, 1, 2]]
        return qpos

    def carried(self, hessians, gradients, twists, values, qpos):
        """Each frame's (H, g) from Newton's reduced dofs onto the unknowns: the root's six columns through
        S^-1 U, U the place's and the body-frame rotation vector's world twists (lin, ang) and S Newton's own."""
        found_h, found_g = hessians.copy(), gradients.copy()
        for frame in range(len(hessians)):
            turn = Rotation.from_quat(qpos[frame, [4, 5, 6, 3]]).as_matrix()
            spun = turn @ right_jacobian(values[frame, self.root[3:]])
            twist = np.zeros((6, 6))
            twist[:3, :3] = np.eye(3)
            twist[:3, 3:] = skew(qpos[frame, :3]) @ spun
            twist[3:, 3:] = spun
            mapped = np.eye(len(self.free_dof))
            mapped[np.ix_(self.root, self.root)] = np.linalg.solve(twists[frame], twist)
            found_h[frame] = mapped.T @ hessians[frame] @ mapped
            found_g[frame] = mapped.T @ gradients[frame]
        return found_h, found_g


def jerk_weights(hessians, step):
    """Each unknown's jerk weight: its data stiffness (the frames' median Gauss-Newton diagonal, floored at
    STIFFNESS_FLOOR of the largest) / (step (2 pi CUTOFF)^6): the chain's cutoff as the solve's half point."""
    stiffness = np.median(np.diagonal(hessians, axis1=1, axis2=2), axis=0)
    stiffness = np.maximum(stiffness, STIFFNESS_FLOOR * stiffness.max())
    return stiffness / (step * (2.0 * np.pi * CUTOFF) ** 6)


def path_jerk(values, joints, rate):
    """The root mean square jerk of the joints' paths (degrees a second cubed; third differences)."""
    if len(values) < 4:
        return 0.0
    return float(np.degrees(np.sqrt(np.mean((np.diff(values[:, joints], 3, axis=0) * rate ** 3) ** 2))))


class Problem:
    """The whole trial on the card: `rows` (`_rows.Rows`), `layout`, the jerk weights, the frame step and the
    landmarks' targets `points` [N, markers, 3] (`landmark_errors`)."""

    def __init__(self, rows, layout, step):
        self.rows, self.layout, self.step = rows, layout, step
        self.weights = self.points = None

    def measured(self, control):
        """(the frames' cost [N], H, g on the unknowns, the poses) at control points `control`."""
        qpos = self.rows.projected(self.layout.poses(_spline.nodes(control)))
        hessians, gradients, costs, twists = self.rows.blocks(qpos)
        hessians, gradients = self.layout.carried(hessians, gradients, twists, self.layout.values(qpos), qpos)
        return costs, hessians, gradients, qpos

    def frames_cost(self, values):
        """The frames' cost [N] of the path whose nodes are `values`."""
        return self.rows.blocks(self.rows.projected(self.layout.poses(values)))[2]

    def total(self, costs, control):
        """The whole cost: the frames' and the jerk's."""
        return float(np.sum(costs)) + _spline.jerk_cost(control, self.step, self.weights)

    def stepped(self, control, hessians, gradients, damping, rows=None):
        """The damped, bounded step from `control`: the frames' and the jerk's normal equations (and `rows`, each
        node's (Jacobians on its control points, residuals): `_spline.Banded.add_rows`), unknowns on a bound their
        descent leaves held, HOLD_PASSES more solves holding those the step pushed out."""
        system = _spline.Banded(len(control), control.shape[1])
        system.add_nodes(hessians, gradients)
        if rows is not None:
            system.add_rows(*rows)
        system.add_jerk(control, self.step, self.weights)
        lower, upper = self.layout.lower[None], self.layout.upper[None]
        held = _spline.held(control, system.gradient, lower, upper)
        step = system.step(damping, held)
        for _ in range(HOLD_PASSES):
            held |= (((control <= lower + _spline.SLACK) & (step < 0))
                     | ((control >= upper - _spline.SLACK) & (step > 0)))
            step = system.step(damping, held)
        return np.clip(control + step, lower, upper)


def solve(problem, control):
    """Levenberg-Marquardt over the control points: (control points, rounds kept, rounds tried, the whole cost)."""
    costs, hessians, gradients, _ = problem.measured(control)
    total, damping, kept, tried = problem.total(costs, control), DAMPING_FIRST, 0, 0
    while tried < MOST_ROUNDS and damping <= DAMPING_MOST:
        trial = problem.stepped(control, hessians, gradients, damping)
        tried += 1
        trial_costs, trial_h, trial_g, _ = problem.measured(trial)
        trial_total = problem.total(trial_costs, trial)
        if trial_total < total:
            lowered = total - trial_total
            control, costs, hessians, gradients, total = trial, trial_costs, trial_h, trial_g, trial_total
            kept, damping = kept + 1, max(damping / DAMPING_RAISE, 1e-12)
            if lowered < RELATIVE_DONE * total:
                break
        else:
            damping *= DAMPING_RAISE
    return control, kept, tried, total


def landmark_errors(clip, qpos, points):
    """Each landmark's root mean square miss over the frames (mm, by its site's name): its site on MuJoCo's model at
    poses qpos [N, nq] against its target `points` [N, markers, 3]; a landmark no frame targets left out."""
    import mujoco
    data, misses = mujoco.MjData(clip.mjm), []
    for pose in qpos:
        data.qpos[:] = pose
        mujoco.mj_kinematics(clip.mjm, data)
        misses.append(data.site_xpos[clip.place_sites] - points[len(misses)])
    squared = np.sum(np.asarray(misses) ** 2, axis=2)
    return {clip.mjm.site(int(site)).name: round(1000.0 * float(np.sqrt(np.nanmean(squared[:, mark]))), 3)
            for mark, site in enumerate(clip.place_sites) if np.isfinite(squared[:, mark]).any()}


def retrusted(clip, trusts):
    """The clip's markers' costs with each landmark's trust times its learned weight (`trusts`, by site name; one
    where not named), the markers' goal (the costs' squared sum) kept: cost_k^2 = goal t_k w_k / sum(t w), t_k the
    clip's own cost_k^2 / goal (`fit._frames.weights`)."""
    costs = np.asarray(clip.place_costs, dtype=float)
    weights = np.array([float(trusts.get(clip.mjm.site(int(site)).name, 1.0)) for site in clip.place_sites])
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError(f"landmark weights {weights}: each must be a positive number")
    goal, weighed = float(np.sum(costs ** 2)), costs ** 2 * weights
    return np.sqrt(goal * weighed / max(float(np.sum(weighed)), 1e-12))


def solved(fitting, turns, points, skins, angles, rate, trusts=None):
    """Every frame's angles [N, nq] of the whole trial's solve, its record and its motion, warm started from the fit's
    `angles` [N, nq] (MuJoCo's own compile of `fitting`'s file): `turns`, `points`, `skins` the clip's `fit.targets`,
    `trusts` each landmark's learned weight (`retrusted`; a job's "motor landmarks", None: the fit's own trusts)."""
    from motor.fit._card import _clip
    from motor.fit._skin import card_device
    if card_device() == "cpu":
        raise RuntimeError("MOTOR solves on the card: Warp sees no CUDA card")
    return solved_on(_clip.for_fitting(fitting, fitting.path), turns, points, skins, angles, rate, trusts)


def solved_on(clip, turns, points, skins, angles, rate, trusts=None):
    """`solved` on a fitted model's `fit._card._clip.Clip`."""
    from motor.trial._rows import Rows
    began = time.perf_counter()
    if trusts:
        clip.place_costs = retrusted(clip, trusts)
    angles = np.asarray(angles, dtype=float)
    step = 1.0 / rate
    layout = Layout(clip.reduction, angles, smooth_turns(angles[:, 3:7], rate))
    problem = Problem(Rows(clip, turns, points, skins, angles), layout, step)
    problem.points = np.asarray(points)
    fitted_values = layout.values(problem.rows.projected(angles))
    control = np.clip(_spline.fitted(fitted_values, step), layout.lower[None], layout.upper[None])
    _, hessians, _, _ = problem.measured(control)
    problem.weights = jerk_weights(hessians, step)
    chain_values = layout.values(problem.rows.projected(layout.poses(low_passed(fitted_values, rate))))
    control, kept, tried, total = solve(problem, control)
    values = _spline.nodes(control)
    qpos = problem.rows.projected(layout.poses(values))
    record = {"method": METHOD, "rounds kept": kept, "rounds tried": tried, "whole cost": round(total, 6),
              "frames cost": {name: round(float(np.sum(problem.frames_cost(path))), 6)
                              for name, path in (("fit", fitted_values), ("fit low-passed", chain_values),
                                                 ("motor", values))},
              "jerk deg/s3": {name: round(path_jerk(path, layout.joints, rate), 1)
                              for name, path in (("fit", fitted_values), ("fit low-passed", chain_values),
                                                 ("motor", values))},
              "off the fit deg": round(float(np.degrees(np.sqrt(np.mean(
                  (values[:, layout.joints] - fitted_values[:, layout.joints]) ** 2)))), 4),
              "off its spline deg": round(float(np.degrees(np.abs(
                  layout.values(qpos)[:, layout.joints] - values[:, layout.joints]).max())), 6),
              "landmark weights": "learned (motor landmarks)" if trusts else "the fit's trusts",
              "landmark errors mm": landmark_errors(clip, qpos, np.asarray(points)),
              "seconds": round(time.perf_counter() - began, 2)}
    return qpos, record, (problem, control)


def dynamics_solved(motion, physics, source, foot_points, sides, limits, height, settings=None):
    """MOTOR step 2's stage A (`_stage_a.solved`) from `solved`'s motion: the trial's movement made dynamically
    consistent with the smooth contact on the seen floor, the pelvis' residual held to `limits`; `settings` a job's
    "motor settings" (`_dynamics.Settings.of`, None: the defaults): (angles, record, its motion, stage B's start)."""
    from motor.trial._dynamics import Settings
    from motor.trial._stage_a import solved as stage_a
    return stage_a(motion, physics, source, foot_points, sides, limits, height, Settings.of(settings))


def muscles_solved(motion, physics, limits, height):
    """MOTOR step 4's stage B (`_stage_b.solved`) from `dynamics_solved`'s motion: the movement, contact and the
    muscles' activations solved together, every joint's reserve penalised: (angles, activations [N, muscles] in
    `physics.muscles`' order, record)."""
    from motor.trial._stage_b import solved as stage_b
    return stage_b(motion, physics, limits, height)
