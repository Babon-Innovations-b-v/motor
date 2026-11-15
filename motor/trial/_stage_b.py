"""MOTOR step 4 (#98), stage B: the whole trial's movement, contact and muscles as one solve, from stage A's answer.

The unknowns: stage A's spline control points (the movement) and every node's muscle activations (`_muscles`). The
cost, on MuJoCo's C (`_dynamics.Exact`; the Jacobians the card's, the speed's with every velocity product): the
fit's own cost and the jerk (step 1's), the pelvis' six residual rows by stage A's augmented Lagrangian
(its multipliers and penalty carried on), every leading joint's reserve (what the joint needs, tau of stage A's rows,
less what the muscles give: the moment arms' sum of MuJoCo's own muscle forces, gain x activation + bias, over
weight x height) squared at RESERVE weight, the muscles' effort (the sum of activations cubed, OpenCap's) and the
activation dynamics (the excitation each span between two nodes needs, `_muscles.excitations`, held inside [0, 1] by
a squared hinge). A joint at an end of its range pushed back in, a knee's ligament-held turn (`muscles.LIGAMENTS`)
and a joint no muscle spans (the neck's) give their torque at STOP_SHARE of a reserve's weight: their stops,
ligaments and passive structures carry it, as the static optimisation's stops do.

Solve: Levenberg-Marquardt, each step eliminating every node's activations from its normal equations (their block is
diagonal plus the reserve rows' rank: Woodbury, one joints x joints solve a node), so the movement's banded system
(`_spline.Banded`) carries what the muscles can and cannot give; the activations then follow from the movement's
step and are clipped to [0, 1]. The activation dynamics' pairing of two nodes is left out of the elimination (each
node keeps its own share: a Gauss-Newton model; the cost is always exact). The movement starts at stage A's answer,
the activations at their own solve with the movement held there (the static optimisation's place: stage B's warm
start and its comparison). Outer rounds as stage A's: the pelvis' multipliers updated, its penalty raised without
PROGRESS, the reserves' weight raised RESERVE_RAISE times while the worst actuated reserve passes RESERVE_LIMIT, the
pelvis' penalty and the excitations' weight with it (raised alone, the reserves' weight bent the movement off the
pelvis' limits on most training trials, 2026-11-08).

The weights are scales of a frame's fit cost at stage A's answer, as stage A's: a reserve at RESERVE_LIMIT on one
joint, all muscles fully active on one node (MUSCLE_EFFORT), a span's excitation EXCITATION_SCALE outside [0, 1],
each weighs a frame's fit; never fitted to the lab's angles or moments."""
import time

import mujoco
import numpy as np

from motor.trial import _spline, landmark_errors
from motor.trial._muscles import Curves, Paths, excitations
from motor.trial._dynamics import Dynamics, Exact
from motor.trial._stage_a import Weighed, residual_record, shares

RESERVE_LIMIT = 0.01  # a joint's reserve over weight x height the solve holds every actuated joint under
STOP_SHARE = 1e-4  # a stop's, a ligament's or an unactuated joint's weight against a reserve's
MUSCLE_EFFORT = 1.0  # every muscle fully active on one node against a frame's fit cost
EXCITATION_SCALE = 0.1  # a span's excitation this far outside [0, 1] weighs a frame's fit
ACTIVATION_START = 0.05
OUTER_MOST = 6
INNER_MOST = 12
WARM_ROUNDS = 40  # the activations' own solve, the movement held
RESERVE_RAISE = 10.0
PENALTY_RAISE, PROGRESS = 10.0, 0.25
DAMPING_FIRST, DAMPING_RAISE, DAMPING_MOST = 1e-4, 10.0, 1e6
RELATIVE_DONE = 1e-6
HOLD_PASSES = 3  # `trial.HOLD_PASSES`
SPANS = 1e-4  # m: a muscle spans a leading joint whose moment arm passes this at a node
STOP_REACH = 0.05  # rad: a joint's stop carries its reserve from this far inside an end of its range, fully at it
METHOD = ("MOTOR step 4 (#98), stage B: stage A's whole-trial solve with every node's muscle activations as "
          "unknowns, their lengths and moment arms MuJoCo's own at every node, MuJoCo's own "
          "muscle curves on the card (Warp), every joint's reserve penalised, the activations' effort (cubed) and "
          "MuJoCo's activation dynamics; Levenberg-Marquardt with each node's activations eliminated (Woodbury) "
          "onto the movement's banded normal equations")


class Muscled:
    """Stage B's rows on stage A's `dynamics`: the person's muscle `paths` and `curves`, the leading joints' layout;
    `loose` the joints a ligament carries or no muscle spans (an arm past SPANS at no node of `qpos`)."""

    def __init__(self, dynamics, paths, curves, weight_height, activation_dynamics, qpos):
        path, model = dynamics.path, dynamics.path.model
        self.dynamics, self.paths, self.curves, self.scale = dynamics, paths, curves, weight_height
        self.leads = np.asarray(path.couplings.leads)[6:]
        joints = model.dof_jntid[self.leads]
        self.lead_at = model.jnt_qposadr[joints]
        self.unknowns = len(path.leads)
        self.selection = np.zeros((len(self.leads), self.unknowns))
        for unknown, column in enumerate(path.leads):
            if column >= 6:
                self.selection[column - 6, unknown] = 1.0
        limited = model.jnt_limited[joints].astype(bool)
        self.lower = np.where(limited, model.jnt_range[joints, 0], -np.inf)
        self.upper = np.where(limited, model.jnt_range[joints, 1], np.inf)
        from motor.muscles import LIGAMENTS
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or "" for joint in joints]
        spanned = np.abs(paths.at(qpos)[1]).max(axis=(0, 1)) > SPANS
        self.loose = np.array([name.endswith(LIGAMENTS) for name in names]) | ~spanned
        self.dynamics_prm = activation_dynamics  # the muscles' dynprm [M, 3]

    def lead_values(self, qpos):
        """The leading joints' values [N, J]."""
        return qpos[:, self.lead_at]

    def muscle_part(self, qpos, qvel, activations):
        """Every node's muscles at the state and `activations` [N, M]: a dict of the paths, the curves, the forces
        and the joints' muscle torques over weight x height (`given` [N, J])."""
        lengths, arms = self.paths.at(qpos)
        speeds = qvel[:, self.leads]
        velocity = (arms @ speeds[:, :, None])[:, :, 0]
        gains, biases, gain_length, gain_speed, bias_length = self.curves.evaluated(lengths, velocity)
        forces = gains * activations + biases
        given = (np.swapaxes(arms, 1, 2) @ forces[:, :, None])[:, :, 0] / self.scale
        return {"arms": arms, "gains": gains, "forces": forces, "by_length": gain_length * activations + bias_length,
                "by_speed": gain_speed * activations, "given": given, "lengths": lengths, "velocity": velocity}

    def weights(self, reserves, qpos, reserve_weight):
        """Each node's each joint's reserve weight [N, J] and its derivative on the joint's value [N, J]:
        `reserve_weight`, STOP_SHARE of it where a ligament or no muscle carries it (`loose`), and between the two
        for a reserve pushing a joint back in from within STOP_REACH of an end of its range (`stop_shares`: its stop
        carries it, the passive structures that hold a straight knee, as the static optimisation's stops; smooth in
        the joint's value, so the cost is too)."""
        values = self.lead_values(qpos)
        low, low_slope = stop_shares(values - self.lower)
        high, high_slope = stop_shares(self.upper - values)
        share = 1.0 - (1.0 - STOP_SHARE) * np.where(reserves > 0, low, 0.0)
        slope = -(1.0 - STOP_SHARE) * np.where(reserves > 0, low_slope, 0.0)
        share_high = 1.0 - (1.0 - STOP_SHARE) * np.where(reserves < 0, high, 0.0)
        slope_high = (1.0 - STOP_SHARE) * np.where(reserves < 0, high_slope, 0.0)
        found = share * share_high
        by_value = slope * share_high + share * slope_high
        found = np.where(self.loose[None], STOP_SHARE, found)
        by_value = np.where(self.loose[None], 0.0, by_value)
        return reserve_weight * found, reserve_weight * by_value

    def muscle_jacobians(self, part):
        """The muscle torques' Jacobians on the leading joints' place and speed (2 x [N, J, J], over weight x
        height): through each muscle's length and speed along its arms (the arms' own change with the pose left
        out, a Gauss-Newton model)."""
        arms = part["arms"]
        crossed = np.swapaxes(arms, 1, 2)
        return [crossed @ (part[kind][:, :, None] * arms) / self.scale for kind in ("by_length", "by_speed")]

    def on_activations(self, part):
        """Each node's muscle torques' Jacobian on its activations [N, J, M] (over weight x height)."""
        return np.swapaxes(part["arms"] * part["gains"][:, :, None], 1, 2) / self.scale


def stop_shares(distance):
    """A stop's share of a joint's reserve at `distance` (rad) inside an end of its range, and its derivative: one at
    the end (or past it), none from STOP_REACH in, a cubic smoothstep between."""
    scaled = np.clip(distance / STOP_REACH, 0.0, 1.0)
    inside = (distance > 0.0) & (distance < STOP_REACH)
    return 1.0 - scaled ** 2 * (3.0 - 2.0 * scaled), -6.0 * scaled * (1.0 - scaled) * inside / STOP_REACH


class Solve:
    """Stage B's whole cost and its steps: `problem` (step 1's), `weighed` (stage A's root rows), `muscled`, the
    weights."""

    def __init__(self, problem, weighed, muscled, reserve_weight, effort, excitation):
        self.problem, self.weighed, self.muscled = problem, weighed, muscled
        self.reserve_weight, self.effort, self.excitation = reserve_weight, effort, excitation

    def measured(self, control, activations, motion=None):
        """At (control, activations): (the whole cost, its pieces); `motion` the movement's measure already taken
        at `control` (the fit's costs, H, g, the dynamics' evaluation), else taken."""
        problem, muscled = self.problem, self.muscled
        if motion is None:
            costs, hessians, gradients, _ = problem.measured(control)
            motion = (costs, hessians, gradients, muscled.dynamics.evaluated(control))
        costs, _, _, evaluated = motion
        torques, _, (qpos, qvel, _), _ = evaluated
        part = muscled.muscle_part(qpos, qvel, activations)
        reserves = torques[:, 6:] - part["given"]
        weights, weights_by_value = muscled.weights(reserves, qpos, self.reserve_weight)
        excited, first, second = excitations(activations, problem.step, muscled.dynamics_prm)
        over = np.maximum(excited - 1.0, 0.0) - np.maximum(-excited, 0.0)
        total = (problem.total(costs, control) + self.weighed.cost(torques[:, :6])
                 + float(np.sum(weights * reserves ** 2)) + self.effort * float(np.sum(activations ** 3))
                 + self.excitation * float(np.sum(over ** 2)))
        return total, {"motion": motion, "part": part, "reserves": reserves, "weights": weights,
                       "weights by value": weights_by_value,
                       "over": over, "first": first, "second": second}

    def activation_system(self, activations, pieces, damping):
        """Each node's activation normal equations as Woodbury takes them: (D [N, M] the damped diagonal, the rows on
        the activations [N, J, M] times sqrt(weight), the gradient [N, M])."""
        muscled = self.muscled
        rows = -muscled.on_activations(pieces["part"]) * np.sqrt(pieces["weights"])[:, :, None]
        residuals = np.sqrt(pieces["weights"]) * pieces["reserves"]
        gradient = (np.swapaxes(rows, 1, 2) @ residuals[:, :, None])[:, :, 0] + 1.5 * self.effort * activations ** 2
        diagonal = 3.0 * self.effort * np.maximum(activations, 0.0)
        over, first, second = pieces["over"], pieces["first"], pieces["second"]
        gradient[:-1] += self.excitation * over * first
        gradient[1:] += self.excitation * over * second
        active = over != 0.0
        diagonal[:-1] += self.excitation * active * first ** 2
        diagonal[1:] += self.excitation * active * second ** 2
        full = diagonal + np.einsum("njm,njm->nm", rows, rows)
        return diagonal + damping * (full + 1.0), rows, gradient

    def eliminated(self, inverse, rows, right):
        """H_aa^-1 `right` [N, M, ...] by Woodbury: D^-1 - D^-1 R^T (I + R D^-1 R^T)^-1 R D^-1, every node at once."""
        scaled = rows * inverse[:, None, :]
        inner = np.eye(rows.shape[1])[None] + scaled @ np.swapaxes(rows, 1, 2)
        first = inverse[:, :, None] * right.reshape(len(right), right.shape[1], -1)
        solved = np.linalg.solve(inner, rows @ first)
        return (first - np.swapaxes(scaled, 1, 2) @ solved).reshape(right.shape)

    def steps(self, control, activations, pieces, damping, motion_rows, move=True):
        """The damped step from (control, activations), unclipped: (the control points' [N + 2, m], the activations'
        [N, M], the control points held, the activations held): each node's activations eliminated onto the
        movement's banded system (`move`; else the movement held and the activations' own step)."""
        problem = self.problem
        diagonal, rows, gradient = self.activation_system(activations, pieces, damping)
        held_activations = _spline.held(activations, gradient, 0.0, 1.0)
        inverse = np.where(held_activations, 0.0, 1.0 / diagonal)
        if not move:
            return (np.zeros_like(control), -self.eliminated(inverse, rows, gradient),
                    np.ones_like(control, dtype=bool), held_activations)
        _, hessians, gradients, _ = pieces["motion"]
        jacobians, residuals = motion_rows  # [N, 3, rows, m], [N, rows]: the root's and the reserves', weighed
        count, _, size, unknowns = jacobians.shape
        stacked = np.swapaxes(jacobians, 1, 2).reshape(count, size, 3 * unknowns)  # rows x (3 m)
        # the reserves' rows on the activations are `rows`: H_ax = rows^T (the reserves' rows on the movement)
        cross = np.swapaxes(rows, 1, 2) @ stacked[:, 6:]  # [N, M, 3m]
        solved_cross = self.eliminated(inverse, rows, cross)
        solved_gradient = self.eliminated(inverse, rows, gradient)
        crossed = np.swapaxes(cross, 1, 2)
        hessian = np.swapaxes(stacked, 1, 2) @ stacked - crossed @ solved_cross
        reduced = (np.swapaxes(stacked, 1, 2) @ residuals[:, :, None] - crossed @ solved_gradient[:, :, None])[:, :, 0]
        # the reserves' weights move with the joints near their ends: half of d(w) / d(value) x reserve^2 a node
        stopping = 0.5 * pieces["weights by value"] * pieces["reserves"] ** 2 @ self.muscled.selection
        system = _spline.Banded(len(control), control.shape[1])
        system.add_nodes(hessians, gradients + stopping)
        system.add_dense(hessian.reshape(count, 3, unknowns, 3, unknowns).transpose(0, 1, 3, 2, 4),
                         reduced.reshape(count, 3, unknowns))
        system.add_jerk(control, problem.step, problem.weights)
        lower, upper = problem.layout.lower[None], problem.layout.upper[None]
        held = _spline.held(control, system.gradient, lower, upper)
        step = system.step(damping, held)
        for _ in range(HOLD_PASSES):
            held |= (((control <= lower + _spline.SLACK) & (step < 0))
                     | ((control >= upper - _spline.SLACK) & (step > 0)))
            step = system.step(damping, held)
        moved = np.stack([step[node:node + 3] for node in range(count)]).reshape(count, 3 * unknowns)
        return step, -(solved_gradient + (solved_cross @ moved[:, :, None])[:, :, 0]), held, held_activations

    def stepped(self, control, activations, pieces, damping, motion_rows, move=True):
        """`steps` taken: the control points inside the movement's bounds, the activations inside [0, 1]."""
        step, activation_step, _, _ = self.steps(control, activations, pieces, damping, motion_rows, move)
        layout = self.problem.layout
        return (np.clip(control + step, layout.lower[None], layout.upper[None]),
                np.clip(activations + activation_step, 0.0, 1.0))

    def motion_rows(self, control, pieces):
        """Each node's root and reserve rows on its three control points, weighed ([N, 3, 6 + J, m]), and their
        residuals [N, 6 + J]."""
        weighed, muscled = self.weighed, self.muscled
        evaluated = pieces["motion"][3]
        torques, touching, states, found = evaluated
        place, speed, pace = weighed.dynamics.jacobians(control, touching, states, found)
        by_place, by_speed = muscled.muscle_jacobians(pieces["part"])
        selection = muscled.selection
        place[:, 6:] -= by_place @ selection
        speed[:, 6:] -= by_speed @ selection
        root = np.full(6, np.sqrt(weighed.penalty / 2.0))
        weights = np.concatenate([np.broadcast_to(root, (len(torques), 6)), np.sqrt(pieces["weights"])], axis=1)
        place, speed, pace = (kind * weights[:, :, None] for kind in (place, speed, pace))
        residuals = np.concatenate([np.sqrt(weighed.penalty / 2.0) * (torques[:, :6]
                                                                       + weighed.multipliers / weighed.penalty),
                                    np.sqrt(pieces["weights"]) * pieces["reserves"]], axis=1)
        return _spline.spread(place, speed, pace, self.problem.step), residuals


def inner(solve, control, activations, move=True, most=INNER_MOST):
    """One Levenberg-Marquardt solve of stage B's whole cost: (control, activations, kept, tried)."""
    total, pieces = solve.measured(control, activations)
    rows = solve.motion_rows(control, pieces) if move else None
    damping, kept, tried = DAMPING_FIRST, 0, 0
    while tried < most and damping <= DAMPING_MOST:
        trial_control, trial_activations = solve.stepped(control, activations, pieces, damping, rows, move)
        tried += 1
        motion = pieces["motion"] if not move else None
        trial_total, trial_pieces = solve.measured(trial_control, trial_activations, motion)
        if trial_total < total:
            lowered = total - trial_total
            control, activations, total, pieces = trial_control, trial_activations, trial_total, trial_pieces
            kept, damping = kept + 1, max(damping / DAMPING_RAISE, 1e-12)
            if lowered < RELATIVE_DONE * total:
                break
            if move:
                rows = solve.motion_rows(control, pieces)
        else:
            damping *= DAMPING_RAISE
    return control, activations, kept, tried


def reserve_record(pieces, muscled):
    """The reserves' measures (%BW*ht): the actuated joints' p50/p95/max over every node, the loose ones' max, and
    the share of nodes whose actuated joints are all within RESERVE_LIMIT."""
    reserves, loose = np.abs(pieces["reserves"]), muscled.loose
    actuated = reserves[:, ~loose]
    return {"actuated %BW*ht p50/p95/max": [round(100 * float(np.percentile(actuated, rank)), 3)
                                            for rank in (50, 95, 100)],
            "loose %BW*ht max": round(100 * float(reserves[:, loose].max(initial=0.0)), 3),
            "frames within": round(float(np.mean(actuated.max(axis=1) <= RESERVE_LIMIT)), 4)}


def worst_reserve(pieces, muscled):
    """The worst actuated reserve over RESERVE_LIMIT."""
    return float(np.abs(pieces["reserves"][:, ~muscled.loose]).max(initial=0.0)) / RESERVE_LIMIT


def prepared(motion, physics, height):
    """Stage B's `Solve` on stage A's `motion` ((its `trial.Problem`, its control points, its `Weighed`)): the
    person's muscle paths on MuJoCo's C, its dynamics evaluated there too (`_dynamics.Exact`), the weights from a
    frame's fit cost at stage A's answer."""
    problem, control, weighed = motion
    model, muscles = physics.model, physics.muscles
    weight = float(np.sum(model.body_mass) * np.linalg.norm(model.opt.gravity))
    path = weighed.dynamics.path
    dynamics = Dynamics(path, weighed.dynamics.contact, weighed.dynamics.weight, weighed.dynamics.height,
                        Exact(model))
    weighed = Weighed(dynamics, weighed.multipliers, weighed.penalty, weighed.effort)
    muscled = Muscled(dynamics, Paths(model, muscles, path.coupled), Curves(model, muscles), weight * height,
                      model.actuator_dynprm[muscles, :3], path.states(control)[0])
    costs = problem.measured(control)[0]
    per_frame = float(np.sum(costs)) / len(costs)
    return Solve(problem, weighed, muscled, per_frame / RESERVE_LIMIT ** 2, MUSCLE_EFFORT * per_frame / len(muscles),
                 per_frame / EXCITATION_SCALE ** 2)


def solved(motion, physics, limits, height):
    """Stage B from stage A's `motion` ((its `trial.Problem`, its control points, its `Weighed`)): (every frame's
    angles [N, nq] in the fitting model's layout, every node's activations [N, muscles] in `physics.muscles`' order,
    the record). `physics` the person's, `limits` the residual's, `height` the centre's standing height."""
    began = time.perf_counter()
    problem, control, weighed = motion
    solve = prepared(motion, physics, height)
    muscled, weighed = solve.muscled, solve.weighed
    costs = problem.measured(control)[0]
    muscles = physics.muscles
    activations = np.full((len(costs), len(muscles)), ACTIVATION_START)
    control, activations, warm_kept, warm_tried = inner(solve, control, activations, move=False, most=WARM_ROUNDS)
    start_total, start = solve.measured(control, activations)
    warm_control, warm_activations = control, activations
    warm = {"reserves": reserve_record(start, muscled), "residual": residual_record(start["motion"][3][0], limits),
            "activations mean": round(float(activations.mean()), 4), "kept": warm_kept, "tried": warm_tried}
    rounds, worst_before = [], np.inf
    for _ in range(OUTER_MOST):
        control, activations, kept, tried = inner(solve, control, activations)
        _, pieces = solve.measured(control, activations)
        torques = pieces["motion"][3][0]
        force, moment = shares(torques)
        worst = float(np.max(np.maximum(force / limits[0], moment / limits[1])))
        reserve = worst_reserve(pieces, muscled)
        rounds.append({"penalty": float(f"{weighed.penalty:.4g}"), "reserve weight": float(f"{solve.reserve_weight:.4g}"),
                       "kept": kept, "tried": tried, "worst residual over its limit": round(worst, 4),
                       "worst reserve over its limit": round(reserve, 4)})
        if worst <= 1.0 and reserve <= 1.0:
            break
        weighed.multipliers = weighed.multipliers + weighed.penalty * torques[:, :6]
        if worst > PROGRESS * worst_before:
            weighed.penalty *= PENALTY_RAISE
        worst_before = worst
        if reserve > 1.0:  # the pelvis' rows and the excitations raised with the reserves, so they stay above them
            solve.reserve_weight *= RESERVE_RAISE
            solve.excitation *= RESERVE_RAISE
            weighed.penalty *= RESERVE_RAISE
    end_total, end = solve.measured(control, activations)
    warm_total = solve.measured(warm_control, warm_activations)[0]  # at the end's weights and multipliers
    values = _spline.nodes(control)
    record = {"method": METHOD, "loose joints": int(muscled.loose.sum()),
              "warm start (movement held)": warm, "rounds": rounds,
              "reserves after": reserve_record(end, muscled),
              "residual after": residual_record(end["motion"][3][0], limits),
              "activations mean": round(float(activations.mean()), 4),
              "excitation outside [0, 1] max": round(float(np.abs(end["over"]).max(initial=0.0)), 4),
              "off stage A deg": round(float(np.degrees(np.sqrt(np.mean(
                  (values[:, problem.layout.joints] - _spline.nodes(motion[1])[:, problem.layout.joints]) ** 2)))), 4),
              "weights": {"reserve": float(f"{solve.reserve_weight:.4g}"), "effort": float(f"{solve.effort:.4g}"),
                          "excitation": float(f"{solve.excitation:.4g}")},
              "cost": {"warm start": round(start_total, 6), "warm start at the end's weights": round(warm_total, 6),
                       "end": round(end_total, 6)}, "seconds": round(time.perf_counter() - began, 2)}
    qpos = problem.rows.projected(problem.layout.poses(values))
    record["landmark errors mm"] = landmark_errors(problem.rows.clip, qpos, problem.points)
    return qpos, activations, record
