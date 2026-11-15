"""MOTOR step 2 (#98), stage A: the whole trial's movement, contact and joint torques as one solve.

Warm started at step 1's spline (the fit's cost over the whole trial, `trial.solved_on`), the same unknowns take the
dynamics' rows (`_dynamics`): the pelvis' six residual rows (force over the weight, moment over weight x the centre's
standing height) held by an augmented Lagrangian (rows sqrt(rho / 2) (c + lambda / rho), lambda += rho c after each
inner solve, rho raised PENALTY_RAISE times when the worst frame did not fall to PROGRESS of the round before) until
every frame is within the stage's residual limits (2 %BW, 1 %BW*ht), and the joints' torques as an effort (rows
sqrt(w) tau / (weight x height)). Inner solve: Levenberg-Marquardt on the banded normal equations (`_spline.Banded`),
the dynamics' Jacobians taken again only at a kept step (a refused one costs the cost alone).

The penalty starts where a frame whose residual force stands at its limit weighs what a frame's fit costs at the warm
start (started where the warm start's whole residual weighed the fit's, a 100 %BW residual left it at 1e-2 and six
rounds ended 34 times over the limit, 2026-11-08), and the effort at the settings' effort share of the fit's cost:
scales, not weights fitted to anything (step 3 tunes the share and the contact's settings, `_dynamics.Settings`,
against force plates; never against reference angles). Before the first round the contact is settled at
the warm start (`Contact.settle`): no body starts inside its ground."""
import time

import numpy as np

from motor.trial import _spline, landmark_errors
from motor.trial._dynamics import DEFAULT, Contact, Dynamics, Path, ground_forces

OUTER_MOST = 8  # augmented Lagrangian rounds
INNER_MOST = 12  # Levenberg-Marquardt rounds tried a round
PENALTY_RAISE = 10.0
PROGRESS = 0.25  # the worst frame's residual share a round must fall to, else the penalty is raised
DAMPING_FIRST, DAMPING_RAISE, DAMPING_MOST = 1e-4, 10.0, 1e6
RELATIVE_DONE = 1e-6
METHOD = ("MOTOR step 2 (#98), stage A: step 1's whole-trial spline solve with every node's inverse dynamics on the "
          "card (MuJoCo Warp), a smooth Hunt-Crossley contact with smoothed Coulomb friction of the skin on the seen "
          "floor settled at the warm start (its settings as MOTOR step 3's tuning kept them), the "
          "pelvis' residual by an augmented Lagrangian to the stage's limits and the joints' torques as an effort, "
          "Levenberg-Marquardt on the banded normal equations")


class Weighed:
    """The dynamics' rows as the solve weighs them: `multipliers` [N, 6], `penalty` rho, `effort` w."""

    def __init__(self, dynamics, multipliers, penalty, effort):
        self.dynamics, self.multipliers, self.penalty, self.effort = dynamics, multipliers, penalty, effort

    def residuals(self, torques):
        """Each node's rows [N, 6 + joints]: the residual's sqrt(rho / 2) (c + lambda / rho), the effort's."""
        return np.concatenate([np.sqrt(self.penalty / 2.0) * (torques[:, :6] + self.multipliers / self.penalty),
                               np.sqrt(self.effort) * torques[:, 6:]], axis=1)

    def cost(self, torques):
        """The rows' cost (their sum of squares, the multipliers' constant left in)."""
        return float(np.sum(self.residuals(torques) ** 2))

    def rows(self, control, evaluated):
        """Each node's (Jacobians on its three control points [N, 3, rows, m], residuals [N, rows])."""
        torques = evaluated[0]
        place, speed, pace = self.dynamics.jacobians(control, *evaluated[1:])
        weights = np.concatenate([np.full(6, np.sqrt(self.penalty / 2.0)),
                                  np.full(torques.shape[1] - 6, np.sqrt(self.effort))])
        place, speed, pace = (kind * weights[None, :, None] for kind in (place, speed, pace))
        return _spline.spread(place, speed, pace, self.dynamics.path.step), self.residuals(torques)


def measured(problem, weighed, control):
    """At `control`: (the whole cost, the frames' H and g on the unknowns, the dynamics' evaluation)."""
    costs, hessians, gradients, _ = problem.measured(control)
    evaluated = weighed.dynamics.evaluated(control)
    return problem.total(costs, control) + weighed.cost(evaluated[0]), hessians, gradients, evaluated


def inner(problem, weighed, control):
    """One Levenberg-Marquardt solve of the whole cost: (control points, rounds kept, rounds tried, Jacobian
    seconds)."""
    total, hessians, gradients, evaluated = measured(problem, weighed, control)
    began = time.perf_counter()
    rows = weighed.rows(control, evaluated)
    jacobian_seconds = time.perf_counter() - began
    damping, kept, tried = DAMPING_FIRST, 0, 0
    while tried < INNER_MOST and damping <= DAMPING_MOST:
        trial = problem.stepped(control, hessians, gradients, damping, rows)
        tried += 1
        trial_total, trial_h, trial_g, trial_evaluated = measured(problem, weighed, trial)
        if trial_total < total:
            lowered = total - trial_total
            control, total, hessians, gradients = trial, trial_total, trial_h, trial_g
            kept, damping = kept + 1, max(damping / DAMPING_RAISE, 1e-12)
            if lowered < RELATIVE_DONE * total:
                break
            began = time.perf_counter()
            rows = weighed.rows(control, trial_evaluated)
            jacobian_seconds += time.perf_counter() - began
        else:
            damping *= DAMPING_RAISE
    return control, kept, tried, jacobian_seconds


def shares(torques):
    """Each node's residual force (over the weight) and moment (over weight x height), from the scaled torques."""
    return np.linalg.norm(torques[:, :3], axis=1), np.linalg.norm(torques[:, 3:6], axis=1)


def residual_record(torques, limits):
    """The residual's measures: force %BW and moment %BW*ht (median, p95, most) and the share of nodes within both."""
    force, moment = shares(torques)
    measure = {name: [round(100 * float(np.percentile(values, rank)), 3) for rank in (50, 95, 100)]
               for name, values in (("force %BW p50/p95/max", force), ("moment %BW*ht p50/p95/max", moment))}
    measure["frames within"] = round(float(np.mean((force <= limits[0]) & (moment <= limits[1]))), 4)
    measure["joint torque rms %BW*ht"] = round(100 * float(np.sqrt(np.mean(torques[:, 6:] ** 2))), 4)
    return measure


def solved(motion, physics, source, foot_points, sides, limits, height, settings=DEFAULT):
    """Stage A from step 1's `motion` ((its `trial.Problem`, its control points)): (every frame's angles [N, nq] in
    the fitting model's layout, the record, its motion: the problem, its control points and its `Weighed`, stage
    B's warm start). `physics` the person's (`muscles.Physics`, its skin credited), `source`
    the fitting model, `foot_points` a foot's skin points (the contact's spread), `sides` each side's foot bodies
    (the record's ground force each), `limits` the residual's (force over the weight, moment over weight x
    `height`, the centre's standing height), `settings` the contact's and the effort's (`_dynamics.Settings`)."""
    began = time.perf_counter()
    problem, control = motion
    model = physics.model
    weight = float(np.sum(model.body_mass) * np.linalg.norm(model.opt.gravity))
    path = Path(problem.layout, source, physics, problem.step)
    dynamics = Dynamics(path, Contact(physics, foot_points, settings), weight, height)
    costs = problem.measured(control)[0]
    dynamics.contact.settle(dynamics.evaluated(control)[3])
    start = dynamics.evaluated(control)
    fitted = float(np.sum(costs))
    penalty = 2.0 * fitted / (len(costs) * limits[0] ** 2)
    effort = settings.effort_share * fitted / max(float(np.sum(start[0][:, 6:] ** 2)), 1e-12)
    weighed = Weighed(dynamics, np.zeros((len(start[0]), 6)), penalty, effort)
    first = control
    rounds, worst_before = [], np.inf
    for _ in range(OUTER_MOST):
        control, kept, tried, jacobian_seconds = inner(problem, weighed, control)
        torques = dynamics.evaluated(control)[0]
        force, moment = shares(torques)
        worst = float(np.max(np.maximum(force / limits[0], moment / limits[1])))
        rounds.append({"penalty": float(f"{weighed.penalty:.4g}"), "kept": kept, "tried": tried,
                       "worst over its limit": round(worst, 4), "jacobian seconds": round(jacobian_seconds, 2)})
        if worst <= 1.0:
            break
        weighed.multipliers = weighed.multipliers + weighed.penalty * torques[:, :6]
        if worst > PROGRESS * worst_before:
            weighed.penalty *= PENALTY_RAISE
        worst_before = worst
    end = dynamics.evaluated(control)
    values, before = _spline.nodes(control), _spline.nodes(first)
    joints = problem.layout.joints
    whole, each = ground_forces(end[1], sides, physics.skin.owners)
    record = {"method": METHOD, "rounds": rounds,
              "residual before": residual_record(start[0], limits), "residual after": residual_record(end[0], limits),
              "frames cost": {"step 1": round(fitted, 6), "stage A": round(float(np.sum(problem.measured(control)[0])),
                                                                            6)},
              "off step 1 deg": round(float(np.degrees(np.sqrt(np.mean((values[:, joints] - before[:, joints]) ** 2)))),
                                      4),
              "ground force N": np.round(whole, 1).tolist(), "ground force by side N": np.round(each, 1).tolist(),
              "settings": {**settings.record(), "effort": float(f"{effort:.4g}"),
                           "first penalty": float(f"{penalty:.4g}")},
              "settled": {"frames": int(np.any(dynamics.contact.sunk < 0, axis=1).sum()),
                          "deepest m": round(float(-dynamics.contact.sunk.min()), 4)},
              "seconds": round(time.perf_counter() - began, 2)}
    qpos = problem.rows.projected(problem.layout.poses(values))
    record["landmark errors mm"] = landmark_errors(problem.rows.clip, qpos, problem.points)
    return qpos, record, (problem, control, weighed)
