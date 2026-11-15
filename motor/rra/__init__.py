"""OpenSim's residual reduction (RRA: Delp et al. 2007, doi:10.1109/TBME.2007.901024; opensim-core's RRATool,
Apache-2.0), in our own stack: RRA_ROUNDS times, the body's mass changed by the mean upward residual and the torso's
centre of mass moved under the mean residual moment (`reweighed`, RRATool::adjustCOMToReduceResiduals), then the
pelvis' and trunk's path moved so the ground's pushes explain the movement, each frame's move capped, the feet kept
where they stood by the legs (`reduced`, `Footing`: mink, as the fit); the result kept within the clip's `follows` limit
(`within_follows`) and the per-frame cap as measured (`within_frames`). Nothing is applied to the body; how far it
moved is the body's `adjusted` record (`records.adjustment_record`, from `turned_off` and `off_clip`)."""
import hashlib

import mujoco
import numpy as np

from motor.model import body_index
from motor.rra._cap import (SKIN_SLACK, cap_rows, held_share, off_clip, skin_excess, smoothed_off, turned_off,  # noqa: F401
                      within_follows, within_frames)
from motor.rra._footed import Footing  # noqa: F401
from motor.rra._linear import adjusted_leads, frame_states, linearised, moved, residuals, sensitivities, trunk_moments  # noqa: F401
from motor.batched import within_balls_card
from motor.solvers import filter_matrix, within_balls

# The residual reduction's rounds (RRA is run again on its own output), the weight of a move the size of its cap
# against a residual the size of its limit (RRA's tracking), RRATool's 0.1 m limit on the torso's centre of mass, and
# the halvings of a step that does not lower the cost, and the most the body's mass may change from the mass it was
# made with over every round together (#70, 2026-10-29: unbounded, a floating body lost 98 to 99.9% of its mass;
# OpenSim's RRA only recommends its change and leaves the model to its user, so a change past this bound is recorded
# as recommended, never made).
RRA_ROUNDS = 3
TRACK = 0.01
COM_MOST = 0.1
RRA_HALVINGS = 3
MASS_MOST = 0.02


def reweighed(physics, path, residuals, home, made):
    """RRA's model step (opensim-core RRATool::adjustCOMToReduceResiduals): the whole body's mass changed by the mean
    upward residual over gravity, spread over the bodies by their masses (their inertias with them), at most
    MASS_MOST from its masses and inertias as made (`made`: (masses, inertias)) over every round, and the torso's
    centre of mass moved under the mean horizontal residual moment across the torso's mean up (`level`), at most RRA's
    0.1 m from where it was made (`home`) over every round; returns the change from the made mass as made and as RRA recommends it (shares)."""
    model, data = physics.model, physics.data
    gravity = -model.opt.gravity[2]
    recommended = (float(model.body_mass.sum()) - float(np.mean(residuals[:, 2])) / gravity) / float(made[0].sum()) - 1
    share = float(np.clip(recommended, -MASS_MOST, MASS_MOST))
    model.body_mass[:] = made[0] * (1 + share)
    model.body_inertia[:] = made[1] * (1 + share)
    torso = body_index(model, "torso")
    weight = model.body_mass[torso] * gravity
    rows, wanted, ups = [], [], []
    for qpos, residual in zip(path, residuals):  # the torso's weight moved by d gives the pelvis (w d_y, -w d_x)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        turn = data.xmat[torso].reshape(3, 3)
        rows += [weight * turn[1], -weight * turn[0]]
        wanted += [-residual[3], -residual[4]]
        ups.append(turn[2].copy())  # up in the torso's frame
    across = level(np.mean(ups, axis=0))
    shift = across.T @ np.linalg.lstsq(np.array(rows) @ across.T, np.array(wanted), rcond=None)[0]
    model.body_ipos[torso] = home + np.clip(model.body_ipos[torso] + shift - home, -COM_MOST, COM_MOST)
    return share, recommended


def level(up):
    """The two directions across `up` (2 x 3, orthonormal): the torso's centre of mass moves only across its own up,
    as RRATool moves it (its body's x and z, never y): a weight's moment about the pelvis does not see a move along its
    own line, so a move up or down is read off the frames' small tilts alone (#98, 2026-11-09: unbounded there, nearly
    every video run took the torso's centre of mass to the 10 cm cap)."""
    return np.linalg.svd(np.asarray(up, dtype=float)[None, :])[2][1:]


def residuals_kept(physics, states, kept):
    """`residuals` of `states` at the body's masses, inertias and torso centre of mass as they stand, worked out once
    and kept in `kept` ({key: residuals}): `reduced` asks them of one path again (a step's cost, then the next round's
    model step on the step taken, at the masses its cost was measured with)."""
    model = physics.model
    key = hashlib.sha256(b"".join(np.ascontiguousarray(part).tobytes() for state in states for part in state) +
                         model.body_mass.tobytes() + model.body_inertia.tobytes() + model.body_ipos.tobytes())
    key = key.hexdigest()
    if key not in kept:
        kept[key] = residuals(physics, states)
    return kept[key]


def reduced(physics, path, rate, limits, most, land, fitted, riders):
    """OpenSim's residual reduction (RRA: Delp et al. 2007, IEEE TBME 54(11):1940-50, doi:10.1109/TBME.2007.901024;
    opensim-core's RRATool, Apache-2.0) on the fitted path, in our own stack: RRA_ROUNDS times, the model step
    (`reweighed`) on the mean residual, then the pelvis' and trunk's path (`adjusted_leads`) moved so the ground
    explains the movement: one least squares over every frame of the residual (force over `limits[0]`, moment over
    `limits[1]`), the lumbar's moments beyond the fitted path's (over `limits[1]`: `demands`) and the move (over its
    cap, TRACK its weight: RRA's tracking), the move low-passed by construction (`filter_matrix`), its feet kept where
    they stood by the legs (`land`: `Footing.footed`), and bounded inside the solve by the owner's per-frame cap
    (`cap_rows`: every segment's turn within `most[1]` and the pelvis within `most[0]` of the fitted path, all rounds
    together; `within_balls`), taken only when it lowers that cost as measured (halved up to RRA_HALVINGS times, else
    the rounds end; a path's residuals at the masses as they stand worked out once, `residuals_kept`, and the step's
    cost before it the one its solve read). Nothing is applied to the body: only its path and masses change. Returns
    the path, the move (frames x adjusted) and the record."""
    model = physics.model
    adjusted = adjusted_leads(physics)
    count, width = len(path), len(adjusted)
    scale = np.concatenate([np.full(3, limits[0]), np.full(3, limits[1])])
    scale = np.tile(np.concatenate([scale, np.full(width - 6, limits[1])]), count)
    track = np.sqrt(TRACK) / np.tile([most[0]] * 3 + [most[1]] * (width - 3), count)
    smooth = np.kron(filter_matrix(count, rate), np.eye(width))
    total, current, recommended = np.zeros(count * width), path, []
    home = model.body_ipos[body_index(model, "torso")].copy()
    made = (model.body_mass.copy(), model.body_inertia.copy())

    kept = {}  # each path's residuals at the masses as they stand (`residuals_kept`)

    def asked(states):  # the residuals, and the trunk's moments beyond the fitted path's: a residual moved into the
        # lumbar is a reserve, counted as one (RRA's own objective weighs the joints' actuators beside the residuals)
        return np.concatenate([residuals_kept(physics, states, kept),
                               trunk_moments(physics, states, adjusted[6:]) - fitted_trunk], axis=1).reshape(-1) / scale

    def cost(moves, states):
        return costed(moves, asked(states))

    def costed(moves, left):
        return float(np.sum(left ** 2) + np.sum((track * moves) ** 2))

    for _ in range(RRA_ROUNDS):
        states = frame_states(physics, current, rate)
        mass, wanted = reweighed(physics, current, residuals_kept(physics, states, kept), home, made)
        recommended.append(wanted)
        fitted_trunk = trunk_moments(physics, frame_states(physics, path, rate), adjusted[6:])
        states = frame_states(physics, current, rate)
        left = asked(states)
        system = linearised(physics, states, adjusted, rate, scale) @ smooth
        rows, values = cap_rows(physics, path, total.reshape(count, width), land, fitted, riders, most)
        rows = np.einsum("fbcw,fwk->fbck", rows, smooth.reshape(count, width, -1)).reshape(-1, smooth.shape[1])
        solved, _ = (within_balls if physics.card is None else within_balls_card)(np.concatenate([system, track[:, None] * smooth]),
                                 np.concatenate([-left, -track * total]), rows, values.reshape(-1))
        step = smooth @ solved
        best = costed(total, left)
        for _ in range(RRA_HALVINGS + 1):
            walked = land(moved(physics, path, (total + step).reshape(count, width), adjusted))
            if cost(total + step, frame_states(physics, walked, rate)) < best:
                total, current = total + step, walked
                break
            step /= 2
        else:
            break
    return current, total.reshape(count, width), {"mass": mass, "mass recommended": recommended,
                                                   "torso": model.body_ipos[body_index(model, "torso")] - home}
