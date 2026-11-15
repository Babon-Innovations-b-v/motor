"""motor: the whole trial's spline solve (MOTOR step 1) on the card's toy (`fit/_card/tests/card_toy.py`: a free root,
three ranged hinges, a polyval follower, a frozen hinge) over a 40-frame clip whose places are 2 mm off and one hinge
driven past its range: the spline's nodes are the values it was fitted to and its banded step the dense one; the
frames' cost carried onto the unknowns (the root's place and body-frame rotation vector included) has the gradient of
the frames' cost by central differences; and the solve, warm started from level 3's per-frame fit, lowers the whole
cost (frames and jerk), shakes less than the fit, and keeps every hinge's spline inside its range (the
projection never moves a joint off it). It runs in Newton's own
Python on a card (MOTOR's own Python, the pins in pyproject.toml).

Breaks (each applied once with the env linked into a sandbox home, seen red, reverted), 2026-11-08:
- `Layout.carried`'s `twist[:3, 3:] = skew(qpos[frame, :3]) @ spun` dropped: the root's turn columns lose the
  place's swing, and the differences' check is red.
- `Layout.carried`'s `right_jacobian(...)` made `np.eye(3)`: red at the rotated pose's root turn columns.
- `_spline.Banded.add_nodes`' `NODE[first] * NODE[second]` made `NODE[first]`: the banded system is no longer
  positive definite (solveh_banded raises).
- `Layout`'s bounds made unbounded (`self.lower[self.joints] = -np.inf` and the upper `np.inf` after the root's):
  h1's spline leaves its range and the projection clips it off the spline.

Step 2 (stage A) on the same toy, built as a person is (`model.built`) with a skin grid under its root on a flat
floor: the contact's force derivatives against central differences, rows spread on a node's control points against
the dense normal equations, the nodes' speeds against their poses' central differences, a slow node's dynamics
Jacobian within 10% of its torques' central differences, and the solve taking the worst frame's residual to under
half of step 1's. Breaks, 2026-11-08, each seen red and reverted:
- `_dynamics.push`'s `deep_over` sign flipped: the contact's derivative and the node's Jacobian are red.
- `Dynamics.jacobians`' mass matrix dropped (`by_pace` 0): the node's Jacobian is 1.0 of its differences off.
- `_spline.Banded.add_rows`' blocks J_b^T J_a made J_a^T J_b: the spread rows' step is red.
- `_stage_a.Weighed.residuals` negated: no step is kept and the residual stays at step 1's.

Step 3's pieces on the same toy: the couplings and the place nudge on all rows at once against `model.Couplings` and
MuJoCo's `mj_integratePos` row by row, the warm start lowered 1 cm settled out of its floor, a learned landmark
weight keeping the markers' goal, a settings name not known refused, every landmark's miss recorded. Breaks,
2026-11-08, each seen red and reverted:
- `_dynamics.integrated`'s turn `q * exp(w)` made `exp(w) * q`: the nudge 3.3e-2 off mj_integratePos.
- `_dynamics.Coupled.maps`' `moved ** (power - 1)` made `moved ** power`: not the couplings' map.
- `_dynamics.Contact.settle`'s sunk made 0: the lowered skin stays 2.8 cm inside its floor.
- `trial.retrusted`'s `costs ** 2 * weights` made `costs ** 2`: the weight 4 changes nothing.

Step 4 (stage B) on the same toy with five muscles (`muscled`): MuJoCo's muscle curves in Warp against
`mju_muscleGain` / `mju_muscleBias` and their tape derivatives against differences, the muscles' torques against
MuJoCo's own actuator forces, the excitations against `mju_muscleDynamics`, the speed Jacobian of the muscle torques,
the activations' elimination against the dense normal equations, the cost's slope along the activations' step, the
reserves' stop weights' slope, the pelvis' rows along a short step within 0.3 of their linear model, and the solve
lowering its cost from its warm start, its worst reserve and residual no higher. Breaks, 2026-11-08, each seen red
and reverted:
- `_muscles._curves`' force-velocity `(speed + one) * (speed + one)` made `(speed + one)`: the gain and the torques
  are off MuJoCo's.
- `_stage_b.Muscled.muscle_part`'s forces without the passive bias: the torques 0.37 off MuJoCo's.
- `_muscles.excitations`' `- constant / step` made `+`: the first node's derivative 3.5 off.
- `_stage_b.Solve.eliminated`'s Woodbury `first - ...` made `first + ...`: the step's solve raises.
- `_dynamics.Dynamics.jacobians` on the damping alone for stage B: the pelvis' rows 1.26 off their model, and the
  worst residual over stage A's.
- `_stage_b.stop_shares`' slope's sign flipped: the stop weights' slope off its differences.

Run: .venv/bin/python -m pytest motor/trial/
"""
import pathlib
import sys

INSIDE = pathlib.Path(__file__).resolve().parents[1]
TOYS = INSIDE / "fit/_card/tests"
FRAMES, RATE = 40, 30.0


def toy_targets(mjm):
    """The toy's 40-frame clip: (true poses, turns [N, 1, 3, 3], places [N, 4, 3] 2 mm off); h1 driven to 1.5."""
    import mujoco
    import numpy as np
    from card_toy import PLACE_SITES
    from scipy.spatial.transform import Rotation
    data, generator = mujoco.MjData(mjm), np.random.default_rng(11)
    poses, turns, places = [], [], []
    for moment in np.linspace(0.0, 1.0, FRAMES):
        qpos = mjm.qpos0.copy()
        qpos[:3] = (0.1 * moment, 0.0, 0.05 * moment)
        qpos[3:7] = np.roll(Rotation.from_rotvec([0.3 * moment, -0.2, 0.6 * moment]).as_quat(), 1)
        qpos[7:10] = (0.8 * np.sin(2.0 * moment), 1.5 * moment, -0.6 * moment)
        qpos[10] = 0.5 * qpos[8] + 0.2 * qpos[8] ** 2
        data.qpos[:] = qpos
        mujoco.mj_kinematics(mjm, data)
        poses.append(qpos)
        turns.append(data.site("t").xmat.reshape(3, 3).copy())
        places.append([data.site(name).xpos + generator.normal(0.0, 0.002, 3) for name in PLACE_SITES])
    return np.array(poses), np.array(turns)[:, None], np.array(places)


def spline_faults():
    """The spline's own maths: nodes of `fitted` against their values, the banded step against a dense solve."""
    import numpy as np
    from motor.trial import _spline
    faults, generator = [], np.random.default_rng(3)
    values = generator.normal(size=(12, 3))
    if not np.allclose(_spline.nodes(_spline.fitted(values, 1 / RATE)), values, atol=1e-8):
        faults.append("the fitted control points' nodes are not their values")
    system = _spline.Banded(14, 3)
    blocks = generator.normal(size=(12, 3, 3))
    system.add_nodes(np.einsum("nij,nkj->nik", blocks, blocks), generator.normal(size=(12, 3)))
    control = generator.normal(size=(14, 3))
    weights = np.array([1e-6, 2e-6, 3e-6])
    system.add_jerk(control, 1 / RATE, weights)
    dense = np.zeros((42, 42))
    for first in range(12):
        spread = np.zeros((3, 42))
        for index, weight in enumerate(_spline.NODE):
            spread[:, (first + index) * 3:(first + index + 1) * 3] = weight * np.eye(3)
        dense += spread.T @ (blocks[first] @ blocks[first].T) @ spread
    for span in range(11):
        spread = np.zeros((3, 42))
        for index, weight in enumerate(_spline.JERK):
            spread[:, (span + index) * 3:(span + index + 1) * 3] = weight * RATE ** 3 * np.eye(3)
        dense += spread.T @ np.diag(weights / RATE) @ spread
    wanted = np.linalg.solve(dense, -system.gradient.reshape(-1))
    if not np.allclose(system.step(0.0, np.zeros((14, 3), dtype=bool)).reshape(-1), wanted, rtol=1e-6, atol=1e-9):
        faults.append("the banded step is not the dense normal equations' step")
    return faults


def carried_faults(problem, poses):
    """The carried gradient (2 g) of the frames' cost against its central differences along every unknown of the
    middle frame, its root turned 0.54 rad off the low-passed turn the rotation vector stands on."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    layout, faults, frame = problem.layout, [], FRAMES // 2
    poses = np.array(poses)
    turned = Rotation.from_quat(poses[frame, [4, 5, 6, 3]]) * Rotation.from_rotvec([0.4, -0.3, 0.2])
    poses[frame, 3:7] = turned.as_quat()[[3, 0, 1, 2]]
    qpos = problem.rows.projected(poses)
    hessians, gradients, _, twists = problem.rows.blocks(qpos)
    values = layout.values(qpos)
    _, gradient = layout.carried(hessians, gradients, twists, values, qpos)
    for column in range(values.shape[1]):
        moved = [values.copy(), values.copy()]
        moved[0][frame, column] += 1e-4
        moved[1][frame, column] -= 1e-4
        ahead, behind = (float(problem.frames_cost(path)[frame]) for path in moved)
        difference = (ahead - behind) / 2e-4
        if abs(difference - 2 * gradient[frame, column]) > 2e-3 * max(1.0, abs(difference)):
            faults.append(f"unknown {column}: gradient {2 * gradient[frame, column]:.6f}, "
                          f"differences {difference:.6f}")
    return faults


def the_trial_solve_holds():
    """Every fault of the spline, the carried gradient and the toy's whole-trial solve."""
    import mujoco
    import numpy as np
    sys.path.insert(0, str(TOYS))
    from motor import trial
    from card_toy import TOY, toy_clip_card
    from motor.trial._rows import Rows
    faults = spline_faults()
    mjm = mujoco.MjModel.from_xml_string(named(TOY))
    clip = toy_clip_card(mjm)
    poses, turns, places = toy_targets(mjm)
    fitted = clip.angles(turns, places, poses, exact=True)[0].astype(float)
    layout = trial.Layout(clip.reduction, fitted, trial.smooth_turns(fitted[:, 3:7], RATE))
    problem = trial.Problem(Rows(clip, turns, places, None, fitted), layout, 1 / RATE)
    faults += carried_faults(problem, poses)
    angles, record, motion = trial.solved_on(clip, turns, places, None, fitted, RATE)
    start = trial._spline.fitted(layout.values(problem.rows.projected(fitted)), 1 / RATE)
    problem.weights = trial.jerk_weights(problem.measured(start)[1], 1 / RATE)
    if not record["whole cost"] < problem.total(problem.measured(start)[0], start):
        faults.append(f"the whole cost {record['whole cost']} is not under the warm start's")
    if not record["jerk deg/s3"]["motor"] < record["jerk deg/s3"]["fit"]:
        faults.append(f"jerk {record['jerk deg/s3']}: the solve shakes no less than the fit")
    hinges = [mjm.jnt_qposadr[mjm.joint(name).id] for name in ("h0", "h1", "h2")]
    if np.abs(angles[:, hinges]).max() > 1.2 + 1e-6:
        faults.append(f"a hinge at {np.abs(angles[:, hinges]).max():.4f} rad, past its 1.2 range")
    if record["off its spline deg"] > 1e-4:
        faults.append(f"the projection moved a joint {record['off its spline deg']} deg off the spline")
    if record["rounds kept"] < 1:
        faults.append(f"no round kept: {record}")
    return faults + push_faults() + rows_faults() + stage_a_faults(mjm, motion)


def push_faults():
    """The contact's force derivatives (on a point's height and speed) against central differences."""
    import numpy as np
    from motor.trial._dynamics import push
    generator, faults = np.random.default_rng(7), []
    over = np.array([-0.004, -0.001, 0.0, 0.0008, 0.003, 0.012])
    speed = generator.normal(0.0, 0.3, (len(over), 3))
    up = np.array([0.05, -0.03, 1.0]) / np.linalg.norm([0.05, -0.03, 1.0]) * np.ones((len(over), 1))
    force, by_over, by_speed = push(over, speed, up, 1e4)
    if not (np.einsum("pi,pi->p", force, up) >= 0).all():
        faults.append(f"a contact pulls: normal forces {np.einsum('pi,pi->p', force, up)}")
    step = 1e-7
    differences = (push(over + step, speed, up, 1e4)[0] - push(over - step, speed, up, 1e4)[0]) / (2 * step)
    if not np.allclose(differences, by_over, rtol=1e-4, atol=1e-3):
        faults.append(f"d force / d height {by_over.round(2).tolist()}, differences {differences.round(2).tolist()}")
    for axis in range(3):
        moved = np.zeros(3)
        moved[axis] = 1e-6
        differences = (push(over, speed + moved, up, 1e4)[0] - push(over, speed - moved, up, 1e4)[0]) / 2e-6
        if not np.allclose(differences, by_speed[:, :, axis], rtol=1e-4, atol=1e-3):
            faults.append(f"d force / d speed {axis}: {by_speed[:, :, axis].round(3).tolist()}, differences "
                          f"{differences.round(3).tolist()}")
    return faults


def rows_faults():
    """Rows spread over a node's control points (`_spline.spread`, `Banded.add_rows`) against the dense J^T J and
    J^T r of the same rows written on the control points."""
    import numpy as np
    from motor.trial import _spline
    generator, step, count, size, rows = np.random.default_rng(9), 1 / RATE, 5, 2, 3
    place, speed, pace = (generator.normal(size=(count, rows, size)) for _ in range(3))
    residuals = generator.normal(size=(count, rows))
    system = _spline.Banded(count + 2, size)
    system.add_rows(_spline.spread(place, speed, pace, step), residuals)
    dense, gradient = np.zeros(((count + 2) * size,) * 2), np.zeros((count + 2) * size)
    for node in range(count):
        jacobian = np.zeros((rows, (count + 2) * size))
        for point in range(3):
            jacobian[:, (node + point) * size:(node + point + 1) * size] = (
                _spline.NODE[point] * place[node] + _spline.SPEED[point] / step * speed[node]
                + _spline.PACE[point] / step ** 2 * pace[node])
        dense += jacobian.T @ jacobian
        gradient += jacobian.T @ residuals[node]
    wanted = np.linalg.solve(dense + np.eye(len(dense)), -gradient)
    system.blocks[:, 0] += np.eye(size)
    found = system.step(0.0, np.zeros((count + 2, size), dtype=bool)).reshape(-1)
    if not np.allclose(found, wanted, rtol=1e-8, atol=1e-10):
        return ["the spread rows' banded step is not the dense rows' step"]
    return []


def named(toy):
    """The toy with its root's free joint named, as every joint of a person's file is (`model.carried` matches by
    name)."""
    return toy.replace("<freejoint/>", '<freejoint name="root"/>')


def muscled(toy, mjm):
    """The toy with five muscles (MuJoCo's own muscle model as a person's file writes it, `general` actuators): a
    pair across h0, a pair across h1 and one across both, each laid over the lengths the toy's clip gives it (2 cm
    either side)."""
    import mujoco
    import numpy as np
    sites = {'<freejoint name="root"/>': [("rp", "0 0.08 0.15"), ("rm", "0 -0.08 0.15")],
             '<joint name="h0" axis="1 0 0" range="-1.2 1.2"/>': [("ap", "0 0.08 0.15"), ("am", "0 -0.08 0.15"),
                                                                    ("axp", "0.08 0 0.15"), ("axm", "-0.08 0 0.15")],
             '<joint name="h1" axis="0 1 0" range="-1.2 1.2"/>': [("bxp", "0.08 0 0.15"), ("bxm", "-0.08 0 0.15")]}
    text = named(toy)
    for after, placed in sites.items():
        text = text.replace(after, after + "".join(f'<site name="{name}" pos="{pos}"/>' for name, pos in placed))
    paths = [("rp", "ap"), ("rm", "am"), ("axp", "bxp"), ("axm", "bxm"), ("rm", "am", "axm", "bxm")]
    tendons = "".join(f'<spatial name="t{number}">' + "".join(f'<site site="{site}"/>' for site in path) + "</spatial>"
                      for number, path in enumerate(paths))
    curves = "0.75 1.05 1000 1 0.5 1.6 10 1.3 1.4"
    actuator = ('<general name="m{0}" tendon="t{0}" dyntype="muscle" gaintype="muscle" biastype="muscle" '
                f'dynprm="0.01 0.04 0" ctrlrange="0 1" ctrllimited="true" gainprm="{curves}" biasprm="{curves}" '
                'lengthrange="{1}"/>')
    probe = mujoco.MjModel.from_xml_string(text.replace("</worldbody>", f"</worldbody><tendon>{tendons}</tendon>"))
    data, seen = mujoco.MjData(probe), []
    for qpos in toy_targets(mjm)[0]:
        data.qpos[:] = qpos
        mujoco.mj_fwdPosition(probe, data)
        seen.append(data.ten_length.copy())
    seen = np.array(seen)
    actuators = "".join(actuator.format(number, f"{seen[:, number].min() - 0.02:.4f} {seen[:, number].max() + 0.02:.4f}")
                        for number in range(len(paths)))
    return text.replace("</worldbody>", f"</worldbody><tendon>{tendons}</tendon><actuator>{actuators}</actuator>")


def toy_physics(mjm, folder):
    """The muscled toy (`muscled`) built as the run builds a person (`model.built`, on the card), a 5 x 5 skin grid
    riding its root 5 cm under it, a flat floor 5.2 cm under the root's first place: (the physics, the fitting model's
    root body)."""
    import numpy as np
    from motor.contact import Skin
    from motor.model import built
    from motor.muscles import Physics
    from card_toy import TOY
    path = pathlib.Path(folder) / "toy.xml"
    path.write_text(muscled(TOY, mjm))
    model = built(path, 9.81)
    grid = np.array([[x, y, -0.05] for x in np.linspace(-0.1, 0.1, 5) for y in np.linspace(-0.1, 0.1, 5)])
    root = int(model.jnt_bodyid[0])
    skin = Skin(grid, np.ones((len(grid), 1)), np.eye(4)[None], [root], np.eye(4)[None])
    ground = np.array([[[-5.0, -0.052, -5.0], [5.0, -0.052, -5.0], [0.0, -0.052, 5.0]]])
    physics = Physics(model, ground, skin)
    return physics, root


def stage_a_faults(mjm, motion):
    """Stage A on the toy (`_stage_a`): the nodes' speeds against central differences of their poses, the dynamics'
    Jacobians on a node's control points against central differences of its torques (a slow node: the speed's
    product terms the Gauss-Newton model leaves out stay small), and the solve lowering the worst frame's residual
    to under half of step 1's."""
    import tempfile
    import numpy as np
    from motor import trial
    from motor.trial import _spline
    from motor.trial._dynamics import Contact, Dynamics, Path
    faults = []
    problem, control = motion
    with tempfile.TemporaryDirectory() as folder:
        physics, root = toy_physics(mjm, folder)
        model, step = physics.model, problem.step
        path = Path(problem.layout, mjm, physics, step)
        weight = float(np.sum(model.body_mass) * 9.81)
        dynamics = Dynamics(path, Contact(physics, len(physics.skin.points)), weight, 0.3)
        qpos, qvel, _ = path.states(control)
        central = np.zeros_like(qvel[1:-1])
        for frame in range(1, len(qpos) - 1):
            mujoco_difference(model, central[frame - 1], 2 * step, qpos[frame - 1], qpos[frame + 1])
        miss = np.abs(central - qvel[1:-1]).max() / max(np.abs(qvel).max(), 1e-9)
        if miss > 0.06:  # the central differences themselves miss by the step squared
            faults.append(f"the nodes' speeds {miss:.3f} of the largest off their poses' central differences")
        evaluated = dynamics.evaluated(control)
        jacobian = _spline.spread(*dynamics.jacobians(control, *evaluated[1:]), step)
        node = int(np.argmin(np.abs(qvel[2:-2]).max(axis=1))) + 2
        differences = np.zeros_like(jacobian[node])
        for point in range(3):
            for column in range(control.shape[1]):
                moved = [control.copy(), control.copy()]
                moved[0][node + point, column] += 1e-4
                moved[1][node + point, column] -= 1e-4
                ahead, behind = (dynamics.evaluated(copy)[0][node] for copy in moved)
                differences[point, :, column] = (ahead - behind) / 2e-4
        miss = np.linalg.norm(differences - jacobian[node]) / max(np.linalg.norm(differences), 1e-9)
        if miss > 0.1:
            faults.append(f"node {node}'s dynamics Jacobian {miss:.3f} of its central differences off")
        angles, record, stage_a_motion = trial.dynamics_solved(motion, physics, mjm, len(physics.skin.points), [[root]],
                                                      (0.02, 0.01), 0.3)
        before, after = record["residual before"], record["residual after"]
        worst_before = max(before["force %BW p50/p95/max"][2] / 2, before["moment %BW*ht p50/p95/max"][2])
        worst_after = max(after["force %BW p50/p95/max"][2] / 2, after["moment %BW*ht p50/p95/max"][2])
        if not worst_after < 0.5 * worst_before:
            faults.append(f"stage A's worst residual {worst_after:.3f} of its limit's percent, from {worst_before:.3f}: "
                          f"{record['rounds']}")
        if angles.shape != qpos.shape[:1] + (mjm.nq,):
            faults.append(f"stage A's angles {angles.shape}, not a fitting model's pose a frame")
        faults += step_3_faults(problem, control, dynamics, record)
        faults += stage_b_faults(stage_a_motion, physics, record)
    return faults


def step_3_faults(problem, control, dynamics, record):
    """Step 3's pieces on the toy: the couplings and the place nudge on all rows at once the same as MuJoCo's and
    `model.Couplings`' row by row; the warm start settled (sunk 1 cm into its floor, no body starts inside it); the
    learned landmark weights keeping the markers' goal and weighing each as asked; a settings name not known refused;
    every landmark's miss recorded."""
    import mujoco
    import numpy as np
    from motor import trial
    from motor.trial._dynamics import Settings, integrated
    faults, path = [], dynamics.path
    model, couplings = path.model, path.couplings
    qpos = path.states(control)[0]
    generator = np.random.default_rng(5)
    poses = np.array([couplings.apply(pose + np.r_[np.zeros(7), generator.normal(0, 0.3, model.nq - 7)])
                      for pose in qpos[:6]])
    if not (np.array_equal(path.coupled.applied(poses), np.array([couplings.apply(pose.copy()) for pose in poses]))
            and np.array_equal(path.coupled.maps(poses), np.array([couplings.map(pose) for pose in poses]))):
        faults.append("the couplings on all rows are not `model.Couplings`' row by row")
    moves = generator.normal(0.0, 0.2, (len(poses), model.nv))
    expected = poses.copy()
    for pose, move in zip(expected, moves):
        mujoco.mj_integratePos(model, pose, move, 1.0)
    miss = np.abs(integrated(model, poses, moves) - expected).max()
    if miss > 1e-12:
        faults.append(f"the place nudge on all rows {miss:.2e} off MuJoCo's mj_integratePos")
    lowered = control.copy()
    lowered[:, problem.layout.root[2]] -= 0.01
    contact = dynamics.contact
    found = dynamics.evaluated(lowered)[3]
    contact.sunk = None
    inside = float(np.min(contact.heights(found)[0]))
    contact.settle(found)
    settled = float(np.min(contact.heights(found)[0]))
    if not (inside < -0.005 and settled > -1e-9):
        faults.append(f"lowered 1 cm the skin's deepest {inside:.4f} m, settled {settled:.4f} m: not settled")
    clip = problem.rows.clip
    names = [clip.mjm.site(int(site)).name for site in clip.place_sites]
    weighed = trial.retrusted(clip, {names[0]: 4.0})
    costs = np.asarray(clip.place_costs)
    if not (np.isclose(np.sum(weighed ** 2), np.sum(costs ** 2))
            and np.isclose((weighed[0] / weighed[1]) ** 2, 4.0 * (costs[0] / costs[1]) ** 2)):
        faults.append(f"the learned weight 4 on {names[0]}: costs {costs} made {weighed}")
    try:
        Settings.of({"stifness": 1.0})
        faults.append("a settings name not known taken")
    except ValueError:
        pass
    errors = record.get("landmark errors mm", {})
    if sorted(errors) != sorted(names) or not all(np.isfinite(list(errors.values()))):
        faults.append(f"stage A's landmark misses {errors} for the landmarks {names}")
    return faults


def curves_faults(model, muscles):
    """MuJoCo's muscle curves in Warp (`_muscles.Curves`) against `mju_muscleGain` and `mju_muscleBias` at lengths
    and speeds across and past every muscle's range, their tape derivatives against central differences."""
    import mujoco
    import numpy as np
    from motor.trial._muscles import Curves
    generator, faults = np.random.default_rng(13), []
    ranges = model.actuator_lengthrange[muscles]
    lengths = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * generator.uniform(-0.3, 1.3, (60, len(muscles)))
    speeds = generator.normal(0.0, 0.6, lengths.shape)
    curves = Curves(model, muscles)
    gains, biases, by_length, by_speed, bias_length = curves.evaluated(lengths, speeds)
    wanted = [np.array([[call(node, muscle, actuator) for muscle, actuator in enumerate(muscles)]
                        for node in range(len(lengths))])
              for call in (lambda node, muscle, actuator: mujoco.mju_muscleGain(
                               lengths[node, muscle], speeds[node, muscle], ranges[muscle],
                               model.actuator_acc0[actuator], model.actuator_gainprm[actuator, :9]),
                           lambda node, muscle, actuator: mujoco.mju_muscleBias(
                               lengths[node, muscle], ranges[muscle], model.actuator_acc0[actuator],
                               model.actuator_biasprm[actuator, :9]))]
    for name, found, want in (("gain", gains, wanted[0]), ("bias", biases, wanted[1])):
        if not np.allclose(found, want, rtol=1e-12, atol=1e-9):
            faults.append(f"the card's muscle {name} {np.abs(found - want).max():.3e} off MuJoCo's")
    step = 1e-7
    differences = [(curves.evaluated(lengths + step, speeds)[0] - curves.evaluated(lengths - step, speeds)[0]),
                   (curves.evaluated(lengths, speeds + step)[0] - curves.evaluated(lengths, speeds - step)[0]),
                   (curves.evaluated(lengths + step, speeds)[1] - curves.evaluated(lengths - step, speeds)[1])]
    for name, found, difference in zip(("gain on length", "gain on speed", "bias on length"),
                                       (by_length, by_speed, bias_length), differences):
        difference = difference / (2 * step)
        if np.abs(found - difference).max() > 1e-5 * max(np.abs(difference).max(), 1.0):
            faults.append(f"the {name} {np.abs(found - difference).max():.3e} off its central differences")
    return faults


def paths_faults(physics, muscled_rows, qpos, qvel, activations):
    """The muscles' torques on the leading joints (`Muscled.muscle_part`) against MuJoCo's own actuator forces
    (`mj_fwdActuation`, the activations as the muscles' act) carried onto the leading joints by the couplings."""
    import mujoco
    import numpy as np
    model, data = physics.model, mujoco.MjData(physics.model)
    given = muscled_rows.muscle_part(qpos, qvel, activations)["given"] * muscled_rows.scale
    maps = muscled_rows.dynamics.path.coupled.maps(qpos)
    worst = 0.0
    for node in range(len(qpos)):
        data.qpos[:], data.qvel[:] = qpos[node], qvel[node]
        data.act[physics.muscles] = activations[node]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        mujoco.mj_fwdActuation(model, data)
        wanted = (maps[node].T @ data.qfrc_actuator)[6:]
        worst = max(worst, float(np.abs(given[node] - wanted).max() / max(np.abs(wanted).max(), 1.0)))
    return [f"the muscles' torques {worst:.3e} of the largest off MuJoCo's actuator forces"] if worst > 1e-9 else []


def excitation_faults(dynamics):
    """The excitations a span needs (`_muscles.excitations`) give MuJoCo's own activation rate
    (`mju_muscleDynamics`) at the span's midpoint, and their derivatives on both nodes are their central
    differences."""
    import mujoco
    import numpy as np
    from motor.trial._muscles import excitations
    generator, faults, step = np.random.default_rng(19), [], 1 / 30
    activations = generator.uniform(0.05, 0.95, (12, len(dynamics)))
    found, first, second = excitations(activations, step, dynamics)
    middle = 0.5 * (activations[:-1] + activations[1:])
    rates = np.array([[mujoco.mju_muscleDynamics(found[span, muscle], middle[span, muscle], dynamics[muscle])
                       for muscle in range(len(dynamics))] for span in range(len(found))])
    inside = (found > 0.0) & (found < 1.0)
    pace = (activations[1:] - activations[:-1]) / step
    if not np.allclose(rates[inside], pace[inside], rtol=1e-9, atol=1e-9):
        faults.append(f"the excitations' rates {np.abs(rates - pace)[inside].max():.3e} off MuJoCo's")
    moved = 1e-7
    for name, parity, derivative in (("first", 0, first), ("second", 1, second)):
        # every other node nudged: span k's first node (k even) or its second (k + 1 even), not both
        ahead, behind = activations.copy(), activations.copy()
        ahead[::2] += moved
        behind[::2] -= moved
        difference = (excitations(ahead, step, dynamics)[0] - excitations(behind, step, dynamics)[0]) / (2 * moved)
        spans = np.arange(len(found)) % 2 == parity
        miss = np.abs(difference[spans] - derivative[spans]).max()
        if miss > 1e-5 * max(np.abs(difference[spans]).max(), 1.0):
            faults.append(f"the excitation's derivative on the {name} node {miss:.3e} off its central differences")
    return faults


def dense_step(solve, control, activations, pieces, rows, damping):
    """Stage B's whole normal equations written densely (the movement's control points, then every node's
    activations), the damping as the solve places it (the activations' on their block, the movement's on its
    eliminated diagonal): (the dense step, the unknowns `steps` held)."""
    import numpy as np
    from motor.trial import _spline
    step, activation_step, held, held_activations = solve.steps(control, activations, pieces, damping, rows)
    _, hessians, gradients, _ = pieces["motion"]
    stopping = 0.5 * pieces["weights by value"] * pieces["reserves"] ** 2 @ solve.muscled.selection
    system = _spline.Banded(len(control), control.shape[1])
    system.add_nodes(hessians, gradients + stopping)
    system.add_rows(*rows)
    system.add_jerk(control, solve.problem.step, solve.problem.weights)
    points, _, size, _ = system.blocks.shape
    count, muscles = activations.shape
    moving = points * size
    hessian, gradient = np.zeros((moving + count * muscles,) * 2), np.zeros(moving + count * muscles)
    for first in range(points):
        for offset in range(min(4, points - first)):
            block = system.blocks[first, offset]
            hessian[(first + offset) * size:(first + offset + 1) * size, first * size:(first + 1) * size] += block
            if offset:
                hessian[first * size:(first + 1) * size, (first + offset) * size:(first + offset + 1) * size] += block.T
    gradient[:moving] = system.gradient.reshape(-1)
    diagonal, on_activations, activation_gradient = solve.activation_system(activations, pieces, damping)
    jacobians = rows[0]
    for node in range(count):
        at = moving + node * muscles
        hessian[at:at + muscles, at:at + muscles] += np.diag(diagonal[node]) + on_activations[node].T @ on_activations[node]
        moved = np.swapaxes(jacobians[node], 0, 1).reshape(jacobians.shape[2], -1)[6:]
        cross = on_activations[node].T @ moved
        hessian[at:at + muscles, node * size:(node + 3) * size] += cross
        hessian[node * size:(node + 3) * size, at:at + muscles] += cross.T
        gradient[at:at + muscles] = activation_gradient[node]
    free = np.concatenate([~held.reshape(-1), ~held_activations.reshape(-1)])
    shown, eliminated = np.flatnonzero(free[:moving]), moving + np.flatnonzero(free[moving:])
    schur = hessian[np.ix_(shown, shown)] - hessian[np.ix_(shown, eliminated)] @ np.linalg.solve(
        hessian[np.ix_(eliminated, eliminated)], hessian[np.ix_(eliminated, shown)])
    hessian[shown, shown] += damping * (np.diag(schur) + 1.0)
    wanted = np.zeros(len(gradient))
    wanted[free] = np.linalg.solve(hessian[np.ix_(free, free)], -gradient[free])
    return wanted, np.concatenate([step.reshape(-1), activation_step.reshape(-1)])


def stage_b_faults(after, physics, stage_a):
    """Stage B on the muscled toy (`_stage_b`): MuJoCo's curves on the card, the muscles' torques against MuJoCo's own
    actuator forces, the excitations, the muscle torques' Jacobian on the joints' speed against central differences
    (on the place a Gauss-Newton model: the arms' own change left out), the step
    with every node's activations eliminated against the dense normal equations, the cost's slope along the
    activations' own step and the reserves' weights' slope on the joints against their differences, the pelvis'
    rows' change along a short step within 0.3 of their linear model (the speed's velocity products in), and the
    solve lowering its whole cost from its warm start (both at the end's weights), its worst actuated reserve no higher and its pelvis' residual no
    higher than stage A's, the activations inside [0, 1]."""
    import numpy as np
    from motor import trial
    from motor.trial import _stage_b
    faults = curves_faults(physics.model, physics.muscles)
    solve = _stage_b.prepared(after, physics, 0.3)
    muscled_rows, (problem, control, _) = solve.muscled, after
    count = len(problem.measured(control)[0])
    qpos = muscled_rows.dynamics.path.states(control)[0]
    faults += excitation_faults(muscled_rows.dynamics_prm)
    activations = np.full((count, len(physics.muscles)), 0.05)
    control, activations, _, _ = _stage_b.inner(solve, control, activations, move=False, most=40)
    activations = np.clip(activations, 0.02, 0.98)
    total, pieces = solve.measured(control, activations)
    _, qvel = pieces["motion"][3][2][:2]
    faults += paths_faults(physics, muscled_rows, qpos, qvel, activations)
    _, by_speed = muscled_rows.muscle_jacobians(pieces["part"])
    for joint in range(len(muscled_rows.leads)):
        for kind, found in (("speed", by_speed),):
            places, speeds = [qpos.copy(), qpos.copy()], [qvel.copy(), qvel.copy()]
            for sign, (place, speed) in zip((1.0, -1.0), zip(places, speeds)):
                if kind == "place":
                    place[:, muscled_rows.lead_at[joint]] += sign * 1e-6
                else:
                    speed[:, muscled_rows.leads[joint]] += sign * 1e-6
            difference = (muscled_rows.muscle_part(places[0], speeds[0], activations)["given"]
                          - muscled_rows.muscle_part(places[1], speeds[1], activations)["given"]) / 2e-6
            if np.abs(difference - found[:, :, joint]).max() > 1e-5 * max(np.abs(difference).max(), 1e-9):
                faults.append(f"the muscle torques' Jacobian on joint {joint}'s {kind} "
                              f"{np.abs(difference - found[:, :, joint]).max():.3e} off its differences")
    rows = solve.motion_rows(control, pieces)
    wanted, found = dense_step(solve, control, activations, pieces, rows, 1e-6)
    if np.abs(found - wanted).max() > 1e-4 * np.abs(wanted).max():
        faults.append(f"the eliminated step {np.abs(found - wanted).max() / np.abs(wanted).max():.3e} of its largest "
                      "off the dense normal equations'")
    _, activation_step, _, _ = solve.steps(control, activations, pieces, 1e-6, rows, move=False)
    moved = 1e-6
    ahead = solve.measured(control, activations + moved * activation_step, pieces["motion"])[0]
    behind = solve.measured(control, activations - moved * activation_step, pieces["motion"])[0]
    slope = 2.0 * float(np.sum(solve.activation_system(activations, pieces, 0.0)[2] * activation_step))
    if abs((ahead - behind) / (2 * moved) - slope) > 1e-3 * abs(slope):
        faults.append(f"the cost's slope along the activations' step {(ahead - behind) / (2 * moved):.6g}, its "
                      f"gradient's {slope:.6g}")
    values = muscled_rows.lead_values(qpos)
    reserves = pieces["reserves"]
    _, by_value = muscled_rows.weights(reserves, qpos, 1.0)
    for joint in range(len(muscled_rows.leads)):
        nudged = [qpos.copy(), qpos.copy()]
        nudged[0][:, muscled_rows.lead_at[joint]] += 1e-7
        nudged[1][:, muscled_rows.lead_at[joint]] -= 1e-7
        difference = (muscled_rows.weights(reserves, nudged[0], 1.0)[0]
                      - muscled_rows.weights(reserves, nudged[1], 1.0)[0])[:, joint] / 2e-7
        if np.abs(difference - by_value[:, joint]).max() > 1e-4 * max(np.abs(difference).max(), 1.0):
            faults.append(f"the reserves' weights' slope on joint {joint} (at {values[:, joint].round(3).max()}) "
                          f"{np.abs(difference - by_value[:, joint]).max():.3e} off its differences")
    step, activation_step, _, _ = solve.steps(control, activations, pieces, 1e4, rows)
    linear = np.einsum("nprm,npm->nr", rows[0][:, :, :6], np.stack([step[node:node + 3] for node in range(count)]))
    changed = (solve.measured(control + step, activations + activation_step)[1]["motion"][3][0][:, :6]
               - pieces["motion"][3][0][:, :6]) * np.sqrt(solve.weighed.penalty / 2.0)
    miss = np.abs(changed - linear).max() / max(np.abs(linear).max(), 1e-12)
    if miss > 0.3:
        faults.append(f"the pelvis' rows change {miss:.3f} of their linear model off along stage B's step")
    angles, found_activations, record = trial.muscles_solved(after, physics, (0.02, 0.01), 0.3)
    warm = record["warm start (movement held)"]
    costs = record["cost"]
    if not costs["end"] < costs["warm start at the end's weights"]:
        faults.append(f"stage B's cost {costs}: the end not under its warm start's")
    if record["reserves after"]["actuated %BW*ht p50/p95/max"][2] > warm["reserves"]["actuated %BW*ht p50/p95/max"][2]:
        faults.append(f"stage B's worst reserve {record['reserves after']} over its warm start's {warm['reserves']}")
    worst = [max(one["force %BW p50/p95/max"][2] / 2, one["moment %BW*ht p50/p95/max"][2])
             for one in (stage_a["residual after"], record["residual after"])]
    if worst[1] > worst[0]:
        faults.append(f"stage B's worst residual {worst[1]:.3f} of its limit's percent over stage A's {worst[0]:.3f}")
    if found_activations.shape != (count, len(physics.muscles)) or found_activations.min() < 0 \
            or found_activations.max() > 1 or angles.shape != (count, problem.rows.clip.mjm.nq):
        faults.append(f"stage B's activations {found_activations.shape} or angles {angles.shape} not as laid")
    return faults


def mujoco_difference(model, found, span, before, after):
    """MuJoCo's `mj_differentiatePos` into `found`."""
    import mujoco
    mujoco.mj_differentiatePos(model, found, span, before, after)


def test_one_spline_solve_lowers_the_trials_cost_and_jerk_within_its_ranges():
    sys.path.insert(0, str(TOYS))
    faults = the_trial_solve_holds()
    assert not faults, "\n".join(faults)
