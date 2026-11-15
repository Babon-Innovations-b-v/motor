"""batched: on a CUDA card, every frame's state at once on MuJoCo Warp (the model Newton puts there, `model.built`) is
MuJoCo's C to float32's rounding: on a toy (a free root, an arm of two hinges with damping and armature, a muscle (a
<general> one, as `myo` writes them) on a spatial tendon wrapped round a cylinder, gravity) over 12 states of moving
angles, speeds and accelerations, the needed generalised forces within 1e-5 of their largest (and within 1e-5 of
MuJoCo's C under `Physics.each`, its state laid in `data`), the muscle's length, speed and moment arms within 1e-6 m,
each body's place within 1e-6 m and turn within 1e-6, the tendon's path points (`records.paths_of`, as the layer draws
it) within 1e-6 m; after RRA's model step moves a mass the card's needed forces follow it (`Card.sync`); and RRA's
sensitivities on the card (`rra.sensitivities`, CARD_STEPS) within 1e-3 of the largest of MuJoCo's C ones at RRA_STEP;
and every frame's least squares at once (`least_all`) is `solvers.least`'s (`least_squares_alike`), and RRA's capped move with its products on the card (`within_balls_card`) is
`solvers.within_balls`' (`balls_alike`). Run to run (`repeats`): the card's states evaluated again, three worlds a
launch, are the same bits; and on Warp's processor device (card or none) the toy's `Card.evaluate`, its own `stages`
in Warp's run-to-run mode, gives every value it reads bit for bit as MuJoCo Warp's whole `fwd_position`,
`fwd_velocity` and `rne` with the accelerations, twice. The card's run is
the processor's within CARD_TOLERANCES (`batched.apart`, measured on #70's clips): a record moved within each reads
equal, past one reads apart by its name (`the_judge_tells_a_moved_record`, run with or without a card). Without a card
it prints that it was not run. It runs in MOTOR's own Python (Newton and MuJoCo, the pins in pyproject.toml).

Breaks (each applied once, seen red, reverted), 2026-11-05:
- Card._chunk's `+ model.dof_armature * qacc` dropped: the needed forces leave out the armature's.
- Card.sync's `target.assign(...)` dropped: the card keeps the masses as built after RRA moves them.
- Card.load's copy of `cdof` dropped: a point's Jacobian on the CPU reads stale motions.
- rra.sensitivities' `CARD_STEPS[0]` made 1e-6: float32's rounding swamps the places' differences.
- _least._free's `held = ...` made `held = False`: a column its bound holds steps past it and is clipped back.
- _least._chosen's `+ wp.float64(1e-4) * along[...]` made `+ 1e4 * abs(...)`: a step that lowers the cost too
  little is taken anyway.
- within_balls_card's `- rho * shifted` dropped: the balls' own offsets left out of the target.
- apart's `if not off <= tolerance` made `if off > 10 * tolerance`: a record moved twice a tolerance reads equal.
- apart's residual moment check dropped: a residual moment moved past its tolerance reads equal.
Breaks, 2026-11-06 (on the processor, the card hidden):
- stages' `mjw.passive(...)` dropped: the passive forces, and so the needed forces, are not MuJoCo Warp's.
- stages' `rne(..., flg_acc=True)` made `flg_acc=False`: the needed forces leave out the accelerations'.
- stages' `_tendon_velocity` launch dropped: the damped tendon's pull reads no speed.

Run: .venv/bin/python motor/batched/test_every_frame_on_the_card_is_mujocos_own_to_its_rounding.py
"""
import pathlib
import sys
import tempfile

INSIDE = pathlib.Path(__file__).resolve().parents[1]
GRAVITY = 9.81
ARM = """<mujoco model="arm">
  <compiler angle="radian"/>
  <default><joint damping="0.5" armature="0.01"/></default>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1" mass="5"/>
      <site name="origin" pos="0.05 0 0.1"/>
      <body name="upper" pos="0 0 0.1">
        <joint name="shoulder_flexion" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom name="wrap" type="cylinder" size="0.03 0.05" euler="1.5708 0 0" mass="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="2"/>
        <body name="lower" pos="0 0 0.3">
          <joint name="elbow_flexion" type="hinge" axis="0 1 0" range="0 2.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.025" mass="1"/>
          <site name="insertion" pos="0.03 0 0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="flexor_tendon">
      <site site="origin"/>
      <geom geom="wrap" sidesite="origin"/>
      <site site="insertion"/>
    </spatial>
  </tendon>
  <actuator>
    <general name="flexor" tendon="flexor_tendon" ctrlrange="0 1" ctrllimited="true" dyntype="muscle"
             dynprm="0.01 0.04" gaintype="muscle" biastype="muscle" gainprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2"
             biasprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2"/>
  </actuator>
</mujoco>
"""


def states(found, count=12):
    """`count` states of the toy: the root moving and turning, the hinges swinging, speeds and accelerations."""
    import mujoco
    import numpy as np
    steps = np.linspace(0.0, 1.0, count)
    qpos = np.tile(found.qpos0, (count, 1))
    qpos[:, 0] += 0.1 * steps
    qpos[:, 2] += 0.05 * np.sin(3 * steps)
    for row, angle in enumerate(0.4 * steps):
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2]), angle)
        qpos[row, 3:7] = quat
    qpos[:, 7] = -0.5 + steps
    qpos[:, 8] = 0.3 + 1.2 * steps
    qvel = np.outer(np.cos(5 * steps), np.linspace(0.5, 2.0, found.nv))
    qacc = np.outer(np.sin(4 * steps), np.linspace(-3.0, 3.0, found.nv))
    return qpos, qvel, qacc


def every_frame_on_the_card_is_mujocos_own():
    """The checks of the docstring, together."""
    import mujoco
    import numpy as np
    from motor import contact
    from motor import model
    from motor import muscles
    from motor import records
    from motor import rra
    from motor.rra._linear import frame_states
    import warp
    if not warp.is_cuda_available():
        print("batched: no CUDA card, the card's checks not run")
        return repeats_on_the_processor() + the_judge_tells_a_moved_record()
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "arm.xml"
        path.write_text(ARM)
        found = model.built(path, GRAVITY)
    ground = np.array([[[-5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [5.0, 0.0, -5.0]]])
    nothing = contact.Skin(np.zeros((0, 3)), np.zeros((0, 1)), [np.eye(4)], [0], [np.eye(4)])
    physics = muscles.Physics(found, ground, nothing)
    if physics.card is None:
        return ["a CUDA card, but the model Newton built has no MuJoCo Warp model kept"]
    qpos, qvel, qacc = states(found)
    card = physics.card.evaluate(qpos, qvel, qacc)
    data = mujoco.MjData(found)
    cpu = {name: [] for name in ("needed", "length", "speed", "moment", "xpos", "xmat", "wrap")}
    for row in range(len(qpos)):
        cpu["needed"].append(physics.needed(qpos[row], qvel[row], qacc[row]))
        own = physics.data
        cpu["length"].append(own.actuator_length.copy())
        cpu["speed"].append(own.actuator_velocity.copy())
        dense = np.zeros((found.nu, found.nv))
        mujoco.mju_sparse2dense(dense, own.actuator_moment, own.moment_rownnz, own.moment_rowadr, own.moment_colind)
        cpu["moment"].append(dense)
        cpu["xpos"].append(own.xpos.copy())
        cpu["xmat"].append(own.xmat.copy())
        cpu["wrap"].append(records.paths_of(own, [0])[0])
    cpu = {name: values if name == "wrap" else np.array(values) for name, values in cpu.items()}
    problems = []

    def within(name, got, wanted, tolerance):
        miss = float(np.abs(np.asarray(got) - np.asarray(wanted)).max())
        if not miss <= tolerance:
            problems.append(f"{name} {miss:.3g} off MuJoCo's C (allowed {tolerance:.3g})")

    largest = np.abs(cpu["needed"]).max()
    within("needed forces", card["needed"], cpu["needed"], 1e-5 * largest)
    within("muscle length", card["actuator_length"], cpu["length"], 1e-6)
    within("muscle speed", card["actuator_velocity"], cpu["speed"], 1e-5 * max(1.0, np.abs(cpu["speed"]).max()))
    within("bodies' places", card["xpos"], cpu["xpos"], 1e-6)
    within("bodies' turns", card["xmat"].reshape(cpu["xmat"].shape), cpu["xmat"], 1e-6)
    laid = []
    for row in range(len(qpos)):
        physics.card.load(card, row, data)
        dense = np.zeros((found.nu, found.nv))
        mujoco.mju_sparse2dense(dense, data.actuator_moment, data.moment_rownnz, data.moment_rowadr,
                                data.moment_colind)
        laid.append((dense, records.paths_of(data, [0])[0]))
        jacobian, cpu_jacobian = np.zeros((3, found.nv)), np.zeros((3, found.nv))
        mujoco.mj_jac(found, data, jacobian, None, data.xpos[3], 3)
        own = mujoco.MjData(found)
        own.qpos[:] = qpos[row]
        mujoco.mj_fwdPosition(found, own)
        mujoco.mj_jac(found, own, cpu_jacobian, None, own.xpos[3], 3)
        within("a point's Jacobian on the laid state", jacobian, cpu_jacobian, 1e-5)
    within("moment arms", [moment for moment, _ in laid], cpu["moment"], 1e-6)
    if any(len(points) != len(cpu["wrap"][row]) for row, (_, points) in enumerate(laid)):
        problems.append("the tendon's path points counted otherwise on the card")
    else:
        within("path points", np.concatenate([points for _, points in laid]), np.concatenate(cpu["wrap"]), 1e-6)
    within("Physics.each", np.array(list(physics.each(qpos, qvel, qacc))), cpu["needed"], 1e-5 * largest)
    found.body_mass[1] *= 1.02
    moved = physics.card.evaluate(qpos, qvel, qacc)["needed"]
    wanted = np.array([physics.needed(*state) for state in zip(qpos, qvel, qacc)])
    within("needed forces after a mass moved", moved, wanted, 1e-5 * largest)
    walked = frame_states(physics, qpos, 30.0)
    adjusted = [0, 1, 2, 3, 4, 5, 6]  # the root's six and the shoulder
    on_card = rra.sensitivities(physics, walked, adjusted)
    kept, physics.card = physics.card, None
    on_cpu = rra.sensitivities(physics, walked, adjusted)
    physics.card = kept
    for kind, name in enumerate(("places", "speeds", "accelerations")):
        wanted = np.array([frame[kind] for frame in on_cpu])
        within(f"sensitivity to {name}", np.array([frame[kind] for frame in on_card]), wanted,
               1e-3 * np.abs(wanted).max())
    problems += least_squares_alike() + balls_alike() + repeats(physics.card, qpos, qvel, qacc)
    return problems + repeats_on_the_processor() + the_judge_tells_a_moved_record()


def repeats(card, qpos, qvel, qacc):
    """`card.evaluate` of the states twice, the second three worlds a launch: every value read the same bits."""
    import numpy as np
    first = card.evaluate(qpos, qvel, qacc)
    kept, card.chunk = card.chunk, 3
    try:
        second = card.evaluate(qpos, qvel, qacc)
    finally:
        card.chunk = kept
    return [f"{name} not the same bits again" for name in first if not np.array_equal(first[name], second[name])]


def repeats_on_the_processor():
    """On Warp's processor device, the toy's `Card.evaluate` (its `stages`, in Warp's run-to-run mode) twice, each
    value it reads the same bits as MuJoCo Warp's whole `fwd_position` (no factorising), `fwd_velocity` and `rne`
    with the accelerations on the same states, its tendon damped so the tendon's speed is read."""
    import mujoco
    import mujoco_warp
    import numpy as np
    import warp
    from motor.batched import COPIED, Card
    found = mujoco.MjModel.from_xml_string(ARM)
    found.tendon_damping[0] = 2.0
    qpos, qvel, qacc = states(found)
    with warp.ScopedDevice("cpu"):
        card = Card(found, mujoco_warp.put_model(found))
        found_once = card.evaluate(qpos, qvel, qacc)
        problems = repeats(card, qpos, qvel, qacc)
        whole = mujoco_warp.make_data(found, nworld=len(qpos), nconmax=1, njmax=1)
        whole.qpos.assign(qpos.astype(np.float32))
        whole.qvel.assign(qvel.astype(np.float32))
        whole.qacc.assign(qacc.astype(np.float32))
        mujoco_warp.fwd_position(card.mjw_model, whole, factorize=False)
        mujoco_warp.fwd_velocity(card.mjw_model, whole)
        mujoco_warp.rne(card.mjw_model, whole, flg_acc=True)
    wanted = {name: getattr(whole, name).numpy() for name in COPIED}
    wanted["needed"] = (whole.qfrc_bias.numpy().astype(float) + found.dof_armature * qacc
                        - whole.qfrc_passive.numpy().astype(float))
    for name, values in wanted.items():
        got = np.asarray(found_once[name])
        if not np.array_equal(got, np.asarray(values).astype(got.dtype)):
            problems.append(f"{name} on the processor not MuJoCo Warp's whole stages' bits")
    return [f"processor: {problem}" for problem in problems]


def entry(frames=4, weight=700.0):
    """A person's out.json entry with what `batched.apart` reads."""
    following = {"bones outside skin %": {"all": 1.6, "hand": 1.5},
                 "joints mm": {"free": [9.7, 21.5, 3619], "at a limit": [12.0, 19.8, 341]},
                 "turns deg": {"free": [1.6, 7.7, 3737], "at a limit": [3.1, 7.9, 403]}}
    return {"model": {"fit": {"following, the path the physics takes": following}}, "weight N": weight,
            "adjusted": {"follows deg": [3.97, 3.99], "kept share": 0.75, "least frame share": 0.3},
            "frames": [{"ground": [1.0, 690.0, -2.0], "residual": [0.1, 2.0, -0.3, 0.5, -0.2, 0.1], "reserve": 3.2,
                        "activations": [0.1, 0.5, 0.9]} for _ in range(frames)]}


def the_judge_tells_a_moved_record():
    """`batched.apart`: a record against itself, and one moved within every tolerance, are not apart; one moved past
    a tolerance in each measure is named by it (and only it)."""
    import copy
    from motor.batched import CARD_TOLERANCES, apart
    base = entry()
    problems = [] if not apart(base, copy.deepcopy(base)) else [f"a record apart from itself: {apart(base, base)}"]
    moves = {"bones outside %": lambda record, by: record["model"]["fit"]["following, the path the physics takes"][
                 "bones outside skin %"].__setitem__("hand", 1.5 + by),
             "joints mm free": lambda record, by: record["model"]["fit"]["following, the path the physics takes"][
                 "joints mm"]["free"].__setitem__(1, 21.5 + by),
             "turns deg at a limit": lambda record, by: record["model"]["fit"]["following, the path the physics takes"][
                 "turns deg"]["at a limit"].__setitem__(0, 3.1 + by),
             "follows deg": lambda record, by: record["adjusted"]["follows deg"].__setitem__(1, 3.99 + by),
             "shares": lambda record, by: record["adjusted"].__setitem__("least frame share", 0.3 + by),
             "ground push / weight": lambda record, by: record["frames"][2]["ground"].__setitem__(1, 690.0 + 700 * by),
             "residual force / weight": lambda record, by: record["frames"][1]["residual"].__setitem__(
                 0, 0.1 + 700 * by),
             "residual moment N m": lambda record, by: record["frames"][3]["residual"].__setitem__(4, -0.2 + by),
             "reserve N m": lambda record, by: record["frames"][0].__setitem__("reserve", 3.2 + by),
             "activation": lambda record, by: record["frames"][2]["activations"].__setitem__(1, 0.5 + by)}
    allowed = {"joints mm free": "joints mm", "turns deg at a limit": "turns deg", "shares": "share"}
    for name, move in moves.items():
        tolerance = CARD_TOLERANCES[allowed.get(name, name)]
        within, past = copy.deepcopy(base), copy.deepcopy(base)
        move(within, 0.5 * tolerance)
        move(past, 2.0 * tolerance)
        if apart(base, within):
            problems.append(f"{name} moved half its tolerance read apart: {apart(base, within)}")
        found = apart(base, past)
        if len(found) != 1 or not found[0].startswith(name):
            problems.append(f"{name} moved twice its tolerance read {found}")
    return problems


def balls_alike():
    """`within_balls_card` on RRA's shape of problem (60 frames' moves of 4 dofs low-passed, a ball a frame on 3 of
    them, the wanted move past the balls) is `solvers.within_balls`' within 1e-8 of its largest entry, its furthest
    ball as far."""
    import numpy as np
    from motor.batched import within_balls_card
    from motor.solvers import filter_matrix, within_balls
    generator = np.random.default_rng(3)
    frames, width = 60, 4
    smooth = np.kron(filter_matrix(frames, 30.0), np.eye(width))
    system = generator.normal(size=(frames * width + 40, frames * width)) @ smooth
    wanted = generator.normal(size=frames * width + 40) * 5.0
    rows = np.zeros((frames, 3, frames * width))
    for frame in range(frames):
        rows[frame, :, frame * width:frame * width + 3] = np.eye(3) * 2.0
    rows = (rows.reshape(-1, frames * width) @ smooth)
    values = generator.normal(size=frames * 3) * 0.1
    wanted_y, wanted_over = within_balls(system, wanted, rows, values)
    found_y, found_over = within_balls_card(system, wanted, rows, values)
    miss = float(np.abs(found_y - wanted_y).max() / np.abs(wanted_y).max())
    problems = [] if miss <= 1e-8 else [f"within_balls_card {miss:.3g} off within_balls"]
    return problems + ([] if abs(found_over - wanted_over) <= 1e-8 else [f"furthest ball {found_over} against "
                                                                          f"{wanted_over}"])


def least_squares_alike():
    """`least_all` on 20 problems of 40 to 300 columns and 30 rows, the first fifth unbounded above (pushes) and the
    rest at most 1 (activations), wanted far past what the bounds allow on half of them so many columns end on a
    bound: each problem's x equals `solvers.least`'s within 1e-7 of its largest entry (float64 both; the solves differ
    only in their Cholesky and LU rounding). Problems `least` settles: with the balance rows weighed 100 against
    costs of 1e-3, random ones end its 1000 rounds unsettled, and two runs of the same steps part by rounding."""
    import numpy as np
    from motor.batched import least_all
    from motor.solvers import least
    generator = np.random.default_rng(7)
    systems, wanteds, costs, highests = [], [], [], []
    for problem in range(20):
        width = int(generator.integers(40, 300))
        system = generator.normal(size=(30, width))
        systems.append(system)
        wanteds.append(generator.normal(size=30) * (40.0 if problem % 2 else 4.0))
        costs.append(np.ones(width))
        highests.append(np.concatenate([np.full(width // 5, np.inf), np.ones(width - width // 5)]))
    found = least_all(systems, wanteds, costs, highests)
    found_problems = []
    bounded = 0
    for problem, x in enumerate(found):
        wanted = least(systems[problem], wanteds[problem], costs[problem], highests[problem])
        bounded += int(np.sum(np.isclose(wanted[len(wanted) // 5:], 1.0)))
        miss = float(np.abs(x - wanted).max() / max(np.abs(wanted).max(), 1e-12))
        if not miss <= 1e-7:
            found_problems.append(f"least_all's problem {problem} {miss:.3g} off least's")
    return found_problems + ([] if bounded else ["no column ended on its bound: the bounds untested"])


def in_newton():
    """The check, in Newton's own Python."""
    failures = every_frame_on_the_card_is_mujocos_own()
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(in_newton())
