"""muscles: a body standing still on a flat ground under gravity is carried by the ground: the static optimisation
gives the ground's push its weight and leaves almost no residual on the pelvis, the movement's needed force is its
weight up, and the touching part's push (`touches`, the ball's body by its name) is that whole push, at the ground under
it; and the static optimisation is well-posed: a muscle has no moment on the root's free joint, so the card's float32
rounding there (1e-3, MuJoCo's C 1e-11) moves no activation, and a 1e-6 nudge of the moments moves them by about 1e-6;
and on the ground too, the ground reaction force solved first (`staged`): two feet on a flat floor asked for a
moment past their toes (the body cannot be balanced, so the balance sits on the edge of the contacts' cone, its toe
points' friction pyramids dependent), a 1e-6 nudge of the contact columns moves the activations under 1e-5 and each
foot carries half the weight; and a knee's free turn is carried by its ligaments, not the muscles crossing it; and a
leader's own moment is its joint's alone, its net moment its followers' with it.
It runs in Newton's own Python (on a model Newton builds; MOTOR's own Python, the pins in pyproject.toml).

Breaks (each applied once with the env linked into a sandbox home, seen red, reverted), 2026-10-28:
- shared's `rows[:6] = BALANCE` made `rows[:6] = 0.0`: the pelvis' balance costs nothing, so the ground pushes
  nothing and the weight is left as residual.
- needed's `found + model.dof_armature * qacc - data.qfrc_passive` made `-found + ...`: the movement asks for its
  weight down.
2026-11-04:
- touches' `centre = sizes @ ...` made `centre = np.zeros(3) + 1.0`: the push stands off the ground under the ball.
2026-11-05:
- touches' `normals = np.divide(normals, lengths, ...)` dropped: the push along its normals reads 3.2 times the push
  (the pyramid's four edges, each `closeness` 0.8 long 1 mm over, summed).
- problem's `arms[root dofs] = 0.0` made `pass`: the muscles' columns stand on the root's rows, and given the card's
  1e-3 of moment there the airborne leg's holding muscle goes from 0.150 to 0.0, pulled to cancel the pelvis'
  residual it cannot touch (2026-11-05, #70: the jump's frame 30, mean 0.115 on MuJoCo's C, 0.206 on the card).
2026-11-09:
- share_of's `whole[self.couplings.leads[6:]]` made `net[6:]`: a lumbar leader's own moment carries its followers'
  (#98: a lab's back moment is the one joint's).
- staged's steps made the one least squares of before (contacts, stops and muscles together, the pelvis' rows
  BALANCE): a 1e-6 nudge of the contacts moves an ankle muscle by 0.145.
- share_asked's SHARE_COST made CONTACT_COST: the nudge moves it by 0.0028 (the balance crowded on a few points).
2026-11-18 (#1):
- share_asked's `SPLIT * split[None]` made `0.0 * split[None]`: the right foot carries 0.85 of the weight.
- split_row's `ups * (1.0 - left)` made `ups * left`: the left foot's push weighed by its own share.
2026-11-07:
- problem's `np.concatenate([self.stops(), self.ligaments()], axis=1)` made `self.stops()`: the turn's 5 N m moves
  the activations (the muscles reach it 1 cm off its axis) and leaves a reserve on it.

Run: .venv/bin/python motor/muscles/test_a_standing_body_is_carried_by_the_ground.py
"""
import pathlib
import sys
import tempfile

INSIDE = pathlib.Path(__file__).resolve().parents[1]
KG = 10.0
GRAVITY = 9.81
BODY = """<mujoco model="block">
  <compiler angle="radian"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.101">
      <freejoint name="root"/>
      <geom type="sphere" size="0.1" mass="10"/>
    </body>
  </worldbody>
</mujoco>
"""

LEG = """<mujoco model="leg">
  <compiler angle="radian"/>
  <worldbody>
    <body name="pelvis" pos="0 0 2">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.05" mass="10"/>
      <site name="front" pos="0.05 0 0"/>
      <site name="back" pos="-0.05 0 0"/>
      <body name="thigh" pos="0 0 -0.05">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="5"/>
        <site name="shin front" pos="0.05 0 -0.1"/>
        <site name="shin back" pos="-0.05 0 -0.1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="flexor"><site site="front"/><site site="shin front"/></spatial>
    <spatial name="extensor"><site site="back"/><site site="shin back"/></spatial>
  </tendon>
  <actuator>
    <general name="flexor" tendon="flexor" lengthrange="0.05 0.2" ctrlrange="0 1" ctrllimited="true"
      dyntype="muscle" gaintype="muscle" biastype="muscle" dynprm="0.01 0.04"
      gainprm="0.75 1.05 500 1 0 2 10 1.5 1.4" biasprm="0.75 1.05 500 1 0 2 10 1.5 1.4"/>
    <general name="extensor" tendon="extensor" lengthrange="0.05 0.2" ctrlrange="0 1" ctrllimited="true"
      dyntype="muscle" gaintype="muscle" biastype="muscle" dynprm="0.01 0.04"
      gainprm="0.75 1.05 500 1 0 2 10 1.5 1.4" biasprm="0.75 1.05 500 1 0 2 10 1.5 1.4"/>
  </actuator>
</mujoco>
"""
ROUNDING = 1e-3  # the card's float32 rounding of a moment arm, at its largest (2026-11-05, #70)
# The leg with a shin turning on its thigh about the thigh's long axis (a knee's free turn, muscles.LIGAMENTS'
# name), the two muscles crossing it 1 cm off its axis.
TURNING = LEG.replace('''        <site name="shin front" pos="0.05 0 -0.1"/>
        <site name="shin back" pos="-0.05 0 -0.1"/>''', '''        <body name="shin" pos="0 0 -0.05">
          <joint name="knee_angle_rotation3_r" type="hinge" axis="0 0 1" range="-0.26 0.26" limited="true"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="3"/>
          <site name="shin front" pos="0.05 0.01 -0.05"/>
          <site name="shin back" pos="-0.05 0.01 -0.05"/>
        </body>''')
TURN_ASKED = 5.0  # N m on the shin's turn


def ball_skin(found):
    """The ball's skin: its surface (0.1 m) in 26 directions at qpos0, riding the pelvis."""
    from motor import contact
    import mujoco
    import numpy as np
    from motor.model import TO_OWN, own_pose
    data = mujoco.MjData(found)
    mujoco.mj_kinematics(found, data)
    directions = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if x or y or z], float)
    points = TO_OWN @ data.xpos[1] + 0.1 * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return contact.Skin(points, np.ones((len(points), 1)), [own_pose(found, data, 1)], [1], [np.eye(4)])


def the_ground_carries_a_still_body():
    """A 10 kg ball standing 1 mm over a flat ground, still: the movement needs 98.1 N up at the pelvis, the ground's
    push sums to it within 1% and the pelvis' residual force is under 1% of it."""
    from motor import model as built_model
    import mujoco
    from motor import muscles
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "block.xml"
        path.write_text(BODY)
        found = built_model.built(path, GRAVITY)
    ground = np.array([[[-5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [5.0, 0.0, -5.0]],
                       [[-5.0, 0.0, -5.0], [-5.0, 0.0, 5.0], [5.0, 0.0, 5.0]]])
    physics = muscles.Physics(found, ground, ball_skin(found))
    still = np.zeros(found.nv)
    needed = physics.needed(found.qpos0.copy(), still, still)
    weight = KG * GRAVITY
    problems = [] if np.allclose(needed[:3], [0.0, 0.0, weight], rtol=1e-6) else [f"needed {needed[:3]}"]
    share = physics.shared(needed)
    pushed = share["pushes"].sum(axis=0)
    if abs(pushed[2] - weight) > 0.01 * weight:
        problems.append(f"the ground pushes {pushed.round(3).tolist()} N, not {weight} N up")
    touched = share["touches"]  # the ball's one body: the whole push, at the ground straight under its lowest point
    ball = touched.get(mujoco.mj_id2name(found, mujoco.mjtObj.mjOBJ_BODY, 1))
    if len(touched) != 1 or ball is None or not np.allclose(ball[0], pushed) or not np.allclose(ball[1], 0.0,
                                                                                               atol=1e-6) or not \
            np.isclose(ball[2], pushed[2], rtol=1e-6):
        problems.append(f"touches {touched}: not the whole push at the ground under the ball")
    residual = float(np.linalg.norm(share["residual"][:3]))
    return problems + ([] if residual < 0.01 * weight else [f"residual force {residual:.3f} N"])


def rounding_moves_no_activation():
    """A body 2 m up (the ground pushes nothing), its hip held bent by two muscles across it: the problem's muscle
    columns are 0 on the root's six rows; with every muscle's moment on the root given the card's rounding (ROUNDING
    of its largest), the activations move under 1e-9; with every moment nudged by 1e-6 of itself, under 1e-5.
    A knee's free turn moves no activation either: its ligaments hold it (`ligament_turn_problems`)."""
    from motor import model as built_model
    import mujoco
    from motor import muscles
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "leg.xml"
        path.write_text(LEG)
        found = built_model.built(path, GRAVITY)
    physics = muscles.Physics(found, np.zeros((0, 3, 3)), None)
    physics.contacts = lambda: []  # 2 m over any ground: nothing touches
    qpos, still = found.qpos0.copy(), np.zeros(found.nv)
    qpos[7] = 0.5
    hip = found.jnt_dofadr[1]
    dense = mujoco.mju_sparse2dense

    def solved(change):
        """The activations with the problem's dense moments (muscles x dofs) changed by `change` as it reads them."""
        def read(moments, *sparse):
            dense(moments, *sparse)
            change(moments)
        needed = physics.needed(qpos, still, still)
        muscles.mujoco.mju_sparse2dense = read
        try:
            problem = physics.problem(needed)
        finally:
            muscles.mujoco.mju_sparse2dense = dense
        return problem, physics.solved_all([problem], card=False)[0][problem["count"] + problem["held"]:]

    plain, before = solved(lambda moments: None)
    problems = [] if np.all(plain["system"][:6, plain["count"] + plain["held"]:] == 0.0) else \
        [f"muscle columns on the root {plain['system'][:6]}"]
    if not 0.0 < before.max() < 1.0:
        problems.append(f"no muscle holds the hip within its range: activations {before}")

    def rounded(moments):
        moments[:, :6] += ROUNDING * np.abs(moments[:, hip]).max()

    off_root = solved(rounded)[1]
    if np.abs(off_root - before).max() > 1e-9:
        problems.append(f"the card's rounding on the root moves the activations {before} to {off_root}")
    nudge = np.random.default_rng(0)
    nudged = solved(lambda moments: moments.__imul__(1 + 1e-6 * nudge.standard_normal(moments.shape)))[1]
    if np.abs(nudged - before).max() > 1e-5:
        problems.append(f"a 1e-6 nudge of the moments moves the activations {before} to {nudged}")
    return problems + ligament_turn_problems() + own_moment_problems()


def ligament_turn_problems():
    """The leg with its shin's turn (`TURNING`) 2 m up, its hip held bent: asked TURN_ASKED more on the turn, the
    activations move under 1e-6 and its reserve stays under 1e-3 N m (the knee's ligaments carry it,
    `muscles.LIGAMENTS`)."""
    from motor import model as built_model
    from motor import muscles
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "turning.xml"
        path.write_text(TURNING)
        found = built_model.built(path, GRAVITY)
    physics = muscles.Physics(found, np.zeros((0, 3, 3)), None)
    physics.contacts = lambda: []
    qpos, still = found.qpos0.copy(), np.zeros(found.nv)
    qpos[7] = 0.5
    turn = found.jnt_dofadr[2]
    needed = physics.needed(qpos, still, still)
    plain = physics.shared(needed)
    needed[turn] += TURN_ASKED
    asked = physics.shared(needed)
    problems = [] if 0.0 < plain["activations"].max() < 1.0 else [f"no muscle holds the hip: {plain['activations']}"]
    if np.abs(asked["activations"] - plain["activations"]).max() > 1e-6:
        problems.append(f"the turn's {TURN_ASKED} N m moves the activations {plain['activations']} to "
                        f"{asked['activations']}")
    left = asked["reserves"][list(physics.couplings.leads[6:]).index(turn)]
    return problems + ([] if abs(left) < 1e-3 else [f"{left:.4f} N m left on the turn as reserve"])


def feet(lean):
    """Two feet's static optimisation as `Physics.problem` lays one out: each foot 12 points on a flat floor (6 along
    it, -5 to 20 cm, two across), each a friction pyramid (`contact.FRICTION`), the pelvis' six rows about the origin,
    one ankle row a foot (its ankle 8 cm up) and two muscles across each ankle (5 cm arms, 3 kN), the weight (700 N)
    asked to stand `lean` metres forward of the ankles."""
    from motor import contact
    from motor import muscles
    import numpy as np
    edges = [np.array(edge, float) for edge in ((contact.FRICTION, 0, 1), (-contact.FRICTION, 0, 1),
                                                (0, contact.FRICTION, 1), (0, -contact.FRICTION, 1))]
    pelvis, ankles = [], []
    for foot, side in enumerate((-0.1, 0.1)):
        ankle = np.array([0.0, side, 0.08])
        for along in np.linspace(-0.05, 0.2, 6):
            for across in (side - 0.04, side + 0.04):
                at = np.array([along, across, 0.0])
                for edge in edges:
                    pelvis.append(np.concatenate([edge, np.cross(at, edge)]))
                    row = np.zeros(2)
                    row[foot] = np.cross(at - ankle, edge)[1]
                    ankles.append(row)
    pelvis, ankles = np.array(pelvis).T, np.array(ankles).T
    arms = 3000.0 * np.array([[0.05, -0.05, 0.0, 0.0], [0.0, 0.0, 0.05, -0.05]])
    count, weight = pelvis.shape[1], 700.0
    return {"system": np.vstack([np.hstack([pelvis, np.zeros((6, 4))]), np.hstack([ankles, arms])]),
            "wanted": np.array([0.0, 0.0, weight, 0.0, -weight * lean, 0.0, 0.0, 0.0]),
            "costs": np.concatenate([np.full(count, muscles.CONTACT_COST), np.ones(4)]),
            "highest": np.concatenate([np.full(count, np.inf), np.ones(4)]), "count": count, "held": 0}


def ground_frames_are_well_posed():
    """`feet` asked for its weight 25 cm forward, past the toes (20 cm): `staged` by `solvers.least` gives each foot
    within 5% of half the weight, and a 1e-6 nudge of the contact columns (five draws) moves no activation over 1e-5."""
    from motor import muscles
    import numpy as np
    from motor.solvers import least
    plain = feet(0.25)
    count = plain["count"]

    def solved(problem):
        return muscles.staged([problem], lambda asked: [least(*each) for each in asked])[0]

    before = solved(plain)
    problems = []
    for foot in (slice(0, count // 2), slice(count // 2, count)):
        carried = float(plain["system"][2, foot] @ before[foot])
        if abs(carried - 350.0) > 0.05 * 350.0:
            problems.append(f"a foot carries {carried:.1f} N of the 700 N")
    moved = 0.0
    for seed in range(5):  # five nudges: which one a fragile solve breaks on is the draw's
        nudged = dict(plain, system=plain["system"].copy())
        nudged["system"][:, :count] *= 1 + 1e-6 * np.random.default_rng(seed).standard_normal((len(plain["system"]),
                                                                                               count))
        moved = max(moved, float(np.abs(solved(nudged)[count:] - before[count:]).max()))
    return problems + ([] if moved < 1e-5 else [f"a 1e-6 nudge of the contacts moves an activation by {moved:.3g}"]) + \
        split_problems()


def split_problems():
    """The feet's split off their contacts' timeline (`split_row`): a left foot of four touching points and a right of
    one, the left's share 1/4 (`contact.feet_shares`), weigh the left's vertical push by 3/4 and the right's by -1/4,
    whatever each foot's points' count, and without a share nothing; and `feet` asked to stand its weight 8 cm toward
    its right foot (past the middle of its sole), which the balance alone gives the right foot 0.8 of, a split of half
    each brings within 0.15 of half the weight while the ground still carries it within 2 % (#1, 2026-11-18: the split
    followed the video body's frontal moment, the left share 0.28 to 0.82 frame to frame through a squat the plates
    held at 0.48; then the contacts' closeness, the flatter sole carrying more)."""
    from motor import muscles
    import numpy as np
    from motor.solvers import least
    pyramid = [np.array([0.0, 0.0, 1.0])] * 4
    contacts = [(1, None, pyramid, None)] * 4 + [(2, None, pyramid, None)]
    row = muscles.split_row(contacts, {1: "LeftFoot", 2: "RightFoot"}, 0.25)
    wanted = np.concatenate([np.full(16, 0.75), np.full(4, -0.25)])
    problems = [] if np.allclose(row, wanted) else [f"split row {np.round(row, 3).tolist()}, not {wanted.tolist()}"]
    if muscles.split_row(contacts, {1: "LeftFoot", 2: "RightFoot"}, None).any():
        problems.append("without a share the split row asks a split")
    plain = feet(0.0)
    count = plain["count"]
    plain["wanted"][3] = 700.0 * 0.08  # about x: the weight 8 cm toward the right foot (y = +0.1)
    ups = plain["system"][2, :count]
    plain["split"] = np.where(np.arange(count) < count // 2, 0.5 * ups, -0.5 * ups)  # equal contact, left first
    for split, low, high in ((0.0, 0.75, 1.0), (muscles.SPLIT, 0.35, 0.65)):
        kept, muscles.SPLIT = muscles.SPLIT, split
        try:
            found = muscles.staged([plain], lambda asked: [least(*each) for each in asked])[0]
        finally:
            muscles.SPLIT = kept
        right, total = float(ups[count // 2:] @ found[count // 2:count]), float(ups @ found[:count])
        if not low <= right / total <= high or abs(total - 700.0) > 14.0:
            problems.append(f"SPLIT {split}: the right foot carries {right / total:.2f} of {total:.0f} N, not "
                            f"{low} to {high} of 700")
    return problems


SPINE = """<mujoco model="spine">
  <compiler angle="radian"/>
  <worldbody>
    <body name="pelvis" pos="0 0 2">
      <freejoint name="root"/>
      <geom type="box" size="0.15 0.1 0.1" mass="8"/>
      <body name="low" pos="0 0 0.1">
        <joint name="low_flex" type="hinge" axis="1 0 0"/>
        <geom type="box" pos="0 0 0.05" size="0.05 0.05 0.05" mass="1"/>
        <body name="high" pos="0 0 0.1">
          <joint name="high_flex" type="hinge" axis="1 0 0"/>
          <geom type="box" pos="0.02 0 0.2" size="0.1 0.1 0.2" mass="20"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="high_flex" joint2="low_flex" polycoef="0 0.5 0 0 0"/>
  </equality>
</mujoco>
"""


def own_moment_problems():
    """A spine of two levels, the upper following the lower at half its turn, the body 2 m up and turning: the leader's
    net moment is its own joint's and half its follower's (the couplings' map), its own moment its joint's alone (the
    moment across that one joint, as a single back joint's inverse dynamics gives it)."""
    from motor import contact
    from motor import model
    from motor import muscles
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "spine.xml"
        path.write_text(SPINE)
        found = model.built(path, 9.81)
    ground = np.array([[[-5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [5.0, 0.0, -5.0]]])
    aloft = contact.Skin(np.zeros((1, 3)), np.ones((1, 1)), [np.eye(4)], [model.body_index(found, "pelvis")],
                         [np.eye(4)])  # one skin point at the pelvis, 2 m over the ground: nothing touches
    physics = muscles.Physics(found, ground, aloft)
    qpos = found.qpos0.copy()
    qpos[7] = 0.3
    qpos = physics.couplings.apply(qpos)
    qvel, qacc = np.zeros(found.nv), np.zeros(found.nv)
    qvel[6], qvel[7] = 1.0, 0.5
    qacc[6], qacc[7] = 4.0, 2.0
    needed = physics.needed(qpos, qvel, qacc)
    problem = physics.problem(needed)
    share = physics.share_of(problem, np.zeros(len(problem["costs"])))
    problems = []
    if not np.allclose(share["moments"], [needed[6] + 0.5 * needed[7]], atol=1e-6):
        problems.append(f"the leader's net moment {share['moments']}, not {needed[6] + 0.5 * needed[7]}")
    if not np.allclose(share["own moments"], [needed[6]], atol=1e-6):
        problems.append(f"the leader's own moment {share['own moments']}, not its joint's {needed[6]}")
    return problems


CHECKS = (the_ground_carries_a_still_body, rounding_moves_no_activation, ground_frames_are_well_posed)


def in_newton():
    """The checks, in Newton's own Python."""
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(in_newton())
