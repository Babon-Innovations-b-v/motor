"""myo: the person's file written from a small body laid out as MyoFullBody's wheel has it (a floor, a body with two
joints, a muscle whose class gives its parameters, names with '-'), sized to a toy SOMA (each thigh and shin a box),
compiles in MuJoCo and builds in Newton with the
same joints, one joint a body, no '-' in a name, no floor, the person's mass, and every muscle's gain and bias in
Newton's model the published model's (#60's import fixes); with the fit's constants given, an inside joint's
body sits moved by its found move, and one sizing (`sized_body`) writes each file as a sizing of its own would; the neck and head carry MyoHead's two joints, the torso the thoracic three, and the
girdle's coupling to the arm's elevation is gone, its joint on its own range, and its phantom follows it at -1
(`myo._movable`); a finger body laid at its joints' rest turns (`myo._rested`) stands at zero where it stood at those
turns and reaches every pose as before, its ranges moved by them; each foot turns on its shin about the shin's long
axis at the ankle, before the ankle's flexion, toes in for a positive turn on both sides, on `FEET`' ranges, and at
no twist stands as it did; each finger's MCP flexion bends back to `KNUCKLES`' extension, its flexion and every other
joint's range kept; each knee's shin turn (knee_angle_rotation3) leads on its published range, no longer set by the
flexion, the knee's other couplings kept (`KNEE`); the neck bends and flexes at its turn's place (`CERVICAL`); each
axis' L3/L4 joint leads, the levels above it tied to it in the model's ratios, so every pose of the old ties is
reached as before, L4/L5 still tied to lumbar5 (`LUMBAR`); the torso's lumbar wrap and its side site stand on
their vertebra where they stood, its tendon as long at the zero pose, the torso's other geoms left (`LUMBAR_WRAPS`);
and each joint of `RANGES` has its published range,
both sides. It runs in Newton's own Python
(MOTOR's own Python, the pins in pyproject.toml).

Breaks (each applied once with the env linked into a sandbox home, seen red, reverted), 2026-10-28:
- person_file's `split(full_muscles(renamed(xml), final))` made `full_muscles(renamed(xml), final)`: a body keeps two
  joints.
- person_file's `split(full_muscles(renamed(xml), final))` made `split(full_muscles(xml, final))`: names keep '-'.
- person_file's `split(full_muscles(renamed(xml), final))` made `split(renamed(xml))`: the muscle's partial
  gainprm loses its class's range and Newton refuses it ("lmin<1<lmax required in muscle").
- person_file's `spec.delete(geom)` made `pass`: the floor stays.
- (2026-10-29) inside_moved's `body.pos = body.pos + joints[filed]` made `pass`: the talus stays where it was.
- (2026-10-31) _fresh's `movable(spec)` made `pass`: no neck or thoracic joint, the girdle still coupled.
- (2026-10-31) _girdle's `elif stem in PHANTOMS:` made `elif False:`: the phantom stays tied to shoulder_elv.
- (2026-11-01) rested's `_carried(joint, moves[index + 1:])` made `pass`: the first of two joints turns about its old
  axis, the tip lands 0.4 to 3.3 mm off.
- (2026-11-01) rested's `- turns[index]` made `- 0.0`: the range stays where it was.
- (2026-11-01) _feet's twist added after the ankle's flexion (the re-adding loop moved before it): the twist's axis
  tips with the foot's flexion, off the shin.
- (2026-11-01) _feet's `(-1.0 if side == "l" else 1.0)` made `1.0`: the left toes turn out for a positive twist.
- (2026-11-01) _knuckles' `joint.range = [...]` made `pass`: the MCP flexion stays at 0.
- (2026-11-01) KNUCKLES' stems made every `mcp{finger}_{turn}` (abduction too): the abduction's range moves.
- (2026-11-04) _knee's `spec.delete(equality)` made `pass`: the shin's turn stays tied to the flexion.
- (2026-11-04) _knee's `equality.name1[:-2] == follower and` dropped: the knee's sideways turn loses its coupling too.
- (2026-11-04) _lumbar's `spec.delete(own)` made `pass`: L3/L4 stays tied to flex_extension.
- (2026-11-04) _lumbar's `float(tie.data[1]) / ratio` made `float(tie.data[1])`: the levels above L3/L4 bend at
  the wrong share of it, off the old ties' poses.
- (2026-11-04) _cervical's `pos=turn.pos` made `pos=[0.0, 0.0, 0.0]`: the neck bends at the neck body's origin, off
  its turn's place.
- (2026-11-04) _ranges' `joint.range = span` made `pass`: the hip's rotation and the toes keep their old ranges.
- (2026-11-07) _lumbar_wraps' `_laid_on(...)` made `pass`: the cylinder stays on the torso.
- (2026-11-07) _laid_on's `_turn(at.quat).apply(place)` made `place`: the cylinder lands off its place, the tendon
  longer.
- (2026-11-06) Sized.spec's `self.sized.copy()` made `self.sized`: the first file's spec is written on again, the
  lunate's record differs.

Run: .venv/bin/python motor/myo/test_the_persons_file_builds_in_newton_as_mujoco_compiles_it.py
"""
import pathlib
import re
import sys
import tempfile

INSIDE = pathlib.Path(__file__).resolve().parents[1]
# Each side's chain: (body, its place in its parent's frame, its parent); x is mirrored for the right side.
CHAIN = (("femur", "0.1 0 -0.1", "pelvis"), ("tibia", "0 0 -0.4", "femur"), ("talus", "0 0 -0.4", "tibia"),
         ("toes", "0 0.1 -0.05", "talus"), ("humerus", "0.2 0 0.4", "torso"), ("ulna", "0 0 -0.3", "humerus"),
         ("lunate", "0 0 -0.25", "ulna"))
KG = 50.0
# The toy's bone groups: each side's thigh and shin (bone_groups.GROUPS' shape); the rest take their parent's scale.
TABLE = {f"{side}{segment}": ((f"{side}{segment}", f"{side}{end}"), (f"{start}_{letter}", f"{finish}_{letter}"),
                              f"{body}_{letter}", (f"{body}_{letter}",))
         for side, letter in (("Left", "l"), ("Right", "r"))
         for segment, end, start, finish, body in (("Leg", "Shin", "hip", "knee", "femur"),
                                                   ("Shin", "Foot", "knee", "ankle", "tibia"))}
# The toy SOMA's joints (own frame, +x left; the right side mirrored): thigh and shin 0.44 m.
JOINTS = {"LeftLeg": (0.1, 1.0, 0.0), "LeftShin": (0.1, 0.56, 0.0), "LeftFoot": (0.1, 0.12, 0.0),
          "LeftArm": (0.2, 1.4, 0.0)}


# A landmark at each SOMA joint (`j_<joint>`), on the model's body at that joint.
ON_BODY = {f"j_{side}{segment}": f"{body}_{side[0].lower()}" for side in ("Left", "Right")
           for segment, body in (("Leg", "femur"), ("Shin", "tibia"), ("Foot", "talus"))}


def toy_rig():
    """The toy SOMA (its joints, a 0.04 m box along each stretch and a point face a landmark at each joint), and its
    landmark table ({name: row}, `landmarks._table.LANDMARKS`' shape)."""
    import numpy as np
    places = {**JOINTS, **{"Right" + name[4:]: (-x, y, z) for name, (x, y, z) in JOINTS.items()}}
    names = list(places)
    bind = np.tile(np.eye(4), (len(names), 1, 1))
    bind[:, :3, 3] = [places[name] for name in names]
    unit = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    points = [unit * [0.02, 0.15, 0.02] + (np.asarray(places[start]) + places[end]) / 2
              for (start, end), _, _, _ in TABLE.values()]
    faces = [faces + 8 * index for index in range(len(points))]
    table, count = {}, 8 * len(points)
    for name, body in ON_BODY.items():
        points.append(np.array([places[name[2:]]], dtype=float))
        faces.append(np.array([[count, count, count]]))
        table[name] = (len(np.concatenate(faces)) - 1, (1.0, 0.0, 0.0), "Hips", body, (0, 0, 0), (0, 0, 0), 1.0, True)
        count += 1
    rig = {"names": names, "bind": bind, "points": np.concatenate(points), "faces": np.concatenate(faces),
           "weights": np.zeros((count, len(names)))}
    return rig, table


def toy_body():
    """The toy body's MJCF as MyoFullBody's wheel has it: a floor, every body a sphere of 1 kg (a moving body needs a
    mass), the spine with two joints and carrying a mesh up to 1.6 m, the torso on it (the neck, head and a girdle
    joint and its phantom coupled to an arm lift on it), each toes body a mesh down to 0 m, one muscle
    of a class with partial parameters from the pelvis to the left tibia, names with '-'."""
    sides = []
    for side, mirror in (("l", 1.0), ("r", -1.0)):
        for body, place, parent in CHAIN:
            x, y, z = (float(part) for part in place.split())
            mesh = '<geom type="mesh" mesh="sole" mass="0.1"/>' if body == "toes" else ""
            mesh += '<site name="rect-fem-insertion" pos="0 0.05 -0.1"/>' if (body, side) == ("tibia", "l") else ""
            sides.append((f"{body}_{side}", f"{mirror * x:g} {y:g} {z:g}",
                          parent if parent in ("pelvis", "torso") else f"{parent}_{side}", mesh))

    def children(parent):
        return "".join(f'<body name="{name}" pos="{place}"><joint type="hinge" axis="1 0 0"/>'
                       f'<geom type="sphere" size="0.03" mass="1"/>{mesh}{children(name)}</body>'
                       for name, place, mine, mesh in sides if mine == parent)
    return f"""<mujoco model="toybody">
  <compiler angle="radian"/>
  <default>
    <default class="muscle">
      <muscle ctrllimited="true" ctrlrange="0 1" scale="200"/>
    </default>
  </default>
  <asset>
    <mesh name="head" vertex="-0.1 -0.1 0  0.1 -0.1 0  0 0.1 0  0 0 0.5"/>
    <mesh name="sole" vertex="-0.03 -0.03 0  0.03 -0.03 0  0 0.03 0  0 0 -0.05"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <site name="rect-fem-origin" pos="0.1 0.1 -0.05"/>
      {children("pelvis")}
      <body name="spine" pos="0 0 0.1">
        <joint name="lumbar-flex" type="hinge" axis="1 0 0"/>
        <joint name="lumbar-bend" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <geom type="mesh" mesh="head" mass="0.1"/>
        <body name="torso">
        <geom type="sphere" size="0.05" mass="1"/>
        {children("torso")}
        <body name="neck" pos="0 0 0.3">
          <geom type="sphere" size="0.03" mass="1"/>
          <body name="head" pos="0 0 0.1"><geom type="sphere" size="0.08" mass="1"/></body>
        </body>
        <body name="clavicle_l" pos="0.1 0 0.3">
          <joint name="sternoclavicular_r2_l" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.02" mass="1"/>
          <body name="clavphant_l" pos="0.05 0 0">
          <joint name="unrotscap_r2_l" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.02" mass="1"/>
          <body name="blade_l">
            <joint name="shoulder_elv_l" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.02" mass="1"/>
          </body>
          </body>
        </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="sternoclavicular_r2_l" joint2="shoulder_elv_l" polycoef="0 -0.242 0 0 0"/>
    <joint joint1="unrotscap_r2_l" joint2="shoulder_elv_l" polycoef="0 0.242 0 0 0"/>
  </equality>
  <tendon>
    <spatial name="rect_fem">
      <site site="rect-fem-origin"/>
      <site site="rect-fem-insertion"/>
    </spatial>
  </tendon>
  <actuator>
    <muscle name="rect_fem" class="muscle" tendon="rect_fem" force="500"/>
  </actuator>
</mujoco>
"""




def written(folder, constants=None):
    """The person's file written from the toy wheel in `folder` (with the fit's `constants`), its text, its record and
    its two compiles: MuJoCo's and Newton's."""
    import mujoco
    from motor import model
    from motor import myo
    from motor import sizing
    source = pathlib.Path(folder) / myo.SOURCE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(toy_body())
    path = pathlib.Path(folder) / "person.xml"
    from motor.landmarks._table import LANDMARKS
    kept = dict(LANDMARKS)
    try:
        rig, table = toy_rig()
        LANDMARKS.clear()
        LANDMARKS.update(table)
        record, _ = myo.person_file(folder, sizing.Size(rig, TABLE), KG, path, constants)
    finally:
        LANDMARKS.clear()
        LANDMARKS.update(kept)
    return path.read_text(), record, mujoco.MjModel.from_xml_path(str(path)), model.built(path, 9.81)


def the_written_file_builds_as_it_compiles():
    """MuJoCo's compile and Newton's build of the written file have the same joints; every body one joint at most; no
    name with '-'; nothing in the world body (the floor gone); the record's mass the person's."""
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        text, record, compiled, built = written(folder)
    problems = [] if (compiled.nq, compiled.nv) == (built.nq, built.nv) else [
        f"MuJoCo nq {compiled.nq} nv {compiled.nv}, Newton nq {built.nq} nv {built.nv}"]
    joints = np.bincount(compiled.jnt_bodyid, minlength=compiled.nbody)
    problems += [] if joints.max() <= 1 else [f"{int((joints > 1).sum())} bodies with several joints"]
    named = [name for name in re.findall(r'\bname="([^"]*)"', text) if "-" in name]
    problems += [] if not named else [f"names with '-': {named}"]
    problems += [] if not (compiled.geom_bodyid == 0).any() else ["a geom in the world body (the floor)"]
    return problems + ([] if abs(record["kg"] - KG) < 1e-3 else [f"record kg {record['kg']}"])


def the_constants_move_the_file():
    """With the left talus's inside move 1 cm along z: the talus sits that far off its place in the file written
    without it."""
    import mujoco
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        plain = written(folder)[2]
        fitted = written(folder, {"joints": {"talus_l": [0.0, 0.0, 0.01]}})[2]

    def place(model, name):
        return model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
    return [] if np.allclose(place(fitted, "talus_l"), place(plain, "talus_l") + [0.0, 0.0, 0.01]) else [
        f"talus at {place(fitted, 'talus_l')} against {place(plain, 'talus_l')}"]


def one_sizing_writes_each_file_as_its_own_would():
    """One sized toy (`myo.sized_body`) written without constants, then with the left talus moved and with the left
    lunate moved: each file's text and record as `person_file` writes it from a sizing of its own."""
    from motor import myo
    from motor import sizing
    from motor.landmarks._table import LANDMARKS
    cases = (None, {"joints": {"talus_l": [0.0, 0.0, 0.01]}}, {"joints": {"lunate_l": [0.01, 0.0, 0.0]}})
    kept, problems = dict(LANDMARKS), []
    with tempfile.TemporaryDirectory() as folder:
        source = pathlib.Path(folder) / myo.SOURCE
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(toy_body())
        path = pathlib.Path(folder) / "person.xml"
        try:
            rig, table = toy_rig()
            LANDMARKS.clear()
            LANDMARKS.update(table)
            alone = []
            for constants in cases:
                record, _ = myo.person_file(folder, sizing.Size(rig, TABLE), KG, path, constants)
                alone.append((path.read_text(), record))
            body = myo.sized_body(folder, sizing.Size(rig, TABLE), KG)
            for constants, (text, record) in zip(cases, alone):
                again, _ = myo.written(body, path, constants)
                problems += [] if path.read_text() == text else [f"{constants}: the file differs"]
                problems += [] if again == record else [f"{constants}: the record differs"]
        finally:
            LANDMARKS.clear()
            LANDMARKS.update(kept)
    return problems


def every_muscle_keeps_its_published_parameters_in_newton():
    """The muscle's gain and bias parameters in Newton's model are the published model's compile (to 1e-6)."""
    import mujoco
    from motor import myo
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        _, _, _, built = written(folder)
        source = mujoco.MjModel.from_xml_path(str(pathlib.Path(folder) / myo.SOURCE))
    problems = []
    for field in ("gainprm", "biasprm"):
        theirs, ours = getattr(source, "actuator_" + field)[:, :9], getattr(built, "actuator_" + field)[:, :9]
        if theirs.shape != ours.shape or not np.allclose(theirs, ours, rtol=1e-6, atol=1e-6):
            problems.append(f"{field}: Newton {ours.round(4).tolist()}, published {theirs.round(4).tolist()}")
    return problems


def the_neck_thorax_and_girdle_move():
    """The written file's neck, head and torso carry `myo._movable`'s joints with their ranges, and the girdle's
    joint leads on GIRDLE's range, its phantom the one coupling left, at -1 on it."""
    import numpy as np
    from motor.myo._movable import CERVICAL, GIRDLE, NECK, THORACIC
    with tempfile.TemporaryDirectory() as folder:
        compiled = written(folder)[2]
    wanted = {joint["name"]: joint["range"] for joint in NECK.values()}
    wanted.update({name: span for name, _, span in THORACIC[3]})
    wanted.update({name: span for name, _, span in CERVICAL[2]})
    wanted["sternoclavicular_r2_l"] = GIRDLE["sternoclavicular_r2"]
    names = [compiled.joint(joint).name for joint in range(compiled.njnt)]
    problems = [f"{name}: missing" for name in wanted if name not in names]
    problems += [f"{name}: range {compiled.joint(name).range} against {span}" for name, span in wanted.items()
                 if name in names and not np.allclose(compiled.joint(name).range, span)]
    tied = [(compiled.joint(int(compiled.eq_obj1id[at])).name, compiled.joint(int(compiled.eq_obj2id[at])).name,
             compiled.eq_data[at, :5].tolist()) for at in range(compiled.neq)]
    turn = compiled.joint(NECK["neck"]["name"]).id
    problems += [f"{name}: at {compiled.jnt_pos[compiled.joint(name).id]}, its turn at {compiled.jnt_pos[turn]}"
                 for name, _, _ in CERVICAL[2] if name in names
                 and not np.allclose(compiled.jnt_pos[compiled.joint(name).id], compiled.jnt_pos[turn])]
    return problems + ([] if tied == [("unrotscap_r2_l", "sternoclavicular_r2_l", [0.0, -1.0, 0.0, 0.0, 0.0])] else
                       [f"couplings {tied}"])


FINGER = """<mujoco><compiler angle="radian"/><worldbody><body name="hand" pos="0.1 0.2 0.3" quat="0.9 0.1 0.3 0.2">
  <geom type="sphere" size="0.01" mass="0.1"/>
  <body name="finger" pos="0.02 -0.03 0.01" quat="0.8 -0.2 0.1 0.4">
    <joint name="flex" type="hinge" axis="1 0.2 0" pos="0.003 0.001 0" range="0 1.5"/>
    <joint name="spread" type="hinge" axis="0 0.3 1" pos="0 -0.002 0.004" range="-0.3 0.3"/>
    <geom type="sphere" size="0.01" mass="0.01"/>
    <body name="tip" pos="0 -0.04 0"><joint name="bend" type="hinge" axis="1 0 0" range="0 1.5"/>
      <geom type="sphere" size="0.01" mass="0.01"/></body>
  </body></body></worldbody></mujoco>"""


def a_rested_finger_reaches_every_pose_as_before():
    """A finger body with two hinges (and a tip on a third) laid at rest turns (flex 0.4, spread -0.2, bend 0.3): at
    each of three poses less the rest turns, the finger and tip stand where the published file put them at those poses
    (to 1e-9 m and in turn); each laid joint's range is its own less its rest turn."""
    import mujoco
    import numpy as np
    from motor.myo._rested import rested
    rest = {"flex": 0.4, "spread": -0.2, "bend": 0.3}
    published = mujoco.MjModel.from_xml_string(FINGER)
    spec = mujoco.MjSpec.from_string(FINGER)
    rested(spec, rest)
    laid = spec.compile()
    problems = []
    for pose in ([0.4, -0.2, 0.3], [0.0, 0.0, 0.0], [1.1, 0.25, 0.9]):
        before, after = mujoco.MjData(published), mujoco.MjData(laid)
        before.qpos[:] = pose
        after.qpos[:] = np.asarray(pose) - [rest["flex"], rest["spread"], rest["bend"]]
        mujoco.mj_kinematics(published, before)
        mujoco.mj_kinematics(laid, after)
        if not (np.allclose(before.xpos, after.xpos, atol=1e-9) and np.allclose(before.xmat, after.xmat, atol=1e-9)):
            problems.append(f"at {pose}: places off by {np.abs(before.xpos - after.xpos).max():.2e} m")
    for name, turn in rest.items():
        if not np.allclose(laid.joint(name).range, published.joint(name).range - turn):
            problems.append(f"{name}: range {laid.joint(name).range} against {published.joint(name).range - turn}")
    return problems


LEG = """<mujoco><compiler angle="radian"/><default><default class="myolegs"><joint damping="0.5"/></default></default>
  <worldbody>{legs}</worldbody></mujoco>"""
SIDE = """<body name="tibia_{s}" pos="0 1 {side}"><geom type="sphere" size="0.02" mass="1"/>
  <body name="talus_{s}" pos="-0.01 -0.4 0"><joint name="ankle_angle_{s}" type="hinge" axis="0.1 0.17 {z}" range="-0.7 0.52"/>
    <geom type="sphere" size="0.02" mass="0.1"/>
    <body name="calcn_{s}" pos="-0.05 -0.04 0"><joint name="subtalar_angle_{s}" type="hinge" axis="-0.79 -0.6 -0.12"
      range="-0.35 0.35"/><geom type="sphere" size="0.02" mass="0.1"/>
      <body name="toes_{s}" pos="0.18 0 0"><geom type="sphere" size="0.01" mass="0.05"/></body></body></body></body>"""


def a_foot_twists_on_its_shin():
    """A MyoFullBody-shaped leg each side (tibia, talus with the ankle hinge, calcaneus with the subtalar one; x
    forward, y up the shin, z to the body's right as OpenSim's, the left side at -z): the talus's first joint is `ankle_rotation`, its world axis along the
    knee-to-ankle line at a flexed ankle too; a positive twist moves the toes toward the middle on both sides; the
    ranges are FEET's; at no twist every body stands as the published leg's."""
    import mujoco
    import numpy as np
    from motor.myo._movable import FEET, _feet
    text = LEG.format(legs="".join(SIDE.format(s=side, side=-0.1 if side == "l" else 0.1, z=0.98 if side == "l" else -0.98)
                                   for side in "lr"))
    published = mujoco.MjModel.from_xml_string(text)
    spec = mujoco.MjSpec.from_string(text)
    _feet(spec)
    laid = spec.compile()
    problems = []
    for side, middle in (("l", 1.0), ("r", -1.0)):
        talus = laid.body(f"talus_{side}").id
        first = laid.joint(int(laid.body_jntadr[talus])).name
        problems += [] if first == f"ankle_rotation_{side}" else [f"{side}: the talus's first joint is {first}"]
        data = mujoco.MjData(laid)
        data.qpos[laid.jnt_qposadr[laid.joint(f"ankle_angle_{side}").id]] = 0.4
        mujoco.mj_kinematics(laid, data)
        shin = data.xpos[laid.body(f"tibia_{side}").id] - data.xpos[talus]
        twist_axis = data.xaxis[laid.joint(f"ankle_rotation_{side}").id]
        if abs(abs(twist_axis @ shin) / np.linalg.norm(shin) - 1.0) > 1e-9:
            problems.append(f"{side}: twist axis {twist_axis} off the shin {shin} at a flexed ankle")
        toes = laid.body(f"toes_{side}").id
        before = data.xpos[toes][2]
        data.qpos[laid.jnt_qposadr[laid.joint(f"ankle_rotation_{side}").id]] = 0.1
        mujoco.mj_kinematics(laid, data)
        if (data.xpos[toes][2] - before) * middle <= 0:
            problems.append(f"{side}: a positive twist moves the toes {data.xpos[toes][2] - before:+.4f} m across, not in")
        ranges = {"ankle_rotation": FEET["twist"][1], **FEET["ranges"]}
        problems += [f"{side}: {stem} range {laid.joint(f'{stem}_{side}').range}" for stem, span in ranges.items()
                     if not np.allclose(laid.joint(f"{stem}_{side}").range, span)]
    rest, plain = mujoco.MjData(laid), mujoco.MjData(published)
    mujoco.mj_kinematics(laid, rest)
    mujoco.mj_kinematics(published, plain)
    return problems + ([] if np.allclose(rest.xpos, plain.xpos, atol=1e-12) else ["at no twist the feet moved"])


HAND = """<mujoco><compiler angle="radian"/><worldbody><body name="hand"><geom type="sphere" size="0.01" mass="0.1"/>
  {fingers}</body></worldbody></mujoco>"""
KNUCKLE = """<body name="finger{finger}_{side}"><geom type="sphere" size="0.005" mass="0.01"/>
  <joint name="mcp{finger}_flexion_{side}" type="hinge" axis="1 0 0" range="0 1.5708"/>
  <joint name="mcp{finger}_abduction_{side}" type="hinge" axis="0 0 1" range="-0.261799 0.261799"/></body>"""


def the_knuckles_bend_back():
    """A hand with MyoFullBody's four finger MCP joints each side (flexion 0 to 90 degrees, abduction 15 each way):
    each flexion's range is -KNUCKLES' extension to the published 90 degrees; each abduction keeps its range."""
    import mujoco
    import numpy as np
    from motor.myo._movable import KNUCKLES, _knuckles
    text = HAND.format(fingers="".join(KNUCKLE.format(finger=finger, side=side) for finger in "2345" for side in "lr"))
    published = mujoco.MjModel.from_xml_string(text)
    spec = mujoco.MjSpec.from_string(text)
    _knuckles(spec)
    laid = spec.compile()
    wanted = {name: ([-KNUCKLES["extension"], 1.5708] if "flexion" in name else published.joint(name).range)
              for name in (published.joint(joint).name for joint in range(published.njnt))}
    return [f"{name}: range {laid.joint(name).range} against {span}" for name, span in wanted.items()
            if not np.allclose(laid.joint(name).range, span)]


KNEE = """<mujoco><compiler angle="radian"/><worldbody>{legs}</worldbody><equality>{ties}</equality></mujoco>"""
KNEE_SIDE = """<body name="femur_{s}"><geom type="sphere" size="0.02" mass="1"/><body name="tibia_{s}" pos="0 -0.4 0">
  <joint name="knee_angle_{s}" type="hinge" axis="0 0 -1" range="0 2.0944"/>
  <joint name="knee_angle_rotation2_{s}" type="hinge" axis="1 0 0" range="-0.00167821 0.0335354"/>
  <joint name="knee_angle_rotation3_{s}" type="hinge" axis="0 1 0" range="-0.262788 0.262788"/>
  <geom type="sphere" size="0.02" mass="0.5"/></body></body>"""
KNEE_TIES = """<joint joint1="knee_angle_rotation2_{s}" joint2="knee_angle_{s}" polycoef="0 0.0791 -0.0328478 -0.0252183 0.0108321"/>
  <joint joint1="knee_angle_rotation3_{s}" joint2="knee_angle_{s}" polycoef="0 0.369499 -0.169478 0.0251643 0"/>"""


def the_knee_turns_the_shin_of_its_own():
    """A MyoFullBody-shaped knee each side (the flexion, the sideways turn and the shin's turn, both tied to the
    flexion by the published polynomials): the shin's turn is tied to nothing and keeps its published range; the
    sideways turn stays tied to the flexion."""
    import mujoco
    import numpy as np
    from motor.myo._movable import _knee
    text = KNEE.format(legs="".join(KNEE_SIDE.format(s=side) for side in "lr"),
                       ties="".join(KNEE_TIES.format(s=side) for side in "lr"))
    spec = mujoco.MjSpec.from_string(text)
    _knee(spec)
    tied = sorted(equality.name1 for equality in spec.equalities)
    laid = spec.compile()
    problems = [] if tied == ["knee_angle_rotation2_l", "knee_angle_rotation2_r"] else [f"tied: {tied}"]
    return problems + [f"{side}: the shin's turn's range {laid.joint(f'knee_angle_rotation3_{side}').range}"
                       for side in "lr"
                       if not np.allclose(laid.joint(f"knee_angle_rotation3_{side}").range, [-0.262788, 0.262788])]


SPINE = """<mujoco><compiler angle="radian"/><worldbody><body name="lumbar5">
  <joint name="flex_extension" type="hinge" axis="0 0 1" range="-1.35 0.75"/><geom type="sphere" size="0.02" mass="1"/>
  {levels}</body>{limbs}</worldbody><equality>{ties}</equality></mujoco>"""
SPINE_LEVELS = (("lumbar4", "L4_L5", 0.185), ("lumbar3", "L3_L4", 0.204), ("lumbar2", "L2_L3", 0.231),
                ("lumbar1", "L1_L2", 0.255))
LIMB = """<body name="limb_{s}" pos="0 0 -1"><geom type="sphere" size="0.02" mass="1"/>
  <joint name="hip_rotation_{s}" type="hinge" axis="0 1 0" range="-0.698132 0.698132"/>
  <joint name="mtp_angle_{s}" type="hinge" axis="1 0 0" range="-0.523599 0.523599"/></body>"""


def the_upper_lumbar_leads_and_ranges_are_human():
    """A MyoFullBody-shaped lumbar spine (lumbar5's leader, each level above tied to it by its ratio) and two limbs:
    L3/L4 leads, L4/L5 stays tied to lumbar5, L2/L3 and L1/L2 are tied to L3/L4 so that every pose of the old ties
    stands as before; each limb's joint of `RANGES` has its range; the lumbar wrap rides its vertebra
    (`lumbar_wrap_problems`)."""
    import mujoco
    import numpy as np
    from motor.myo._movable import RANGES, _lumbar, _ranges
    levels = "".join(f'<body name="{body}" pos="0 0.05 0"><joint name="{level}_FE" type="hinge" axis="0 0 1"/>'
                     f'<geom type="sphere" size="0.02" mass="1"/>' for body, level, _ in SPINE_LEVELS)
    levels += "</body>" * len(SPINE_LEVELS)
    ties = "".join(f'<joint joint1="{level}_FE" joint2="flex_extension" polycoef="0 {share} 0 0 0"/>'
                   for _, level, share in SPINE_LEVELS)
    text = SPINE.format(levels=levels, limbs="".join(LIMB.format(s=side) for side in "lr"), ties=ties)
    spec = mujoco.MjSpec.from_string(text)
    _lumbar(spec)
    _ranges(spec)
    laid = spec.compile()
    tied = sorted((equality.name1, equality.name2) for equality in spec.equalities)
    problems = [] if tied == [("L1_L2_FE", "L3_L4_FE"), ("L2_L3_FE", "L3_L4_FE"), ("L4_L5_FE", "flex_extension")] \
        else [f"tied: {tied}"]
    shares = dict((level, share) for _, level, share in SPINE_LEVELS)
    for equality in spec.equalities:
        lead = shares.get(equality.name2.removesuffix("_FE"), 1.0)
        if not np.isclose(lead * equality.data[1], shares[equality.name1.removesuffix("_FE")]):
            problems.append(f"{equality.name1}: {equality.data[1]} of {equality.name2}, off the old ties' pose")
    problems += lumbar_wrap_problems()
    return problems + [f"{stem}_{side}: range {laid.joint(f'{stem}_{side}').range} against {RANGES[stem]}"
                       for stem in ("hip_rotation", "mtp_angle") for side in "lr"
                       if not np.allclose(laid.joint(f"{stem}_{side}").range, RANGES[stem])]


WRAPPED = """<mujoco><compiler angle="radian"/><default><default class="wrap"><geom contype="0" conaffinity="0"
  group="3"/></default></default><worldbody><body name="sacrum"><geom type="sphere" size="0.02" mass="1"/>
  <site name="low" pos="0.05 -0.1 0"/>
  <body name="lumbar2" pos="0 0.1 0" quat="0.98 0.1 0.15 0"><joint name="L2_L3_FE" axis="0 0 1"/>
    <geom type="sphere" size="0.02" mass="1"/><site name="high" pos="0.06 0.05 0"/>
    <body name="lumbar1" pos="0.01 0.08 0" quat="0.97 0 0.1 0.2"><joint name="L1_L2_FE" axis="0 0 1"/>
      <geom type="sphere" size="0.02" mass="1"/>
      <body name="torso" pos="0.0025 -0.089 0" quat="0.96 0.2 0 0.1">
        <geom name="ribs" type="sphere" size="0.05" mass="1"/>
        <geom name="back_cylinder_l" class="wrap" type="cylinder" size="0.08 0.1" pos="-0.11 0.04 0.01"
          quat="0.9 0.1 0.3 0.2"/>
        <site name="back_cylinder_l_IL_L2_r_1_sidesite" pos="-0.04 -0.07 -0.12"/>
      </body></body></body></body></worldbody>
  <tendon><spatial name="IL"><site site="low"/>
    <geom geom="back_cylinder_l" sidesite="back_cylinder_l_IL_L2_r_1_sidesite"/>
    <site site="high"/></spatial></tendon></mujoco>"""


def lumbar_wrap_problems():
    """A torso on lumbar1 on lumbar2 carrying a wrap cylinder, its side site and a geom of its own, a tendon from the
    sacrum to lumbar2 wrapped on it: after `_lumbar_wraps` the cylinder and its side site are lumbar2's, standing where
    they stood at the zero pose (the tendon as long), the cylinder its size and class, the torso's own geom kept."""
    import mujoco
    import numpy as np
    from motor.myo._movable import LUMBAR_WRAPS, _lumbar_wraps
    before = mujoco.MjModel.from_xml_string(WRAPPED)
    spec = mujoco.MjSpec.from_string(WRAPPED)
    _lumbar_wraps(spec)
    after = spec.compile()
    datas = [mujoco.MjData(model) for model in (before, after)]
    for model, data in zip((before, after), datas):
        mujoco.mj_forward(model, data)
    problems = []
    for kind, name in ((mujoco.mjtObj.mjOBJ_GEOM, "back_cylinder_l"),
                       (mujoco.mjtObj.mjOBJ_SITE, "back_cylinder_l_IL_L2_r_1_sidesite")):
        ids = [mujoco.mj_name2id(model, kind, name) for model in (before, after)]
        geom = kind == mujoco.mjtObj.mjOBJ_GEOM
        owner = (after.geom_bodyid if geom else after.site_bodyid)[ids[1]]
        if after.body(owner).name != LUMBAR_WRAPS["back_cylinder_l"]:
            problems.append(f"{name}: on {after.body(owner).name}")
        places = [(data.geom_xpos if geom else data.site_xpos)[at] for data, at in zip(datas, ids)]
        turns = [(data.geom_xmat if geom else data.site_xmat)[at] for data, at in zip(datas, ids)]
        if not (np.allclose(*places, atol=1e-9) and np.allclose(*turns, atol=1e-9)):
            problems.append(f"{name}: at {places[1]} against {places[0]}")
    cylinder = after.geom("back_cylinder_l")
    if not (np.allclose(cylinder.size, before.geom("back_cylinder_l").size) and cylinder.contype[0] == 0
            and cylinder.group[0] == 3):
        problems.append(f"back_cylinder_l: size {cylinder.size}, contype {cylinder.contype}, group {cylinder.group}")
    if after.body(after.geom("ribs").bodyid[0]).name != "torso":
        problems.append("ribs: moved off the torso")
    if not np.isclose(datas[0].ten_length[0], datas[1].ten_length[0], atol=1e-9):
        problems.append(f"IL: {datas[1].ten_length[0]} against {datas[0].ten_length[0]}")
    return problems


CHECKS = (the_upper_lumbar_leads_and_ranges_are_human,
          the_knee_turns_the_shin_of_its_own, the_knuckles_bend_back,
          a_foot_twists_on_its_shin,
          a_rested_finger_reaches_every_pose_as_before,
          the_neck_thorax_and_girdle_move, the_constants_move_the_file, the_written_file_builds_as_it_compiles,
          every_muscle_keeps_its_published_parameters_in_newton, one_sizing_writes_each_file_as_its_own_would)


def in_newton():
    """The checks, in Newton's own Python."""
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(in_newton())
