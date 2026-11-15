"""records: the record's pieces come out as worked by hand: a pose as place and quaternion, a segment's turn against
its parent in degrees, a mesh's near vertices merged with its flat faces dropped, RRA's `adjusted` record's pelvis
centimetres, least share and mass percent, the arrays named for the person, and a swaying quiet stance's pushes one a
foot, summing to its weight, each centre on the ground under its foot and none jumping frame to frame (`push_faults`,
on the solve's own pushes through `steadied`). It runs in Newton's own Python
(MuJoCo pinned there; MOTOR's own Python, the pins in pyproject.toml).

Breaks (each applied once with the env linked into a sandbox home, seen red, reverted), 2026-10-28:
- merged's flat-face filter `faces = faces[(faces[:, 0] != faces[:, 1]) & ...]` dropped: a face merged flat stays.
- adjustment_record's `pelvis = 100 * np.linalg.norm(moved_by, axis=1)` made `np.linalg.norm(moved_by, axis=1)`: the
  pelvis' move in metres, not centimetres.
- local_turns' `worlds[parent][:3, :3].T @ worlds[index][:3, :3]` made `worlds[index][:3, :3]`: a turn not against
  its parent.
2026-11-05:
- muscles.parts_of's `segment.removesuffix("ToeBase").removesuffix("Foot") + "Foot"` made `segment`: each toes' push
  is its own part, four arrows where two belong.
- muscles.touches' `sizes @ under / sizes.sum()` made `sizes @ under`: the centre is the push's newtons times the
  ground's points, metres off the ground and outside the foot.
- contact.pushes' `closeness(over[point]) * edge` made `edge`: a forefoot crossing GAP takes its whole share at once;
  steadied, a foot's push still jumps 2.5% of the weight and its centre 6.4 mm a frame (6.2% and 15.8 mm unsteadied).

- records.paths_of's `at[kept]` made `at`, contact.lowests' `heights[points]` made `heights`, local_turns'
  `above.transpose(0, 2, 1)` made `above`: each a one-pass piece no longer equal to the item-by-item one (2026-11-05).

Run: .venv/bin/python motor/records/test_a_record_keeps_poses_meshes_and_moves_as_worked_by_hand.py
"""
import pathlib
import sys
import tempfile

INSIDE = pathlib.Path(__file__).resolve().parents[1]


def placed(turn, place):
    """A turn and a place as a 4 x 4."""
    import numpy as np
    found = np.eye(4)
    found[:3, :3], found[:3, 3] = turn, place
    return found


def a_pose_and_a_turn_read_as_worked_by_hand():
    """A 90 degree turn about z at (1, 2, 3) is [1, 2, 3, 0, 0, sin 45, cos 45]; a child turned 30 degrees about x under
    a parent turned 90 about z is [30, 0, 0] against it; the root has no turn of its own."""
    import numpy as np
    from motor import records
    from scipy.spatial.transform import Rotation
    parent = placed(Rotation.from_euler("z", 90, degrees=True).as_matrix(), [1.0, 2.0, 3.0])
    child = parent @ placed(Rotation.from_euler("x", 30, degrees=True).as_matrix(), [0.0, 0.0, 0.0])
    pose = records.segment_poses([parent])[0]
    half = np.sqrt(0.5)
    problems = [] if np.allclose(pose, [1.0, 2.0, 3.0, 0.0, 0.0, half, half]) else [f"pose {pose}"]
    turns = records.local_turns([parent, child], [-1, 0])
    return problems + ([] if len(turns) == 1 and np.allclose(turns[0], [30.0, 0.0, 0.0]) else [f"turns {turns}"])


def near_vertices_merge_and_flat_faces_go():
    """Two vertices 1 mm apart (within BONE_CELL, 5 mm) become one; the face between them falls flat and goes; the
    other face stays on the three vertices left."""
    import numpy as np
    from motor import records
    points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    kept, faces = records.merged(points, np.array([[0, 1, 2], [0, 2, 3]]))
    return [] if len(kept) == 3 and len(faces) == 1 and sorted(faces[0]) == [0, 1, 2] else [
        f"{len(kept)} vertices, faces {faces.tolist()}"]


def the_adjusted_record_reads_in_its_units():
    """The pelvis moved 1 and 3 cm on two frames: median 2 cm and most 3 cm; frame shares 1 and 0.5: least 0.5; a
    mass share of 0.02: 2 %, RRA's recommended -0.5 and -0.25: -50 and -25 %; no segment turned."""
    import numpy as np
    from motor import records
    still = np.tile(np.eye(4), (2, 1, 1, 1))
    found = records.adjustment_record(np.array([[0.01, 0.0, 0.0], [0.03, 0.0, 0.0]]), 1.0, np.array([1.0, 0.5]),
                                      {"mass": 0.02, "mass recommended": [-0.5, -0.25], "torso": np.zeros(3)},
                                      still, still, ["Hips"], [1.0, 2.0])
    wanted = {"pelvis cm": [2.0, 3.0], "least frame share": 0.5, "mass %": 2.0, "largest turn deg": 0.0,
              "kept share": 1.0, "mass % recommended": [-50.0, -25.0]}
    return [f"{key}: {found[key]}, not {value}" for key, value in wanted.items() if found[key] != value]


def the_arrays_are_named_for_the_person():
    """Two frames of one body and one muscle path of two points: `<name>/bones` 2 x 1 x 7, `/paths` 4 x 3, `/counts`
    2 x 1, and each bone's mesh points and faces."""
    import numpy as np
    from motor import records
    found = records.arrays("walker", [[[0.0] * 7]] * 2, [[[0.0] * 7]] * 2, [np.zeros((2, 3))] * 2, [[2], [2]],
                           {"femur_l": (np.zeros((3, 3)), np.array([[0, 1, 2]]))})
    shapes = {name: array.shape for name, array in found.items()}
    wanted = {"walker/bones": (2, 1, 7), "walker/segments": (2, 1, 7), "walker/paths": (4, 3),
              "walker/counts": (2, 1), "walker/mesh/femur_l/points": (3, 3), "walker/mesh/femur_l/faces": (1, 3)}
    return [] if shapes == wanted else [f"arrays {shapes}"]


PULLEYED = """<mujoco model="pulleyed">
  <worldbody>
    <body name="fixed"><geom size="0.05"/><site name="one"/><site name="two" pos="0 0 0.1"/></body>
    <body name="free" pos="0.1 0.2 0.3"><freejoint/><geom size="0.05"/><site name="three"/>
      <site name="four" pos="0.1 0 0"/></body>
  </worldbody>
  <tendon>
    <spatial name="plain"><site site="one"/><site site="three"/></spatial>
    <spatial name="pulleyed"><site site="one"/><site site="three"/><pulley divisor="2"/><site site="two"/>
      <site site="four"/></spatial>
  </tendon>
</mujoco>
"""


def the_one_pass_pieces_equal_the_item_by_item_ones():
    """The record's one-pass pieces give bit for bit what one item at a time gave before them (#70, 2026-11-05): the
    muscles' paths gathered at once (a pulley's break left out, the tendons in the order asked) against each tendon's
    own slice, the poses and turns in one rotation call against one call each, and the skin's lowest points placed
    once against the skin placed again for each foot."""
    import mujoco
    import numpy as np
    from motor import contact
    from motor import records
    from motor import model as built_model
    from motor.ground import body_reach, heights_over
    from scipy.spatial.transform import Rotation
    problems = []
    pulleyed = mujoco.MjModel.from_xml_string(PULLEYED)
    data = mujoco.MjData(pulleyed)
    data.qpos[3:7] = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_quat(scalar_first=True)
    mujoco.mj_forward(pulleyed, data)
    eaches = []
    for tendon in (1, 0):
        span = slice(data.ten_wrapadr[tendon], data.ten_wrapadr[tendon] + data.ten_wrapnum[tendon])
        eaches.append(data.wrap_xpos.reshape(-1, 3)[span][data.wrap_obj.reshape(-1)[span] != -2])
    points, counts = records.paths_of(data, [1, 0])
    if not (np.array_equal(points, np.concatenate(eaches)) and counts.tolist() == [len(each) for each in eaches]
            and counts.tolist() == [4, 2]):
        problems.append(f"paths {counts.tolist()}, not {[len(each) for each in eaches]}")
    worlds = np.tile(np.eye(4), (5, 1, 1))
    worlds[:, :3, :3] = Rotation.random(5, random_state=7).as_matrix()
    worlds[:, :3, 3] = np.arange(15.0).reshape(5, 3)
    parents = [-1, 0, 1, 1, 3]
    each = [[*world[:3, 3], *Rotation.from_matrix(world[:3, :3]).as_quat()] for world in worlds]
    if not np.array_equal(records.segment_poses(worlds), np.array(each)):
        problems.append("poses differ from one call each")
    each = [Rotation.from_matrix(worlds[parent][:3, :3].T @ worlds[index][:3, :3]).as_rotvec(degrees=True)
            for index, parent in enumerate(parents) if parent >= 0]
    if not np.array_equal(records.local_turns(worlds, parents), np.array(each)):
        problems.append("turns differ from one call each")
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "stander.xml"
        path.write_text(STANDER)
        stander = built_model.built(path, 9.81)
    data = mujoco.MjData(stander)
    mujoco.mj_kinematics(stander, data)
    bodies = [built_model.body_index(stander, body) for body in FEET.values()]
    rng = np.random.default_rng(8)
    skin_points = rng.uniform([-0.2, -0.01, -0.2], [0.2, 0.3, 0.2], (200, 3))
    weights = rng.dirichlet(np.ones(len(bodies)), 200)
    skin = contact.Skin(skin_points, weights, [built_model.own_pose(stander, data, body) for body in bodies], bodies,
                        [np.eye(4)] * len(bodies))
    ground = np.array([[[-0.1, 0.0, -0.5], [0.5, 0.01, 0.5], [0.5, 0.0, -0.5]],
                       [[-0.1, 0.0, -0.5], [-0.1, 0.02, 0.5], [0.5, 0.01, 0.5]]])  # off the skin's left part
    parts = {"some": np.arange(0, 200, 3), "none": np.zeros(0, int), "off": np.flatnonzero(skin_points[:, 0] < -0.15)}
    placed = skin.placed(stander, data)

    def lowest(heights):
        return None if np.isnan(heights).all() else float(np.nanmin(heights))
    wanted = (lowest(heights_over(placed, ground, body_reach(placed))),
              {part: lowest(heights_over(placed[mine], ground, body_reach(placed))) if len(mine) else None
               for part, mine in parts.items()})
    found = contact.lowests(stander, data, skin, ground, parts)
    if found != wanted or wanted[1]["off"] is not None or wanted[1]["some"] is None:
        problems.append(f"lowests {found}, not {wanted}")
    return problems


STANDER = """<mujoco model="stander">
  <compiler angle="radian"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.9">
      <freejoint name="root"/>
      <geom type="sphere" size="0.1" mass="60"/>
      <body name="calcn_l" pos="0.1 0.08 -0.88"><geom type="box" size="0.04 0.06 0.02" mass="1"/>
        <body name="toes_l" pos="0 0.09 0"><geom type="box" size="0.04 0.03 0.02" mass="0.2"/></body>
      </body>
      <body name="calcn_r" pos="-0.1 -0.08 -0.88"><geom type="box" size="0.04 0.06 0.02" mass="1"/>
        <body name="toes_r" pos="0 0.09 0"><geom type="box" size="0.04 0.03 0.02" mass="0.2"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
FEET = {"LeftFoot": "calcn_l", "LeftToeBase": "toes_l", "RightFoot": "calcn_r", "RightToeBase": "toes_r"}
RATE, SWAY = 30.0, 0.0015  # frames a second; the stance's sway up and down (m, once a second)
FOREFOOT = 0.0045  # m: each forefoot's flat sole over the ground at rest, so swaying it comes within GAP and leaves


def swaying_stance():
    """A 62.4 kg stander's quiet stance on a flat ground, two seconds: its pelvis swaying SWAY up and down once a
    second (0.15% of g), its left foot 8 cm ahead and its right 8 cm behind (so how each foot's push is spread over
    its sole is the solve's choice), each a sole of skin points under its calcaneus and toes, its heel 2 mm into the ground
    and its toe tip 4 mm over it, so its points come within GAP and leave it as it sways. Returns the record's pushes
    (`steadied`), the solve's (`touched`'s), the movement's needed force on the pelvis each frame and the weight."""
    import mujoco
    from motor import muscles
    import numpy as np
    from motor import records
    from motor import contact
    from motor import model as built_model
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "stander.xml"
        path.write_text(STANDER)
        found = built_model.built(path, 9.81)
    ground = np.array([[[-5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [5.0, 0.0, -5.0]],
                       [[-5.0, 0.0, -5.0], [-5.0, 0.0, 5.0], [5.0, 0.0, 5.0]]])
    bodies = {segment: built_model.body_index(found, body) for segment, body in FEET.items()}
    data = mujoco.MjData(found)
    mujoco.mj_kinematics(found, data)
    points, owners = [], []
    for side, x, y in (("Left", 0.1, 0.08), ("Right", -0.1, -0.08)):  # the left foot ahead, the right behind
        for across in np.linspace(x - 0.03, x + 0.03, 4):
            for along in np.linspace(-0.06, 0.12, 13):  # MuJoCo's y forward: the heel 2 mm in, the forefoot over
                points.append(built_model.TO_OWN @ [across, y + along, -0.002 if along < 0.03 else FOREFOOT])
                owners.append(f"{side}Foot" if along < 0.06 else f"{side}ToeBase")
    segments = list(FEET)
    weights = np.array([[float(owner == segment) for segment in segments] for owner in owners])
    skin = contact.Skin(points, weights, [built_model.own_pose(found, data, bodies[segment]) for segment in segments],
                        [bodies[segment] for segment in segments], [np.eye(4)] * len(segments))
    physics = muscles.Physics(found, ground, skin, muscles.parts_of(found, bodies))
    raw, needs = [], []
    for frame in range(int(2 * RATE)):
        turn = 2 * np.pi * frame / RATE
        qpos, qvel, qacc = found.qpos0.copy(), np.zeros(found.nv), np.zeros(found.nv)
        qpos[2] += SWAY * np.sin(turn)
        qvel[2], qacc[2] = SWAY * 2 * np.pi * np.cos(turn), -SWAY * (2 * np.pi) ** 2 * np.sin(turn)
        needed = physics.needed(qpos, qvel, qacc)
        raw.append(records.touched(physics.shared(needed)["touches"]))
        needs.append(built_model.TO_OWN @ needed[:3])
    weight = float(sum(found.body_mass) * 9.81)
    return records.steadied(raw, RATE, ground), raw, needs, weight, ground


def a_quiet_stance_pushes_its_weight_steadily_under_each_foot():
    """The swaying stance's record: one push a foot (its calcaneus and toes as one, by its Foot segment's name) on
    every frame, and none of `push_faults`: the pushes sum to the weight within 5% on its quiet frames, each centre of
    pressure on the ground and under its foot's touching skin, no push or centre jumping from frame to frame."""
    from motor import records
    steady, raw, needs, weight, ground = swaying_stance()
    problems = [f"frame {frame}: parts {sorted(touches)}" for frame, touches in enumerate(steady)
                if sorted(touches) != ["LeftFoot", "RightFoot"]]
    measures = records.push_measures(steady, raw, needs, weight, ground)
    if measures["quiet frames"] < len(steady):
        problems.append(f"{measures['quiet frames']} of {len(steady)} frames quiet")
    return problems + [f"{fault} ({measures})" for fault in records.push_faults(measures)]


CHECKS = (a_pose_and_a_turn_read_as_worked_by_hand, near_vertices_merge_and_flat_faces_go,
          the_adjusted_record_reads_in_its_units, the_arrays_are_named_for_the_person,
          a_quiet_stance_pushes_its_weight_steadily_under_each_foot, the_one_pass_pieces_equal_the_item_by_item_ones)


def in_newton():
    """The checks, in Newton's own Python."""
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(in_newton())
