"""sizing: a small body built like MyoFullBody's legs and lower trunk (a pelvis, lumbar5 and lumbar3, each side's
femur, tibia, talus and toes), sized to a SOMA made of one thin box along each bone group's stretch, has every group's
stretch between its two end joints (the child bodies' joint anchors) SOMA's rig joints apart (the foot 1.5 times the
model's), every group's
mass its box's volume at Dempster's density (all brought to the person's mass with the bodies of no group kept), each
group's centre of mass on its stretch's middle and each one-body group's inertia its box's, each limb body as wide
across its long axis as its length scale times its box's width over its length over SOMA's mean shape's, each foot
across its forefoot as SOMA's metatarsal heads stand apart across it over the model's (0.06 over 0.08 m), and a box's
inertia under an uneven scale the scaled box's (all worked out by hand from the boxes); the hip joints as far apart
as the pelvis' scale puts the model's (0.15 m), not SOMA's hip joints (0.2 m), and a body with its femurs at SOMA's
width judged off; a toy neck placed by its centre on a scaled toy torso, that centre where the torso's scale puts it; and a body on its inertia's triangle edge (MyoFullBody's lumbar2) kept off it through a float32 round trip. It runs in Newton's own Python (MuJoCo pinned there; MOTOR's own Python, the pins in pyproject.toml).

Breaks (each applied once with the env linked into a sandbox home, seen red, reverted):
- (2026-10-28) sized's `np.asarray(body.pos) * (scale if body.name in OWN_PLACED else above)` made
  `np.asarray(body.pos)`: no group grows.
- (2026-10-28) weigh's `shift = target - masses @ data.xipos[index] / mass` made `shift = np.zeros(3)`: the centres
  stay the model's.
- (2026-10-28) volumes.weighed's `density(group) * 1000.0 * volume` made `1000.0 * volume`: every density one.
- (2026-10-28) weigh's `max((trace - spread) / max(own, 1e-12), INERTIA_FLOOR)` made `1.0`: the inertias stay the
  model's.
- (2026-10-29) landmarks.marker's `+ np.asarray(row[5], dtype=float)` (the link) dropped: the foot sizes 1.513, not 1.5.
- (2026-10-29) soma_lengths' `soma_point(end, rig) - soma_point(start, rig)` made `soma_point(end, rig)`: each
  stretch sized to its end joint's distance from SOMA's origin, not its two joints apart.

- (2026-10-31) sizing.filled's cache dropped (`volumes.filled` called every time): the mesh is filled once per sizing.
- (2026-11-01) body_scales' `np.full(3, scales[owner[name]] * across)` made `np.full(3, scales[owner[name]])`: the
  limbs as wide as long.
- (2026-11-01) long_axes' `np.argmax(np.abs(local))` made `np.argmin(np.abs(local))`: the length laid across the
  bone, the foot sized along its width.
- (2026-11-01) _inertia's `* _stretched(scale, quat) ** 2` dropped: the box's inertia only by its mass.
- (2026-11-02) body_scales' `found[name][fixed[name][0]] = fixed[name][1]` dropped: the foot as wide as its girth asks,
  not as SOMA's forefoot.
- (2026-11-02) forefoot_width's `- along * (apart @ along)` dropped: the heads' plain distance apart (0.767, not 0.75).
- (2026-11-04) balanced's `inertia[largest] = most` dropped: lumbar2's A + B - C stays at 0 after float32.
- (2026-11-05) joint_placed's `parent_scale * in_parent` made `in_parent`: the neck's centre stays the source's place on
  a shrunk torso (the mean body's cervical bones 35 mm over its T1 at rest).
- (2026-11-10, #98) sized's femur places moved to SOMA's hip joints apart after sizing (9cca1f06's `hip_width`): the
  hips 0.2 m apart, not the pelvis' 0.15.

Run: .venv/bin/python motor/sizing/test_a_sized_group_is_somas_length_and_weight.py
"""
import pathlib
import sys

INSIDE = pathlib.Path(__file__).resolve().parents[1]
KG = 60.0
# The toy's bone groups, as bone_groups.GROUPS writes them (lumbar3 and toes in no group: their parent's scale).
TABLE = {"Hips": (("hipmid", "Spine1"), ("hipmid", "L5_S1"), "pelvis", ("pelvis",)),
         "Spine1": (("Spine1", "Spine2"), ("L5_S1", "L3_L4"), "lumbar5", ("lumbar5",)),
         **{f"{side}{segment}": ((f"{side}{segment}", f"{side}{end}"), (f"{start}_{letter}", f"{finish}_{letter}"),
                                 f"{body}_{letter}", (f"{body}_{letter}",))
            for side, letter in (("Left", "l"), ("Right", "r"))
            for segment, end, start, finish, body in (("Leg", "Shin", "hip", "knee", "femur"),
                                                      ("Shin", "Foot", "knee", "ankle", "tibia"),
                                                      ("Foot", "ToeBase", "ankle", "mtp", "talus"))}}
# SOMA's joints (own frame: Y up, +x left; the right side mirrored), and so its stretches: Hips 0.15, Spine1 0.1, Leg
# and Shin 0.45, Foot 0.15 m.
JOINTS = {"Hips": (0, 1.0, 0), "Spine1": (0, 1.15, 0), "Spine2": (0, 1.25, 0), "LeftLeg": (0.1, 1.0, 0),
          "LeftShin": (0.1, 0.55, 0), "LeftFoot": (0.1, 0.1, 0), "LeftToeBase": (0.1, 0.1, 0.15)}
LENGTHS = {"Hips": 0.15, "Spine1": 0.1, "Leg": 0.45, "Shin": 0.45, "Foot": 0.15}
TOE_LIFT = 0.02  # the toe landmark over its joint on SOMA (own y) and its link over the bone point (MuJoCo z)
HIPS_APART = 0.15  # the toy's hips (0.2 m apart) by the pelvis' scale, SOMA's Hips 0.15 over the toy's 0.2
FOOT_SCALE = 1.5  # the toy's foot: 0.15 m on SOMA over 0.1 m on the model
# Each landmark's MyoFullBody body and bone point (the toy's joints: MuJoCo's frame, the model's left +x).
ON_BODY = {"j_hipmid": ("pelvis", (0, 0, -0.1)), "j_Spine1": ("lumbar5", (0, 0, 0)), "j_Spine2": ("lumbar3", (0, 0, 0)),
           **{f"j_{side}{segment}": (f"{body}_{side[0].lower()}", (0, 0, 0)) for side in ("Left", "Right")
              for segment, body in (("Leg", "femur"), ("Shin", "tibia"), ("Foot", "talus"), ("ToeBase", "toes"))}}
DENSITY = {"Hips": 1.01, "Spine1": 1.01, "Leg": 1.05, "Shin": 1.09, "Foot": 1.10}  # Winter 2009 Table 4.1
# The foot's width landmarks (the left's; the right mirrored): SOMA's skin point (own frame) and the talus's bone point
# (MuJoCo's frame, forward -y). Across the heel to the heads' middle SOMA's heads stand 0.06 m apart, the model's 0.08.
FOOT_MARKS = {"heel_back": ((0.1, 0.1, -0.03), (0, 0.03, 0)), "MT1_head": ((0.07, 0.1, 0.13), (-0.04, -0.09, 0)),
              "MT5_head": ((0.13, 0.1, 0.11), (0.04, -0.07, 0))}
FOOT_ACROSS = 0.75  # the foot's scale along the talus's x
# Each group's box: 0.04 x 0.04 m across, along the middle 60% of its stretch, so its centre is the stretch's middle.
SIDE, SHARE = 0.04, 0.6


def joints():
    found = dict(JOINTS)
    for name, (x, y, z) in JOINTS.items():
        if name.startswith("Left"):
            found["Right" + name[4:]] = (-x, y, z)
    return found


def box(start, end):
    """A closed box (8 points, 12 faces) along the middle SHARE of start to end (an axis line), SIDE across."""
    import numpy as np
    start, end = np.asarray(start, float), np.asarray(end, float)
    middle, along = (start + end) / 2, np.abs(end - start) * SHARE / 2
    half = np.where(along > 0, along, SIDE / 2)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + middle
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
             [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return corners, np.array(faces), half * 2


def toy_rig():
    """The toy SOMA: its joints, one box a group (the Hips' from the hip joints' middle) and a point face (three
    corners on one point) a landmark, at its joint, the toes' TOE_LIFT over it; and the toy landmark table."""
    import numpy as np
    places = joints()
    names = list(places)
    bind = np.tile(np.eye(4), (len(names), 1, 1))
    for index, name in enumerate(names):
        bind[index, :3, 3] = places[name]
    points, faces, sizes = [], [], {}
    for group, ((start, end), _, _, _) in TABLE.items():
        first = (np.asarray(places["LeftLeg"]) + places["RightLeg"]) / 2 if start == "hipmid" else places[start]
        corners, mine, size = box(first, places[end])
        faces.append(mine + 8 * len(points))
        points.append(corners)
        sizes[group] = size
    table, count = {}, 8 * len(points)
    for name, (body, local) in ON_BODY.items():
        joint = name[2:]
        at = ((np.asarray(places["LeftLeg"]) + places["RightLeg"]) / 2 if joint == "hipmid" else
              np.asarray(places[joint], dtype=float) + ([0, TOE_LIFT, 0] if joint.endswith("ToeBase") else 0))
        faces.append(np.array([[count, count, count]]))
        points.append(np.array([at]))
        link = (0, 0, TOE_LIFT) if joint.endswith("ToeBase") else (0, 0, 0)
        table[name] = (len(np.concatenate(faces)) - 1, (1.0, 0.0, 0.0), "Hips", body, local, link, 1.0, True)
        count += 1
    for name, (at, local) in FOOT_MARKS.items():
        for letter, mirror in (("l", 1), ("r", -1)):
            faces.append(np.array([[count, count, count]]))
            points.append(np.array([[mirror * at[0], at[1], at[2]]]))
            table[f"{name}_{letter}"] = (len(np.concatenate(faces)) - 1, (1.0, 0.0, 0.0), "Hips", f"talus_{letter}",
                                         (mirror * local[0], local[1], local[2]), (0, 0, 0), 1.0, True)
            count += 1
    return {"names": names, "bind": bind, "points": np.concatenate(points), "faces": np.concatenate(faces),
            "weights": np.zeros((count, len(names)))}, sizes, table


def toy_body():
    """The toy MJCF (MuJoCo's frame, Z up, the model's left +x): every body a 1 kg sphere; the hips 0.1 m down and
    0.1 m to each side of the pelvis, lumbar5 0.1 m up, lumbar3 0.08 m over it; femur and tibia 0.4 m, the toes 0.1 m
    forward (MuJoCo's -y) of the talus. Its stretches: Hips 0.2, Spine1 0.08, Leg and Shin 0.4, Foot 0.1 m."""
    sides = ""
    for letter, mirror in (("l", 1), ("r", -1)):
        sides += (f'<body name="femur_{letter}" pos="{0.1 * mirror} 0 -0.1"><joint type="hinge" axis="1 0 0"/>'
                  f'<geom type="sphere" size="0.03" mass="1"/><body name="tibia_{letter}" pos="0 0 -0.4">'
                  f'<joint type="hinge" axis="1 0 0"/><geom type="sphere" size="0.03" mass="1"/>'
                  f'<body name="talus_{letter}" pos="0 0 -0.4"><joint type="hinge" axis="1 0 0"/>'
                  f'<geom type="sphere" size="0.03" mass="1"/><body name="toes_{letter}" pos="0 -0.1 0">'
                  f'<joint type="hinge" axis="1 0 0"/><geom type="sphere" size="0.02" mass="1"/>'
                  f'</body></body></body></body>')
    return f"""<mujoco model="toybody"><worldbody><body name="pelvis" pos="0 0 1"><freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>{sides}
      <body name="lumbar5" pos="0 0 0.1"><joint type="hinge" axis="1 0 0"/><geom type="sphere" size="0.03" mass="1"/>
        <body name="lumbar3" pos="0 0 0.08"><joint type="hinge" axis="1 0 0"/>
          <geom type="sphere" size="0.03" mass="1"/></body></body></body></worldbody></mujoco>"""


def sized_toy():
    """The toy sized and weighed to the toy SOMA, its landmarks the toy's (laid into the landmark table for the
    while): (final model, its body scales, the boxes' sizes)."""
    import mujoco
    from motor import sizing
    from motor.landmarks._table import LANDMARKS
    from motor.model import TO_OWN
    rig, sizes, table = toy_rig()
    kept = dict(LANDMARKS)
    LANDMARKS.clear()
    LANDMARKS.update(table)
    try:
        size = sizing.Size(rig, TABLE, turns={group: TO_OWN for group in TABLE})
        source = mujoco.MjModel.from_xml_string(toy_body())
        scales = size.scales(source)
        spec = mujoco.MjSpec.from_string(toy_body())
        sizing.sized(spec, source, scales, KG, 0)
        grown = spec.compile()
        size.weigh(spec, grown, scales, KG - sizing.ungrouped_kg(grown, TABLE))
        final = spec.compile()
        lengths = sizing.myo_lengths(final, scales, TABLE)
    finally:
        LANDMARKS.clear()
        LANDMARKS.update(kept)
    return final, scales, sizes, lengths


def every_group_is_somas_length_mass_and_centre():
    """Each stretch between its landmarks SOMA's length within 1 mm, the foot FOOT_SCALE times the model's within
    0.001; each group's mass KG less the bodies of no group (lumbar3, the toes)
    shared by box volume times density within 0.1%; the whole body KG; each group's centre of mass on its stretch's
    middle within 1 mm; each group's inertia's trace its box's, m (a2 + b2 + c2) / 6, within 0.5%; the hips where the
    pelvis' scale puts them (`the_hips_stand_where_the_pelvis_scale_puts_them`)."""
    import mujoco
    import numpy as np
    from motor import sizing
    from motor.model import Couplings, default_pose
    final, scales, sizes, found = sized_toy()
    along = scales["talus_l"][1]  # the toy's foot runs along the talus's y
    problems = [] if abs(along - FOOT_SCALE) < 1e-3 else [f"the foot sized {along:.4f}"]
    for group, length in found.items():
        if abs(length - LENGTHS[group.removeprefix("Left").removeprefix("Right")]) > 0.001:
            problems.append(f"{group} {length:.4f} m")
    loose = sum(final.body_mass[mujoco.mj_name2id(final, mujoco.mjtObj.mjOBJ_BODY, name)]
                for name in ("lumbar3", "toes_l", "toes_r"))
    weighted = {group: DENSITY[group.removeprefix("Left").removeprefix("Right")] * np.prod(sizes[group])
                for group in TABLE}
    data = default_pose(final, Couplings(final))
    for group, (_, (start, end), frame, _) in TABLE.items():
        body = mujoco.mj_name2id(final, mujoco.mjtObj.mjOBJ_BODY, frame)
        wanted = (KG - loose) * weighted[group] / sum(weighted.values())
        if abs(final.body_mass[body] - wanted) > 0.001 * wanted:
            problems.append(f"{group} {final.body_mass[body]:.4f} kg, not {wanted:.4f}")
        middle = (sizing.myo_point(final, data, start, scales) + sizing.myo_point(final, data, end, scales)) / 2
        if np.linalg.norm(data.xipos[body] - middle) > 0.001:
            problems.append(f"{group}'s centre {np.round(data.xipos[body], 4)}, not {np.round(middle, 4)}")
        trace = final.body_mass[body] * float(np.sum(sizes[group] ** 2)) / 6
        if abs(final.body_inertia[body].sum() - trace) > 0.005 * trace:
            problems.append(f"{group}'s inertia trace {final.body_inertia[body].sum():.6f}, not {trace:.6f}")
    problems += the_hips_stand_where_the_pelvis_scale_puts_them()
    return problems + ([] if abs(final.body_mass.sum() - KG) < 1e-6 else [f"mass {final.body_mass.sum()} kg"])


def a_mesh_is_filled_once_and_sizes_alike():
    """Two sizings of one rig fill its mesh once and give the same volumes; a moved mesh is filled anew."""
    import numpy as np
    from motor import sizing
    from motor import volumes
    from motor.model import TO_OWN
    from motor.landmarks._table import LANDMARKS
    rig, _, table = toy_rig()
    calls, original, kept = [], volumes.filled, dict(LANDMARKS)
    volumes.filled = lambda *given: calls.append(1) or original(*given)
    LANDMARKS.clear()
    LANDMARKS.update(table)
    try:
        sizing.FILLED.clear()
        first, second = (sizing.Size(rig, TABLE, turns={group: TO_OWN for group in TABLE}) for _ in range(2))
        moved = {**rig, "points": np.asarray(rig["points"]) + 0.001}
        sizing.Size(moved, TABLE, turns={group: TO_OWN for group in TABLE})
    finally:
        volumes.filled = original
        LANDMARKS.clear()
        LANDMARKS.update(kept)
    problems = [] if len(calls) == 2 else [f"the mesh filled {len(calls)} times for two meshes"]
    same = all(volume == again and np.array_equal(centre, centre_again) and spread == spread_again
               for (volume, centre, spread), (again, centre_again, spread_again) in zip(first.found, second.found))
    return problems + ([] if same else ["the second sizing's volumes differ"])


# Each toy limb's long axis in its body's frame: the femur and tibia run along z, the talus to the toes along y.
LONG = {"femur": 2, "tibia": 2, "talus": 1}


def a_limb_is_as_wide_as_its_girth_asks():
    """Each limb body's scale across its long axis its length scale times its box's SIDE over its stretch's length
    over MEAN_SLENDERNESS (the cells read the box's section within 2%), but the foot's across its forefoot (the
    talus's x) FOOT_ACROSS, SOMA's heads apart over the model's."""
    import numpy as np
    from motor import sizing
    _, scales, _, _ = sized_toy()
    problems = []
    for side, letter in (("Left", "l"), ("Right", "r")):
        for segment, body in (("Leg", "femur"), ("Shin", "tibia"), ("Foot", "talus")):
            scale = scales[f"{body}_{letter}"]
            wanted = scale[LONG[body]] * SIDE / LENGTHS[segment] / sizing.MEAN_SLENDERNESS[segment]
            across = [axis for axis in range(3) if axis != LONG[body] and not (body == "talus" and axis == 0)]
            if not np.allclose(scale[across], wanted, rtol=0.02):
                problems.append(f"{side}{segment}: {np.round(scale, 4)}, across not {wanted:.4f}")
        if abs(scales[f"talus_{letter}"][0] - FOOT_ACROSS) > 1e-9:
            problems.append(f"{side}Foot across its forefoot {scales[f'talus_{letter}'][0]:.4f}, not {FOOT_ACROSS}")
    return problems


def a_box_stretched_unevenly_has_the_stretched_boxs_inertia():
    """A 1 kg box 0.1 x 0.2 x 0.3 m, its inertial frame turned a quarter about z, scaled (2, 1, 1) in its body's
    frame with its mass doubled: the 0.2 x 0.2 x 0.3 m box's of 2 kg, m (b2 + c2) / 12 axis by axis."""
    import numpy as np
    from motor import sizing
    def box(a, b, c, kg):
        return kg * np.array([b * b + c * c, a * a + c * c, a * a + b * b]) / 12
    quarter = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # the inertial x along the body's y
    found = sizing._inertia(box(0.1, 0.2, 0.3, 1.0), quarter, np.array([1.0, 2.0, 1.0]), 2.0)
    wanted = box(0.2, 0.2, 0.3, 2.0)
    return [] if np.allclose(found, wanted, rtol=1e-9) else [f"{found}, not {wanted}"]


def a_body_on_its_inertias_edge_is_kept_off_it():
    """A body with MyoFullBody's lumbar2 inertia (its largest the other two's sum) after `balanced` and a float32
    round trip: its largest under the other two's sum by more than 1e-6 of it, and moved at most BALANCE_MARGIN; a
    body off the edge untouched."""
    import mujoco
    import numpy as np
    from motor import sizing
    spec = mujoco.MjSpec()
    edge, off = (0.00365967, 0.00675583, 0.0104155), (0.01, 0.02, 0.025)
    for name, inertia in (("lumbar2", edge), ("off", off)):
        body = spec.worldbody.add_body(name=name, mass=1.0, inertia=list(inertia))
        body.explicitinertial = True
    sizing.balanced(spec)
    bodies = {body.name: np.asarray(body.inertia, dtype=float) for body in spec.bodies[1:]}
    rounded = bodies["lumbar2"].astype(np.float32).astype(float)
    problems = []
    if not rounded[0] + rounded[1] - rounded[2] > 1e-6 * rounded[2]:
        problems.append(f"lumbar2's inertia {rounded.tolist()} on its edge")
    if np.abs(bodies["lumbar2"] - edge).max() > sizing.BALANCE_MARGIN * max(edge):
        problems.append(f"lumbar2's inertia moved to {bodies['lumbar2'].tolist()}")
    if not np.allclose(bodies["off"], off, rtol=0, atol=0):
        problems.append(f"a body off its edge moved to {bodies['off'].tolist()}")
    return problems


def a_joint_placed_body_keeps_its_centre_where_its_parent_puts_it():
    """A toy neck (JOINT_PLACED) on a toy torso, its origin 1 cm off the torso's and its centre C7/T1 on the torso at
    MyoFullBody's point: with the torso scaled 0.8 and the neck 1.25, the neck's place puts its centre (its own frame's
    point scaled by its own scale) where the torso's scale puts it (worked out by hand: 0.8 c - 1.25 (c - origin))."""
    import mujoco
    import numpy as np
    from motor import sizing
    from motor.landmarks import CENTRES
    spec = mujoco.MjSpec()
    torso = spec.worldbody.add_body(name="torso", pos=[0, 0, 0])
    torso.add_geom(size=[0.05, 0, 0])
    neck = torso.add_body(name="neck", pos=[0.01, 0, 0])
    neck.add_geom(size=[0.03, 0, 0], pos=[0, 0.45, 0])
    centre = np.asarray(CENTRES["C7_T1"][1], dtype=float)
    scales = {"torso": np.full(3, 0.8), "neck": np.full(3, 1.25)}
    wanted = 0.8 * centre - 1.25 * (centre - [0.01, 0, 0])
    found = sizing.joint_placed(spec.compile(), scales)["neck"]
    return [] if np.allclose(found, wanted, atol=1e-9) else [f"the neck placed at {found.round(5).tolist()}, not {wanted.round(5).tolist()}"]


def hips_off(model, source, scales):
    """How far the sized `model`'s hip joints (the femurs' places) stand apart off the pelvis' scale times the
    `source` model's ([] within 0.1 mm): the pelvis' muscles are attached by that scale, so must its hips be."""
    import mujoco
    import numpy as np
    def apart(found):
        places = [found.body_pos[mujoco.mj_name2id(found, mujoco.mjtObj.mjOBJ_BODY, name)] for name in ("femur_l", "femur_r")]
        return float(np.linalg.norm(places[0] - places[1]))
    wanted = apart(source) * float(scales["pelvis"][0])
    found = apart(model)
    return [] if abs(found - wanted) < 1e-4 else [f"the hips {found:.4f} m apart, the pelvis' scale puts them {wanted:.4f}"]


def the_hips_stand_where_the_pelvis_scale_puts_them():
    """The sized toy's hip joints HIPS_APART apart (the toy's 0.2 m by the pelvis' 0.75), judged so by `hips_off`; the
    same body with its femurs at SOMA's hip joints apart (0.2 m, the rule 9cca1f06 brought) judged off."""
    import mujoco
    import numpy as np
    final, scales, _, _ = sized_toy()
    source = mujoco.MjModel.from_xml_string(toy_body())
    problems = hips_off(final, source, scales)
    if abs(float(scales["pelvis"][0]) * 0.2 - HIPS_APART) > 1e-6:
        problems.append(f"the pelvis scaled {np.round(scales['pelvis'], 4).tolist()}, not {HIPS_APART / 0.2}")
    wide = mujoco.MjModel.from_xml_string(toy_body())
    for name, mirror in (("femur_l", 1), ("femur_r", -1)):
        wide.body_pos[mujoco.mj_name2id(wide, mujoco.mjtObj.mjOBJ_BODY, name)] = final.body_pos[
            mujoco.mj_name2id(final, mujoco.mjtObj.mjOBJ_BODY, name)] * [0, 1, 1] + [mirror * 0.1, 0, 0]
    return problems + ([] if hips_off(wide, source, scales) else ["femurs at SOMA's hip joints apart judged in place"])


CHECKS = (every_group_is_somas_length_mass_and_centre, a_joint_placed_body_keeps_its_centre_where_its_parent_puts_it, a_mesh_is_filled_once_and_sizes_alike,
          a_limb_is_as_wide_as_its_girth_asks, a_box_stretched_unevenly_has_the_stretched_boxs_inertia,
          a_body_on_its_inertias_edge_is_kept_off_it)


def in_newton():
    """The checks, in Newton's own Python."""
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(in_newton())
