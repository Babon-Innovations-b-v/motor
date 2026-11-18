"""contact: the ground pushes the person's skin (`Skin`, carried by the body's bones) only where it comes within GAP
of the drawn ground, whichever drawn surface that is (a seat as a floor), inside a friction pyramid when slow and
sliding against its slip when fast, a point blended over two bodies moving with both by its weights, its push's
directions as long as it is near, so a point's push fades to nothing at GAP (worked by hand); every state's pushes at
once (`pushes_all`, the card's) are each state's own, and so are the balance's touching points (`touching_all`).
It runs in Newton's own Python (MuJoCo pinned there; MOTOR's own Python, the pins in pyproject.toml).

Breaks (each applied once with the env linked into a sandbox home, seen red, reverted), 2026-10-28:
- pushes' `<= GAP` made `<= 1.0`: a ball 6 mm over the ground is pushed.
- pushes' `np.linalg.norm(moving) > SLIDE_MOST` made `< SLIDE_MOST`: a slow point slides and a fast one sticks.
2026-11-01:
- pushes' `own` made the colliders' points (`surface(model, data, range(model.ngeom)) @ TO_OWN.T`, the contact before
  the skin): the ball's 3 cm collider stands 24 mm over its seat inside its 5 cm skin, and nothing is pushed.
- Skin.jacobian's `found += self.weights[point, segment] * along` made `found += along`: the half-and-half point
  moves as fast as the ball.
2026-11-05:
- pushes_all's `motion *= chains.moves[...]` dropped: a point on the world is moved by the ball's dofs.
- Chains' `dof = model.dof_parentid[dof]` made `dof -= 1`: a chain runs through dofs that are not the body's.
- pushes' `closeness(over[point]) * edge` made `edge`: the point 4 mm over is pushed as freely as one on the ground,
  so a point crossing GAP takes its whole share at once (a foot's centre of pressure 3 to 6 cm a frame).
2026-11-06:
- touching_all's `speed[:, 3:] + np.cross(...)` made `speed[:, 3:]`: the rolling ball's lowest point read at its
  centre's 1 m/s, so it sticks on a pyramid where its push slides.
- pyramids' `edges[sliding, 0] = ...` dropped: a sliding point keeps its pyramid's first edge.
2026-11-07 (#98):
- pushes' `over = over - skin.credit[...]` dropped: the foot seen standing 20 mm up is not pushed.
- pushes_all's `- credit[state]` dropped: its state differs from `pushes`'.
- kabsch's `middle_wanted - turn @ middle_given` made `middle_wanted - middle_given`: the registered riding is off.
2026-11-17 (#1):
- feet_touching's `and rising[frame] < RISING` dropped: a foot lifting off slowly at a clip's start touches.
- credited's `AWAY` made `0.0`: a foot not touching, 2 mm over the ground, is pushed.
- pyramids' `np.sqrt(areas)` dropped: a point four times another's area pushes along edges as long as its.
- feet_touching's `deep or` dropped: a foot landing fast 40 mm into the floor does not touch.
- feet_touching's `deep or landed or` made `deep or`: a foot landing fast 15 mm into the floor does not touch.
- feet_touching's `and not held` dropped: a landed foot still under its floor leaves on its speed.
- feet_touching's `held = down and (since < FALLEN or rising[frame] < RISING)` made `held = down`: a foot standing,
  then lifting off fast from under its floor, stays touching.
- feet_touching's `since < FALLEN or ` dropped: a landed foot settling up out of its floor leaves on its speed.
2026-11-18 (#1):
- feet_shares' `found[frames] = over` made `found[frames] = 0.5`: a step's left shares 0.5, not 1/6, 1/2, 5/6.
- feet_shares' `/ ramp` made `/ 1.0`: a late foot's shares 0.333, 0.5, not 1/7, 1/3, 5/11.

Run: .venv/bin/python motor/contact/test_only_a_point_within_gap_is_pushed.py
"""
import pathlib
import sys

INSIDE = pathlib.Path(__file__).resolve().parents[1]
BALL = """<mujoco model="ball">
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.03" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


PADS = """<mujoco model="pads">
  <worldbody>
    <body name="ball" pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.03" mass="1"/>
      <body name="left" pos="0.1 0 -0.2">
        <joint name="left_hinge" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.02" mass="0.2"/>
      </body>
      <body name="right" pos="-0.1 0 -0.25">
        <joint name="right_hinge" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
SKIN = 0.05  # the ball's skin's radius (m), its collider 3 cm inside it


def flat():
    """The ground: a 10 m square at height 0 (the person's frame, Y up), two triangles."""
    import numpy as np
    return np.array([[[-5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [5.0, 0.0, -5.0]],
                     [[-5.0, 0.0, -5.0], [-5.0, 0.0, 5.0], [5.0, 0.0, 5.0]]])


def seated():
    """The flat ground with a seat: a 50 cm square top 0.45 m over it, round the ball's line."""
    import numpy as np
    top = np.array([[[-0.25, 0.45, -0.25], [0.25, 0.45, 0.25], [0.25, 0.45, -0.25]],
                    [[-0.25, 0.45, -0.25], [-0.25, 0.45, 0.25], [0.25, 0.45, 0.25]]])
    return np.concatenate([flat(), top])


def ball_at(height, speed=0.0):
    """The ball's model and its data with its centre `height` metres up (MuJoCo's frame, Z up), moving `speed` m/s
    along x."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(BALL)
    data = mujoco.MjData(model)
    data.qpos[2], data.qvel[0] = height, speed
    mujoco.mj_forward(model, data)
    return model, data


def skin_of(model, data, shares=((1, 1.0),)):
    """The ball's skin: SOMA-like points SKIN round its centre in 26 directions, bound where it stands now, each
    point's weight `shares` [(body, share)], every segment riding its body's frame."""
    from motor import contact
    from motor import model as built_model
    import numpy as np
    directions = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if x or y or z], float)
    centre = built_model.TO_OWN @ data.xpos[1]
    points = centre + SKIN * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    bodies = [body for body, _ in shares]
    weights = np.tile([share for _, share in shares], (len(points), 1))
    return contact.Skin(points, weights, [built_model.own_pose(model, data, body) for body in bodies], bodies,
                        [np.eye(4)] * len(bodies))


def pushed(height, ground, speed=0.0):
    """The ground's pushes on the ball's skin, its centre `height` up."""
    from motor import contact
    from motor import ground as grounds
    model, data = ball_at(height, speed)
    return contact.pushes(model, data, ground, grounds.normals(ground), skin_of(model, data))


def only_a_point_within_gap_is_pushed():
    """The ball's skin (5 cm) with its bottom 4 mm over the ground: its one lowest point is pushed, along the four
    edges of the friction pyramid (normal up, plus or minus FRICTION across), each a fifth long (1 mm left of GAP's
    5); 6 mm over it, nothing is pushed. A touching foot is pushed from its sole
    (`seen_standing_problems`); a registered skin lies on SOMA's (`registered_skin_problems`)."""
    import numpy as np
    near, far = pushed(0.054, flat()), pushed(0.056, flat())
    problems = [] if len(near) == 1 and len(near[0][2]) == 4 else [f"4 mm over: {len(near)} points pushed"]
    if near and not np.allclose(sorted(np.round(edge, 9).tolist() for edge in near[0][2]),
                                sorted([[0.0, 0.1, 0.2], [0.0, -0.1, 0.2], [0.1, 0.0, 0.2], [-0.1, 0.0, 0.2]])):
        problems.append(f"4 mm over: edges {[edge.tolist() for edge in near[0][2]]}")
    return problems + ([] if not far else [f"6 mm over: {len(far)} points pushed"]) + seen_standing_problems() + \
        registered_skin_problems()


def a_seat_carries_the_skin_on_it():
    """The ball's skin 4 mm over a seat 0.45 m up (its collider 24 mm over it): the seat pushes its one lowest point,
    straight up (MuJoCo's z, a fifth long: 1 mm left of GAP), booked on the ball; the floor under the seat pushes
    nothing."""
    import numpy as np
    found = pushed(0.45 + 0.054, seated())
    if len(found) != 1:
        return [f"on the seat: {len(found)} points pushed"]
    body, along, edges, _ = found[0]
    problems = [] if body == 1 else [f"booked on body {body}"]
    return problems + ([] if np.allclose(np.mean(edges, axis=0), [0.0, 0.0, 0.2]) else [f"edges {edges}"])


def a_fast_point_slides_against_its_slip():
    """The same ball moving 2 m/s along x (over SLIDE_MOST, 1.2 m/s): one edge, the normal and FRICTION of it
    against the slip, (-0.5, 0, 1), a fifth long (4 mm over)."""
    import numpy as np
    found = pushed(0.054, flat(), 2.0)
    edges = found[0][2] if found else []
    return [] if len(edges) == 1 and np.allclose(edges[0], [-0.1, 0.0, 0.2]) else [f"sliding edges {edges}"]


def a_blended_point_moves_by_its_weights():
    """A skin point weighed half on the ball and half on the still world: its Jacobian is half the ball's point's."""
    import mujoco
    import numpy as np
    model, data = ball_at(0.054)
    half = skin_of(model, data, ((1, 0.5), (0, 0.5)))
    whole = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, whole, None, data.xpos[1] + [0.0, 0.0, -SKIN], 1)
    lowest = int(np.argmin(half.points[:, 1]))
    found = half.jacobian(model, data, half.by_segment(model, data), lowest)
    return [] if np.allclose(found, 0.5 * whole) else [f"Jacobian {found.round(3).tolist()}, not half the ball's"]


def every_state_at_once_is_each_states_own():
    """`pushes_all` on six states of the ball (4 and 6 mm over, 1 mm into the ground, sliding at 2 m/s, turned 0.4 rad
    and spinning, rolling back at 10 rad/s while moving 1 m/s so its lowest point slides at 1.5), its
    skin half on the ball and half on the still world for some points: each state's pushes are `pushes`' on that state
    alone (bodies, Jacobians to 1e-12, edges, the ground's points), its lowest point `lowests`', and `touching_all`'s
    points (the balance's: no Jacobians, each point's speed from the bodies' `cvel`) are the same points, edges and
    places to 1e-12; and on three states of two sibling pads on their
    own hinges under a ball, a pad's skin blended with the still world (a point's chain is its own body's dofs, never
    its sibling's)."""
    import mujoco
    import numpy as np
    from motor import contact
    from motor import ground as grounds
    model, data = ball_at(0.054)
    skin = skin_of(model, data, ((1, 0.5), (0, 0.5)))
    skin.weights[::2] = [1.0, 0.0]  # every other point on the ball alone
    states, wanted = {"xpos": [], "xmat": [], "cdof": [], "cvel": [], "subtree_com": [], "qvel": []}, []
    for height, speed, turn in ((0.054, 0.0, 0.0), (0.056, 0.0, 0.0), (0.053, 2.0, 0.0), (0.0535, 0.5, 0.4), (0.049, 0.0, 0.0),
                                (0.05, 1.0, -10.0)):  # the last rolling back: its lowest point 1.5 m/s, sliding
        data.qpos[:] = model.qpos0
        data.qpos[2] = height
        mujoco.mju_axisAngle2Quat(data.qpos[3:7], np.array([0.0, 1.0, 0.0]), turn)
        data.qvel[:] = [speed, 0.0, 0.0, 0.0, turn, 0.3]
        mujoco.mj_forward(model, data)
        for name in states:
            states[name].append(getattr(data, name).copy())
        wanted.append(contact.pushes(model, data, flat(), grounds.normals(flat()), skin))
    pads = mujoco.MjModel.from_xml_string(PADS)  # two sibling pads, each on its own hinge: a chain is its own
    own = mujoco.MjData(pads)
    own.qpos[2] = 0.3 - 0.25 - 0.002  # the right pad's centre 2 mm under its ground, its skin 1 mm round it
    mujoco.mj_forward(pads, own)
    from motor import model as built_model
    centre = built_model.TO_OWN @ own.xpos[3]
    directions = np.array([[0.0, -1.0, 0.0], [0.6, -0.8, 0.0], [0.0, -0.8, 0.6], [-0.6, -0.8, 0.0]])
    pad_skin = contact.Skin(centre + 0.001 * directions, np.tile([0.7, 0.3], (4, 1)),
                            [built_model.own_pose(pads, own, body) for body in (3, 0)], [3, 0], [np.eye(4)] * 2)
    pad_states, pad_wanted = {name: [] for name in states}, []
    for turn in (0.0, 0.2, -0.3):
        own.qpos[7:] = 0.5 * turn, turn
        own.qvel[:] = [0.1, 0.0, -0.05, 0.2, turn, 0.1, 1.0, -0.5]
        mujoco.mj_forward(pads, own)
        for name in pad_states:
            pad_states[name].append(getattr(own, name).copy())
        pad_wanted.append(contact.pushes(pads, own, flat(), grounds.normals(flat()), pad_skin))
    arrays = lambda found: {name: np.array(values) for name, values in found.items()}  # noqa: E731
    found, over = contact.pushes_all(model, arrays(states), flat(), grounds.normals(flat()), skin,
                                     contact.Chains(model))
    pad_found, _ = contact.pushes_all(pads, arrays(pad_states), flat(), grounds.normals(flat()), pad_skin,
                                      contact.Chains(pads))
    touching = contact.touching_all(model, arrays(states), flat(), grounds.normals(flat()), skin)
    touching += contact.touching_all(pads, arrays(pad_states), flat(), grounds.normals(flat()), pad_skin)
    found += pad_found
    wanted += pad_wanted
    problems = []
    for state, ((places, summed, edges, counts), theirs) in enumerate(zip(touching, wanted)):
        if len(places) != len(theirs):
            problems.append(f"state {state}: {len(places)} points touching, pushed {len(theirs)}")
            continue
        for place, weight, edge, count, other in zip(places, summed, edges, counts, theirs):
            if count != len(other[2]) or not np.allclose(edge[:count], other[2], atol=1e-12) or not np.allclose(
                    place[:2], other[3][:2], atol=1e-12) or not np.isclose(weight, 1.0, atol=1e-12):
                problems.append(f"state {state}: a touching point differs from its push")
                break
    for state, height in enumerate((0.054, 0.056, 0.053, 0.0535, 0.049)):
        data.qpos[:] = model.qpos0
        data.qpos[2] = height
        mujoco.mj_forward(model, data)  # its turn left out: the lowest point's height does not turn with it here
        lowest, _ = contact.lowests(model, data, skin, flat(), {})
        if state in (0, 1, 4) and abs(np.nanmin(over[state]) - lowest) > 1e-12:
            problems.append(f"state {state}: lowest {np.nanmin(over[state])} against lowests'")
    for state, (mine, theirs) in enumerate(zip(found, wanted)):
        if len(mine) != len(theirs):
            problems.append(f"state {state}: {len(mine)} points pushed, alone {len(theirs)}")
            continue
        for one, other in zip(mine, theirs):
            if one[0] != other[0] or not np.allclose(one[1], other[1], atol=1e-12) or not np.allclose(
                    one[2], other[2], atol=1e-12) or not np.allclose(one[3], other[3], atol=1e-12):
                problems.append(f"state {state}: a point's push differs from its state's alone")
                break
    return problems + ([] if sum(len(theirs) for theirs in wanted) >= 3 else ["too few points pushed to tell"])


def seen_standing_problems():
    """The ball's skin 20 mm over the ground: unpushed as it is; credited as a touching foot with its lowest point 20
    mm up (`Skin.credited`), its lowest point is pushed as if on the ground (edges whole: (0, 0, 1) plus or minus
    FRICTION across), on `pushes` and on `pushes_all`'s state alike; credited touching with its lowest point 15 mm
    into the ground, only that point is pushed (the sole laid on the ground, no point pushed from inside it); not
    touching, nothing is pushed, even 2 mm over. Contact from the foot's own height and speed (`feet_touching`), area
    by area (`area_problems`)."""
    import numpy as np
    from motor import contact
    from motor import ground as grounds
    up = grounds.normals(flat())
    problems = [] if not pushed(0.07, flat()) else ["20 mm over, uncredited: pushed"]
    for height, touching, lowest, wanted in ((0.07, True, 0.02, 1), (0.07, False, 0.02, 0), (0.035, True, -0.015, 1),
                                             (0.052, False, 0.002, 0)):
        model, data = ball_at(height)
        skin = skin_of(model, data)
        skin.credited(np.array([[lowest]]), np.array([[touching]]), [[1]], 30.0)
        found = contact.pushes(model, data, flat(), up, skin)
        if len(found) != wanted:
            problems.append(f"touching {touching}, lowest {lowest}: {len(found)} points pushed, not {wanted}")
        elif wanted and not np.allclose(np.mean(found[0][2], axis=0), [0.0, 0.0, 1.0]):
            problems.append(f"touching, lowest {lowest}: edges {np.round(found[0][2], 3).tolist()}, not whole")
        if touching and lowest == 0.02:
            states = {name: np.array([getattr(data, name).copy()]) for name in
                      ("xpos", "xmat", "cdof", "cvel", "subtree_com", "qvel")}
            every, _ = contact.pushes_all(model, states, flat(), up, skin, contact.Chains(model))
            if len(every[0]) != len(found) or not np.allclose(every[0][0][2], found[0][2], atol=1e-12):
                problems.append("touching: pushes_all's state differs from pushes'")
    return problems + touching_problems() + area_problems()


def touching_problems():
    """A foot standing 30 mm over its floor (the video's), slow, then lifted fast to 100 mm and back: its floor is
    fitted at its standing height, so it touches while it stands, leaves when it moves fast or rises past LEAVE_HIGH,
    and touches again on landing slow; one rising past RISING does not come into contact, however low and slow (a
    foot lifting off at a clip's start); one landing fast DEEP into the floor touches at once and stays, and so does
    one falling onto it fast, under LANDED but not DEEP (a drop jump's landing), while it stays under its floor; one
    that stood and lifts off fast from under its floor leaves on its speed (a walking push-off)."""
    import numpy as np
    from motor import contact
    heights = np.array([0.03, 0.03, 0.031, 0.04, 0.10, 0.10, 0.05, 0.032, 0.03])
    speeds = np.array([0.0, 0.1, 0.2, 1.5, 2.0, 2.0, 0.9, 0.2, 0.0])
    wanted = [True, True, True, False, False, False, True, True, True]
    found = contact.feet_touching(heights[:, None], speeds[:, None], 30.0)[:, 0].tolist()
    problems = [] if found == wanted else [f"touching {found}, not {wanted}"]
    held = contact.feet_touching(np.array([[0.03], [0.03], [0.11], [0.03]]), np.array([[0.1], [0.1], [0.1], [0.1]]),
                                 30.0)
    if held[:, 0].tolist() != [True, True, False, True]:
        problems.append(f"rising past LEAVE_HIGH slow: {held[:, 0].tolist()}, not left")
    lifting = contact.feet_touching(np.array([[0.06], [0.065], [0.07]]), np.full((3, 1), 0.3), 30.0)
    if lifting.any():
        problems.append(f"lifting off slowly: {lifting[:, 0].tolist()}, touching")
    landing = contact.feet_touching(np.array([[0.0], [0.0], [0.0], [0.10], [-0.04], [-0.04]]),
                                    np.array([[0.1], [0.1], [0.1], [3.0], [1.5], [1.5]]), 30.0)
    problems += [] if landing[:, 0].tolist() == [True, True, True, False, True, True] else \
        [f"landing fast 40 mm into the floor: {landing[:, 0].tolist()}, not touching"]
    falling = contact.feet_touching(np.array([[0.0], [0.0], [0.0], [0.10], [-0.015], [-0.01], [0.0], [0.0]]),
                                    np.array([[0.1], [0.1], [0.1], [3.0], [2.0], [1.5], [0.3], [0.1]]), 30.0)
    problems += [] if falling[:, 0].tolist() == [True, True, True, False, True, True, True, True] else \
        [f"landing fast 15 mm into the floor: {falling[:, 0].tolist()}, not touching at once"]
    pushing = contact.feet_touching(np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [-0.01], [-0.02], [-0.01], [0.01]]),
                                    np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.5], [1.0], [2.0], [2.2]]), 30.0)
    if pushing[:, 0].tolist() != [True] * 7 + [False, False]:
        problems.append(f"lifting off fast from under the floor: {pushing[:, 0].tolist()}, still touching")
    return problems + shares_problems()


def shares_problems():
    """The feet's split while both touch (`feet_shares`): a step, the left foot down on frame 5 while the right lifts
    off after frame 7, hands the load over linearly (left 1/6, 1/2, 5/6); a stance both feet hold throughout shares it
    equally; a foot coming down on frame 2 of a long stance ramps in over RAMP (3 frames at 30 a second: 1/7, 1/3, 5/11,
    then 1/2); one foot alone, none (#1, 2026-11-18: walking's trailing foot held its load to its lift-off, a cliff)."""
    import numpy as np
    from motor import contact
    problems = []
    step = np.zeros((12, 2), bool)
    step[:8, 1], step[5:, 0] = True, True
    for name, touching, wanted in (
            ("a step", step, [np.nan] * 5 + [1 / 6, 1 / 2, 5 / 6] + [np.nan] * 4),
            ("a stance", np.ones((4, 2), bool), [0.5] * 4),
            ("a late foot", np.concatenate([[[False, True]] * 2, np.ones((6, 2), bool)]),
             [np.nan, np.nan, 1 / 7, 1 / 3, 5 / 11, 0.5, 0.5, 0.5])):
        found = contact.feet_shares(touching, 30.0)
        if not np.allclose(found, wanted, equal_nan=True):
            problems.append(f"{name}: left shares {np.round(found, 3).tolist()}, not {np.round(wanted, 3).tolist()}")
    return problems


def area_problems():
    """A square skin of two triangles sharing a diagonal (its two ends in both, so twice the others' area): its points'
    areas are a third of each triangle's at each corner over their mean, and a pushed point's directions are its
    area's square root long, so the least squares that shares a load gives each point its area's share."""
    import numpy as np
    from motor import contact
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    areas = contact.point_areas(points, faces)
    wanted = np.array([2.0, 1.0, 2.0, 1.0]) / 1.5
    problems = [] if np.allclose(areas, wanted) else [f"areas {areas.round(3).tolist()}, not {wanted.round(3).tolist()}"]
    if not np.allclose(contact.point_areas(points, None), 1.0):
        problems.append("no faces: areas not all one")
    edges, _ = contact.pyramids(np.array([[0.0, 1.0, 0.0]] * 2), np.zeros((2, 3)), np.zeros(2), np.array([4.0, 1.0]))
    lengths = np.linalg.norm(edges[:, 0], axis=1)
    return problems + ([] if np.isclose(lengths[0] / lengths[1], 2.0) else [f"edges {lengths.round(3)}: not area's root"])


def registered_skin_problems():
    """The ball's skin (its segment's frame 22 cm off its middle) carried on three frames whose SOMA frames stand turned 0.1 rad and shifted (1, -2, 3) cm off
    the carried ones: `Skin.registered` rides it by that turn and shift (to 1e-9), so it lies on SOMA's."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from motor import model as built_model
    from motor import contact
    model, data = ball_at(0.3)
    ball = skin_of(model, data)
    start = np.linalg.inv(ball.unbound[0])
    start[:3, 3] += [0.2, 0.0, 0.1]  # its segment's frame off the skin's middle, as a foot's at its ankle
    skin = contact.Skin(ball.points, ball.weights, [start], ball.bodies, ball.riding)
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_rotvec([0.0, 0.1, 0.05]).as_matrix()
    offset[:3, 3] = [0.01, -0.02, 0.03]
    carried = []
    for height in (0.3, 0.32, 0.35):
        data.qpos[2] = height
        data.qpos[3:7] = Rotation.from_rotvec([0.0, height, 0.0]).as_quat(scalar_first=True)
        import mujoco
        mujoco.mj_forward(model, data)
        carried.append([built_model.own_pose(model, data, 1)])
    carried = np.array(carried)
    found = skin.registered(carried, carried @ offset)
    return [] if np.allclose(found.riding[0], offset, atol=1e-9) else [f"riding {np.round(found.riding[0], 4)}"]


CHECKS = (only_a_point_within_gap_is_pushed, a_seat_carries_the_skin_on_it, a_fast_point_slides_against_its_slip,
          a_blended_point_moves_by_its_weights, every_state_at_once_is_each_states_own)


def in_newton():
    """The checks, in Newton's own Python."""
    failures = [f"{check.__name__}: {problem}" for check in CHECKS for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(in_newton())
