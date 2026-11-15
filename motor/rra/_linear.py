"""RRA's linear part: what a state asks of the body (the needed wrench at the pelvis and the lumbar's needed moments),
what the ground's pushes alone leave of it (the residual, every frame's columns at once), each frame's state, the
dofs RRA moves, a path moved by them, and the residual's and lumbar's change with each frame's move, linearised."""
import mujoco
import numpy as np

from motor.batched import root_pose
from motor.bone_groups import TRUNK
from motor.contact import touching_all
from motor.model import limited, rates
from motor.muscles import BALANCE, CONTACT_COST
from motor.solvers import stencils

RRA_STEP = 1e-5  # the central differences' step (m, rad)
# On the card (MuJoCo Warp, float32) the steps its rounding allows: a place's 1e-3 (its rounding, about 1e-6 of a
# needed force over the step, and the second order term's under 1e-4 of the change; RRA_STEP's 1e-5 would leave the
# float32 rounding at 5% of it), and a speed's and an acceleration's 1: the needed forces are quadratic in the speeds
# and linear in the accelerations, so a central difference is exact at any step
CARD_STEPS = (1e-3, 1.0, 1.0)


def about(model, data, generalised, origin):
    """The pelvis' six rows of a generalised force as a world wrench (force, then moment about `origin`, MuJoCo's
    frame): the free joint's force is the world's, its moment the root body's about its own origin."""
    root = model.jnt_bodyid[0]
    force = generalised[:3]
    moment = data.xmat[root].reshape(3, 3) @ generalised[3:6] + np.cross(data.xpos[root] - origin, force)
    return np.concatenate([force, moment])


def demands(physics, qpos, qvel, qacc, origin, trunk):
    """What a state asks of the body: the needed wrench about `origin` (6) and the needed moments at the `trunk`
    leading dofs (the lumbar's, N m: what its muscles must give, or its reserves)."""
    needed = physics.needed(qpos, qvel, qacc)
    return np.concatenate([about(physics.model, physics.data, needed, origin),
                           (physics.couplings.map(qpos).T @ needed)[trunk]])


def trunk_moments(physics, states, trunk):
    """Each frame's needed moments at the `trunk` leading dofs (frames x trunk), every frame at once on the card
    (`Physics.each`)."""
    path, velocities, accelerations, _ = zip(*states)
    return np.array([(physics.couplings.map(qpos).T @ needed)[trunk]
                     for qpos, needed in zip(path, physics.each(path, velocities, accelerations, contacts=False))])


def sensitivity(physics, qpos, qvel, qacc, origin, adjusted):
    """d(what the state asks: `demands`, the trunk the adjusted lumbar leaders) / d(each `adjusted` leading dof's
    place, speed and acceleration): three (6 + trunk) x adjusted matrices, by central differences of inverse
    dynamics."""
    model, couplings = physics.model, physics.couplings
    mapped = couplings.map(qpos)
    found = [np.zeros((len(adjusted), len(adjusted))) for _ in range(3)]  # 6 wrench rows and the trunk's
    for column, lead in enumerate(adjusted):
        step = mapped[:, lead] * RRA_STEP
        for sign in (1.0, -1.0):
            placed = qpos.copy()
            mujoco.mj_integratePos(model, placed, sign * step, 1.0)
            for kind, state in enumerate(((couplings.apply(placed), qvel, qacc), (qpos, qvel + sign * step, qacc),
                                          (qpos, qvel, qacc + sign * step))):
                found[kind][:, column] += sign * demands(physics, *state, origin, adjusted[6:]) / (2 * RRA_STEP)
    return found


def sensitivities(physics, states, adjusted):
    """Each frame's `sensitivity` (frames x 3 matrices): on the card every frame's every nudged state at once
    (`batched.Card`, CARD_STEPS), each state's couplings' map the frame's own where its leaders stand as the frame's
    (a speed's or an acceleration's nudge, a root's place), else frame by frame."""
    if physics.card is None:
        return [sensitivity(physics, *state, adjusted) for state in states]
    model, couplings, trunk = physics.model, physics.couplings, adjusted[6:]
    width = len(adjusted)
    leaders = [model.jnt_qposadr[leader] for _, leader, _ in couplings.followers if leader >= 0]
    nudged, origins, maps = [], [], []
    for qpos, qvel, qacc, origin in states:
        mapped = couplings.map(qpos)
        for lead in adjusted:
            for sign in (1.0, -1.0):
                placed = qpos.copy()
                mujoco.mj_integratePos(model, placed, sign * CARD_STEPS[0] * mapped[:, lead], 1.0)
                placed = couplings.apply(placed)
                nudged += [(placed, qvel, qacc),
                           (qpos, qvel + sign * CARD_STEPS[1] * mapped[:, lead], qacc),
                           (qpos, qvel, qacc + sign * CARD_STEPS[2] * mapped[:, lead])]
                origins += [origin] * 3
                # the map reads only the leaders' coordinates: a nudge that leaves them (the root's) keeps the frame's
                same = placed[leaders].tobytes() == qpos[leaders].tobytes()
                maps += [mapped if same else couplings.map(placed), mapped, mapped]
    qpos, qvel, qacc = (np.array(values) for values in zip(*nudged))
    found = physics.card.evaluate(qpos, qvel, qacc, ("xpos", "xmat"))
    turns, places = root_pose(model, found)
    origins = np.array(origins)
    force = found["needed"][:, :3]
    moment = np.einsum("nij,nj->ni", turns, found["needed"][:, 3:6]) + np.cross(places - origins, force)
    leads = np.array([(mapped.T @ needed)[trunk] for mapped, needed in zip(maps, found["needed"])])
    asked = np.concatenate([force, moment, leads], axis=1).reshape(len(states), width, 2, 3, -1)
    steps = np.array(CARD_STEPS)[:, None]
    differences = (asked[:, :, 0] - asked[:, :, 1]) / (2 * steps)  # frames x adjusted x kind x asked
    return [[differences[frame, :, kind].T for kind in range(3)] for frame in range(len(states))]


def adjusted_leads(physics):
    """The leading dofs RRA adjusts here (the owner's: the pelvis and the trunk): the root's six and the lumbar's
    leaders (TRUNK), as columns of the couplings' map."""
    model, leads = physics.model, physics.couplings.leads
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.dof_jntid[dof]) for dof in leads]
    trunk = [column for column, name in enumerate(names) if column >= 6 and name.endswith(TRUNK)]
    if len(trunk) != len(TRUNK):
        raise ValueError(f"the model's lumbar leaders {TRUNK} not found once each: {len(trunk)} found")
    return [0, 1, 2, 3, 4, 5, *trunk]


def moved(physics, path, adjustment, adjusted):
    """`path` with each frame's adjustment (frames x adjusted, the leading dofs' own units: metres, radians) added; or
    several adjustments of it (copies x frames x adjusted), each copy's path stacked (copies * frames), each frame's
    couplings' map worked out once for every copy (`cap_rows`' nudges)."""
    model, couplings = physics.model, physics.couplings
    adjustment = np.asarray(adjustment, dtype=float)
    maps = [couplings.map(qpos) for qpos in path]
    found = []
    for copy in adjustment.reshape(-1, len(path), adjustment.shape[-1]):
        for qpos, mapped, change in zip(path, maps, copy):
            leads = np.zeros(len(couplings.leads))
            leads[adjusted] = change
            qpos = qpos.copy()
            mujoco.mj_integratePos(model, qpos, mapped @ leads, 1.0)
            found.append(limited(model, qpos, couplings))
    return np.array(found)


def frame_states(physics, path, rate):
    """Each frame's speeds, accelerations (`rates`) and the pelvis' origin (MuJoCo's frame), the residuals' point."""
    model, data = physics.model, physics.data
    velocities, accelerations = rates(model, path, rate)
    origins = []
    for qpos in path:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        origins.append(data.xpos[model.jnt_bodyid[0]].copy())
    return list(zip(path, velocities, accelerations, origins))


def residuals(physics, states):
    """Each frame's residual after the ground's pushes alone (frames x 6): the wrench the movement needs at the
    pelvis (about its origin) less what the touching points' pushes (`contact.touching_all`) give of it at best, the
    pelvis' rows of `Physics.shared` alone (muscles and joint stops put no net force on the whole body): a push e at
    a point p, its weights summing to s, gives (s e, (p - s origin) x e), every frame's columns at once; every
    frame's state on the card at once where there is one (`posed`), the least squares too (`Physics.least_each`)."""
    path, velocities, accelerations, origins = (np.array(values) for values in zip(*states))
    found = posed(physics, path, velocities, accelerations)
    turns, places = root_pose(physics.model, found)
    force = found["needed"][:, :3]
    needed = np.concatenate([force, np.einsum("nij,nj->ni", turns, found["needed"][:, 3:6]) +
                             np.cross(places - origins, force)], axis=1)
    pushes = [wrenches(*touching, origin) for touching, origin in
              zip(touching_all(physics.model, found, physics.ground, physics.up, physics.skin), origins)]
    solved = physics.least_each([(BALANCE * one, BALANCE * wanted, np.full(one.shape[1], CONTACT_COST),
                                  np.full(one.shape[1], np.inf)) for one, wanted in zip(pushes, needed)])
    return np.array([wanted - one @ pushed for wanted, one, pushed in zip(needed, pushes, solved)])


def wrenches(places, summed, edges, counts, origin):
    """The pushes' wrenches on the root about `origin` (6 x pushes, `pushes`' order: point by point, its edges in
    turn), from `contact.touching_all`'s points (places, their weights' sums, push directions and counts)."""
    used = np.arange(edges.shape[1])[None, :] < counts[:, None]
    force = (summed[:, None, None] * edges)[used]
    moment = np.cross((places - summed[:, None] * origin)[:, None, :], edges)[used]
    return np.concatenate([force, moment], axis=1).T


def posed(physics, path, velocities, accelerations):
    """Each state's needed generalised forces and the bodies' poses and speeds `residuals` reads (`evaluate`'s dict:
    needed, xpos, xmat, cvel, subtree_com, qvel), every state at once on the card, else frame by frame on MuJoCo's
    C (`Physics.needed`)."""
    names = ("xpos", "xmat", "cvel", "subtree_com")
    if physics.card is not None:
        return physics.card.evaluate(path, velocities, accelerations, names)
    found = {name: [] for name in ("needed", *names)}
    for qpos, qvel, qacc in zip(path, velocities, accelerations):
        found["needed"].append(physics.needed(qpos, qvel, qacc))
        for name in names:
            found[name].append(getattr(physics.data, name).copy())
    found = {name: np.array(values) for name, values in found.items()}
    found["qvel"] = np.asarray(velocities, float)
    return found


def linearised(physics, states, adjusted, rate, scale):
    """The residuals' and the trunk's moments' change with each frame's move of the `adjusted` dofs ((6 + trunk)
    frames x frames adjusted, over `scale`): each frame's sensitivity to its place, speed and acceleration, the speeds
    and accelerations `rates`' own differences of the places."""
    count, width = len(states), len(adjusted)
    rows = width
    speeds, accelerations = stencils(count, rate)
    system = np.zeros((rows * count, count * width))
    for frame, (places, speed, acceleration) in enumerate(sensitivities(physics, states, adjusted)):
        for other in range(max(frame - 2, 0), min(frame + 3, count)):
            block = speed * speeds[frame, other] + acceleration * accelerations[frame, other]
            system[rows * frame:rows * (frame + 1), other * width:(other + 1) * width] = block + (
                places if other == frame else 0.0)
    return system / scale[:, None]
