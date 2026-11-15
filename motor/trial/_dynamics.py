"""MOTOR step 2 (#98), stage A's dynamics rows: every frame's inverse dynamics on the card, a smooth contact on the
seen floor, the pelvis' residual and the joints' torques, as rows of the whole trial's spline solve.

The spline's unknowns (`trial.Layout`: each leading coordinate, the root's place and its body-frame rotation vector
on the fit's low-passed turn) give every node a pose, speed and acceleration (`Path.states`: the spline's own
derivatives, the poses a quarter step either side by its second-order Taylor series, the reference turn's own
turning included, MuJoCo's `mj_differentiatePos` of them; followers set by the person's couplings). The card gives
the forces each node needs (`batched.Card.evaluate`: rne + armature x qacc - passive), the contact (`Contact`) what
the ground gives: each skin point within BAND of its ground (its height measured from the seen floor: the job's
stance credit, `contact.Skin.credited`) pushed by a smooth Hunt-Crossley spring and damper with a smoothed Coulomb
friction, as OpenCap's tracking simulation's contact (Serrancoli et al. 2019; its dissipation and friction), a force
of the motion alone (no force unknowns, no per-frame assignment). On the leading coordinates (the couplings' map G):
tau = G^T (needed - J_c^T f_c); its root's six rows are the pelvis' residual (constrained by the augmented
Lagrangian in `trial.dynamics_solved`), the rest the joints' torques (the effort).

The rows' Jacobians on a node's place, speed and acceleration: the needed forces' on the place by one-sided
differences on the card (`PLACE_STEP`, every frame's every unknown at once), on the acceleration the mass matrix
(MuJoCo's `mj_fullM`, exact: they are linear in it) and on the speed the joints' damping (the velocity product terms
left out: a Gauss-Newton model, the cost itself always exact); the contact's analytic through each point's Jacobian.

Card only: the needed forces are the card's (`Physics.card`)."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from motor.contact import Chains, on_ground_all
from motor.model import TO_OWN, carried
from motor.trial import _spline

# The contact's settings, physical; MOTOR step 3 tuned them against force plates on training data (`Settings`, a
# job's "motor settings"), 2026-11-08: nine variants, 2e6 to 2e7 N/m, 0.01 to 0.5 m/s, friction 0.4 to 1.0 and
# effort 0.01 to 1 tried. These first values held the pelvis' residual within its limits best; a stiffer set scored
# the plates a little better but lost the sit-to-stand's residual: not taken.
# N/m: a whole foot's, spread over its skin points: OpenCap's six contact spheres a foot at 1e6 N/m each (Falisse et
# al. 2019). 2e5 (a heel pad's own, Aerts et al. 1995) spread so gave a stance foot under 1 N a point at the warm start:
# the pelvis' residual stood near the whole weight on walks (2026-11-08)
FOOT_STIFFNESS = 6e6
SMOOTH = 0.001  # m: the penetration's softplus width (a point SMOOTH over its floor still pushes 0.31 of SMOOTH deep)
BAND = 0.02  # m: a point higher than this over its floor pushes under 1e-8 N and is left out
DISSIPATION = 2.0  # s/m: Hunt-Crossley's (OpenCap's contact spheres', Falisse et al. 2019)
DISSIPATION_SMOOTH = 0.05  # the damping factor (1 - c v) softplus width: a fast lift-off pushes nothing, never pulls
FRICTION = 0.8  # OpenCap's contact spheres' static and dynamic friction
SLIP_SMOOTH = 0.05  # m/s: the smoothed Coulomb's transition speed (|f_t| = mu f_n v / sqrt(v^2 + SLIP_SMOOTH^2))
EFFORT_SHARE = 0.1  # the joints' effort against the fit's cost at the warm start (`_stage_a`)


class Settings:
    """The contact's physical settings and the solve's effort share (MOTOR step 3's tuned ones): `of(mapping)` takes
    a job's "motor settings" over these defaults and refuses a name it does not know."""
    NAMES = ("foot_stiffness", "smooth", "band", "dissipation", "dissipation_smooth", "friction", "slip_smooth",
             "effort_share")

    def __init__(self, foot_stiffness=FOOT_STIFFNESS, smooth=SMOOTH, band=BAND, dissipation=DISSIPATION,
                 dissipation_smooth=DISSIPATION_SMOOTH, friction=FRICTION, slip_smooth=SLIP_SMOOTH,
                 effort_share=EFFORT_SHARE):
        self.foot_stiffness, self.smooth, self.band = float(foot_stiffness), float(smooth), float(band)
        self.dissipation, self.dissipation_smooth = float(dissipation), float(dissipation_smooth)
        self.friction, self.slip_smooth, self.effort_share = float(friction), float(slip_smooth), float(effort_share)

    @classmethod
    def of(cls, mapping):
        """The settings of a job's "motor settings" (None: the defaults), refused on a name not in NAMES."""
        mapping = dict(mapping or {})
        unknown = sorted(set(mapping) - set(cls.NAMES))
        if unknown:
            raise ValueError(f"motor settings {unknown}: not among {list(cls.NAMES)}")
        return cls(**mapping)

    def record(self):
        """The settings as the record keeps them."""
        return {name: getattr(self, name) for name in self.NAMES}


DEFAULT = Settings()
PLACE_STEP = 1e-3  # m, rad: the card's float32 allows (`rra._linear.CARD_STEPS`)
SPEED_STEP = 1e-2  # m/s, rad/s: stage B's speed differences (the velocity products' share is quadratic in it)
SHIFT = 0.25  # of the step: the poses either side a node's speed and acceleration are read off


def softplus(values, width):
    """width log(1 + exp(values / width)), overflow-safe."""
    scaled = np.asarray(values, dtype=float) / width
    return width * (np.maximum(scaled, 0.0) + np.log1p(np.exp(-np.abs(scaled))))


def sigmoid(values):
    """1 / (1 + exp(-values)), overflow-safe."""
    values = np.asarray(values, dtype=float)
    return np.where(values >= 0, 1.0 / (1.0 + np.exp(-np.abs(values))), np.exp(-np.abs(values))
                    / (1.0 + np.exp(-np.abs(values))))


def push(over, speed, up, stiffness, settings=DEFAULT):
    """Each point's force (points x 3, MuJoCo's frame), its derivative on the point's height (points x 3) and on its
    speed (points x 3 x 3): `over` its height over its floor (m), `speed` its velocity, `up` its floor's normal."""
    smooth, dissipation, friction = settings.smooth, settings.dissipation, settings.friction
    deep = softplus(-over, smooth)
    deep_over = -sigmoid(-over / smooth)
    along = np.einsum("pi,pi->p", speed, up)
    damping = softplus(1.0 - dissipation * along, settings.dissipation_smooth)
    damping_along = -dissipation * sigmoid((1.0 - dissipation * along) / settings.dissipation_smooth)
    normal = stiffness * deep * damping
    slip = speed - along[:, None] * up
    length = np.sqrt(np.einsum("pi,pi->p", slip, slip) + settings.slip_smooth ** 2)
    sliding = slip / length[:, None]
    force = normal[:, None] * (up - friction * sliding)
    by_over = (stiffness * deep_over * damping)[:, None] * (up - friction * sliding)
    normal_speed = (stiffness * deep * damping_along)[:, None] * up  # d normal / d speed
    flat = np.eye(3)[None] - np.einsum("pi,pj->pij", up, up)
    turning = (np.eye(3)[None] / length[:, None, None] - np.einsum("pi,pj->pij", slip, slip) / length[:, None, None] ** 3)
    by_speed = (np.einsum("pi,pj->pij", up - friction * sliding, normal_speed)
                - friction * normal[:, None, None] * turning @ flat)
    return force, by_over, by_speed


class Coupled:
    """A model's couplings (`model.Couplings`) on many poses at once: `applied` its `apply` and `maps` its `map`, row
    by row the same arithmetic in the same order (so the same numbers), as array operations over the rows."""

    def __init__(self, couplings):
        self.couplings, model = couplings, couplings.model
        self.leads = couplings.leads
        column = {dof: index for index, dof in enumerate(self.leads)}
        self.followers = [(int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]),
                           int(model.jnt_qposadr[leader]) if leader >= 0 else -1,
                           column.get(int(model.jnt_dofadr[leader])) if leader >= 0 else None, poly)
                          for joint, leader, poly in couplings.followers]

    def applied(self, qpos):
        """Poses [K, nq] with every follower set from its leader (a copy)."""
        qpos, rest = np.array(qpos, dtype=float), self.couplings.model.qpos0
        for at, _, lead_at, _, poly in self.followers:
            moved = qpos[:, lead_at] - rest[lead_at] if lead_at >= 0 else 0.0
            qpos[:, at] = rest[at] + sum(poly[power] * moved ** power for power in range(5))
        return qpos

    def maps(self, qpos):
        """d(every dof) / d(leading dof) at poses [K, nq] ([K, nv, leads])."""
        model, rest = self.couplings.model, self.couplings.model.qpos0
        found = np.zeros((len(qpos), model.nv, len(self.leads)))
        found[:, self.leads, np.arange(len(self.leads))] = 1.0
        for _, dof, lead_at, lead_column, poly in self.followers:
            if lead_at < 0:
                continue
            moved = qpos[:, lead_at] - rest[lead_at]
            found[:, dof, lead_column] = sum(power * poly[power] * moved ** (power - 1) for power in range(1, 5))
        return found


def integrated(model, qpos, moves):
    """Poses [K, nq] each moved by its dof vector [K, nv] as MuJoCo's `mj_integratePos` (scale 1): a free joint's
    place by its first three, its turn and a ball's by the body-frame rotation vector (q exp(w)), a hinge or slide by
    its one."""
    qpos = np.array(qpos, dtype=float)
    for joint in range(model.njnt):
        kind, at, dof = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        if kind in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            qpos[:, at] += moves[:, dof]
            continue
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            qpos[:, at:at + 3] += moves[:, dof:dof + 3]
            at, dof = at + 3, dof + 3
        turns = Rotation.from_quat(qpos[:, at:at + 4][:, [1, 2, 3, 0]]) * Rotation.from_rotvec(moves[:, dof:dof + 3])
        qpos[:, at:at + 4] = turns.as_quat()[:, [3, 0, 1, 2]]
    return qpos


class Path:
    """The spline's nodes as the physics model's states: `index` each of its qpos entries' place in the fitting
    model's (`model.carried`), `columns` each unknown's dof and leading column, `spin` and `swing` the reference
    turn's own body-frame turning speed and acceleration a node."""

    def __init__(self, layout, source, physics, step):
        self.layout, self.model, self.couplings, self.step = layout, physics.model, physics.couplings, step
        self.coupled = Coupled(physics.couplings)
        self.index = carried(source, self.model, np.arange(source.nq, dtype=float)[None])[0].astype(int)
        if list(self.index[:7]) != list(range(7)):
            raise ValueError(f"the two models' roots are not their first seven coordinates: {self.index[:7]}")
        model, leads = self.model, list(self.couplings.leads)
        self.dofs = np.zeros(len(layout.free_dof), dtype=int)
        self.dofs[layout.root] = np.arange(6)
        for column in layout.joints:
            at = int(np.flatnonzero(self.index == layout.free_coord[column])[0])
            joint = int(np.flatnonzero(model.jnt_qposadr == at)[0])
            self.dofs[column] = model.jnt_dofadr[joint]
        missing = [int(dof) for dof in self.dofs if dof not in leads]
        if missing:
            raise ValueError(f"unknowns on dofs {missing} that follow another in the physics model")
        self.leads = np.array([leads.index(dof) for dof in self.dofs])
        turns = layout.reference
        ahead = (turns[:-1].inv() * turns[1:]).as_rotvec()
        ahead = np.concatenate([ahead, ahead[-1:]])
        behind = np.concatenate([-ahead[:1], (turns[1:].inv() * turns[:-1]).as_rotvec()])
        self.spin = (ahead - behind) / (2.0 * step)
        self.swing = (ahead + behind) / step ** 2

    def poses(self, values, shift):
        """The physics model's poses [N, nq] of the unknowns' values, the reference turn moved `shift` seconds on."""
        source = self.layout.poses(values)
        moved = Rotation.from_rotvec(shift * self.spin + 0.5 * shift ** 2 * self.swing)
        turns = self.layout.reference * moved * Rotation.from_rotvec(values[:, self.layout.root[3:]])
        source[:, 3:7] = turns.as_quat()[:, [3, 0, 1, 2]]
        return self.coupled.applied(source[:, self.index])

    def states(self, control):
        """Every node's (qpos, qvel, qacc), the physics model's, of control points `control`."""
        values = _spline.nodes(control)
        speed, pace = _spline.speeds(control, self.step), _spline.paces(control, self.step)
        shift = SHIFT * self.step
        behind, here, ahead = (self.poses(values + sign * shift * speed + 0.5 * shift ** 2 * pace * abs(sign),
                                          sign * shift) for sign in (-1.0, 0.0, 1.0))
        qvel, back, forth = (np.zeros((len(here), self.model.nv)) for _ in range(3))
        for frame in range(len(here)):
            mujoco.mj_differentiatePos(self.model, qvel[frame], 2.0 * shift, behind[frame], ahead[frame])
            mujoco.mj_differentiatePos(self.model, back[frame], shift, behind[frame], here[frame])
            mujoco.mj_differentiatePos(self.model, forth[frame], shift, here[frame], ahead[frame])
        return here, qvel, (forth - back) / shift

    def tangents(self, values, qpos):
        """Each node's d(the physics model's dofs) / d(the unknowns) [N, nv, m]: a joint's its lead's column of the
        couplings' map, the root's place the world's, its rotation vector's through SO(3)'s right Jacobian."""
        from motor.trial import right_jacobian
        found = self.coupled.maps(qpos)[:, :, self.leads]
        root = self.layout.root
        for frame in range(len(qpos)):
            found[frame, :6, root] = 0.0
            found[frame, :3, root[:3]] = np.eye(3)
            found[frame, 3:6, root[3:]] = right_jacobian(values[frame, root[3:]])
        return found


class Contact:
    """The smooth contact of a person's skin on its ground: `stiffness` a point's (the settings' foot stiffness over
    a foot's points), `chains` the bodies' dofs (`contact.Chains`), `sunk` each state's each point's floor lowered
    (m, states x points; `settle`)."""

    def __init__(self, physics, foot_points, settings=DEFAULT):
        self.physics, self.chains, self.settings = physics, Chains(physics.model), settings
        self.stiffness = settings.foot_stiffness / foot_points
        self.up = TO_OWN.T @ np.array([0.0, 1.0, 0.0])
        self.sunk = None

    def heights(self, found):
        """Every state's every point's height over its floor (states x points; inf where it has none), the stance
        credit and `sunk` taken off, and on_ground_all's pieces."""
        physics = self.physics
        count = len(found["qvel"])
        pieces = on_ground_all(physics.model, found, physics.ground, physics.skin)
        own, heights = pieces[5], pieces[6]
        over = own[:, :, 1] - np.nan_to_num(heights, nan=np.inf) - physics.skin.credit_of(count)
        return over - (0.0 if self.sunk is None else self.sunk), pieces

    def settle(self, found):
        """`sunk` from the warm start's states (`found`): on each state, a body whose lowest skin point sits under its
        floor has its points' floor lowered by that depth, so no body starts inside its ground (a seat or a box laid
        at a known height under a skin the video sat lower otherwise pushes many times the body's weight
        from the seat, 2026-11-08); the stance credit lays the feet so already."""
        self.sunk = None
        over = self.heights(found)[0]
        owners = self.physics.skin.owners
        sunk = np.zeros_like(over)
        for body in np.unique(owners):
            mine = owners == body
            lowest = np.min(np.where(np.isfinite(over[:, mine]), over[:, mine], np.inf), axis=1)
            sunk[:, mine] = np.minimum(lowest, 0.0)[:, None]
        self.sunk = sunk

    def pushed(self, found):
        """Every state's touching points (`found`: the card's xpos, xmat, cdof, subtree_com and qvel a state): a list
        a state of (force [p, 3], d force / d height [p, 3], d force / d speed [p, 3, 3], Jacobians [p, 3, nv], their
        floors' normals [p, 3], the points' indices in the skin [p]), MuJoCo's frame."""
        physics, model = self.physics, self.physics.model
        count = len(found["qvel"])
        over_all, (points, _, shares, bodies, carried_at, _, _, triangles) = self.heights(found)
        cdof = found["cdof"].reshape(count, model.nv, 6)
        com = found["subtree_com"].reshape(count, -1, 3)
        touching_all = []
        for state in range(count):
            over = over_all[state]
            near = np.flatnonzero(over <= self.settings.band)
            pairs = np.flatnonzero(np.isin(points, near))
            at = carried_at[state, pairs] @ TO_OWN
            offset = at - com[state, self.chains.roots[bodies[pairs]]]
            motion = np.cross(cdof[state, None, :, :3], offset[:, None, :]) + cdof[state, None, :, 3:]
            motion *= self.chains.moves[bodies[pairs]][:, :, None]
            local = np.searchsorted(near, points[pairs])
            jacobians = np.zeros((len(near), 3, model.nv))
            np.add.at(jacobians, local, shares[pairs, None, None] * np.swapaxes(motion, 1, 2))
            ups = physics.up[triangles[state, near]] @ TO_OWN
            speed = jacobians @ found["qvel"][state]
            force, by_over, by_speed = push(over[near], speed, ups, self.stiffness, self.settings)
            touching_all.append((force, by_over, by_speed, jacobians, ups, near))
        return touching_all


class Exact:
    """Every state's needed forces and the bodies' frames on MuJoCo's C (float64), as `batched.Card.evaluate` gives
    them on the card (float32): the cost of a solve whose weights reach past the card's rounding (stage B: a step
    lowering the reserves by less than the card's 1e-6 of the pelvis' constraint force was lost in it, 2026-11-08)."""

    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)

    def evaluate(self, qpos, qvel, qacc, names):
        """`batched.Card.evaluate`'s dict (`needed` and `names`, one row a state) on MuJoCo's C."""
        model, data = self.model, self.data
        found = {name: [] for name in (*names, "needed")}
        rne = np.zeros(model.nv)
        for pose, speed, pace in zip(qpos, qvel, qacc):
            data.qpos[:], data.qvel[:], data.qacc[:] = pose, speed, pace
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            mujoco.mj_rne(model, data, 1, rne)
            found["needed"].append(rne + model.dof_armature * pace - data.qfrc_passive)
            for name in names:
                found[name].append(getattr(data, name).copy())
        found = {name: np.array(values) for name, values in found.items()}
        found["qpos"], found["qvel"], found["qacc"] = np.asarray(qpos), np.asarray(qvel), np.asarray(qacc)
        return found


class Dynamics:
    """The whole trial's dynamics rows: `path` (`Path`), `contact` (`Contact`), the body's weight (N) and its centre's
    standing height (m), the rows' scales: the root's force over the weight, its moment over weight x height, a
    joint's torque over weight x height; `exact` (`Exact`) the rows' own evaluation when given, else the card's (the
    Jacobians always the card's, on the speed then every velocity product by differences: at stage A's answer the
    damping alone left the pelvis' rows 1.3 of their change off along stage B's step, and no step was kept)."""

    def __init__(self, path, contact, weight, height, exact=None):
        self.path, self.contact, self.weight, self.height, self.exact = path, contact, weight, height, exact
        self.scale = np.array([weight] * 3 + [weight * height] * 3)

    def evaluated(self, control):
        """At control points `control`: (each node's torques on the leading coordinates [N, leads], scaled by
        `scale` on the root's six and weight x height on the joints'; the contact's per state; the states; the card's
        found)."""
        physics = self.contact.physics
        qpos, qvel, qacc = self.path.states(control)
        found = (self.exact or physics.card).evaluate(qpos, qvel, qacc, ("xpos", "xmat", "cdof", "subtree_com"))
        touching = self.contact.pushed(found)
        given = np.array([np.einsum("pin,pi->n", jacobians, force)
                          for force, _, _, jacobians, _, _ in touching])
        torques = np.einsum("knl,kn->kl", self.path.coupled.maps(qpos), found["needed"] - given)
        torques[:, :6] /= self.scale
        torques[:, 6:] /= self.weight * self.height
        return torques, touching, (qpos, qvel, qacc), found

    def jacobians(self, control, touching, states, found):
        """Each node's scaled torques' Jacobians on its unknowns' place, speed and acceleration (3 x [N, leads, m])."""
        physics, model, path = self.contact.physics, self.contact.physics.model, self.path
        qpos, qvel, qacc = states
        values = _spline.nodes(control)
        tangents = path.tangents(values, qpos)
        count, unknowns = len(qpos), tangents.shape[2]
        nudged = path.coupled.applied(integrated(model, np.repeat(qpos, unknowns, axis=0),
                                                 PLACE_STEP * np.swapaxes(tangents, 1, 2).reshape(-1, model.nv)))
        moved = physics.card.evaluate(nudged, np.repeat(qvel, unknowns, axis=0), np.repeat(qacc, unknowns, axis=0),
                                      ())["needed"].reshape(count, unknowns, model.nv)
        base = found["needed"] if self.exact is None else physics.card.evaluate(qpos, qvel, qacc, ())["needed"]
        by_place = np.swapaxes(moved - base[:, None], 1, 2) / PLACE_STEP  # N x nv x m
        data, by_pace = mujoco.MjData(model), np.zeros((count, model.nv, unknowns))
        inertia = np.zeros((model.nv, model.nv))
        for frame in range(count):
            data.qpos[:] = qpos[frame]
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fullM(model, data, inertia)
            by_pace[frame] = inertia @ tangents[frame]
        if self.exact is None:  # stage A's model: the joints' damping alone
            by_speed = model.dof_damping[None, :, None] * tangents
        else:  # stage B's: every velocity product term by one-sided differences on the card
            speeding = physics.card.evaluate(np.repeat(qpos, unknowns, axis=0), np.repeat(qvel, unknowns, axis=0)
                                             + SPEED_STEP * np.swapaxes(tangents, 1, 2).reshape(-1, model.nv),
                                             np.repeat(qacc, unknowns, axis=0), ())["needed"]
            by_speed = np.swapaxes(speeding.reshape(count, unknowns, model.nv) - base[:, None], 1, 2) / SPEED_STEP
        for frame, (force, by_over, by_point_speed, jacobians, ups, _) in enumerate(touching):
            stacked = jacobians.reshape(-1, model.nv)  # (p x 3) x nv
            rises = np.einsum("pi,pin->pn", ups, jacobians) @ tangents[frame]  # d height / d unknowns, p x m
            moving = (stacked @ tangents[frame]).reshape(len(jacobians), 3, unknowns)  # p x 3 x m
            by_place[frame] -= stacked.T @ (by_over[:, :, None] * rises[:, None, :]).reshape(-1, unknowns)
            by_speed[frame] -= stacked.T @ np.matmul(by_point_speed, moving).reshape(-1, unknowns)
        scale = np.concatenate([self.scale, np.full(model.nv, self.weight * self.height)])
        maps = path.coupled.maps(qpos)
        mapped = [np.einsum("knl,knm->klm", maps, kind) for kind in (by_place, by_speed, by_pace)]
        leads = mapped[0].shape[1]
        return [kind / scale[:leads][None, :, None] for kind in mapped]


def ground_forces(touching, sides, owners):
    """Each state's ground reaction force, the whole body's and each side's (states x 3, states x sides x 3, the
    person's frame, N): `sides` each side's bodies, `owners` (`Skin.owners`) the body each point's force is booked on,
    `touching` `Contact.pushed`'s."""
    whole = np.array([force.sum(axis=0) for force, *_ in touching]) @ TO_OWN.T
    each = np.array([[force[np.isin(owners[near], side)].sum(axis=0) for side in sides]
                     for force, _, _, _, _, near in touching]) @ TO_OWN.T
    return whole, each
