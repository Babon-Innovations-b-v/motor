"""The physics of one person's movement, frame by frame, on the MuJoCo model Newton builds (nothing stepped):
`Physics.needed` the generalised forces the movement needs (inverse dynamics, armature and passive forces in),
`Physics.stops` the joints at an end of their range, and `Physics.shared` the static optimisation in OpenSim's two
stages (`staged`): the ground reaction force first, from the pelvis' six balance rows alone (RRA's balance on the
ground's contacts, `contact.pushes`, then that balance shared over the touching points), then the muscles' activations
and the joints' stops with that ground reaction force fixed (least squared activation); what they cannot give is the
pelvis' residual and each joint's reserve, measured and never applied."""
import time

import mujoco
import numpy as np

from motor.batched import Card, least_all, same_layout
from motor.bone_groups import FEET
from motor.contact import Chains, lowest_of, lowests, pushes, pushes_all
from motor.ground import normals
from motor.model import Couplings, on_card
from motor.solvers import least

CONTACT_COST = 1e-3  # the solve's weight of a newton of contact force (or N m of a joint's stop) against an activation
# The weight of the pelvis' residual (a newton or N m) against a joint's reserve (N m) and in RRA's balance: the ground
# must balance the whole body first, so a movement it cannot carry shows as residual, and one the muscles cannot make as
# reserves.
BALANCE = 100.0
# The weight of a newton of contact force when the balance's wrench is shared over the touching points (`staged`'s
# second step), against the balance's rows, and the times it is asked again for what the last sharing left of the
# balance (iterated Tikhonov): the balance alone, its contacts at CONTACT_COST, sits on the edge of the contacts' cone
# whenever the body cannot be balanced, where the points carrying it are near dependent (a flat floor's friction
# pyramids), and a 1e-6 nudge of the contact columns moved its ground reaction force by up to 2.4 kN and an
# activation by 0.74 (2026-11-05, #70: the pickup's dumped frames). Shared at this weight in two rounds, on every dumped
# ground frame of the pickup, jump and env1's standing person, a 1e-6 nudge of the contact columns moves an activation
# by at most 5e-3, 3e-3 and 5e-5 (medians 7e-5, 1e-5, 6e-7; one problem, at most 1.0, 0.65 and 0.08), and the pelvis'
# residual force grows by at most 0.35, 1.8 and 0.07 N; at 0.01 the force crowded onto a few points again (0.02), at
# 0.1 in one round the residual grew by 7.9 N.
SHARE_COST = 0.03
SHARE_ROUNDS = 2
# the static optimisation's two stages, as their wall seconds are named (`staged`'s `seconds`)
STAGES = ("static optimisation 1, ground reaction", "static optimisation 2, stops and muscles")
AT_STOP = 1e-4  # radians or metres: a joint this near an end of its range stands against it
# The joints a knee's ligaments and condyles hold, both ways, at CONTACT_COST a newton metre, as a stop (#98,
# 2026-11-07): the shin's turn on the femur, freed of the knee's flexion (myo._movable's KNEE), where MyoFullBody's
# knee extensors reach it by 0.3 to 0.5 cm and no muscle by much more; left to the muscles, its few newton metres
# decided the vasti's split (at a deep squat one vastus fully active, another at 0).
LIGAMENTS = ("knee_angle_rotation3_r", "knee_angle_rotation3_l")


def parts_of(model, bodies):
    """The part each body's push is booked on ({body: name}): a foot is one part, its toes' body under its foot's
    name (`bodies`: {segment: body}, the FEET segments among them), every other body its own MuJoCo name."""
    found = {body: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) for body in range(model.nbody)}
    for segment, body in bodies.items():
        if segment.endswith(FEET):
            found[body] = segment.removesuffix("ToeBase").removesuffix("Foot") + "Foot"
    return found


def touches(contacts, each, parts):
    """Each touching part's push and where it acts ({part: (force N, centre of pressure m, normal push N, the ground
    points under its touching skin m)}, MuJoCo's frame): its points' pushes (`each`, the solve's, a row a contact)
    summed over the part (`parts`: a foot's calcaneus and toes together, `parts_of`), the centre their ground points
    under them weighed by each push along its contact's normal (its edges' sum: a friction pyramid's four edges sum
    to four times the normal); a part the solve pushes nothing along its normals is left out."""
    found = {}
    for part in sorted({parts[contact[0]] for contact in contacts}):
        mine = [number for number, contact in enumerate(contacts) if parts[contact[0]] == part]
        normals = np.array([np.sum(contacts[number][2], axis=0) for number in mine])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)  # 0 for a point at GAP, `contact.closeness`
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
        sizes = np.einsum("ij,ij->i", each[mine], normals)
        if sizes.sum() > 0:
            under = np.array([contacts[number][3] for number in mine])
            found[part] = (each[mine].sum(axis=0), sizes @ under / sizes.sum(), float(sizes.sum()), under)
    return found


class Physics:
    """One person's MuJoCo model as Newton built it, and the frame by frame share of its movement's forces."""

    def __init__(self, model, ground, skin, parts=None):
        """`parts` the part each body's push is booked on (`parts_of`; by default every body its own name)."""
        self.model, self.data = model, mujoco.MjData(model)
        self.parts = parts_of(model, {}) if parts is None else parts
        self.couplings = Couplings(model)
        self.ground, self.up = ground, normals(ground)
        self.skin = skin  # the person's drawn surface the ground pushes (`contact.Skin`)
        self.muscles = np.flatnonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
        card = on_card(model)
        if card is not None and not same_layout(model, card):
            raise ValueError("MuJoCo Warp's model of the body is not the CPU model's layout: the card reads another")
        self.card = Card(model, card) if card is not None else None
        self.chains = Chains(model) if self.card is not None else None
        self.laid = None  # the state `each` laid in `data`: its contacts, worked out with every state's at once
        self.laid_over = None  # ... and its skin points' heights over their ground
        self.row = 0  # the frame `each` laid in `data` frame by frame on MuJoCo's C (the skin's credit is a frame's)

    def each(self, path, velocities, accelerations, contacts=True):
        """Each state's needed forces (`needed`), its state laid in `data` before it is given (for `contacts`,
        `stops`, `shared` and every other read of the frame): with the model on the card every state at once
        (`batched.Card`), else frame by frame on MuJoCo's C; `contacts` False when the caller reads none (the
        ground's pushes worked out for nothing)."""
        if self.card is None:
            try:
                for row, (qpos, qvel, qacc) in enumerate(zip(path, velocities, accelerations)):
                    self.row = row
                    yield self.needed(qpos, qvel, qacc)
            finally:
                self.row = 0
            return
        found = self.card.evaluate(path, velocities, accelerations)
        touching, over = pushes_all(self.model, found, self.ground, self.up, self.skin, self.chains) if contacts else \
            (None, None)
        try:
            for row in range(len(found["needed"])):
                self.laid, self.laid_over = (touching[row], over[row]) if contacts else (None, None)
                yield self.card.load(found, row, self.data)
        finally:
            self.laid, self.laid_over = None, None

    def lowests(self, parts):
        """The lowest of the skin's points over its ground now, and of each part's ({part: its points' indices}),
        `contact.lowests`' (inside `each` on the card, read off every state's ground worked out at once)."""
        if self.laid_over is None:
            return lowests(self.model, self.data, self.skin, self.ground, parts)
        return lowest_of(self.laid_over), {part: lowest_of(self.laid_over[points]) if len(points) else None
                                           for part, points in parts.items()}

    def needed(self, qpos, qvel, qacc):
        """The state set, and the generalised forces the movement needs beyond the passive ones (nv)."""
        model, data = self.model, self.data
        data.qpos[:], data.qvel[:], data.qacc[:] = qpos, qvel, qacc
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        found = np.zeros(model.nv)
        mujoco.mj_rne(model, data, 1, found)
        return found + model.dof_armature * qacc - data.qfrc_passive

    def contacts(self):
        """The points the ground pushes now (`contact.pushes` on this person's skin and ground; inside `each` on the
        card, the laid state's, worked out with every state's at once: `contact.pushes_all`)."""
        if self.laid is not None:
            return self.laid
        return pushes(self.model, self.data, self.ground, self.up, self.skin, self.row)

    def stops(self):
        """Each leading hinge or slide at an end of its range now, as a one-sided generalised force pushing it back
        in (nv x stops): MuJoCo's joint limit, the passive structures that hold a straight knee, say."""
        model, data = self.model, self.data
        found = []
        for dof in self.couplings.leads:
            joint = model.dof_jntid[dof]
            if not model.jnt_limited[joint] or model.jnt_type[joint] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                                                              mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            at, (low, high) = data.qpos[model.jnt_qposadr[joint]], model.jnt_range[joint]
            for sign, end in ((1.0, at <= low + AT_STOP), (-1.0, at >= high - AT_STOP)):
                if end:
                    column = np.zeros(model.nv)
                    column[dof] = sign
                    found.append(column)
        return np.array(found).T if found else np.zeros((model.nv, 0))

    def ligaments(self):
        """Each leading joint of LIGAMENTS as a two-sided generalised force (nv x 2 per joint): what the knee's
        ligaments and condyles carry of its shin's turn."""
        model, found = self.model, []
        for dof in self.couplings.leads:
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.dof_jntid[dof]) or "").endswith(LIGAMENTS):
                for sign in (1.0, -1.0):
                    column = np.zeros(model.nv)
                    column[dof] = sign
                    found.append(column)
        return np.array(found).T if found else np.zeros((model.nv, 0))

    def shared(self, needed):
        """The needed forces shared out: each contact's push (N, MuJoCo's frame, by body), each touching body's push
        and its centre of pressure (`touches`: a foot one part), each muscle's activation
        and force (N, tension), the residual on the pelvis (force N and moment N m, MuJoCo's frame) and each leading
        joint's reserve and net moment (N m): the net moment its coordinate needs with its followers' (the couplings'
        map), and its own joint's alone ("own moments": the moment across that one joint, as a single joint's inverse
        dynamics gives it; the two differ only on a leader with followers, a lumbar level spread over the others)."""
        problem = self.problem(needed)
        return self.share_of(problem, self.solved_all([problem], card=False)[0])

    def solved_all(self, problems, card=True, seconds=None):
        """Each `problem`'s static optimisation solved in its two stages (`staged`, each stage's wall seconds added
        into `seconds` when given), each stage's least squares every frame's at once (`least_each`)."""
        return staged(problems, lambda asked: self.least_each(asked, card), seconds)

    def least_each(self, asked, card=True):
        """Each least squares of `asked` (system, wanted, costs, highest) solved: every one at once on the card
        (`batched.least_all`) when the model is there and `card`, else one by one (`solvers.least`)."""
        if not asked:
            return []
        return least_all(*zip(*asked)) if self.card is not None and card else [least(*one) for one in asked]

    def problem(self, needed):
        """The static optimisation of the state laid in `data` (`shared`'s), unsolved: its system (the pelvis' six rows
        first, then the leading joints'; the contacts' columns, the stops', the muscles'), wanted, costs and bounds,
        which `staged` solves, and what `share_of` reads of the state."""
        model, data = self.model, self.data
        mapped = self.couplings.map(data.qpos)
        moments = np.zeros((model.nu, model.nv))
        mujoco.mju_sparse2dense(moments, data.actuator_moment, data.moment_rownnz, data.moment_rowadr,
                                data.moment_colind)
        gains, biases = [], []
        for actuator in self.muscles:
            gains.append(mujoco.mju_muscleGain(data.actuator_length[actuator], data.actuator_velocity[actuator],
                                               model.actuator_lengthrange[actuator], model.actuator_acc0[actuator],
                                               model.actuator_gainprm[actuator, :9]))
            biases.append(mujoco.mju_muscleBias(data.actuator_length[actuator], model.actuator_lengthrange[actuator],
                                                model.actuator_acc0[actuator], model.actuator_biasprm[actuator, :9]))
        gains, biases = np.array(gains), np.array(biases)
        arms = moments[self.muscles].T  # nv x muscles
        # A muscle pulls between two of the body's own bodies: no net force or moment on the whole body, so none on
        # the root's free joint. MuJoCo's C gives 1e-11 there, the card's float32 1e-3, and the pelvis' residual,
        # BALANCE^2 dearer, then drove the activations off that rounding (2026-11-05, #70: a frame's mean 0.05 to 0.21).
        arms[model.jnt_dofadr[0]:model.jnt_dofadr[0] + 6] = 0.0
        contacts = self.contacts()
        columns = []
        for _, along, edges, _ in contacts:
            columns += [along.T @ edge for edge in edges]
        pushes = np.array(columns).T if columns else np.zeros((model.nv, 0))
        stops = np.concatenate([self.stops(), self.ligaments()], axis=1)
        wanted = mapped.T @ (needed - arms @ biases)
        system = mapped.T @ np.concatenate([pushes, stops, arms * gains], axis=1)
        count, held = pushes.shape[1], stops.shape[1]
        costs = np.concatenate([np.full(count + held, CONTACT_COST), np.ones(len(self.muscles))])
        highest = np.concatenate([np.full(count + held, np.inf), np.ones(len(self.muscles))])
        return {"costs": costs, "highest": highest, "system": system, "wanted": wanted,
                "count": count, "held": held, "contacts": contacts, "gains": gains, "biases": biases,
                "pushes": pushes, "mapped": mapped, "needed": needed,
                "root": data.xmat[model.jnt_bodyid[0]].reshape(3, 3).copy()}

    def share_of(self, problem, solved):
        """`shared`'s record of a `problem` solved as `solved` (the least squares' x)."""
        model, contacts, count, held = self.model, problem["contacts"], problem["count"], problem["held"]
        pushed, activations = solved[:count], solved[count + held:]
        left = problem["wanted"] - problem["system"] @ solved
        forces = np.zeros((model.nbody, 3))
        each = np.zeros((len(contacts), 3))
        edges = [(number, body, edge) for number, (body, _, mine, _) in enumerate(contacts) for edge in mine]
        for (number, body, edge), weight in zip(edges, pushed):
            forces[body] += weight * edge
            each[number] += weight * edge
        whole = problem["needed"] - problem["pushes"] @ pushed
        net = problem["mapped"].T @ whole
        return {"pushes": forces, "touches": touches(contacts, each, self.parts), "activations": activations,
                "forces": -(problem["gains"] * activations + problem["biases"]),
                "residual": np.concatenate([left[:3], problem["root"] @ left[3:6]]), "reserves": left[6:],
                "moments": net[6:], "own moments": whole[self.couplings.leads[6:]]}


def balance_asked(problem):
    """`staged`'s first step: the contacts' forces that best give the pelvis' six rows (RRA's balance: its rows
    BALANCE, each newton CONTACT_COST); its least squares' system, wanted, costs and bounds."""
    count = problem["count"]
    pelvis = problem["system"][:6, :count]
    return BALANCE * pelvis, BALANCE * problem["wanted"][:6], np.full(count, CONTACT_COST), np.full(count, np.inf)


def share_asked(problem, wanted):
    """`staged`'s second step: the wrench `wanted` on the pelvis' six rows (BALANCE times theirs) shared over the
    touching points at SHARE_COST a newton: the frame's ground reaction force."""
    count = problem["count"]
    return BALANCE * problem["system"][:6, :count], wanted, np.full(count, SHARE_COST), np.full(count, np.inf)


def joints_asked(problem, ground):
    """`staged`'s third step: the joints' stops and the muscles' activations giving the leading joints' rows with the
    ground reaction force `ground` (the contacts' forces) fixed; costs and bounds the problem's."""
    count = problem["count"]
    system = problem["system"][6:]
    return (system[:, count:], problem["wanted"][6:] - system[:, :count] @ ground, problem["costs"][count:],
            problem["highest"][count:])


def staged(problems, solve, seconds=None):
    """Each `problem`'s static optimisation as OpenSim's (Hicks et al. 2015): the ground reaction force first, from the
    pelvis' rows alone (`balance_asked`, then `share_asked`), then the stops and muscles with it fixed (`joints_asked`),
    least squared activation; `solve` solves a list of least squares (system, wanted, costs, highest) at once. Each
    problem's x: the contacts' forces, the stops', the activations. Solved as one problem the contacts' columns, at
    CONTACT_COST, made it ill-posed: a 1e-6 nudge of them moved an activation by up to 0.79 (2026-11-05, #70).
    `seconds`, when given, has each stage's wall seconds added (STAGES: the paper's per-step timing, #98)."""
    started = time.perf_counter()
    pelvis = [BALANCE * problem["system"][:6, :problem["count"]] for problem in problems]
    balanced = [rows @ found for rows, found in zip(pelvis, solve([balance_asked(problem) for problem in problems]))]
    wanted = balanced
    for _ in range(SHARE_ROUNDS):  # each round asks again for what the last one's sharing left of the balance
        ground = solve([share_asked(problem, one) for problem, one in zip(problems, wanted)])
        wanted = [one + target - rows @ found for one, target, rows, found in zip(wanted, balanced, pelvis, ground)]
    middle = time.perf_counter()
    rest = solve([joints_asked(problem, found) for problem, found in zip(problems, ground)])
    if seconds is not None:
        for stage, spent in zip(STAGES, (middle - started, time.perf_counter() - middle)):
            seconds[stage] = seconds.get(stage, 0.0) + spent
    return [np.concatenate([found, other]) for found, other in zip(ground, rest)]
