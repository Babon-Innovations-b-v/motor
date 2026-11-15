"""The person's MyoFullBody as Newton builds it (SolverMuJoCo's MuJoCo model, on MuJoCo Warp with a card; nothing stepped)
and what every later step reads off it: a body by its name (`body_index`), MuJoCo's joint couplings (`Couplings`),
the joints kept in range (`limited`), a body's pose in the person's frame (`own_pose`, TO_OWN), the default pose
(`default_pose`), each SOMA segment riding its bone along a path (`posed_path`), the path low-passed (`filtered`) and
its velocities and accelerations (`rates`), a path of MuJoCo's own compile carried onto it (`carried`), and the MuJoCo
Warp model Newton put on the card (`on_card`)."""
import contextlib

import mujoco
import newton
import numpy as np
from scipy.signal import butter, filtfilt

from motor.solvers import CUTOFF

DEVICE = "cpu"  # MuJoCo's CPU backend: bit for bit, and MuJoCo Warp's deterministic mode breaks over 60 dofs (#60)
# With a CUDA card the model is also put on it, MuJoCo Warp, `on_card`: Newton's solver not in its deterministic mode
# (its inertia solve is wrong over 60 dofs, #60, and its bound sizes MyoFullBody's records past Warp's limit); the
# stages `batched.Card` reads run in Warp's run-to-run mode on their own (`batched.deterministic`).
CARD = "cuda:0"
ON_CARD = {}  # {id(MuJoCo model): (it, MuJoCo Warp's model of it)}, the models `built` put on the card
# MuJoCo's frame (Z up, the model's own: facing -y, left +x) into the person's (Y up, facing +z).
TO_OWN = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _short_names(model, kind, count):
    """Each object's own name in a model Newton built (its label's last part)."""
    found = []
    for index in range(count):
        name = mujoco.mj_id2name(model, kind, index) or ""
        found.append(name)
    return found


def body_index(model, name):
    """A body of the model Newton built, by the MyoFullBody body's own name (Newton names it by its path)."""
    names = _short_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    found = [index for index, full in enumerate(names) if full == name or full.endswith("_" + name)]
    if len(found) != 1:
        raise ValueError(f"no single body {name} in the model Newton built: {len(found)} found")
    return found[0]


def carried(source, target, path):
    """A path of `source`'s qpos (frames x nq) as `target`'s: each joint's coordinates by its body's and its own name
    (Newton names a joint by its path, `..._<body>_<joint>`), the two models' joints one for one."""
    names = _short_names(target, mujoco.mjtObj.mjOBJ_JOINT, target.njnt)
    found = np.empty((len(path), target.nq))
    for joint in range(source.njnt):
        name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT, joint)
        body = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_BODY, source.jnt_bodyid[joint])
        matched = [index for index, full in enumerate(names) if full == name or full.endswith(f"_{body}_{name}")]
        if len(matched) != 1:
            raise ValueError(f"no single joint {name} in the target model: {len(matched)} found")
        width = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(
            int(source.jnt_type[joint]), 1)
        start, at = source.jnt_qposadr[joint], target.jnt_qposadr[matched[0]]
        found[:, at:at + width] = path[:, start:start + width]
    if source.njnt != target.njnt:
        raise ValueError(f"{source.njnt} joints carried onto a model of {target.njnt}")
    return found


# Every call that adds, attaches or deletes a body in a spec: `listed_once` forgets its kept list at each.
CHANGES_BODIES = ((mujoco.MjSpec, "attach"), (mujoco.MjSpec, "delete"), (mujoco.MjsBody, "add_body"),
                  (mujoco.MjsBody, "attach_frame"), (mujoco.MjsFrame, "add_body"), (mujoco.MjsFrame, "attach_body"),
                  (mujoco.MjsSite, "attach_body"))


@contextlib.contextmanager
def listed_once():
    """Inside it, a spec's `bodies` is listed once and kept until a body is added, attached or deleted
    (CHANGES_BODIES): Newton's SolverMuJoCo names each excluded body pair by `spec.bodies[...]`, a new list of
    every body at each read (MjSpec's own), twice a pair, and MyoFullBody's bodies without colliders exclude nearly
    every pair (#70: about 4 of the 4.3 s of its build on this PC's CPU). The same list, so the same model."""
    kept = {}
    listing = mujoco.MjSpec.bodies
    originals = [(owner, name, getattr(owner, name)) for owner, name in CHANGES_BODIES]

    def bodies(spec):
        found = kept.get(id(spec))
        if found is None or found[0] is not spec:
            found = kept[id(spec)] = (spec, listing.fget(spec))
        return found[1]

    def forgetting(call):
        def changed(*args, **kwargs):
            kept.clear()
            return call(*args, **kwargs)
        return changed
    mujoco.MjSpec.bodies = property(bodies)
    for owner, name, call in originals:
        setattr(owner, name, forgetting(call))
    try:
        yield
    finally:
        mujoco.MjSpec.bodies = listing
        for owner, name, call in originals:
            setattr(owner, name, call)


def built(path, gravity):
    """MuJoCo's model of the person's body as Newton builds it (SolverMuJoCo): no drives, nothing stepped. With a CUDA
    card Newton builds it there, its SolverMuJoCo on MuJoCo Warp, and the MuJoCo Warp model it puts on the card is kept
    (`on_card`) for every frame's state at once (`batched`); the MuJoCo model it compiles on the way is the same either
    way, MuJoCo's C (float64) for every frame's own reads. Without one, MuJoCo's CPU backend. Its spec's bodies are
    listed once (`listed_once`)."""
    import warp
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -gravity))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    # Bone meshes are drawn from MuJoCo's own model of the file; Newton is given the colliders, wraps and joints.
    builder.add_mjcf(str(path), convert_mjc_equality_constraints=False, parse_meshes=False)
    card = warp.is_cuda_available()
    with listed_once():
        solver = newton.solvers.SolverMuJoCo(builder.finalize(device=CARD if card else DEVICE),
                                             use_mujoco_cpu=not card, use_mujoco_contacts=False, integrator="euler")
    if card:
        ON_CARD[id(solver.mj_model)] = (solver.mj_model, solver.mjw_model)
    return solver.mj_model


def on_card(model):
    """MuJoCo Warp's model of a model `built` put on the card, else None."""
    kept = ON_CARD.get(id(model))
    return kept[1] if kept is not None and kept[0] is model else None


class Couplings:
    """MuJoCo's joint couplings of a model: which dofs lead, and each follower's polynomial in its leader."""

    def __init__(self, model):
        self.model = model
        self.followers = []  # (follower joint, leader joint or -1, polynomial)
        for equality in range(model.neq):
            if model.eq_type[equality] == mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[equality]:
                poly = model.eq_data[equality, :5].copy()
                # a lock (no leader) reaches Newton's model with the last joint as its leader (#60's import bug B)
                leader = int(model.eq_obj2id[equality]) if poly[1:].any() else -1
                self.followers.append((int(model.eq_obj1id[equality]), leader, poly))
        following = {model.jnt_dofadr[joint] for joint, _, _ in self.followers}
        self.leads = np.array([dof for dof in range(model.nv) if dof not in following])

    def apply(self, qpos):
        """`qpos` with every follower set from its leader."""
        model = self.model
        for joint, leader, poly in self.followers:
            at = model.jnt_qposadr[joint]
            moved = qpos[model.jnt_qposadr[leader]] - model.qpos0[model.jnt_qposadr[leader]] if leader >= 0 else 0.0
            qpos[at] = model.qpos0[at] + sum(poly[power] * moved ** power for power in range(5))
        return qpos

    def map(self, qpos):
        """d(every dof) / d(leading dof) at `qpos` (nv x leads)."""
        model = self.model
        found = np.zeros((model.nv, len(self.leads)))
        found[self.leads, np.arange(len(self.leads))] = 1.0
        column = {dof: index for index, dof in enumerate(self.leads)}
        for joint, leader, poly in self.followers:
            if leader < 0:
                continue
            moved = qpos[model.jnt_qposadr[leader]] - model.qpos0[model.jnt_qposadr[leader]]
            found[model.jnt_dofadr[joint], column[model.jnt_dofadr[leader]]] = sum(
                power * poly[power] * moved ** (power - 1) for power in range(1, 5))
        return found


def limited(model, qpos, couplings):
    """`qpos` with every limited hinge or slide inside its range, its followers set again."""
    for joint in range(model.njnt):
        if model.jnt_limited[joint] and model.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE,
                                                                   mujoco.mjtJoint.mjJNT_SLIDE):
            at = model.jnt_qposadr[joint]
            qpos[at] = np.clip(qpos[at], *model.jnt_range[joint])
    return couplings.apply(qpos)


def own_pose(model, data, body):
    """A body's frame in the person's frame (4 x 4)."""
    found = np.eye(4)
    found[:3, :3] = TO_OWN @ data.xmat[body].reshape(3, 3)
    found[:3, 3] = TO_OWN @ data.xpos[body]
    return found


def default_pose(model, couplings):
    """The model's default pose (MuJoCo's frame), its couplings applied: MyoSuite's own `stand` keyframe."""
    data = mujoco.MjData(model)
    data.qpos[:] = couplings.apply(model.qpos0.copy())
    mujoco.mj_kinematics(model, data)
    return data


def posed_path(model, path, segments, bodies, riding):
    """Each SOMA segment riding its bone, every frame of a path (frames x segments x 4 x 4, the person's frame)."""
    data, found = mujoco.MjData(model), []
    for qpos in path:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        found.append([own_pose(model, data, bodies[segment]) @ riding[segment] for segment in segments])
    return np.array(found)


def filtered(model, couplings, path, rate):
    """The path low-passed at CUTOFF before anything is differentiated (biomechanics' order: Winter 2009): every
    leading coordinate but the root's turn through a 2nd-order Butterworth run forward and back (zero lag, 4th order),
    the root's quaternion likewise, its sign kept continuous and its length one; the followers set again."""
    low, high = butter(2, CUTOFF / (rate / 2))
    if len(path) <= 3 * max(len(low), len(high)):
        return path.copy()
    found = path.copy()
    turn = found[:, 3:7]
    for frame in range(1, len(turn)):  # q and -q are one turn: keep each next to the one before
        if turn[frame] @ turn[frame - 1] < 0:
            turn[frame] *= -1
    leads = [model.jnt_qposadr[model.dof_jntid[dof]] for dof in couplings.leads[6:]]
    found[:, [0, 1, 2, *leads]] = filtfilt(low, high, found[:, [0, 1, 2, *leads]], axis=0)
    found[:, 3:7] = filtfilt(low, high, turn, axis=0)
    found[:, 3:7] /= np.linalg.norm(found[:, 3:7], axis=1, keepdims=True)
    return np.array([couplings.apply(qpos) for qpos in found])


def rates(model, path, rate):
    """The path's velocities and accelerations (frames x nv each): central differences of the filtered path."""
    count = len(path)
    velocities = np.zeros((count, model.nv))
    for frame in range(count):
        before, after = path[max(frame - 1, 0)], path[min(frame + 1, count - 1)]
        span = (min(frame + 1, count - 1) - max(frame - 1, 0)) / rate
        if span:
            mujoco.mj_differentiatePos(model, velocities[frame], span, before, after)
    return velocities, np.gradient(velocities, 1.0 / rate, axis=0)
