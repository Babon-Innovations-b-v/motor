"""A clip's level 3 on the card (#70, plan step 5's first piece): the card loop (`_solve`) set up from a fitted MuJoCo
model as `../_frames.Frames` sets mink up (the turns' and the markers' sites and costs, the couplings held, the leading
joints in their followers-free ranges, the frozen dofs), every frame one row seeded on its own (`_seeds`). The
prototype's setup (the card build's gate.py) made production; mink in `../_frames.py` stays the reference."""
import weakref
from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from ._forward import Exact
from ._objective import PlacePerProblem, TurnPerProblem
from ._project import Projection
from ._seeds import newton_order
from ._skin_step import BoneSkin
from ._solve import solved_in_chunks
from ._step import Reduction

BIG = 1e30  # a bound that never holds
DEVICE = "cuda:0"
LIMITED_TYPES = (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE))


def couplings(mjm):
    """The model's active joint couplings [(follower coord, leader coord or -1, polycoef[5])]."""
    return [(int(mjm.jnt_qposadr[mjm.eq_obj1id[index]]),
             int(mjm.jnt_qposadr[mjm.eq_obj2id[index]]) if mjm.eq_obj2id[index] >= 0 else -1,
             np.array(mjm.eq_data[index, :5], dtype=float))
            for index in range(mjm.neq)
            if int(mjm.eq_type[index]) == int(mujoco.mjtEq.mjEQ_JOINT) and mjm.eq_active0[index]]


def coordinate_bounds(limited):
    """(lower, upper) per coordinate [nq]: a limited slide's or hinge's range, else +-BIG."""
    lower, upper = np.full(limited.nq, -BIG), np.full(limited.nq, BIG)
    for joint in range(limited.njnt):
        if limited.jnt_limited[joint] and int(limited.jnt_type[joint]) in LIMITED_TYPES:
            lower[limited.jnt_qposadr[joint]], upper[limited.jnt_qposadr[joint]] = limited.jnt_range[joint]
    return lower, upper


def followers_free_ranges(mjm, followers):
    """What `coordinate_bounds` reads of `../_model.followers_free(mjm, followers)` (its joints' types, coordinates,
    ranges and limited flags, the followers' off), without its deep copy of the whole model (2.5 s of a tall case's
    card run, 2026-10-29: the meshes copied)."""
    limited = np.array(mjm.jnt_limited, copy=True)
    limited[list(followers)] = 0
    return SimpleNamespace(nq=mjm.nq, njnt=mjm.njnt, jnt_limited=limited, jnt_type=mjm.jnt_type,
                           jnt_qposadr=mjm.jnt_qposadr, jnt_range=mjm.jnt_range)


def dof_coordinates(mjm):
    """Each dof's coordinate [nv]: a free joint's six dofs all its first coordinate, a slide's or hinge's its own."""
    found = np.zeros(mjm.nv, dtype=int)
    for joint in range(mjm.njnt):
        if int(mjm.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
            found[mjm.jnt_dofadr[joint]:mjm.jnt_dofadr[joint] + 6] = mjm.jnt_qposadr[joint]
        else:
            found[mjm.jnt_dofadr[joint]] = mjm.jnt_qposadr[joint]
    return found


def reduction_of(mjm, limited, frozen):
    """The reduced dofs (`_step.Reduction`): the frozen dofs and the followers out, each follower on its leader."""
    coupled, (lower, upper), coordinate = couplings(mjm), coordinate_bounds(limited), dof_coordinates(mjm)
    dof_of = {int(coordinate[dof]): dof for dof in range(mjm.nv) if dof >= 6 or coordinate[dof] != 0}
    followers = {dof_of[follower] for follower, _, _ in coupled}
    free = [dof for dof in range(mjm.nv) if dof not in set(frozen) and dof not in followers]
    rooted = lambda dof: dof < 6 and int(mjm.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    return Reduction(free, [int(coordinate[dof]) for dof in free],
                     [-BIG if rooted(dof) else lower[coordinate[dof]] for dof in free],
                     [BIG if rooted(dof) else upper[coordinate[dof]] for dof in free],
                     [dof_of[follower] for follower, _, _ in coupled],
                     [dof_of[leader] if leader >= 0 else -1 for _, leader, _ in coupled],
                     [leader for _, leader, _ in coupled],
                     [mjm.qpos0[leader] if leader >= 0 else 0.0 for _, leader, _ in coupled],
                     np.array([poly for _, _, poly in coupled]).reshape(-1, 5), DEVICE)


def projection_of(mjm, limited, frozen):
    """The exact projection (`_project.Projection`): ranges (the root's coordinates never), followers, frozen."""
    coupled, (lower, upper), coordinate = couplings(mjm), coordinate_bounds(limited), dof_coordinates(mjm)
    lower[:7], upper[:7] = -BIG, BIG
    return Projection(list(lower), list(upper), [follower for follower, _, _ in coupled],
                      [leader for _, leader, _ in coupled], [mjm.qpos0[follower] for follower, _, _ in coupled],
                      [mjm.qpos0[leader] if leader >= 0 else 0.0 for _, leader, _ in coupled],
                      np.array([poly for _, _, poly in coupled]).reshape(-1, 5),
                      [int(coordinate[dof]) for dof in frozen], DEVICE)


def card(values, kind):
    """A float32 card array of `values` as Warp type `kind`."""
    return wp.array(np.ascontiguousarray(values, dtype=np.float32), dtype=kind, device=DEVICE)


class Clip:
    """The card loop's parts for one fitted model: `newton_model` built from the model's own MJCF (its bodies where
    the constants put them), `mjm` the MuJoCo model with the fit's sites, `frame_sites` / `place_sites` its turn and
    marker sites, `turn_cost` / `place_costs` `_frames.weights`', `frozen` dofs, `limited` the followers-free copy."""

    def __init__(self, newton_model, mjm, frame_sites, place_sites, turn_cost, place_costs, frozen, limited):
        if newton_model.joint_coord_count != mjm.nq or newton_model.joint_dof_count != mjm.nv:
            raise ValueError(f"Newton's {newton_model.joint_coord_count} coords / {newton_model.joint_dof_count} dofs"
                             f" against MuJoCo's {mjm.nq} / {mjm.nv}")
        self.model, self.mjm = newton_model, mjm
        self.frame_sites, self.place_sites = list(frame_sites), list(place_sites)
        self.turn_cost, self.place_costs = float(turn_cost), np.asarray(place_costs, dtype=float)
        self.body = {label.split("/")[-1]: index for index, label in enumerate(newton_model.body_label)}
        self.reduction = reduction_of(mjm, limited, frozen)
        self.projection = projection_of(mjm, limited, frozen)
        self._bones = None
        self._exact = None

    def bones(self):
        """The model's bone points for the skin's rows (`_skin_step.BoneSkin`), made once; None without bone meshes."""
        if self._bones is None:
            self._bones = BoneSkin(self)
        return self._bones if self._bones.count else None

    def skin_of(self, skins):
        """`solved_in_chunks`' `skin` for rows posed in `skins` (`fit._skin.PosedSkins`, a frame a row), None when
        none are given or the model has no bone meshes."""
        if skins is None or self.bones() is None:
            return None
        return self.bones(), skins

    def link(self, site):
        """The Newton body a MuJoCo site rides."""
        return self.body[self.mjm.body(int(self.mjm.site_bodyid[site])).name]

    def optimizer_for(self, turns, points):
        """optimizer_for(start, stop) for `_solve.solved_in_chunks`: Newton's IKOptimizerLM on rows start:stop."""
        from newton import ik

        def made(start, stop):
            rows = stop - start
            objectives = [TurnPerProblem(self.link(site), card(np.tile(self.mjm.site_quat[site][[1, 2, 3, 0]],
                                                                       (rows, 1)), wp.quat),
                                         card(Rotation.from_matrix(turns[start:stop, group]).as_quat(), wp.vec4),
                                         card(np.full(rows, self.turn_cost), wp.float32))
                          for group, site in enumerate(self.frame_sites)]
            objectives += [PlacePerProblem(self.link(site), card(np.tile(self.mjm.site_pos[site], (rows, 1)), wp.vec3),
                                           card(points[start:stop, mark], wp.vec3),
                                           card(np.full(rows, cost), wp.float32))
                           for mark, (site, cost) in enumerate(zip(self.place_sites, self.place_costs))]
            return ik.IKSolver(self.model, rows, objectives, optimizer="lm",
                               jacobian_mode=ik.IKJacobianType.ANALYTIC)._impl
        return made

    def exact(self):
        """The residuals' link frames from MuJoCo's own float64 model (`_forward.Exact`), made once."""
        if self._exact is None:
            self._exact = Exact(self.model, self.mjm)
        return self._exact

    def angles(self, turns, points, seeds, skins=None, exact=False):
        """Every frame's angles (frames x nq, MuJoCo's layout) and steps, each frame from its own seed [frames, nq]:
        `turns` frames x groups x 3 x 3, `points` frames x markers x 3, `skins` each frame's posed skin
        (`fit._skin.PosedSkins`; None: no skin rows); `exact`: the residuals on MuJoCo's float64 frames (`exact()`,
        level 3's, mink's own model), else on Newton's float32-built joint places."""
        seeds = np.asarray(seeds, dtype=float)
        return solved_in_chunks(self.optimizer_for(np.asarray(turns), np.asarray(points)), self.reduction,
                                self.projection, newton_order(seeds), seeds, skin=self.skin_of(skins),
                                exact=self.exact() if exact else None)


MAPPED = 1e-5  # m, the plan's site tolerance: Newton's float32 sites at rest off MuJoCo's (heavy: 0.0073 mm)


def site_miss(clip):
    """The largest distance (m) between Newton's and MuJoCo's place of the clip's sites at the model's qpos0."""
    import newton
    data = mujoco.MjData(clip.mjm)
    mujoco.mj_kinematics(clip.mjm, data)
    state = clip.model.state()
    clip.model.joint_q.assign(newton_order(clip.mjm.qpos0[None])[0].astype(np.float32))
    newton.eval_fk(clip.model, clip.model.joint_q, clip.model.joint_qd, state)
    body_q = state.body_q.numpy().astype(float)
    sites = clip.frame_sites + clip.place_sites
    placed = [Rotation.from_quat(body_q[clip.link(site), 3:7]).apply(clip.mjm.site_pos[site]) + body_q[clip.link(site), :3]
              for site in sites]
    return float(np.max(np.linalg.norm(np.array(placed) - data.site_xpos[sites], axis=1)))


def moved_inside(builder, fitting):
    """Newton's joints into `fitting`'s inside bodies moved in their parents by the fit's current joint moves
    (`Fitting.apply` writes them to MuJoCo's body_pos only, so the file at its path has them not)."""
    joints = {builder.body_label[child].split("/")[-1]: joint for joint, child in enumerate(builder.joint_child)}
    moves = fitting.model.body_pos[fitting.inside] - fitting.base[0][fitting.inside]
    for body, move in zip(fitting.inside, moves):
        joint = joints[fitting.model.body(int(body)).name]
        place = builder.joint_X_p[joint]
        builder.joint_X_p[joint] = wp.transform(wp.vec3(*(np.asarray(wp.transform_get_translation(place)) + move)),
                                                wp.transform_get_rotation(place))


def inside_joints(newton_model, fitting):
    """The Newton joint into each of `fitting.inside`'s bodies, in its order (one a body: Newton merges a body's
    several MuJoCo joints into one)."""
    labels = [label.split("/")[-1] for label in newton_model.body_label]
    into = {labels[child]: joint for joint, child in enumerate(newton_model.joint_child.numpy())}
    return np.array([into[fitting.model.body(int(body)).name] for body in fitting.inside], dtype=int)


def set_places(places, joints, held, moves):
    """A nudged set's Newton joint places ([joints, 7], translation first) from `places`, which hold the inside
    joints' moves `held`, with those joints moved by `moves` instead: no Newton build per set."""
    moved = np.array(places, dtype=np.float32, copy=True)
    moved[joints, :3] += np.asarray(moves) - np.asarray(held)
    return moved


BUILT = weakref.WeakKeyDictionary()  # {Fitting: (its file's path, Newton's model of it, the inside moves it holds)}


def newton_model_for(fitting, path, device=DEVICE):
    """Newton's model of the person's file at `path` with `fitting`'s current inside joint moves, built once per
    `Fitting` and file and kept (`BUILT`): a later call moves its inside joints to the fitting's current moves in
    place (`set_places`), as level 1's sets ask for it once for level 2's sampled frames and again, the constants laid
    on, for the clips (2026-10-31: two Newton builds a set, ~0.9 s each)."""
    import newton
    moves = fitting.model.body_pos[fitting.inside] - fitting.base[0][fitting.inside]
    kept = BUILT.get(fitting)
    if kept is not None and kept[0] == str(path):
        model, held = kept[1], kept[2]
        model.joint_X_p.assign(set_places(model.joint_X_p.numpy(), inside_joints(model, fitting), held, moves))
    else:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_mjcf(str(path), convert_mjc_equality_constraints=False, parse_meshes=False)
        moved_inside(builder, fitting)
        model = builder.finalize(device=device)
    BUILT[fitting] = (str(path), model, moves.copy())
    return model


def for_fitting(fitting, path):
    """The `Clip` of a `..._model.Fitting` compiled from the person's file at `path` with its current inside joint moves:
    Newton's model of the same file (`newton_model_for`), refused when its site places miss MuJoCo's by over MAPPED."""
    from .._frames import weights
    turn_cost, place_costs = weights(fitting)
    clip = Clip(newton_model_for(fitting, path), fitting.model, fitting.frame_sites, fitting.place_sites, turn_cost,
                place_costs, fitting.frozen, followers_free_ranges(fitting.model, fitting.followers))
    miss = site_miss(clip)
    if miss > MAPPED:
        raise ValueError(f"{path}: Newton's sites {miss * 1000:.4f} mm off MuJoCo's at rest")
    return clip
