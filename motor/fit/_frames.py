"""Every frame's joint angles by mink (github.com/kevinzakka/mink, Apache-2.0, pinned 1.3.0 in pyproject.toml), MuJoCo's
inverse kinematics: per frame one quadratic program a step, run to convergence (an offline fit, no real-time rate), the
bone groups' turns (orientation-only `FrameTask`s on the calibration sites, the main term), the landmarks' markers
(position-only `FrameTask`s on the marker sites) weighed as OpenSim's IK weighs
its goals (each goal's squared errors over its own count: `weights`), the model's couplings held exactly
(`EqualityConstraintTask` as a constraint, OpenSim's default), the leading joints kept in their ranges
(`ConfigurationLimit`), the joints no target sees frozen (`DofFreezingTask`), and given the frames' posed skins the
bones held inside them (`SkinTask`: each kept bone vertex's depth outside, `_skin_rows`, #70 2026-11-01). Each frame
starts from the one before."""
import mink
import mujoco
import numpy as np

from motor.fit._skin import SkinMeshes, card_device
from motor.fit._skin_rows import BonePoints

SOLVER = "daqp"  # the QP solver every mink example uses; pinned with mink
STEP_DONE = 1e-8  # a step this small (its tangent norm, or its change to the weighted errors) ends a frame's fit
MOST_STEPS = 300  # a frame not converged by then fails the run: never kept silently
LM_DAMPING = 1e-2  # mink's Levenberg-Marquardt term on each task: shapes a step, never moves the optimum
DAMPING = 1e-9
# level 1's global balance (#70): the markers' goal's (the landmarks' and the group ends' together) total weight against
# the turns' goal's 1. OpenSim's IK weighs its goals equally, so 1: the turns stay the main term (#70) unless level 1's
# least squares over many bodies, judged on following SOMA, finds otherwise; never set by hand to pass a score
MARKER_GOAL = 2.1169
# poses whose skin measure a `SkinTask` keeps (#70 speed, 2026-11-05): a judged step asks its error twice and its QP
# objective once at one pose, and a turned-back one asks again at the kept pose, ~3.3 closest-point launches a step
# (pickup: 5,438 for 1,660 skin steps, 35 s of level 3's 50 on an L4); the kept pose and the trial are the two asked
SEEN = 2


class NotConverged(RuntimeError):
    """A frame whose fit did not converge in MOST_STEPS."""


def weighted_errors(configuration, tasks):
    """Every task's and constraint's error times its cost, stacked: what of the objective a step can change."""
    return np.concatenate([task.cost * task.compute_error(configuration) for task in tasks])


def solved(configuration, tasks, limits, constraints, refresh=None, judged_from=None, gain=None):
    """mink's steps on `configuration` (from where it stands) until a step is under STEP_DONE, its own norm or the
    change it makes to the tasks' and constraints' weighted errors (a step along a direction no task sees, a gimbal's
    two cancelling joints at the shoulder's zero elevation, never ends by its own norm: #70's arms_raise): its qpos;
    `refresh`, when given, sets the targets again before each step. With a `SkinTask`, or given `judged_from` (the
    judged steps' first added damping, lowered on every kept step: `judged_solve`'s `decaying`, with its `gain`), the
    steps are judged (`judged_solve`). Not converged in MOST_STEPS raises NotConverged."""
    if judged_from is not None:
        return judged_solve(configuration, tasks, limits, constraints, refresh, judged_from, decaying=True, gain=gain)
    if any(isinstance(task, SkinTask) for task in tasks):
        return judged_solve(configuration, tasks, limits, constraints, refresh)
    seen = tasks + [constraint for constraint in constraints if isinstance(constraint, mink.EqualityConstraintTask)]
    for _ in range(MOST_STEPS):
        if refresh is not None:
            refresh()
        before = weighted_errors(configuration, seen)
        velocity = mink.solve_ik(configuration, tasks, 1.0, SOLVER, damping=DAMPING, limits=limits,
                                 constraints=constraints)
        configuration.integrate_inplace(velocity, 1.0)
        moved = np.linalg.norm(weighted_errors(configuration, seen) - before)
        if np.linalg.norm(velocity) < STEP_DONE or moved < STEP_DONE:
            return configuration.q.copy()
    raise NotConverged(f"a frame's fit did not converge in {MOST_STEPS} steps (its last step "
                       f"{np.linalg.norm(velocity)})")


LAM_FIRST, LAM_RAISE, LAM_MOST = 1e-3, 10.0, 1e8  # the judged steps' added damping (`fit._card._solve`'s)


def judged_solve(configuration, tasks, limits, constraints, refresh=None, first=LAM_FIRST, decaying=False,
                 gain=None):
    """`solved` with each step judged, Levenberg-Marquardt's control (the skin's depth bends where its closest point
    jumps: plain steps cycled for good, #70 2026-11-01): a step whose tasks' weighted cost is not under the kept pose's
    is turned back and the added damping (on mink's `damping`) raised LAM_RAISE times from `first` (LAM_FIRST), a
    kept one lowers it (to 0 under `first`; `decaying`, LAM_RAISE times, never put back to 0: Levenberg-Marquardt's own
    rule) and ends the fit when it, or its change to the weighted errors, is under STEP_DONE, or, given `gain` and
    taken at no more than `first`, it lowered the cost by under `gain` of it (a target no pose reaches leaves the cost
    large, and Gauss-Newton's steps along what the reached targets see only weakly go on for good); a turned-back fit
    ends on a step under STEP_DONE or past LAM_MOST (no step lowers its cost). The card's (`fit._card._solve`) the same
    rule."""
    seen = tasks + [constraint for constraint in constraints if isinstance(constraint, mink.EqualityConstraintTask)]
    kept, lam, cost = configuration.q.copy(), 0.0, np.inf
    before = weighted_errors(configuration, seen)
    for _ in range(MOST_STEPS):
        if refresh is not None:
            refresh()
        errors = weighted_errors(configuration, seen)
        trial = float(np.sum(weighted_errors(configuration, tasks) ** 2))
        if trial < cost:
            moved = np.linalg.norm(errors - before) if np.isfinite(cost) else np.inf
            flat = gain is not None and lam <= first and cost - trial < gain * trial
            kept, cost, before = configuration.q.copy(), trial, errors
            lam = lam / LAM_RAISE if decaying or lam > first else 0.0
            if moved < STEP_DONE or flat:
                return kept
        else:
            configuration.update(kept)
            if refresh is not None:
                refresh()
            lam = max(lam * LAM_RAISE, first)
            if lam > LAM_MOST:
                return kept
        velocity = mink.solve_ik(configuration, tasks, 1.0, SOLVER, damping=DAMPING + lam, limits=limits,
                                 constraints=constraints)
        if np.linalg.norm(velocity) < STEP_DONE:
            return kept
        configuration.integrate_inplace(velocity, 1.0)
    raise NotConverged(f"a frame's fit did not converge in {MOST_STEPS} steps (its last step "
                       f"{np.linalg.norm(velocity)})")


def weights(fitting):
    """Each turn task's cost and each marker's: OpenSim's IK normalises each goal by its own weight sum (Simbody's
    OrientationSensors and Markers goals), here two goals, the turns (every weight 1, as OpenSense's) and the markers
    (each weight its landmark's trust, `landmarks.trust`; SOMA's own joints are no goal: they are the measure); a mink
    cost multiplies the error, so a cost is the square root of its weight over its goal's sum; the markers' goal (the
    landmarks and the group ends, one goal) sums to the fitting's own `marker_goal`, the turns' to 1."""
    trusts = np.asarray(fitting.trusts, dtype=float)
    places = np.sqrt(fitting.marker_goal * trusts / max(trusts.sum(), 1e-12))
    return 1.0 / np.sqrt(max(len(fitting.frame_sites), 1)), places


class SkinTask(mink.Task):
    """The bones inside the frame's posed skin as a mink task (`_skin_rows.BonePoints`): its error each kept bone
    vertex's cost times its depth outside (what the stop rule and the judged steps read), its QP objective the three
    offset rows a point (the offset's Jacobian the projector off the skin's closest feature, so the skin's bend at
    its edges and corners is in the step) with mink's LM damping on their squared sum; exact, SOMA does not move with
    the pose."""

    def __init__(self, model):
        self.points = BonePoints(model)
        super().__init__(cost=np.ones(self.points.count), gain=1.0, lm_damping=LM_DAMPING)
        self.meshes, self.device, self.seen = None, card_device(), {}

    def set_skin(self, skin):
        """The frame's posed skin (`_skin.PosedSkins` of one frame) as the target."""
        self.meshes, self.seen = SkinMeshes(skin, self.device), {}

    def _apart(self, configuration):
        """The points (world), their offsets off the skin and those offsets' projectors at `configuration`: measured
        once for points of the same bytes among the last SEEN (the kernel is deterministic, so the same answer)."""
        placed = self.points.placed(configuration.data)
        key = placed.tobytes()
        if key not in self.seen:
            measured = self.meshes.apart(placed, np.zeros(len(placed), dtype=np.int32))
            for array in measured:
                array.flags.writeable = False
            if not SEEN:
                return (placed, *measured)
            while len(self.seen) >= SEEN:
                self.seen.pop(next(iter(self.seen)))  # the longest not asked
        else:
            measured = self.seen.pop(key)  # asked again: the last asked, so the newest kept
        self.seen[key] = measured
        return (placed, *measured)

    def compute_error(self, configuration):
        return self.points.depths(self._apart(configuration)[1])

    def compute_jacobian(self, configuration):
        """The depth rows' Jacobian: each outside point's unit off the skin through its offset's rows."""
        placed, apart, bends = self._apart(configuration)
        found = np.zeros((self.points.count, configuration.model.nv))
        outside, rows = self.points.nv_rows(configuration.model, configuration.data, placed, apart, bends)
        units = apart[outside] / np.linalg.norm(apart[outside], axis=1)[:, None]
        found[outside] = np.einsum("pi,pij->pj", units, rows)
        return found

    def compute_qp_residual(self, configuration):
        return None  # the offset rows' objective, not the depth rows' (`compute_qp_objective`)

    def compute_qp_objective(self, configuration):
        _, apart, bends = self._apart(configuration)
        square, pull = self.points.objective(configuration.model, configuration.data, apart, bends)
        damping = self.lm_damping * float(np.sum(self.points.depths(apart) ** 2))
        return mink.tasks.task.Objective(square + damping * np.eye(configuration.model.nv), pull)


class Frames:
    """mink's tasks, limits and constraints on a `Fitting`; `fitted` gives the clip's angles (frames x nq)."""

    def __init__(self, fitting):
        self.fitting = fitting
        model = fitting.model
        turn_cost, place_costs = weights(fitting)
        self.turns = [mink.FrameTask(site, "site", position_cost=0.0, orientation_cost=turn_cost,
                                     lm_damping=LM_DAMPING) for site in fitting.frame_sites]
        self.marks = [mink.FrameTask(site, "site", position_cost=cost, orientation_cost=0.0, lm_damping=LM_DAMPING)
                      for site, cost in zip(fitting.place_sites, place_costs)]
        self.limits = [mink.ConfigurationLimit(fitting.limited_model(), gain=1.0)]
        self.constraints = [mink.EqualityConstraintTask(model, cost=1.0, gain=1.0)] if model.neq else []
        if fitting.frozen:
            self.constraints.append(mink.DofFreezingTask(model, dof_indices=fitting.frozen))
        self.configuration = mink.Configuration(model)
        self.skin = None

    def _targets(self, turns, points):
        """The frame's targets set: each group site's world turn (MuJoCo's frame); each marker's place, its target
        turn its site's own, so mink's SE(3) error is the plain place error (its V^-1 term vanishes)."""
        for task, turn in zip(self.turns, turns):
            task.set_target(mink.SE3.from_rotation(mink.SO3.from_matrix(turn)))
        self._marked(points)

    def _marked(self, points):
        """Each place task's target: its point, turned as its site turns now."""
        configuration = self.configuration
        for task, point in zip(self.marks, points):
            turn = configuration.get_transform_frame_to_world(task.frame_name, "site").rotation()
            task.set_target(mink.SE3.from_rotation_and_translation(turn, point))

    def frame(self, qpos, turns, points, skin=None):
        """One frame's angles (qpos) from `qpos`, to convergence; `skin` the frame's posed skin (`_skin.PosedSkins`
        of one frame) holds the bones inside it (`SkinTask`)."""
        self.configuration.update(qpos)
        self._targets(turns, points)
        tasks = self.turns + self.marks
        if skin is not None:
            self.skin = self.skin or SkinTask(self.fitting.model)
            self.skin.set_skin(skin)
            tasks = tasks + ([self.skin] if self.skin.points.count else [])
        return solved(self.configuration, tasks, self.limits, self.constraints, lambda: self._marked(points))

    def start(self, turns, points):
        """The first frame's start: the standing pose, its root turned so the first group's site turns as its target
        and moved so the markers' mean is the targets'."""
        model, data = self.fitting.model, mujoco.MjData(self.fitting.model)
        qpos = self.fitting.standing()
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        site = self.fitting.frame_sites[0]
        now = data.site_xmat[site].reshape(3, 3)
        turn = turns[0] @ now.T @ data.xmat[model.jnt_bodyid[0]].reshape(3, 3)
        mujoco.mju_mat2Quat(qpos[3:7], turn.reshape(-1))
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        qpos[:3] += np.mean(points, axis=0) - np.mean(data.site_xpos[self.fitting.place_sites], axis=0)
        return qpos

    def fitted(self, turns, points, start=None, skins=None):
        """Every frame's angles (frames x nq): `turns` frames x groups x 3 x 3 and `points` frames x markers x 3, both
        MuJoCo's frame, `skins` each frame's posed skin (`_skin.PosedSkins`, or None: no skin rows); each frame from
        the one before, the first from `start` (else `start()`), its skin rows from its own fit without them: the
        skin pulls a bone to the nearest skin, whichever part's, so from the standing start it pulled env1's seated
        astronaut's forearms to his thighs' skin (arms 23 to 43 degrees off, 2026-11-01), and from the frame before
        it pulled a fast-swinging arm onto the hips' or chest's skin and held it there (running's wrist 94 to 168 mm
        off SOMA's, kimodo's 309, 2026-11-04): every frame's, not the first's only."""
        qpos = self.start(turns[0], points[0]) if start is None else start
        found = []
        for frame, (frame_turns, frame_points) in enumerate(zip(turns, points)):
            qpos = self.frame(qpos, frame_turns, frame_points)
            if skins is not None:
                qpos = self.frame(qpos, frame_turns, frame_points, skins[frame:frame + 1])
            found.append(qpos)
        return np.array(found)
