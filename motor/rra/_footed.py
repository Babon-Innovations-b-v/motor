"""The feet kept where they stood by the legs (`Footing`): a move of the pelvis or trunk leaves the feet, and so the
ground's contacts, as they were. mink per frame, set up as the fit's (`fit.solved`, its solver, step and steps): each
foot body's place and turn a `FrameTask`, the pelvis' height held to its own by FOOTED_HEIGHT (a `FrameTask` on the
root's body, only its height free: mink's `PostureTask` leaves a free joint out), every leading dof but the legs' and the
root's height frozen (`DofFreezingTask`), a leg's directions neither a foot's task nor the height's sees held where
the frame stands (`FreeHold`), the couplings held exactly (`EqualityConstraintTask` as a constraint) and the legs'
leading joints in their ranges (`ConfigurationLimit`); a frame that does not converge fails the run
(`fit.NotConverged`), never kept silently. Before #70's one-solver rule (2026-10-29) this was a handwritten damped least
squares, 40 fixed steps, ranges by clipping."""
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from motor import fit

FOOTED_HEIGHT = 0.1  # the weight of a metre of the pelvis' height against one of a foot's place (or a radian of turn)
# the weight of a radian of a leg's free direction from where the frame stands (`FreeHold`) against a foot's metre or
# radian: a leg has seven joints for its foot's six directions (the ankle's twist, 2026-11-01; with the knee's shin turn
# freed, 2026-11-04, eight), so a direction moves no foot, and at the fit's damping (1e-9) a step along it is unbounded:
# on env1's seated astronaut, his left subtalar on its stop, the hip's turn and adduction, the subtalar and the ankle's
# twist swung 0.05 rad back and forth for good (#70, 2026-11-04); held, that frame converges, its feet's weighted cost
# 8.92e-9 as at damping 1e-6. Held only along what no foot and not the pelvis' height sees, it moves neither (a posture
# held on every leg joint, as mink's PostureTask holds, at 1e-2 left the toy's feet 33 micrometres off and its pelvis
# 6 mm high)
FOOTED_HOLD = 1.0
ROOT_HEIGHT = 2  # the free root's vertical dof (MuJoCo's frame is Z up)
# the footing's steps are judged (`fit.judged_solve`, Levenberg-Marquardt's control: a step that raises the weighted
# cost is turned back and damped) from this added damping, lowered ten times on every kept step and never put back to
# 0 (`decaying`): at a straight leg the undamped step bends the knee by 0.7 to 1 rad and is turned back, the fit's own
# first 1e-3 then takes 3 mm steps, and the two alternated for 300 steps (#98, 2026-11-07: a video walk,
# its knees straight at heel strike); from 1e-6 put back to 0 after each kept step, every other step was the undamped
# one turned back, and five drop jumps' near-straight legs (the shin's turn, the hip's rotation and the ankle's twist
# all but one axis) crept 300 steps at 1e-12 to 8e-7 of cost; a damping held at 1e-6 to 1e-4 instead crept on
# another frame of theirs (steps 9e-5 rad, the cost 0.7% lower a step). Lowered on
# every kept step, all their frames converge, their feet as before. Damping shapes a step, never the optimum.
FOOTED_LAM_FIRST = 1e-5
# a frame whose feet already stand within this (their places' and turns' squared offsets summed, m and rad) of where
# they stood is its own answer (`Footing.settled`): a micrometre, under the card's float32 rounding of them
FOOTED_SETTLED = 1e-12
# a kept step at no more than FOOTED_LAM_FIRST's damping that lowers the weighted cost by under this share of it ends
# the frame: a foot that cannot be reached (the legs toy's right foot, its legs turning only about one axis, under a
# pelvis turned 0.05 rad about the vertical) leaves the cost large, and the other leg's steps along its weakly seen
# turns went on 300 steps, 1e-4 rad each, its foot 25 micrometres off and the cost falling 7e-10 of itself a step
# (#98, 2026-11-07); ended, its foot stands 42 micrometres off
FOOTED_GAIN = 1e-8


class Footing:
    """mink's tasks, limits and constraints holding `feet` (body ids) on a model and its `couplings`; `footed` gives a
    path with its feet put back. Given the person's file (`path`) and a CUDA card (`fit.on_card`), every frame is
    solved at once on the card with the same tasks (`_footed_card.CardFooting`); mink stays the reference and the
    solve where there is no card."""

    def __init__(self, model, couplings, feet, path=None):
        self.model, self.feet = model, list(feet)
        ancestors = set()
        for body in self.feet:
            while body > 0:
                ancestors.add(body)
                body = model.body_parentid[body]
        legs = {ROOT_HEIGHT} | {int(dof) for dof in couplings.leads[6:] if model.dof_bodyid[dof] in ancestors}
        self.tasks = [mink.FrameTask(model.body(body).name, "body", position_cost=1.0, orientation_cost=1.0,
                                     lm_damping=fit.LM_DAMPING) for body in self.feet]
        self.root = int(model.jnt_bodyid[0])
        # mink's error is in the body's own frame; with the root's place frozen but its height, its whole place error
        # is the height's, whatever the pelvis' tilt
        self.height = mink.FrameTask(model.body(self.root).name, "body", position_cost=FOOTED_HEIGHT,
                                     orientation_cost=0.0, lm_damping=fit.LM_DAMPING)
        frozen = [int(dof) for dof in couplings.leads if int(dof) not in legs]
        # only the moving legs' ranges held: a frozen leader the low-pass carried a hair past its range end (a freed
        # girdle joint on its end, 0.01 deg, 2026-10-31) makes its freeze and its range one infeasible QP (daqp)
        unheld = [joint for joint, _, _ in couplings.followers] + [int(model.dof_jntid[dof]) for dof in frozen]
        self.limits = [mink.ConfigurationLimit(fit.followers_free(model, unheld), gain=1.0)]
        self.ranges = self.limits[0].lower.copy(), self.limits[0].upper.copy()
        self.constraints = [mink.EqualityConstraintTask(model, cost=1.0, gain=1.0)] if model.neq else []
        self.constraints.append(mink.DofFreezingTask(model, dof_indices=frozen))
        self.hold = FreeHold(model, couplings, sorted(legs), self.tasks + [self.height])
        self.configuration = mink.Configuration(model)
        self.data = mujoco.MjData(model)
        self.card = None
        if path is not None and fit.on_card():
            from motor.rra._footed_card import CardFooting
            self.card = CardFooting(model, path, couplings, self.feet, sorted(legs), frozen,
                                    (1.0, FOOTED_HEIGHT, FOOTED_HOLD))

    def footed(self, path, before):
        """`path` with each foot body put back where it stands in `before`, place and turn, frame by frame, by the
        legs' joints, a leg's free directions held where `path` stands them (`FreeHold`); the pelvis' height held to its
        own in `path` by FOOTED_HEIGHT: a pelvis raised past what a straight leg reaches comes down, or its feet would
        leave the ground (1 cm up lifted them 9 mm, 600 N of residual, 2026-10-28). Each leg joint is held within its
        range, or, on a frame where `before` stands past a range end (the low-pass carries the fitted path a little
        past a stop), within where `before` stands there: never further out, and never pulled in when nothing moved (2026-11-03: camp's walking astronaut's knee
        0.38 degrees past its 0 stop, pulled in at no move, turned the hip 18 degrees to keep the foot, past the cap
        before any of RRA's move). A frame already footed (`settled`) is its own answer, on the card as by mink."""
        path, before = np.asarray(path, dtype=float), np.asarray(before, dtype=float)
        found, going = path.copy(), np.flatnonzero(~self.settled(path, before))
        if not len(going):
            return found
        if self.card is not None:
            solved, unstopped = self.card.footed(path[going], before[going])
            if len(unstopped):  # the frames the card did not stop: mink's, the reference
                solved[unstopped] = self._mink(path[going][unstopped], before[going][unstopped])
            found[going] = solved
            return found
        found[going] = self._mink(path[going], before[going])
        return found

    def settled(self, path, before):
        """Per frame whether `path` already stands its feet where `before` does: each foot's place (m) and turn (rad)
        off it, squared and summed (its tasks' weighted cost; the pelvis' height and the hold are zero there), under
        FOOTED_SETTLED, where mink's judged steps would end at once. The card's float32 legs never come back as the
        frame's own, so such a frame is not sent there: in RRA's frame check a move lowered to nothing came back from
        the card a rounding off the fitted path (#98, 2026-11-07)."""
        return self._squared(path, before) < FOOTED_SETTLED

    def _squared(self, path, before):
        """Per frame the feet's places' and turns' squared offsets from `before`, summed over the feet."""
        found = []
        for qpos, still in zip(path, before):
            here, there = self._poses(qpos), self._poses(still)
            found.append(sum(float(np.sum((a[0] - b[0]) ** 2)) + Rotation.from_matrix(a[1].T @ b[1]).magnitude() ** 2
                             for a, b in zip(here, there)))
        return np.array(found)

    def _mink(self, path, before):
        """`footed` by mink, frame by frame."""
        found = []
        limit, (lower, upper) = self.limits[0], self.ranges
        for qpos, still in zip(path, before):
            limit.lower, limit.upper = np.minimum(lower, still), np.maximum(upper, still)
            self.data.qpos[:] = still
            mujoco.mj_kinematics(self.model, self.data)
            for task, body in zip(self.tasks, self.feet):
                task.set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(self.data.xmat[body].reshape(3, 3)), self.data.xpos[body].copy()))
            self.configuration.update(qpos)
            self.height.set_target(self.configuration.get_transform_frame_to_world(self.model.body(self.root).name,
                                                                                   "body"))
            self.hold.set_target(self.configuration)
            found.append(fit.solved(self.configuration, self.tasks + [self.height, self.hold], self.limits,
                                    self.constraints, judged_from=FOOTED_LAM_FIRST, gain=FOOTED_GAIN))
        return np.array(found)

    def off(self, path, before):
        """How far each frame's feet are from where they stand in `before`: the largest place (m) and turn (rad) over
        the feet, frames x 2."""
        found = []
        for qpos, still in zip(path, before):
            here = self._poses(qpos)
            there = self._poses(still)
            found.append([max(np.linalg.norm(a[0] - b[0]) for a, b in zip(here, there)),
                          max(Rotation.from_matrix(a[1].T @ b[1]).magnitude() for a, b in zip(here, there))])
        return np.array(found)

    def _poses(self, qpos):
        """Each foot body's world place and turn at `qpos`."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return [(self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()) for body in self.feet]


class FreeHold(mink.Task):
    """`moving` leading dofs held where a target pose stands them along the directions none of the `primary` tasks
    sees there: the null space, over those dofs carried through the couplings, of the tasks' cost-weighted Jacobian at
    the target pose (to numpy's rank tolerance), so a step it shapes moves no primary task's error to first order. The
    directions are the target pose's, fixed for its whole solve, never found again at each step: at a straight leg
    the bend of hip, knee and ankle together (and the twists along the leg) leaves the feet's rank and comes back a
    hair off it, so a projector found at each step held the bend on one step and let it go on the next, and mink's
    steps cycled between the two poses for good (#98, 2026-11-07: video drop jumps, their knees straight
    at take-off and landing, `fit.NotConverged` with the hold's error 5e-4 and 6.5e-2 on alternate steps)."""

    def __init__(self, model, couplings, moving, primary):
        super().__init__(cost=np.full(len(moving), FOOTED_HOLD))
        self.model, self.couplings, self.moving, self.primary = model, couplings, np.array(moving), primary
        self.columns = np.searchsorted(couplings.leads, self.moving)
        self.target, self.free = None, None

    def set_target(self, configuration):
        """The pose the free directions are held to, `configuration`'s (standing there, the primary tasks' targets
        set), and those directions, found there."""
        self.target = configuration.q.copy()
        self.free = self._free(configuration)

    def _free(self, configuration):
        """The projector onto the moving dofs' directions no primary task sees, at `configuration` (moving x moving)."""
        spread = self.couplings.map(configuration.q)[:, self.columns]
        seen = np.vstack([task.cost[:, None] * task.compute_jacobian(configuration) for task in self.primary]) @ spread
        _, sizes, turns = np.linalg.svd(seen)
        rank = int(np.sum(sizes > sizes.max() * max(seen.shape) * np.finfo(float).eps))
        free = turns[rank:].T
        return free @ free.T

    def compute_error(self, configuration):
        """The moving dofs' offset from the target, along the free directions."""
        offset = np.empty(self.model.nv)
        mujoco.mj_differentiatePos(self.model, offset, 1.0, self.target, configuration.q)
        return self.free @ offset[self.moving]

    def compute_jacobian(self, configuration):
        """The free directions' projector on the moving dofs' columns (moving x nv)."""
        found = np.zeros((len(self.moving), self.model.nv))
        found[:, self.moving] = self.free
        return found
