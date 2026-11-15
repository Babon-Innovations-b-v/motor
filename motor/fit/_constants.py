"""The person's constants (#70's level 2) by scipy's `least_squares` with bounds, solved together with the angles of
the frames given as one sparse problem (the recipe of 2026-10-28: alternating mink's angles and the constants
converged linearly, where the joint problem reached machine precision in 17 evaluations on 30 frames). The unknowns
are the constants (each free marker's move on its body and each joint inside a bone group's move in its parent; the
lengths are SOMA's, from the sizing, never solved here) and each frame's leading coordinates (the root's place and its turn as a rotation
vector off mink's, each leading hinge and slide; a follower set from its leader by the model's coupling, the held
fingers as they stand); the held markers' moves stand as given. The rows are every frame's turn errors and marker
place errors, weighed as mink weighs them (`_frames.weights`, OpenSim's), then a prior row per unknown constant: a marker's move to its landmark's link at its marker's own
weight (a move and its body's place trade one for one), within `MARK_MOST`; an inside joint's move to its file's
place at the markers' mean weight, within `MARK_MOST` too (as small as a landmark's correction). Warm from mink's
angles at the start constants (Cotton et al. 2026, the idea only: cold, theirs explained the markers with a wrong pose).
The leading joints' ranges are the box bounds; the followers' own are not held (`Fitting.limited_model`)."""
import mujoco
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh, qr
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from motor.fit._frames import weights

POLISHED = 1e-10  # the free gradient a Gauss-Newton polish takes level 2's end to (scipy's trf/lsmr stalls 2.5-3 mm short, #70)
POLISH_ROUNDS = 80  # Gauss-Newton converges linearly here (about 0.64 a round): 30 rounds left 21 of 70 tuning cases at 1.5e-10 to 4.2e-9 (2026-10-31)
STALLED_ROUNDS = 8  # rounds with no new lowest free gradient after which the polish stands at float64's floor
FLOOR = 1e-8  # the free gradient a stalled polish may stand at (14 of 70 tuning cases stood at 1.5e-10 to 7.45e-10 the
# same at 30 and at 80 rounds, 2026-10-31: the residuals' float64 noise, not too few rounds)
BOUND_HELD = 1e-5  # an unknown this near its bound is held there by the polish
MARK_MOST = 0.03  # metres a marker may move on its body off its landmark's link
INSIDE = 1e-9  # a start this far inside its bounds (least_squares wants a start within them)
TOLD = 1e-8  # an inside joint's column this far, relative, from the others' span is one the data cannot tell (tall,
# 2026-10-29: the untold at 1e-15, the least told, a scapula's y, at 4e-3)
ON_CARD = True  # level 2 by `_card._level2_lm` (the owner's "yes make", 2026-11-03) wherever `wide`; proved on every tuning body (fit/CLAUDE.md)
WELL_TOLD = 1e-4  # the others' smallest singular value over their largest above which their span is whole (far over TOLD:
# the Gram matrix's eigenvalues are good to ~1e-13 of the largest; heavy_kimodo 2026-10-29 at 0.07)


class Problem:
    """The constants' problem on a `Fitting` over the frames of `path` (frames x nq, mink's, the charts' centres) with
    their targets `turns` (frames x groups x 3 x 3) and `points` (frames x markers x 3); `given` the constants vector
    the held markers' moves are read from, `free` the markers that may move (by index)."""

    def __init__(self, fitting, path, turns, points, given, free=()):
        self.fitting, self.turns, self.points = fitting, turns, points
        self.path = np.asarray(path, dtype=float)  # float32 angles (the card's) rounded every qpos the residuals read (#70)
        self.given = np.asarray(given, dtype=float)
        model = fitting.model
        frozen = set(fitting.frozen)
        self.root = model.jnt_qposadr[0] if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else None
        self.leading = [joint for joint in fitting.leaders if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
                        and model.jnt_dofadr[joint] not in frozen]
        self.width = (6 if self.root is not None else 0) + len(self.leading)
        marked = 3 * len(fitting.marks)
        self.slots = np.array([3 * mark + axis for mark in free for axis in range(3)]
                              + [marked + 3 * joint + axis for joint in range(len(fitting.inside)) for axis in range(3)],
                              dtype=int)
        self.free = list(free)
        self.costs = weights(fitting)
        self.data = mujoco.MjData(model)

    def constants(self, unknowns):
        """The whole constants vector: the given one with the unknowns laid in."""
        found = self.given.copy()
        found[self.slots] = unknowns[:len(self.slots)]
        return found

    def qpos(self, frame, values):
        """A frame's qpos from its leading coordinates."""
        qpos = self.path[frame].copy()
        start = 0
        if self.root is not None:
            at = self.root
            qpos[at:at + 3] = values[:3]
            turn = Rotation.from_quat(self.path[frame][at + 3:at + 7], scalar_first=True) * Rotation.from_rotvec(
                values[3:6])
            qpos[at + 3:at + 7] = turn.as_quat(scalar_first=True)
            start = 6
        qpos[self.fitting.model.jnt_qposadr[self.leading]] = values[start:]
        return self.fitting.coupled(qpos)

    def start(self):
        """The unknowns' start: the constants as given, then each frame's coordinates at the path's."""
        model, found = self.fitting.model, [self.given[self.slots]]
        for qpos in self.path:
            root = [qpos[self.root:self.root + 3], np.zeros(3)] if self.root is not None else []
            found.append(np.concatenate([*root, qpos[model.jnt_qposadr[self.leading]]]))
        return np.concatenate(found)

    def bounds(self):
        """The constants' bounds (within MARK_MOST), then each frame's: the leading joints' ranges (unbounded where unlimited)."""
        model = self.fitting.model
        low, high = np.full(len(self.slots), -MARK_MOST), np.full(len(self.slots), MARK_MOST)
        limited = model.jnt_limited[self.leading].astype(bool)
        joint_low = np.where(limited, model.jnt_range[self.leading, 0], -np.inf)
        joint_high = np.where(limited, model.jnt_range[self.leading, 1], np.inf)
        free = np.full(6 if self.root is not None else 0, np.inf)
        frames = len(self.path)
        return (np.concatenate([low, np.tile(np.concatenate([-free, joint_low]), frames)]),
                np.concatenate([high, np.tile(np.concatenate([free, joint_high]), frames)]))

    def exact(self, unknowns):
        """The residuals' exact Jacobian at `unknowns` on the host's MuJoCo kinematics (`_card._level2.jacobian`)."""
        from motor.fit._card._level2 import jacobian
        return jacobian(self, unknowns)

    def residuals(self, unknowns):
        """Every frame's weighed turn errors (the geodesic angle's rotation vector, mink's) and marker errors, then the
        constants' prior rows."""
        fitting, data, model = self.fitting, self.data, self.fitting.model
        constants = self.constants(unknowns)
        fitting.apply(constants)
        turn_cost, place_costs = self.costs
        held = len(self.slots)
        found = []
        for frame in range(len(self.path)):
            values = unknowns[held + frame * self.width:held + (frame + 1) * self.width]
            data.qpos[:] = self.qpos(frame, values)
            mujoco.mj_kinematics(model, data)
            seen = data.site_xmat[fitting.frame_sites].reshape(-1, 3, 3)
            off = Rotation.from_matrix(np.transpose(seen, (0, 2, 1)) @ self.turns[frame]).as_rotvec()
            found += [turn_cost * off.ravel(),
                      (place_costs[:, None] * (data.site_xpos[fitting.place_sites] - self.points[frame])).ravel()]
        moved = 3 * len(self.free)
        found += [(place_costs[self.free][:, None] * unknowns[:moved].reshape(-1, 3)).ravel(),
                  float(np.mean(place_costs)) * unknowns[moved:held]]
        return np.concatenate(found)

    def sparsity(self):
        """Which unknowns each row reads: a frame's turn rows its own coordinates; its place rows those, a marker's
        its own move and the moves of the inside joints it rides below;
        a prior row its own constant."""
        fitting, held = self.fitting, len(self.slots)
        groups, marks = len(fitting.frame_sites), len(fitting.place_sites)
        per = 3 * (groups + marks)
        frames = len(self.path)
        found = lil_matrix((frames * per + held, held + frames * self.width), dtype=int)
        for frame in range(frames):
            found[frame * per:(frame + 1) * per, held + frame * self.width:held + (frame + 1) * self.width] = 1
            placed = frame * per + 3 * groups
            for index, mark in enumerate(self.free):
                found[placed + 3 * mark:placed + 3 * mark + 3, 3 * index:3 * index + 3] = 1
            for joint, riders in enumerate(fitting.riding):
                column = 3 * (len(self.free) + joint)
                for mark in riders:
                    found[placed + 3 * mark:placed + 3 * mark + 3, column:column + 3] = 1
        for slot in range(held):
            found[frames * per + slot, slot] = 1
        return found


def told(problem, matrix):
    """The unknowns the data can tell (column indices of the whole unknowns, in order) from `matrix`, the problem's
    Jacobian at the start (#70 level 2, one rule): an inside joint's column (one axis of its move) whose part off the
    span of every other column (the markers' moves, the frames' coordinates, root place included; `off_span`) and of
    the inside columns before it in the model's order (QR without pivoting) is under TOLD of its own length is
    dropped; the constants' prior rows are left out, since they tell every column apart by construction. The model's
    order, never the pivoted QR's best-told-first: of two axes that move the markers alike (humphant_l's y and
    humphant1_l's on tall, 2026-10-31) the pivot kept whichever a 5 % change of one landmark's trust made a hair longer,
    so level 2's constants jumped by a centimetre between level 1's nudged sets and its trust columns read the jump."""
    held, marked = len(problem.slots), 3 * len(problem.free)
    data = csc_matrix(matrix)[:matrix.shape[0] - held]
    inside = np.arange(marked, held)
    others = np.setdiff1d(np.arange(data.shape[1]), inside)
    columns = data[:, inside].toarray()
    lengths = np.maximum(np.linalg.norm(columns, axis=0), 1e-300)
    _, upper = qr(off_span(data[:, others], columns) / lengths, mode="economic")
    kept = inside[np.abs(np.diag(upper)) >= TOLD] if len(inside) else inside
    return np.sort(np.concatenate([others, kept]))


def off_span(others, columns):
    """`columns` less their part in the span of the sparse `others`' columns, each scaled to unit length, as
    `qr_off_span` takes it: by the scaled columns' Gram matrix (sparse, Cholesky) when their smallest singular value
    is over WELL_TOLD of their largest, since then the pivoted QR's every diagonal is over TOLD of its first (|r_11| <=
    the largest, |r_kk| >= the smallest) and it keeps the whole span; else by that QR itself (2026-10-29: heavy_kimodo's
    11,610 x 1,350 at a ratio of 0.07, 0.2 s against the QR's 2.5 to 5.6 s, the same 58 of 96 axes kept)."""
    norms = np.sqrt(np.asarray(others.multiply(others).sum(axis=0))).ravel()
    scaled = others @ diags(1 / np.maximum(norms, 1e-300))
    gram = (scaled.T @ scaled).toarray()
    values = eigvalsh(gram)
    if len(values) and values[0] > (WELL_TOLD ** 2) * values[-1]:
        return columns - scaled @ cho_solve(cho_factor(gram), scaled.T @ columns)
    return qr_off_span(others.toarray(), columns)


def qr_off_span(others, columns):
    """`columns` less their part in the span of `others`' columns scaled to unit length, that span the column-pivoted
    QR's columns whose diagonal is over TOLD of its first (the rule's reference path)."""
    basis, upper, _ = qr(others / np.maximum(np.linalg.norm(others, axis=0), 1e-300), mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(upper))
    basis = basis[:, :int((diagonal > TOLD * diagonal[0]).sum())] if len(diagonal) else basis[:, :0]
    return columns - basis @ (basis.T @ columns)


def staged(fitting, path, turns, points, given, free=(), jacobian=None, wide=False, card=None):
    """The constants (the whole vector, as `Fitting.unpack` reads it) and the frames' angles (frames x nq) that best
    meet the frames' targets together, from `given` and mink's path; the model is left with the constants laid on,
    and the solve's evaluation count. Only the unknowns the data can tell are solved (`told`, from the exact Jacobian
    at the start in float64; an untold inside joint axis stays at its file's place). `jacobian(problem, unknowns)` (the
    exact one, `_card._level2.jacobian`) replaces scipy's 2-point differences in the problem's sparse pattern, the
    reference, when given. `wide`: the residuals and the exact Jacobian for least_squares and the polish
    from `_card._level2_wide.Wide` (every frame's float64 kinematics on the card; equal to the host's to float64's
    rounding), `jacobian` then unused; `told` reads the host's either way. `card` (default ON_CARD when `wide`; needs `wide`):
    scipy's least_squares replaced by `_card._level2_lm` (Levenberg-Marquardt, the frames eliminated on the card by
    the Schur complement), the same polish after it."""
    problem, begin, keep, bounds = prepared(fitting, path, turns, points, given, free, wide)
    if (ON_CARD and wide if card is None else card):
        if not wide:
            raise ValueError("level 2's card solve reads the card's float64 evaluator: staged(wide=True)")
        from motor.fit._card._level2_lm import Set, solved
        (unknowns,), (evaluations,) = solved([Set(problem, begin, keep, bounds)])
        return finished(problem, polished(problem, unknowns, keep, bounds), evaluations)
    low, high = bounds

    def whole(unknowns):
        """The whole unknowns: `begin` with the told ones laid in."""
        found = begin.copy()
        found[keep] = unknowns
        return found

    if wide:
        jacobian = lambda problem, unknowns: problem.exact(unknowns)  # noqa: E731
    exact = {"jac": lambda unknowns: jacobian(problem, whole(unknowns))[:, keep]} if jacobian is not None else {
        "jac_sparsity": problem.sparsity().tocsc()[:, keep]}
    # scipy's own tolerances: the polish settles (at ftol 1e-12 trf took ~400 s a real body and still stalled, #70)
    found = least_squares(lambda unknowns: problem.residuals(whole(unknowns)), begin[keep],
                          bounds=(low[keep], high[keep]), x_scale="jac", method="trf", tr_solver="lsmr", **exact)
    return finished(problem, polished(problem, whole(found.x), keep, bounds), int(found.nfev))


def staged_across(cases):
    """`staged(..., wide=True, card=True)` of several sets at once (`cases` [(fitting, path, turns, points, given,
    free)], one model): every set's frames eliminated in one card launch a round (`_card._level2_lm.solved`), each
    set polished on its own after. [(constants, angles, evaluations)] in the cases' order."""
    from motor.fit._card._level2_lm import Set, solved
    ready = [prepared(*case, True) for case in cases]
    found, evaluations = solved([Set(*one) for one in ready])
    return [finished(problem, polished(problem, unknowns, keep, bounds), count)
            for (problem, _, keep, bounds), unknowns, count in zip(ready, found, evaluations)]


def prepared(fitting, path, turns, points, given, free, wide):
    """A set's (problem, start, told columns, bounds): the problem the host's or, `wide`, the card's float64
    evaluator; the start the path's, just inside the bounds; `told` on the host's exact Jacobian there."""
    host = Problem(fitting, path, turns, points, given, free)
    low, high = host.bounds()
    begin = np.clip(host.start(), np.where(np.isfinite(low), low + INSIDE, low),
                    np.where(np.isfinite(high), high - INSIDE, high))
    # always the host's MuJoCo double kinematics: the card's floats blur TOLD, and even the card's float64 order of
    # sums flips the pivoted QR's pick between two equally told axes (tall 2026-10-31: the root's x or the thorax's,
    # 2.85 mm, the same residuals; the lumbar bones between them move)
    keep = told(host, host.exact(begin))
    if wide:
        from motor.fit._card._level2_wide import Wide
        return Wide(host), begin, keep, (low, high)
    return host, begin, keep, (low, high)


def finished(problem, unknowns, evaluations):
    """(constants, angles frames x nq, evaluations) at the solved whole `unknowns`, the constants laid on the model."""
    constants = problem.constants(unknowns)
    problem.fitting.apply(constants)
    held = len(problem.slots)
    angles = np.array([problem.qpos(frame, unknowns[held + frame * problem.width:held + (frame + 1) * problem.width])
                       for frame in range(len(problem.path))])
    return constants, angles, evaluations


def off_bounds(unknowns, keep, bounds):
    """The told unknowns (of `keep`) further than BOUND_HELD inside both their bounds."""
    low, high = bounds
    inside = (unknowns[keep] > low[keep] + BOUND_HELD) & (unknowns[keep] < high[keep] - BOUND_HELD)
    return np.asarray(keep)[inside]


def polished(problem, unknowns, keep, bounds):
    """`unknowns` moved by Gauss-Newton steps (the exact float64 Jacobian, `problem.exact`: the host's or `_card._level2_wide`'s, a sparse
    direct solve) on the told unknowns off their bounds, each step halved until the cost falls and kept within the
    bounds, until their gradient is under POLISHED, or the lowest it reached once STALLED_ROUNDS find none lower and
    that is under FLOOR; raises when POLISH_ROUNDS do neither."""
    low, high = bounds
    found = unknowns.copy()
    best, lowest, since = found, np.inf, 0
    for _round in range(POLISH_ROUNDS):
        free = off_bounds(found, keep, bounds)
        residuals = problem.residuals(found)
        matrix = problem.exact(found)[:, free].tocsc()
        gradient = matrix.T @ residuals
        if not len(free) or np.abs(gradient).max() < POLISHED:
            return found
        best, lowest, since = (found, np.abs(gradient).max(), 0) if np.abs(gradient).max() < lowest else (
            best, lowest, since + 1)
        if since >= STALLED_ROUNDS and lowest < FLOOR:
            return best
        step = spsolve((matrix.T @ matrix).tocsc(), -gradient)
        moved = descended(problem, found, free, step, (low, high))
        found = moved if moved is not found else steadied(problem, found, free, step, (low, high), gradient)
    raise ArithmeticError(f"level 2's polish left a free gradient of {np.abs(gradient).max():.2e} after {POLISH_ROUNDS} rounds")


def descended(problem, unknowns, free, step, bounds):
    """`unknowns` with `step` laid on its `free` entries, clipped into the bounds and halved until the cost falls (the
    unknowns as they stand when no halving does). A fall is strict: an equal cost is one the cost cannot see, left to
    `steadied` (heavy_kimodo, 2026-10-30: with `<=` a halving 1e-13 long at an equal cost was taken every round, so the
    polish stood at a free gradient of 1.48e-10 for its last 10 rounds and `steadied` was never asked)."""
    low, high = bounds
    cost = float(np.sum(problem.residuals(unknowns) ** 2))
    for halving in range(30):
        moved = unknowns.copy()
        moved[free] = np.clip(unknowns[free] + step / 2 ** halving, low[free], high[free])
        if float(np.sum(problem.residuals(moved) ** 2)) < cost:
            return moved
    return unknowns


def steadied(problem, unknowns, free, step, bounds, gradient):
    """`unknowns` with the whole `step` laid on its `free` entries (clipped into the bounds) when there its free
    gradient is smaller than `gradient`, else `unknowns`: the polish's test once the cost cannot tell the step's fall
    (mean_kimodo, 2026-10-30: at a free gradient of 1.59e-10 the Gauss-Newton step's predicted fall was 4.4e-17 on a
    cost of 0.867, under the residuals' own float64 noise, so every halving was refused and the polish stood still)."""
    low, high = bounds
    moved = unknowns.copy()
    moved[free] = np.clip(unknowns[free] + step, low[free], high[free])
    reached = problem.exact(moved)[:, free].T @ problem.residuals(moved)
    return moved if np.abs(reached).max() < np.abs(gradient).max() else unknowns
