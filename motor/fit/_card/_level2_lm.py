"""Level 2's person constants by our own Levenberg-Marquardt with the Schur complement (#70, the card plan's step 2;
the owner's "yes make", 2026-11-03: an exception to fit's "no handwritten iterative solver" rule, `../CLAUDE.md`).

The unknowns are the told constants (`_constants.told`) and every sampled frame's leading coordinates; a frame's rows
read only its own coordinates and the constants, so the damped normal equations are an arrow: one dense block per
frame (width x width), the constants' block, and the frames' couplings to the constants. Each round, per set: the
float64 residuals and exact Jacobian (`_level2_wide.Wide`: every frame's kinematics on the card), the blocks as
sparse products on the host, Marquardt's damping (lambda times the diagonal, as scipy's x_scale="jac"), an unknown
on a bound its descent pushes past held (its row and column the identity's, its step 0); then on
the card, for every frame of every set in one launch, the frame's block factored by Warp's tile Cholesky in float64
(level 3's `_step` pattern) and solved for its gradient and its constants' columns, and the Schur complement summed
over each set's frames; one dense Cholesky for each set's constants (scipy, float64, on the host); the frames' steps
back from it. The step is clipped into the bounds and kept when the cost falls (lambda by Nielsen's rule), else
lambda grows. The host decides the stop: the free gradient under `_constants.POLISHED`, or lambda past LAMBDA_MOST,
or ROUNDS; `_constants.polished` then settles as it does after scipy's solve."""
import functools

import numpy as np
import warp as wp
from scipy.linalg import cho_factor, cho_solve

LAMBDA_START = 1e-3  # Marquardt's lambda at the start, of each unknown's own diagonal
LAMBDA_MOST = 1e12  # past it no damped step lowers the cost: the solve stands where it is, the polish takes it on
ROUNDS = 200  # damped solves a set at most (each refused step one)
DIAGONAL_LEAST = 1e-12  # a diagonal entry this small damps as this (an unknown no row reads yet)
COLUMNS = 16  # right-hand columns a block of the frames' Cholesky solves at once (shared memory: width^2 + 2 x width x COLUMNS doubles)
TILE_THREADS = 32


@functools.cache
def _frames_kernel(width: int):
    """Every frame's block factored (tile Cholesky, float64) and solved for COLUMNS of its right-hand side."""
    size, columns = wp.constant(width), wp.constant(COLUMNS)

    def solve(blocks: wp.array3d[wp.float64], sides: wp.array3d[wp.float64], found: wp.array3d[wp.float64]):
        frame, part = wp.tid()
        factor = wp.tile_cholesky(wp.tile_load(blocks[frame], shape=(size, size)))
        side = wp.tile_load(sides[frame], shape=(size, columns), offset=(0, part * columns))
        wp.tile_store(found[frame], wp.tile_cholesky_solve(factor, side), offset=(0, part * columns))

    solve.__name__ = solve.__qualname__ = f"level2_frames_cholesky_{width}"
    return wp.kernel(enable_backward=False, module="unique")(solve)


@wp.kernel(enable_backward=False)
def _schur(couplings: wp.array3d[wp.float64], solved: wp.array3d[wp.float64], frame_set: wp.array[wp.int32],
           reduced: wp.array3d[wp.float64]):
    """reduced[set, i, j] = sum over the set's frames of couplings[f]^T solved[f] (column j of `solved`):
    Hcf Hff^-1 [-g_f | Hfc], the part the Schur complement and its right-hand side take off."""
    own, first, second = wp.tid()
    total = wp.float64(0.0)
    for frame in range(frame_set.shape[0]):
        if frame_set[frame] == own:
            for row in range(couplings.shape[1]):
                total += couplings[frame, row, first] * solved[frame, row, second]
    reduced[own, first, second] = total


class Arrow:
    """One set's damped normal equations at a point, in the told unknowns' order (constants first, then each
    frame's `width` coordinates): `frames` (F x W x W), `couplings` (F x W x C, the frames' rows of Hfc), `constants`
    (C x C), `gradient` (J^T r), the free mask and the diagonal Marquardt's damping scales."""

    def __init__(self, matrix, residuals, told_constants, frames, width, free):
        jacobian = matrix.tocsc()
        constant_part, frame_part = jacobian[:, :told_constants], jacobian[:, told_constants:]
        self.count, self.width, self.told_constants = frames, width, told_constants
        self.gradient = jacobian.T @ residuals
        self.free = free
        frame_square = (frame_part.T @ frame_part).tocoo()
        own = frame_square.row // width == frame_square.col // width
        self.frames = np.zeros((frames, width, width))
        self.frames[frame_square.row[own] // width, frame_square.row[own] % width,
                    frame_square.col[own] % width] = frame_square.data[own]
        self.couplings = (frame_part.T @ constant_part).toarray().reshape(frames, width, told_constants)
        self.constants = (constant_part.T @ constant_part).toarray()
        self.diagonal = np.maximum(np.concatenate([np.diagonal(self.constants),
                                                   np.diagonal(self.frames, axis1=1, axis2=2).ravel()]),
                                   DIAGONAL_LEAST)

    def damped(self, lam):
        """(frames, couplings, constants, gradient) with lambda times the diagonal added and the held unknowns'
        rows and columns made the identity's, their gradient 0."""
        damping = lam * self.diagonal
        told, width, count = self.told_constants, self.width, self.count
        held_constants, held_frames = ~self.free[:told], ~self.free[told:].reshape(count, width)
        constants = self.constants + np.diag(damping[:told])
        constants[held_constants, :] = 0.0
        constants[:, held_constants] = 0.0
        constants[held_constants, held_constants] = 1.0
        frames = self.frames.copy()
        rows = np.arange(width)
        frames[:, rows, rows] += damping[told:].reshape(count, width)
        frames = np.where(held_frames[:, :, None] | held_frames[:, None, :], 0.0, frames)
        frame, place = np.nonzero(held_frames)
        frames[frame, place, place] = 1.0
        couplings = np.where(held_frames[:, :, None] | held_constants[None, None, :], 0.0, self.couplings)
        gradient = np.where(self.free, self.gradient, 0.0)
        return frames, couplings, constants, gradient


def steps(arrows, lams):
    """Each set's damped Gauss-Newton step (the told unknowns' order) for its `Arrow` and lambda, every frame of every
    set eliminated on the card in one launch, then each set's constants by one dense Cholesky."""
    width = arrows[0].width
    if any(arrow.width != width for arrow in arrows):
        raise ValueError("level 2's card solve: every set's frames must be one width (one model)")
    told_most = max(arrow.told_constants for arrow in arrows)
    padded = -(-(1 + told_most) // COLUMNS) * COLUMNS
    damped = [arrow.damped(lam) for arrow, lam in zip(arrows, lams)]
    blocks, sides, couplings, frame_set = [], [], [], []
    for number, (arrow, (frames, coupling, _, gradient)) in enumerate(zip(arrows, damped)):
        side = np.zeros((arrow.count, width, padded))
        side[:, :, 0] = -gradient[arrow.told_constants:].reshape(arrow.count, width)
        side[:, :, 1:1 + arrow.told_constants] = coupling
        wide_coupling = np.zeros((arrow.count, width, padded))
        wide_coupling[:, :, :arrow.told_constants] = coupling
        blocks.append(frames)
        sides.append(side)
        couplings.append(wide_coupling)
        frame_set += [number] * arrow.count
    blocks, sides = np.concatenate(blocks), np.concatenate(sides)
    on_card = wp.array(sides, dtype=wp.float64)
    found = wp.zeros_like(on_card)
    wp.launch_tiled(_frames_kernel(width), dim=[len(blocks), padded // COLUMNS],
                    inputs=[wp.array(np.ascontiguousarray(blocks), dtype=wp.float64), on_card], outputs=[found],
                    block_dim=TILE_THREADS)
    reduced = wp.zeros((len(arrows), padded, padded), dtype=wp.float64)
    wp.launch(_schur, dim=(len(arrows), padded, padded),
              inputs=[wp.array(np.concatenate(couplings), dtype=wp.float64), found,
                      wp.array(np.array(frame_set, dtype=np.int32), dtype=wp.int32)],
              outputs=[reduced])
    solved, reduced = found.numpy(), reduced.numpy()
    found_steps, first = [], 0
    for number, (arrow, (_, _, constants, gradient)) in enumerate(zip(arrows, damped)):
        told = arrow.told_constants
        own = solved[first:first + arrow.count]
        first += arrow.count
        schur = constants - reduced[number, :told, 1:1 + told]
        side = -gradient[:told] - reduced[number, :told, 0]
        constant_step = cho_solve(cho_factor(schur), side)
        frame_step = own[:, :, 0] - own[:, :, 1:1 + told] @ constant_step
        found_steps.append(np.concatenate([constant_step, frame_step.ravel()]))
    return found_steps


class Set:
    """One set's solve on its told unknowns: `problem` (a `Wide`), `begin` (the whole unknowns), `keep` (told
    columns), `bounds` (the whole unknowns' (low, high)); its point, residuals, cost, lambda and Arrow."""

    def __init__(self, problem, begin, keep, bounds):
        self.problem, self.whole, self.keep = problem, begin.copy(), np.asarray(keep)
        held = len(problem.slots)
        self.told_constants = int((self.keep < held).sum())
        frames = self.keep[self.told_constants:]
        if not np.array_equal(frames, np.arange(held, held + len(problem.path) * problem.width)):
            raise ValueError("level 2's card solve: every frame coordinate must be told")
        self.low, self.high = bounds[0][self.keep], bounds[1][self.keep]
        self.residuals = problem.residuals(self.whole)
        self.cost = float(self.residuals @ self.residuals)
        self.lam, self.nu, self.evaluations, self.rounds = LAMBDA_START, 2.0, 1, 0
        self.arrow, self.matrix, self.stop, self.gradient = None, None, None, None

    def point(self):
        """The told unknowns now."""
        return self.whole[self.keep]

    def linearised(self):
        """The Arrow at the point (the exact Jacobian's told columns), its free mask: an unknown on a bound its
        descent (-gradient) pushes past is held; one near it moves on, its step clipped onto it (2026-11-03: held
        within `_constants.BOUND_HELD` of it, the constants stood short of their bound and the real bodies' cost 1.1e-8
        of itself over scipy's)."""
        self.matrix = self.problem.exact(self.whole)[:, self.keep].tocsr()
        gradient = self.matrix.T @ self.residuals
        at = self.point()
        held = ((at <= self.low) & (gradient > 0)) | ((at >= self.high) & (gradient < 0))
        self.arrow = Arrow(self.matrix, self.residuals, self.told_constants, len(self.problem.path),
                           self.problem.width, ~held)

    def settled(self):
        """The host's stop, its reason kept in `stop`: the free gradient under POLISHED ("settled"), lambda past
        LAMBDA_MOST ("stalled") or ROUNDS taken ("rounds")."""
        from motor.fit._constants import POLISHED
        free_gradient = np.abs(self.arrow.gradient[self.arrow.free])
        self.gradient = float(free_gradient.max()) if free_gradient.size else 0.0
        self.stop = ("settled" if self.gradient < POLISHED else "stalled" if self.lam > LAMBDA_MOST
                     else "rounds" if self.rounds >= ROUNDS else None)
        return self.stop is not None

    def tried(self, step):
        """The step clipped into the bounds, kept when the cost falls (lambda by Nielsen's rule from the gain
        ratio), else lambda raised; the Arrow made again at a new point."""
        self.rounds += 1
        moved = np.clip(self.point() + step, self.low, self.high)
        whole = self.whole.copy()
        whole[self.keep] = moved
        residuals = self.problem.residuals(whole)
        self.evaluations += 1
        cost = float(residuals @ residuals)
        model = self.residuals + self.matrix @ (moved - self.point())
        predicted = self.cost - float(model @ model)
        if cost < self.cost and predicted > 0:
            gain = (self.cost - cost) / predicted
            self.whole, self.residuals, self.cost = whole, residuals, cost
            self.lam *= max(1 / 3, 1 - (2 * gain - 1) ** 3)
            self.nu = 2.0
            self.linearised()
        else:
            self.lam *= self.nu
            self.nu *= 2.0


def solved(sets):
    """Each `Set`'s whole unknowns at its stop, all sets' frames eliminated together each round; and each set's
    residual evaluation count (each Set keeps its `stop`, `gradient` and `rounds`)."""
    for one in sets:
        one.linearised()
    while True:
        active = [one for one in sets if not one.settled()]
        if not active:
            return [one.whole for one in sets], [one.evaluations for one in sets]
        for one, step in zip(active, steps([one.arrow for one in active], [one.lam for one in active])):
            one.tried(step)
