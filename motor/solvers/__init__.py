"""The sim body's numerical solvers and filters (pure numpy and scipy): the bounded least squares the static
optimisation shares the forces with (`least`), the least squares over a product of balls RRA's capped move is
(`within_balls`), the path's low-pass (`smoothed`, CUTOFF) and the low-pass and the central differences as matrices
(`filter_matrix`, `stencils`)."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, filtfilt

CUTOFF = 6.0  # Hz (Winter, Biomechanics and Motor Control of Human Movement, 4th ed., 2009: 6 Hz for gait)
SOLVE_ROUNDS = 1000
SOLVE_TOLERANCE = 1e-9  # the solve's projected gradient against its largest linear term
CAP_INSIDE = 0.95  # the solve's balls this far inside the cap, for its linearisation; `within_frames` checks the rest
ADMM_ROUNDS = 3000
ADMM_TOLERANCE = 1e-3
ADMM_RIDGE = 1e-8  # of the mean curvature: a low-passed move leaves its high frequencies free


def least(system, wanted, costs, highest):
    """The x in [0, highest] least in |system x - wanted|^2 + |costs x|^2: a bounded quadratic, solved by projected
    Newton steps (Bertsekas 1982): the bounds that hold the gradient back stay fixed, a Newton step on the rest (its
    inverse by Woodbury's identity, the system having far fewer rows than columns), the step halved until the cost
    falls; done when the projected gradient is under SOLVE_TOLERANCE of the largest linear term."""
    squares = costs ** 2
    linear = system.T @ wanted
    lowest = np.zeros(len(costs))
    enough = SOLVE_TOLERANCE * max(1.0, float(np.abs(linear).max(initial=0.0)))

    def cost(x):
        left = system @ x - wanted
        return 0.5 * (left @ left + squares @ x ** 2)

    x = np.zeros(len(costs))
    for _ in range(SOLVE_ROUNDS):
        gradient = system.T @ (system @ x) + squares * x - linear
        projected = np.abs(x - np.clip(x - gradient, lowest, highest)).max(initial=0.0)
        if projected <= enough:
            break
        near = min(projected, 1e-6)  # Bertsekas' epsilon: a bound this near, pushed on, is held
        held = ((x <= lowest + near) & (gradient > 0)) | ((x >= highest - near) & (gradient < 0))
        free = np.flatnonzero(~held)
        part = system[:, free] / squares[free]
        inner = np.eye(len(system)) + part @ system[:, free].T
        step = np.zeros(len(x))
        step[free] = -(gradient[free] / squares[free] - part.T @ np.linalg.solve(inner, part @ gradient[free]))
        size, before = 1.0, cost(x)
        while True:
            moved = np.clip(x + size * step, lowest, highest)
            if cost(moved) <= before + 1e-4 * gradient @ (moved - x) or size < 1e-10:
                break
            size /= 2
        x = moved
    return x


def within_balls(system, wanted, rows, values):
    """The y least in |system y - wanted|^2 with every ball (`rows` y + `values`, groups of three) of norm at most
    CAP_INSIDE: a convex quadratic over a product of balls, solved by ADMM (Boyd et al. 2011, section 5), its penalty
    balanced on the primal and dual residuals; the y and how far its furthest ball is over one."""
    hessian, linear = system.T @ system, system.T @ wanted
    hessian += ADMM_RIDGE * np.trace(hessian) / len(hessian) * np.eye(len(hessian))  # the filter's null directions
    rho = float(np.trace(hessian) / max(np.trace(rows.T @ rows), 1e-12))
    gram = rows.T @ rows
    y = np.zeros(system.shape[1])
    z = (rows @ y + values).reshape(-1, 3)
    u = np.zeros_like(z)
    factor = cho_factor(hessian + rho * gram)
    for round_ in range(ADMM_ROUNDS):
        target = linear + rho * rows.T @ (z - u).reshape(-1) - rho * rows.T @ values
        y = cho_solve(factor, target)
        reached = (rows @ y + values).reshape(-1, 3)
        before = z
        norms = np.linalg.norm(reached + u, axis=1, keepdims=True)
        z = (reached + u) * np.minimum(1.0, CAP_INSIDE / np.maximum(norms, 1e-12))
        u += reached - z
        primal = np.linalg.norm(reached - z)
        dual = rho * np.linalg.norm(rows.T @ (z - before).reshape(-1))
        if primal < ADMM_TOLERANCE * np.sqrt(len(z)) and dual < ADMM_TOLERANCE * np.sqrt(len(y)):
            break
        if round_ % 25 == 24 and (primal > 10 * dual or dual > 10 * primal):
            change = 2.0 if primal > dual else 0.5
            rho, u = rho * change, u / change
            factor = cho_factor(hessian + rho * gram)
    return y, float(np.linalg.norm((rows @ y + values).reshape(-1, 3), axis=1).max())


def smoothed(moves, rate):
    """A move low-passed as the path is (`model.filtered`: CUTOFF, zero lag), so it adds no shake to the movement."""
    low, high = butter(2, CUTOFF / (rate / 2))
    return filtfilt(low, high, moves, axis=0) if len(moves) > 3 * max(len(low), len(high)) else moves


def filter_matrix(count, rate):
    """`smoothed` as a matrix on a per-frame value (count x count): the low-pass is linear, so a move made as this
    matrix times any values is low-passed by construction."""
    return smoothed(np.eye(count), rate)


def stencils(count, rate):
    """The path's speeds and accelerations as `model.rates` takes them, as matrices on a per-frame value (count x
    count)."""
    speeds = np.zeros((count, count))
    for frame in range(count):
        before, after = max(frame - 1, 0), min(frame + 1, count - 1)
        if after > before:
            speeds[frame, after] += rate / (after - before)
            speeds[frame, before] -= rate / (after - before)
    return speeds, np.gradient(speeds, 1.0 / rate, axis=0)
